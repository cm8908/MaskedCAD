import torch
import torch.optim as optim
from tqdm import tqdm
from model.clmlm_autoencoder import CLMLM_CADTransformer
from .base import BaseTrainer
from .loss import CadCLMLMLoss, contrastive_loss
from .scheduler import GradualWarmupScheduler
from cadlib.macro import *
from torch.cuda.amp import GradScaler


class TrainerCLMLM(BaseTrainer):
    def build_net(self, cfg):
        self.net = CLMLM_CADTransformer(cfg).cuda()

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)
        # self.scheduler = None
        self.scaler = GradScaler(enabled=self.cfg.fp16)

    def set_loss_function(self):
        self.device = 'cuda'
        self.mlm_loss_func = CadCLMLMLoss(self.cfg).to(self.device)
        self.cont_loss_func = contrastive_loss

    def forward(self, data):
        commands1 = data['command'].cuda() # (N, S)
        args1 = data['args'].cuda()  # (N, S, N_ARGS)
        if 'mask_labels' not in data.keys():
            assert not self.net.training
            outputs = self.net(commands1, args1)
            return outputs, None
        mask_labels1 = data['mask_labels'].cuda()
        commands2 = data['command2'].cuda() # (N, S)
        args2 = data['args2'].cuda()  # (N, S, N_ARGS)
        mask_labels2 = data['mask_labels2'].cuda()
        # commands2 = data['command1'].cuda() # (N, S)
        # args2 = data['args1'].cuda()  # (N, S, N_ARGS)
        
        # print("commands.shape: ", commands.shape)
        # print("args.shape: ", args.shape)
        # print("commands2.shape: ", commands2.shape)
        # print("args2.shape: ", args2.shape)

        outputs1 = self.net(commands1, args1)
        loss_dict1 = self.mlm_loss_func(outputs1['command_logits'], outputs1['args_logits'], mask_labels1, data)

        outputs2 = self.net(commands2, args2)
        loss_dict2 = self.mlm_loss_func(outputs2['command_logits'], outputs2['args_logits'], mask_labels2, data)
        
        loss_dict = {}
        loss_dict['mlm_loss_cmd_1'] = loss_dict1['loss_cmd']
        loss_dict['mlm_loss_args_1'] = loss_dict1['loss_args']
        loss_dict['mlm_loss_cmd_2'] = loss_dict2['loss_cmd']
        loss_dict['mlm_loss_args_2'] = loss_dict2['loss_args']
        if not self.cfg.no_cl:
            loss_dict['cont_loss'] = self.cont_loss_func(outputs1['representation'], outputs2['representation'], self.cfg)

        return None, loss_dict

    def encode(self, data, is_batch=False):
        """encode into latent vectors"""
        commands = data['command'].cuda()
        args = data['args'].cuda()
        if not is_batch:
            commands = commands.unsqueeze(0)
            args = args.unsqueeze(0)
        z = self.net(commands, args, encode_mode=True)
        return z

    def decode(self, z):
        """decode given latent vectors"""
        outputs = self.net(None, None, z=z, return_tgt=False)
        return outputs

    def logits2vec(self, outputs, refill_pad=True, to_numpy=True):
        """network outputs (logits) to final CAD vector"""
        out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)  # (N, S)
        out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
        if refill_pad: # fill all unused element to -1
            mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
            out_args[mask] = -1

        out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
        if to_numpy:
            out_cad_vec = out_cad_vec.detach().cpu().numpy()
        return out_cad_vec

    def evaluate(self, test_loader):
        """evaluatinon during training"""
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                commands = data['command'].cuda() # (N, S)
                args = data['args'].cuda()  # (N, S, N_ARGS)
                outputs = self.net(commands, args)
                out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1
                out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)

            gt_commands = commands.squeeze(1).long().detach().cpu().numpy() # (N, S)
            gt_args = args.squeeze(1).long().detach().cpu().numpy() # (N, S, n_args)

            ext_pos = np.where(gt_commands == EXT_IDX)
            line_pos = np.where(gt_commands == LINE_IDX)
            arc_pos = np.where(gt_commands == ARC_IDX)
            circle_pos = np.where(gt_commands == CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(np.int)
            all_ext_args_comp.append(args_comp[ext_pos][:, -N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :N_ARGS_PLANE])
        sket_trans_acc = np.mean(all_ext_args_comp[:, N_ARGS_PLANE:N_ARGS_PLANE+N_ARGS_TRANS])
        extent_one_acc = np.mean(all_ext_args_comp[:, -N_ARGS_EXT_PARAM])
        line_acc = np.mean(np.concatenate(all_line_args_comp, axis=0))
        arc_acc = np.mean(np.concatenate(all_arc_args_comp, axis=0))
        circle_acc = np.mean(np.concatenate(all_circle_args_comp, axis=0))

        self.val_tb.add_scalars("args_acc",
                                {"line": line_acc, "arc": arc_acc, "circle": circle_acc,
                                 "plane": sket_plane_acc, "trans": sket_trans_acc, "extent": extent_one_acc},
                                global_step=self.clock.epoch)
