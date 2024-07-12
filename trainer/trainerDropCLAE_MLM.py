import torch
import torch.optim as optim
from tqdm import tqdm
from model.dropclmlm_autoencoder import DropCLCADContrastiveDropoutTransformer
from .base import BaseTrainer
from .cl_loss import *
from .loss import CadMLMLoss
from .scheduler import GradualWarmupScheduler
from cadlib.macro import *
from torch.cuda.amp import GradScaler


class TrainerDropoutCLAE_MLM(BaseTrainer):
    def build_net(self, cfg):
        self.net = DropCLCADContrastiveDropoutTransformer(cfg).cuda()

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)
        # self.scheduler = None
        self.scaler = GradScaler(enabled=self.cfg.fp16)

    def set_loss_function(self):
        self.device = 'cuda'
        self.loss_func = CADContrastiveLoss(self.cfg, self.device, self.cfg.batch_size, self.cfg.temperature).to(self.device)
        self.mlm_loss = CadMLMLoss(self.cfg).cuda()

    def forward(self, data):
        commands = data['command'].cuda() # (N, S)
        args = data['args'].cuda()  # (N, S, N_ARGS)
        if self.cfg.is_train:
            command_masked = data['command_masked'].cuda() # (N, S)
            args_masked = data['args_masked'].cuda()  # (N, S, N_ARGS)
        else:
            command_masked, args_masked = None, None
        # commands2 = data['command1'].cuda() # (N, S)
        # args2 = data['args1'].cuda()  # (N, S, N_ARGS)
        
        # print("commands.shape: ", commands.shape)
        # print("args.shape: ", args.shape)
        # print("commands2.shape: ", commands2.shape)
        # print("args2.shape: ", args2.shape)

        outputs = self.net.forward(commands, args, command_masked, args_masked)
        # outputs = self.net(commands, args, commands2.detach(), args2.detach())
        loss_dict = self.loss_func(outputs)
        if self.cfg.is_train:
            mlm_loss_dict = self.mlm_loss(outputs['command_logits_mlm'], outputs['args_logits_mlm'], data)
            loss_dict['loss_cmd_mlm'] = mlm_loss_dict['loss_cmd']
            loss_dict['loss_args_mlm'] = mlm_loss_dict['loss_args']

        return outputs, loss_dict

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
                command_masked = data['command_masked'].cuda() # (N, S)
                args_masked = data['args_masked'].cuda()  # (N, S, N_ARGS)
                outputs = self.net(commands, args, command_masked, args_masked)
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
