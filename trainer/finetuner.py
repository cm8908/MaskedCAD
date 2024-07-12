import os
import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from cadlib.macro import *
from model.autoencoder import CADDecoder, CADEncoder, CADTransformer, Encoder
from model.classifier import LinearClassifier
from model.my_skex_encoders import SkexEncoder
from trainer import TrainerCL
from trainer.loss import CADLoss
from trainer.scheduler import GradualWarmupScheduler
from torch import optim
from torch.cuda.amp import GradScaler

from utils.file_utils import ensure_dirs

class EmptyScheduler:
    def step(self):
        pass
    def state_dict(self):
        return {}
    def load_state_dict(self, none):
        pass

CLASSIFIERS = {
    'transformer_decoder': CADDecoder,
    'linear': LinearClassifier
}

class Finetuner(TrainerCL):
    def build_net(self, cfg):
        if cfg.encoder_type == 'DeepCAD':
            self.net = CADEncoder(cfg).cuda()
        elif cfg.encoder_type == 'SkexGen':
            self.net = SkexEncoder(cfg).cuda()
        
        for param in self.net.parameters():
            param.requires_grad = False
            
        self.classifier = CLASSIFIERS[cfg.classifier_type](cfg).cuda()
        
        self.ft_model_dir = os.path.join(self.model_dir, 'finetuned_'+cfg.classifier_type)
        if cfg.tag:
            self.ft_model_dir += '_'+cfg.tag
        ensure_dirs(self.ft_model_dir)
        
    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        if not cfg.is_test:
            self.optimizer = optim.Adam(self.classifier.parameters(), cfg.ft_lr)
            self.scheduler = EmptyScheduler()
            # self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)
            self.scaler = None
    
    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
    
    def load_classifier(self, name=None):
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.ft_model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            try:
                load_path = os.path.join(os.path.join(self.model_dir, 'finetuned'), "{}.pth".format(name))
            except:
                raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading classifier checkpoint from {} ...".format(load_path))
        model_state_dict = checkpoint['classifier_state_dicts']
        if isinstance(self.net, nn.DataParallel):
            self.classifier.module.load_state_dict(model_state_dict)
        else:
            self.classifier.load_state_dict(model_state_dict)
            
    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.ft_model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.ft_model_dir, "{}.pth".format(name))

        if isinstance(self.net, nn.DataParallel):
            model_state_dict = self.classifier.module.cpu().state_dict()
        else:
            model_state_dict = self.classifier.cpu().state_dict()

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'classifier_state_dicts': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)

        self.net.cuda()
        self.classifier.cuda()
        
    def decode(self, z):
        cmd_logits, args_logits = self.classifier(z)
        output = {
            'command_logits': cmd_logits,
            'args_logits': args_logits
        }
        return output
    
    def forward(self, data):
        commands = data['command'].cuda()
        args = data['args'].cuda()

        kwargs = {}
        if self.cfg.encoder_type == 'SkexGen':
            kwargs['epoch'] = self.clock.epoch

        with torch.no_grad():
            output = self.net(commands, args, **kwargs)  # (N, S, D)
        cmd_logits, args_logits = self.classifier(output['representation'])

        output = {
            'tgt_commands': commands,
            'tgt_args': args,
            'command_logits': cmd_logits,
            'args_logits': args_logits
        }
        losses = self.loss_func(output)
        return output, losses
    
    def train_func(self, data):
        """one step of training"""
        self.classifier.train()

        outputs, losses = self.forward(data)

        self.update_network(losses)
        if self.clock.step % 10 == 0:
            self.record_losses(losses, 'finetune')

        return outputs, losses
    
    def set_loss_function(self):
        self.loss_func = CADLoss(self.cfg).cuda()
    
    def evaluate(self, test_loader):
        self.net.eval()
        self.classifier.eval()

        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                commands = data['command'].cuda()
                args = data['args'].cuda()
                output = self.net(commands, args)
                _, args_logits = self.classifier(output['representation'])
                args_logits = args_logits = args_logits.reshape(*args.shape, -1)
                out_args = torch.argmax(torch.softmax(args_logits, dim=-1), dim=-1) - 1
                out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)

            gt_commands = commands.squeeze(1).long().detach().cpu().numpy() # (N, S)
            gt_args = args.squeeze(1).long().detach().cpu().numpy() # (N, S, n_args)

            ext_pos = np.where(gt_commands == EXT_IDX)
            line_pos = np.where(gt_commands == LINE_IDX)
            arc_pos = np.where(gt_commands == ARC_IDX)
            circle_pos = np.where(gt_commands == CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(int)
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
