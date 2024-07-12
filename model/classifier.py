from torch import nn
import torch

class LinearClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.args_dim = cfg.args_dim + 1
        self.n_args = cfg.n_args
        self.n_commands = cfg.n_commands
        self.max_total_len = cfg.max_total_len
        num_classes = self.max_total_len * (self.n_commands + self.n_args * self.args_dim)
        self.classifier = nn.Linear(cfg.dim_z, num_classes)
    def forward(self, z):
        if z.size(1) > 1:
            z = z.mean(1, keepdim=True)
        assert z.size(1) == 1 and len(z.shape) == 3, f'Size issue: {z.shape}'
        logits = self.classifier(z)  # (N, 1, num_classes)
        logits = logits.squeeze(1).reshape(logits.size(0), self.max_total_len, -1)  # (N, max_total_len, n_commands+n_args*args_dim)
        cmd_logits, args_logits = torch.split(logits, [self.n_commands, self.n_args*self.args_dim], dim=-1)
        args_logits = args_logits.reshape(*logits.shape[:2], self.n_args, self.args_dim)
        return cmd_logits, args_logits
        
