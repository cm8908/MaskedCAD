import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import _get_padding_mask, _get_visibility_mask
from cadlib.macro import CMD_ARGS_MASK


class CADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights

        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

    def forward(self, output):
        # Target & predictions
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        command_logits, args_logits = output["command_logits"], output["args_logits"]

        mask = self.cmd_args_mask[tgt_commands.long()]

        loss_cmd = F.cross_entropy(command_logits[padding_mask.bool()].reshape(-1, self.n_commands), tgt_commands[padding_mask.bool()].reshape(-1).long())
        loss_args = F.cross_entropy(args_logits[mask.bool()].reshape(-1, self.args_dim), tgt_args[mask.bool()].reshape(-1).long() + 1)  # shift due to -1 PAD_VAL

        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args

        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return res

class CadMLMLoss(CADLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.weights['loss_cmd_weight'] = cfg.loss_weights['loss_cmd_weight_mlm']
        self.weights['loss_args_weight'] = cfg.loss_weights['loss_args_weight_mlm']
        
    def forward(self, cmd_logits, args_logits, data):
        """
        cmd_logits: (batch_size, seq_len, n_commands)
        args_logits: (batch_size, seq_len, n_args)
        """
        mask_labels = data['mask_labels'].cuda()  # (N, S)
        tgt_cmd, tgt_args = data['tgt_cmd'].cuda(), data['tgt_args'].cuda()  # (N, S), (N, S, N_ARGS)
        
        args_mask = self.cmd_args_mask[tgt_cmd.long()] * mask_labels.unsqueeze(-1)
        
        # TEST
        # len_command = tgt_cmd[0][tgt_cmd[0] != 3].shape[0]
        # print(mask_labels[0])
        # print(args_mask[0].long())
        # print('-'*30)

        loss_cmd = F.cross_entropy(cmd_logits[mask_labels.bool()].reshape(-1, self.n_commands), tgt_cmd[mask_labels.bool()].reshape(-1).long())
        loss_args = F.cross_entropy(args_logits[args_mask.bool()].reshape(-1, self.args_dim), tgt_args[args_mask.bool()].reshape(-1).long() + 1)  # shift due to -1 PAD_VAL

        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args

        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return res

class CadCLMLMLoss(CADLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def forward(self, cmd_logits, args_logits, mask_labels, data):
        """
        cmd_logits: (batch_size, seq_len, n_commands)
        args_logits: (batch_size, seq_len, n_args)
        """
        tgt_cmd, tgt_args = data['tgt_cmd'].cuda(), data['tgt_args'].cuda()  # (N, S), (N, S, N_ARGS)
        
        args_mask = self.cmd_args_mask[tgt_cmd.long()] * mask_labels.unsqueeze(-1)
        
        # TEST
        # len_command = tgt_cmd[0][tgt_cmd[0] != 3].shape[0]
        # print(mask_labels[0])
        # print(args_mask[0].long())
        # print('-'*30)

        loss_cmd = F.cross_entropy(cmd_logits[mask_labels.bool()].reshape(-1, self.n_commands), tgt_cmd[mask_labels.bool()].reshape(-1).long())
        loss_args = F.cross_entropy(args_logits[args_mask.bool()].reshape(-1, self.args_dim), tgt_args[args_mask.bool()].reshape(-1).long() + 1)  # shift due to -1 PAD_VAL

        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args

        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return res
    
def contrastive_loss(z1, z2, cfg):
    z1 = F.normalize(z1.mean(1))  # (N, D)
    z2 = F.normalize(z2.mean(1))  # (N, D)

    batch_size = z1.size(0)
    labels = F.one_hot(torch.arange(batch_size), batch_size * 2).float().cuda()
    masks = F.one_hot(torch.arange(batch_size), batch_size).cuda()
    
    logits_aa = torch.matmul(z1, z1.T) / cfg.temperature
    logits_aa = logits_aa - masks * 1e9
    logits_bb = torch.matmul(z2, z2.T) / cfg.temperature
    logits_bb = logits_bb - masks * 1e9
    logits_ab = torch.matmul(z1, z2.T) / cfg.temperature
    logits_ba = torch.matmul(z2, z1.T) / cfg.temperature

    loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
    loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
    loss = (loss_a + loss_b).mean()
    return loss
    
def sup_con_loss(z1, z2, cfg, labels=None, mask=None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    assert z1.size(1) == z2.size(1) == 1
    features = torch.cat(
        [z1.mean(1, keepdim=True),  # (N, 1, D)
        z2.mean(1, keepdim=True)],  # (N, 1, D)
    dim=1
    )  # (N, 2, D) where 2=n_views
    device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))
    
    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]  # 2
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (2N, D)
    if cfg.contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif cfg.contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(cfg.contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        cfg.temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (cfg.temperature / cfg.base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss

def info_nce_loss(z1, z2, cfg):
    features = torch.cat(
        [z1.mean(1),  # (N, D)
         z2.mean(1)],  # (N, D)
        dim=0
    )  # (2N, D)
    labels = torch.cat([torch.arange(features.size(0)//2) for _ in range(2)], dim=0)  # (2N) 
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda()

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)  # (2N, 2N)

    mask = torch.eye(labels.size(0), dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.size(0), -1)  # (2N, 2N-1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.size(0), -1)  # (2N, 2N-1)
    
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # (2N, 1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # (2N, 2N-2)

    logits = torch.cat([positives, negatives], dim=1)  # (2N, 2N-1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()  # (2N)

    logits = logits / cfg.temperature
    return logits, labels

