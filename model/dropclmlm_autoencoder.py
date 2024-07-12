from model.classifier import LinearClassifier
from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .model_utils import (
    _make_seq_first, 
    _make_batch_first, 
    _get_padding_mask, 
    _get_key_padding_mask, 
    _get_group_mask
)

class CADEmbedding(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""
    def __init__(self, cfg, seq_len, use_group=False, group_len=None):
        super().__init__()


        n_commands = cfg.n_commands + 1  # +1 for MLM
        args_dim = cfg.args_dim + 1 + 1  # +1 for MLM
        self.command_embed = nn.Embedding(n_commands, cfg.d_model)
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)

        # use_group: additional embedding for each sketch-extrusion pair
        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len+2)

    def forward(self, commands, args, groups=None):
        S, N = commands.shape

        src = self.command_embed(commands.long()) + \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))  # shift due to -1 PAD_VAL

        if self.use_group:
            src = src + self.group_embed(groups.long())

        src = self.pos_encoding(src)

        return src


class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        self.d_model = cfg.d_model
        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        seq_len = cfg.max_total_len
        self.use_group = cfg.use_group_emb
        self.embedding = CADEmbedding(cfg, seq_len, use_group=self.use_group)

        encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

    def forward(self, commands, args):
        padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(commands, seq_dim=0)
        group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None

        src = self.embedding(commands, args, group_mask)

        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask)

        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True) # (1, N, dim_z)
        return z


class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * args_dim)

    def forward(self, out):
        S, N, _ = out.shape

        command_logits = self.command_fcn(out)  # Shape [S, N, n_commands]

        args_logits = self.args_fcn(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return command_logits, args_logits


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        self.embedding = ConstEmbedding(cfg, cfg.max_total_len)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        args_dim = cfg.args_dim + 1
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)

    def forward(self, z):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)

        command_logits, args_logits = self.fcn(out)

        out_logits = (command_logits, args_logits)
        return out_logits


class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_z),
            # nn.Tanh()
        )

    def forward(self, z):
        return self.bottleneck(z)
    
class ProjectionLayer(nn.Module):
    def __init__(self, cfg):
        super(ProjectionLayer, self).__init__()

        self.proj = nn.Linear(cfg.dim_z, cfg.dim_z)
        
    def forward(self, z):
        return self.proj(z)

class ProjectionHeadMultipleLayered(nn.Module):
    def __init__(self, cfg):
        super(ProjectionHeadMultipleLayered, self).__init__()

        self.n_layers = cfg.n_phead_layers
        self.linear_proj = nn.ModuleList([nn.Linear(cfg.dim_z, cfg.dim_z) for _ in range(self.n_layers)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(self.n_layers - 1)] + [nn.Identity()])
        # self.proj = nn.Linear(cfg.dim_z, cfg.dim_z)
        
    def forward(self, z):
        for i in range(self.n_layers):
            z = self.activations[i](self.linear_proj[i](z))
        return z

class DropCLCADContrastiveDropoutTransformer(nn.Module):
    def __init__(self, cfg):
        super(DropCLCADContrastiveDropoutTransformer, self).__init__()
        
        self.cfg = cfg
        self.args_dim = cfg.args_dim + 1
        self.encoder = Encoder(cfg)
        self.bottleneck = Bottleneck(cfg)
        if cfg.phead_type == 'legacy':
            self.proj = ProjectionLayer(cfg)
        else:
            self.proj = ProjectionHeadMultipleLayered(cfg)
        self.decoder = Decoder(cfg)
        self.dropout = nn.Dropout(cfg.latent_dropout)
        self.tanh = nn.Tanh()
        if cfg.include_mlm:
            self.mlm_classifier = LinearClassifier(cfg)

    def forward(
        self, 
        commands_enc1, 
        args_enc1,
        command_masked=None,
        args_masked=None,
        z=None, 
        return_tgt=True, 
        encode_mode=False,
    ):
        commands_enc1_, args_enc1_, command_masked, args_masked = \
            _make_seq_first(commands_enc1, args_enc1, command_masked, args_masked)  # Possibly None, None

        if z is None:
            _z = self.encoder(commands_enc1_, args_enc1_)
            # _z = self.bottleneck(_z)
            
            z = self.proj(_z)
            
            proj_z1 = self.dropout(z) 
            proj_z2 = self.dropout(z) 
            z = self.tanh(z)
        else:
            proj_z1, proj_z2 = None, None
            z = _make_seq_first(z)

        if encode_mode: 
            return _make_batch_first(z)
        
        out_logits = self.decoder(z)
        out_logits = _make_batch_first(*out_logits)

        if command_masked is not None:
            _z_masked = self.encoder(command_masked, args_masked)
            # out_logits_mlm = self.decoder(_z_masked)
            _z_masked = _make_batch_first(_z_masked)
            out_logits_mlm = self.mlm_classifier(_z_masked)
            # out_logits_mlm = _make_batch_first(*out_logits_mlm)

        res = {
            "command_logits": out_logits[0],
            "args_logits": out_logits[1],
            "proj_z1": proj_z1,
            "proj_z2": proj_z2
        }
        if command_masked is not None:
            res['command_logits_mlm'] = out_logits_mlm[0]
            res['args_logits_mlm'] = out_logits_mlm[1]
        if return_tgt:
            res["tgt_commands"] = commands_enc1_
            res["tgt_args"] = args_enc1_
        
        return res
