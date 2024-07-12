import torch.nn as nn
import torch
import torch.nn.functional as F

from model.model_utils import _get_key_padding_mask
from .layers.transformer import *
from .layers.improved_transformer import *

PIX_PAD = 4
CMD_PAD = 3
COORD_PAD = 4
EXT_PAD = 1
EXTRA_PAD = 1
R_PAD = 2
NUM_FLAG = 9 
INITIAL_PASS = 25


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) 
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon


    def forward(self, inputs):
      """
      inputs: (S, N, E)
      Returns
        loss: (S, N, E)
        quantized: (S, N, E)
        encoding_flat: (S, N, num_code)
        encoding_indices: (S*N, 1)
      """        
      seqlen, bs = inputs.shape[0], inputs.shape[1]
        
      # Flatten input
      flat_input = inputs.reshape(-1, self._embedding_dim)  # (S*N, E)
        
      # Calculate distances  
      distances = (torch.sum(flat_input**2, dim=1, keepdim=True)  # (S*N, 1)
                  + torch.sum(self._embedding.weight**2, dim=1)  # (num_code,)
                  - 2 * torch.matmul(flat_input, self._embedding.weight.t()))  # (S*N, num_code)
      # (S*N, num_code)

      # Encoding
      encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (S*N, 1)
      encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)  # (S*N, num_code)
      encodings.scatter_(1, encoding_indices, 1)  # (S*N, num_code) set 1 to `encoding_indices`
        
      # Quantize and unflatten
      quantized = torch.matmul(encodings, self._embedding.weight).reshape(seqlen, bs, self._embedding_dim)  # (S, N, E)

      encodings_flat = encodings.reshape(inputs.shape[0], inputs.shape[1], -1)  # (S, N, num_code)
        
      # Use EMA to update the embedding vectors
      if self.training:
          self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                    (1 - self._decay) * torch.sum(encodings, 0)
            
          # Laplace smoothing of the cluster size
          n = torch.sum(self._ema_cluster_size.data)
          self._ema_cluster_size = (
              (self._ema_cluster_size + self._epsilon)
              / (n + self._num_embeddings * self._epsilon) * n)
            
          dw = torch.matmul(encodings.t(), flat_input)
          self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
          self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
      # Loss
      e_latent_loss = F.mse_loss(quantized.detach(), inputs)
      loss = self._commitment_cost * e_latent_loss  # (S, N, E)
        
      # Straight Through Estimator
      quantized = inputs + (quantized - inputs).detach() 
      # convert quantized from BHWC -> BCHW
      return loss, quantized.contiguous(), encodings_flat, encoding_indices


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


class CMDEncoder(nn.Module):

  def __init__(self,
               cfg,
               code_len = 4,
              #  max_len = 80,
              #  num_code = 500,
               ):
    """Initializes Encoder Model.

    Args:
    """
    super(CMDEncoder, self).__init__()
    self.embed_dim = cfg.d_model
    self.dropout = cfg.dropout
    self.max_len = cfg.max_total_len

    self.code_len = code_len
    self.const_embed = Embedder(self.code_len, self.embed_dim)

    self.c_embed = Embedder(cfg.n_commands, self.embed_dim)
    self.pos_embed = PositionalEncoding(d_model=self.embed_dim, max_len=self.max_len+self.code_len)
   
    # Transformer encoder
    encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=cfg.n_heads, 
                                             dim_feedforward=cfg.dim_feedforward, dropout=self.dropout)
    encoder_norm = LayerNorm(self.embed_dim)
    self.encoder = TransformerEncoder(encoder_layers, cfg.n_layers, encoder_norm)
    
    commitment_cost = 0.25
    decay = 0.99
    self.codebook_dim = 128
    self.vq_vae = VectorQuantizerEMA(cfg.num_code, self.codebook_dim, commitment_cost, decay) 
    self.down = nn.Linear(self.embed_dim, self.codebook_dim)
    self.up = nn.Linear(self.codebook_dim, self.embed_dim)
    

  def forward(self, command, mask, epoch):
    """ 
    shape(command) = (N, S)
    shape(mask) = (N, S)
    num_code = 4 for commands by default
    Returns:
      quantized_up: (N, 4, E)
      vq_loss: Number 0.0 | (4, N, E_down)
      selection: (4*N, E_down)
    """ 
    bs, seq_len = command.shape[0], command.shape[1]

    # Command embedding 
    c_embeds = self.c_embed(command.flatten()).view(bs, seq_len, -1)   # (N, S, E)

    embeddings = c_embeds.transpose(0,1)  # (S, N, E)
    z_embed = self.const_embed(torch.arange(0, self.code_len).long().cuda()).unsqueeze(1).repeat(1, bs, 1)  # (4, N, E)
    embed_input = torch.cat([z_embed, embeddings], dim=0)  # (S+4, N, E)
    encoder_input = self.pos_embed(embed_input)   # (S+4, N, E)

    # Pass through transformer encoder
    if mask is not None:
        mask = torch.cat([(torch.zeros([bs, self.code_len])==1).cuda(), mask], axis=1)  # (N, S+4) mask out except for first `num_code` tokens
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=mask)  # (S+4, N, E)
    z_encoded = outputs[0:self.code_len]  # (4, N, E) 

    if epoch < INITIAL_PASS:
      vq_loss = 0.0 
      selection = None 
      quantized_up = self.up(self.down(z_encoded)).transpose(0,1)  # (N, 4, E)
    else:
      vq_loss, quantized, _, selection = self.vq_vae(self.down(z_encoded))  # (4, N, E_down), '', (4*N, E_down)
      quantized_up = self.up(quantized).transpose(0,1)  # (N, 4, E)
    return quantized_up, vq_loss, selection 


  def get_code(self, command, mask, return_np=True):
    """ extracting codes
    shape(command) = (N, S)
    shape(mask) = (N, S)
    num_code = 4 for commands by default
    Return:
      labels: (N, 4)
    """
    bs, seq_len = command.shape[0], command.shape[1]

    # Command embedding 
    c_embeds = self.c_embed(command.flatten()).view(bs, seq_len, -1)  # (N, S, E) 

    embeddings = c_embeds.transpose(0,1)  # (S, N, E)
    z_embed = self.const_embed(torch.arange(0, self.code_len).long().cuda()).unsqueeze(1).repeat(1, bs, 1)   # (4, N, E)
    embed_input = torch.cat([z_embed, embeddings], dim=0)  # (S+4, N, E)
    encoder_input = self.pos_embed(embed_input)   # (S+4, N, E)

    # Pass through transformer encoder
    if mask is not None:
        mask = torch.cat([(torch.zeros([bs, self.code_len])==1).cuda(), mask], axis=1)  # (N, S+4)
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=mask)  
    z_encoded = outputs[0:self.code_len]  # (4, N, E)

    _, _, one_hot, _ = self.vq_vae(self.down(z_encoded))  # [seqlen, bs, one-hot] = (4, N, num_code)
    labels = torch.argmax(one_hot, dim=2).transpose(0,1)  # [bs, seqlen] = (N, 4)
    if return_np:
      return labels.detach().cpu().numpy().astype(int)
    else:
      return labels


class PARAMEncoder(nn.Module):

  def __init__(self,
               cfg,
              #  quantization_bits,
              #  num_code = 100,
               code_len = 4,
              #  max_len = 80,
               ):
    """Initializes Encoder Model.

    Args:
    """
    super(PARAMEncoder, self).__init__()
    self.embed_dim = cfg.d_model
    # self.bits = cfg.quantization_bits
    self.dropout = cfg.dropout
    self.max_len = cfg.max_total_len

    commitment_cost = 0.25
    decay = 0.99
    self.codebook_dim = 128
    self.vq_vae = VectorQuantizerEMA(cfg.num_code, self.codebook_dim, commitment_cost, decay)

    self.code_len = code_len
    self.const_embed = Embedder(self.code_len, self.embed_dim)

    args_dim = cfg.args_dim + 1
    self.param_embed = nn.Embedding(args_dim, self.embed_dim, padding_idx=0)
    self.param_embed_fcn = nn.Linear(self.embed_dim * cfg.n_args, self.embed_dim)
    # self.coord_embed_x = Embedder(2**self.bits+COORD_PAD+EXTRA_PAD, self.embed_dim)
    # self.coord_embed_y = Embedder(2**self.bits+COORD_PAD+EXTRA_PAD, self.embed_dim)

    # self.pixel_embed = Embedder(2**self.bits * 2**self.bits+PIX_PAD+EXTRA_PAD, self.embed_dim)
    self.pos_embed = PositionalEncoding(max_len=self.max_len+code_len, d_model=self.embed_dim)
   
    # Transformer encoder
    encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=cfg.n_heads, 
                                             dim_feedforward=cfg.dim_feedforward, dropout=self.dropout)
    encoder_norm = LayerNorm(self.embed_dim)
    self.encoder = TransformerEncoder(encoder_layers, cfg.n_layers, encoder_norm)
    self.down = nn.Linear(self.embed_dim, self.codebook_dim)
    self.up = nn.Linear(self.codebook_dim, self.embed_dim)
    

  def forward(self, parameters, mask, epoch):
    """ forward pass 
    Inputs:
      parameters: (N, S, N_ARGS)
    Returns:
      quantized_up: (N, 2, E)
      vq_loss: Number 0.0 | (2, N, E_down)
      selection: (2*N, E_down)
    """
    bs, seqlen = parameters.shape[0], parameters.shape[1]

    # embedding 
    embeddings = self.param_embed(parameters + 1).transpose(0,1)  # (S, N, N_ARGS, E)
    embeddings = self.param_embed_fcn(embeddings.view(seqlen, bs, -1))  # (S, N, E)
    # coord_embed = self.coord_embed_x(xy_v[...,0]) + self.coord_embed_y(xy_v[...,1]) # [bs, vlen, dim] = (N, S, E)
    # pixel_embed = self.pixel_embed(pixel_v)  # (N, S, E)
    # embeddings = (coord_embed+pixel_embed).transpose(0,1)  # (S, N, E)
    z_embed = self.const_embed(torch.arange(0, self.code_len).long().cuda()).unsqueeze(1).repeat(1, bs, 1)  # (2, N, E)
    embed_input = torch.cat([z_embed, embeddings], dim=0)  # (S+2, N, E)
    encoder_input = self.pos_embed(embed_input)  # (S+2, N, E)

    # Pass through encoder
    if mask is not None:
        mask = torch.cat([(torch.zeros([bs, self.code_len])==1).cuda(), mask], axis=1)
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=mask)  # [seq_len, bs, dim].transpose(1,0)
    z_encoded = outputs[0:self.code_len]   # (2, N, E)

    if epoch < INITIAL_PASS:
      vq_loss = 0.0
      selection = None 
      quantized_up = self.up(self.down(z_encoded)).transpose(0,1)  # (N, 2, E)
    else:
      vq_loss, quantized, _, selection = self.vq_vae(self.down(z_encoded))  # (2, N, E_down), '', (2*N, E_down)
      quantized_up = self.up(quantized).transpose(0,1)  # (N, 2, E)
    return quantized_up, vq_loss, selection 


  def get_code(self, parameters, mask, return_np=True):
    """
    Return:
      labels: (N, 2)
    """
    bs, seqlen = parameters.shape[0], parameters.shape[1]

    # embedding 
    embeddings = self.param_embed(parameters + 1).transpose(0,1)  # (S, N, E)
    # coord_embed = self.coord_embed_x(xy_v[...,0]) + self.coord_embed_y(xy_v[...,1]) # [bs, vlen, dim]
    # pixel_embed = self.pixel_embed(pixel_v)
    # embeddings = (coord_embed+pixel_embed).transpose(0,1)
    z_embed = self.const_embed(torch.arange(0, self.code_len).long().cuda()).unsqueeze(1).repeat(1, bs, 1) 
    embed_input = torch.cat([z_embed, embeddings], dim=0)
    encoder_input = self.pos_embed(embed_input) 

    # Pass through encoder
    if mask is not None:
        mask = torch.cat([(torch.zeros([bs, self.code_len])==1).cuda(), mask], axis=1)
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=mask)  # [seq_len, bs, dim].transpose(1,0)
    
    z_encoded = outputs[0:self.code_len]
    _, _, one_hot, _ = self.vq_vae(self.down(z_encoded))  # [seqlen, bs, one-hot] 
    labels = torch.argmax(one_hot, dim=2).transpose(0,1)  # [bs, seqlen]
    if return_np:
      return labels.detach().cpu().numpy().astype(int)
    else:
      return labels


class EXTEncoder(nn.Module):

  def __init__(self,
               config,
               quantization_bits,
               num_code = 100,
               code_len = 4,
               max_len = 80,
               ):
    """Initializes Encoder Model.

    Args:
    """
    super(EXTEncoder, self).__init__()
    self.embed_dim = config['embed_dim']
    self.bits = quantization_bits
    self.dropout = config['dropout_rate']
    self.max_len = max_len

    commitment_cost = 0.25
    decay = 0.99
    self.codebook_dim = 128
    self.vq_vae = VectorQuantizerEMA(num_code, self.codebook_dim, commitment_cost, decay)

    self.code_len = code_len
    self.const_embed = Embedder(self.code_len, self.embed_dim)

    self.ext_embed = Embedder(2**self.bits+EXT_PAD+EXTRA_PAD, self.embed_dim)
    self.flag_embed = Embedder(8, self.embed_dim)
    self.pos_embed = PositionalEncoding(max_len=self.max_len+self.code_len, d_model=self.embed_dim)
   
    # Transformer encoder
    encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=config['num_heads'], 
                                             dim_feedforward=config['hidden_dim'], dropout=self.dropout)
    encoder_norm = LayerNorm(self.embed_dim)
    self.encoder = TransformerEncoder(encoder_layers, config['num_layers'], encoder_norm)
    self.down = nn.Linear(256, self.codebook_dim)
    self.up = nn.Linear(self.codebook_dim, 256)
   

  def forward(self, ext_seq, flag_seq, mask, epoch):
    """ forward pass """
    bs, seqlen = ext_seq.shape[0], ext_seq.shape[1]

    # embedding 
    ext_embeds = self.ext_embed(ext_seq)
    flag_embeds = self.flag_embed(flag_seq)
    embeddings = (ext_embeds+flag_embeds).transpose(0,1) #ext_embeds.transpose(0,1) #
    z_embed = self.const_embed(torch.arange(0, self.code_len).long().cuda()).unsqueeze(1).repeat(1, bs, 1) 
    embed_input = torch.cat([z_embed, embeddings], dim=0)
    encoder_input = self.pos_embed(embed_input) 

    # Pass through encoder
    if mask is not None:
        mask = torch.cat([(torch.zeros([bs, self.code_len])==1).cuda(), mask], axis=1)
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=mask)  # [seq_len, bs, dim].transpose(1,0)
    z_encoded = outputs[0:self.code_len]
    
    if epoch < INITIAL_PASS:
      vq_loss = 0.0
      selection = None 
      quantized_up = self.up(self.down(z_encoded)).transpose(0,1)
    else:
      vq_loss, quantized, _, selection = self.vq_vae(self.down(z_encoded))
      quantized_up = self.up(quantized).transpose(0,1)

    return quantized_up, vq_loss, selection 


  def get_code(self, ext_seq, flag_seq, mask, return_np=True):
    """ forward pass """
    bs, seqlen = ext_seq.shape[0], ext_seq.shape[1]

    # embedding 
    ext_embeds = self.ext_embed(ext_seq)
    flag_embeds = self.flag_embed(flag_seq)
    embeddings = (ext_embeds+flag_embeds).transpose(0,1)
    z_embed = self.const_embed(torch.arange(0, self.code_len).long().cuda()).unsqueeze(1).repeat(1, bs, 1) 
    embed_input = torch.cat([z_embed, embeddings], dim=0)
    encoder_input = self.pos_embed(embed_input) 

    # Pass through encoder
    if mask is not None:
        mask = torch.cat([(torch.zeros([bs, self.code_len])==1).cuda(), mask], axis=1)
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=mask)  # [seq_len, bs, dim].transpose(1,0)
    
    z_encoded = outputs[0:self.code_len]
    _, _, one_hot, _ = self.vq_vae(self.down(z_encoded))  # [seqlen, bs, one-hot] 
    labels = torch.argmax(one_hot, dim=2).transpose(0,1)  # [bs, seqlen]
    if return_np:
      return labels.detach().cpu().numpy().astype(int)
    else:
      return labels

class SkexEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cmd_encoder = CMDEncoder(cfg, code_len=4)  # TODO: add `cfg.code_len`
        self.param_encoder = PARAMEncoder(cfg, code_len=2)
    
    def forward(self, command, parameters, epoch): 
        """
        shape(command) = (N, S)
        shape(parameters) = (N, S, N_ARGS)
        """
        mask = _get_key_padding_mask(command, seq_dim=1)
        cmd_codes, cmd_vq_loss, _ = self.cmd_encoder.forward(command, mask, epoch)
        param_codes, param_vq_loss, _ = self.param_encoder.forward(parameters, mask, epoch)
        output = torch.cat([cmd_codes, param_codes], dim=1)  # (N, 4+2, E)
        vq_loss = cmd_vq_loss + param_vq_loss
        return {
            'representation': output,
            'vq_loss': vq_loss
        }
        