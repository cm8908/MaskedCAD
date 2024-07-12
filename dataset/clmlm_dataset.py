import random
import os
import h5py
import torch
from .cad_dataset import CADDataset
from cadlib.macro import *
from torch.utils.data import DataLoader

def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = CadMLMDataset(phase, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader

class CadMLMDataset(CADDataset):
    def __init__(self, phase, config):
        super(CadMLMDataset, self).__init__(phase, config)
        self.mask_ratio = config.mask_ratio
        self.max_total_len = config.max_total_len
        self.include_eos = config.include_eos

    def get_random_token(self):
        data_id2 = self.all_data[random.randint(0, len(self.all_data) - 1)]
        h5_path2 = os.path.join('../datasets/cad_data/cad_vec', data_id2 + ".h5")
        with h5py.File(h5_path2, "r") as fp:
            cad_vec2 = fp["vec"][:]
        command2 = cad_vec2[:, 0]
        len_command2 = command2[command2 != EOS_IDX].shape[0]
        return cad_vec2[random.randint(0, len_command2 - 1)]

    def get_masked_token(self, cad_vec):
        tokens = []
        labels = []
        len_command = cad_vec[cad_vec[:, 0] != EOS_IDX].shape[0]
        if self.include_eos:
            len_command += 1
        for i, token in enumerate(cad_vec): 
            if i < len_command:
                prob = random.random()
                if prob < self.mask_ratio:
                    prob /= self.mask_ratio
                    
                    if prob < 0.8:
                        tokens.append(MASK_TOKEN)
                    
                    elif prob < 0.9:
                        tokens.append(self.get_random_token())
                    
                    else:
                        tokens.append(token)
                    
                    labels.append(1)
                else:
                    tokens.append(token)
                    labels.append(0)
            else:
                tokens.append(EOS_VEC)
                labels.append(0)
        return tokens, labels
        
    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)
        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        
        if self.phase in ['train', 'validation']:
            tokens1, labels1 = self.get_masked_token(cad_vec)
            tokens2, labels2 = self.get_masked_token(cad_vec)

            tokens1 = torch.LongTensor(np.stack(tokens1, axis=0))
            labels1 = torch.LongTensor(labels1)
            tokens2 = torch.LongTensor(np.stack(tokens2, axis=0))
            labels2 = torch.LongTensor(labels2)
            tgt_cmd = torch.LongTensor(cad_vec[:, 0])
            tgt_args = torch.LongTensor(cad_vec[:, 1:])
            return {
                'command': tokens1[:, 0],
                'args': tokens1[:, 1:],
                'mask_labels': labels1,
                'command2': tokens2[:, 0],
                'args2': tokens2[:, 1:],
                'mask_labels2': labels2,
                'tgt_cmd': tgt_cmd,
                'tgt_args': tgt_args,
                'id': data_id
            }
        else:
            return {
                'command': torch.LongTensor(cad_vec[:, 0]),
                'args': torch.LongTensor(cad_vec[:, 1:]),
                'id': data_id
            }
        
                