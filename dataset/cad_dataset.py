from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
import _pickle as pickle
from cadlib.macro import *
from dataset.augmentations import augment, dataset_augment
from dataset.linear_transformations import scale_transform, transform


def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    if config.dataset_type == 'clean':
        dataset = CleanCADDataset(phase, config)
    elif config.dataset_type == 'arc':
        dataset = ArcCADDataset(phase, config)
    elif config.dataset_type == 'default':
        dataset = CADDataset(phase, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader

class CADDataset(Dataset):
    def __init__(self, phase, config):
        super(CADDataset, self).__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.aug = config.augment
        self.path = os.path.join(config.data_root, "train_val_test_split.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        self.max_n_loops = config.max_n_loops          # Number of paths (N_P)
        self.max_n_curves = config.max_n_curves            # Number of commands (N_C)
        self.max_total_len = config.max_total_len
        self.size = 256
        
        self.cfg = config

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)

        if self.aug and self.phase == "train":
            cad_vec = dataset_augment(cad_vec, self.cfg.dataset_augment_type, self.cfg.dataset_augment_prob)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        if not self.cfg.is_train or self.cfg.augment_method is None:  # if not trainnig or no augmentation
            return {"command": command, "args": args, "id": data_id}
        
        command_org, args_org = command.copy(), args.copy()
        if self.cfg.augment_method in ['manual', 'randaug', 'randaug_oneside']:
            command, args, command_aug, args_aug = augment(command, args, self.cfg)
        elif self.cfg.augment_method in ['scale_transform', 'flip_sketch', 'random_transform', 'random_flip', 'rotate_transform']:
            command, args, command_aug, args_aug = transform(self.cfg, data_id)

        if np.array_equal(command_org, command) and np.array_equal(args_org, args):
            is_first_intact = True
            if self.cfg.cl_loss == 'supcon':
                assert is_first_intact

        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        command_aug = torch.tensor(command_aug, dtype=torch.long)
        args_aug = torch.tensor(args_aug, dtype=torch.long)
        
        assert command.shape == command_aug.shape and args.shape == args_aug.shape
        return {"command": command, "args": args, "command_aug": command_aug, "args_aug": args_aug, "id": data_id}

    def __len__(self):
        return len(self.all_data)

class ArcCADDataset(Dataset):
    def __init__(self, phase, config):
        super().__init__()
        self.cfg = config
        self.pickle_path = os.path.join(config.data_root, "cad_vec_arc")
        with open(os.path.join(self.pickle_path, phase+'.pkl'), "rb") as fp:
            self.total_data = pickle.load(fp)
        self.phase = phase
        self.aug = config.augment
        self.max_n_loops = config.max_n_loops          # Number of paths (N_P)
        self.max_n_curves = config.max_n_curves            # Number of commands (N_C)
        self.max_total_len = config.max_total_len
    
    def __len__(self):
        return len(self.total_data)
    
    def __getitem__(self, index):
        data = self.total_data[index]
        # data_id = data['name']
        # cad_vec = data['vec']
        data_id = -1
        cad_vec = data
        
        if self.aug and self.phase == "train":
            # raise NotImplementedError
            cad_vec = dataset_augment(cad_vec, self.cfg.dataset_augment_type, self.cfg.dataset_augment_prob)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        if not self.cfg.is_train or self.cfg.augment_method is None:  # if not trainnig or no augmentation
            return {"command": command, "args": args, "id": data_id}
        
        command_org, args_org = command.copy(), args.copy()
        if self.cfg.augment_method in ['manual', 'randaug', 'randaug_oneside']:
            command, args, command_aug, args_aug = augment(command, args, self.cfg)
        elif self.cfg.augment_method in ['scale_transform', 'flip_sketch', 'random_transform', 'random_flip', 'rotate_transform']:
            command, args, command_aug, args_aug = transform(self.cfg, data_id)

        if np.array_equal(command_org, command) and np.array_equal(args_org, args):
            is_first_intact = True
            if self.cfg.cl_loss == 'supcon':
                assert is_first_intact

        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        command_aug = torch.tensor(command_aug, dtype=torch.long)
        args_aug = torch.tensor(args_aug, dtype=torch.long)
        
        assert command.shape == command_aug.shape and args.shape == args_aug.shape
        return {"command": command, "args": args, "command_aug": command_aug, "args_aug": args_aug, "id": data_id}
        
class CleanCADDataset(Dataset):
    def __init__(self, phase, config):
        super(CleanCADDataset, self).__init__()
        self.cfg = config
        # self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.pickle_path = os.path.join(config.data_root, "cad_vec_clean")
        assert phase != 'test'
        with open(os.path.join(self.pickle_path, phase+'_clean.pkl'), 'rb') as f:
            self.total_data = pickle.load(f)
        self.phase = phase
        self.aug = config.augment
        self.max_n_loops = config.max_n_loops          # Number of paths (N_P)
        self.max_n_curves = config.max_n_curves            # Number of commands (N_C)
        self.max_total_len = config.max_total_len

    def __getitem__(self, index):
        data = self.total_data[index]
        data_id = data['name']
        cad_vec = data['vec']
        
        if self.aug and self.phase == "train":
            cad_vec = dataset_augment(cad_vec, self.cfg.dataset_augment_type, self.cfg.dataset_augment_prob)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        if not self.cfg.is_train or self.cfg.augment_method is None:  # if not trainnig or no augmentation
            return {"command": command, "args": args, "id": data_id}
        
        command_org, args_org = command.copy(), args.copy()
        if self.cfg.augment_method in ['manual', 'randaug', 'randaug_oneside']:
            command, args, command_aug, args_aug = augment(command, args, self.cfg)
        elif self.cfg.augment_method in ['scale_transform', 'flip_sketch', 'random_transform', 'random_flip', 'rotate_transform']:
            command, args, command_aug, args_aug = transform(self.cfg, data_id)

        if np.array_equal(command_org, command) and np.array_equal(args_org, args):
            is_first_intact = True
            if self.cfg.cl_loss == 'supcon':
                assert is_first_intact

        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        command_aug = torch.tensor(command_aug, dtype=torch.long)
        args_aug = torch.tensor(args_aug, dtype=torch.long)
        
        assert command.shape == command_aug.shape and args.shape == args_aug.shape
        return {"command": command, "args": args, "command_aug": command_aug, "args_aug": args_aug, "id": data_id}

    def __len__(self):
        return len(self.total_data)

   