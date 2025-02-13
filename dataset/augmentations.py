from functools import partial
import json
import os
import random
import torch
import torch.nn.functional as F
import h5py
from cadlib.macro import *


def verify(*cmdargs):
    for cmd, arg in cmdargs:
        assert (arg[cmd == EOS_IDX] == PAD_VAL).all() and (arg[cmd == SOL_IDX] == PAD_VAL).all(), \
            'EOS and SOL should have all PAD_VAL args'
        assert (arg[cmd == EXT_IDX, :5] == PAD_VAL).all(), \
            'EXT should have non-PAD_VAL args from 6th arg to the end'
        assert (arg[cmd == LINE_IDX, 2:] == PAD_VAL).all() and (arg[cmd == LINE_IDX, :2] > PAD_VAL).all(), \
            'LINE should have non-PAD_VAL args from first to second arg'
        assert (arg[cmd == CIRCLE_IDX, 2:4] == PAD_VAL).all() and (arg[cmd == CIRCLE_IDX, 5:] == PAD_VAL).all(), \
            'CIRCLE should have non-PAD_VAL args on first, second and 5th arg'
        assert (arg[cmd == CIRCLE_IDX][:,[0,1,4]] > PAD_VAL).all(), \
            'CIRCLE should have non-PAD_VAL args on first, second and 5th arg'
        assert (arg[cmd == ARC_IDX, 4:] == PAD_VAL).all() and (arg[cmd == ARC_IDX, :4] > PAD_VAL).all(), \
            'ARC should have non-PAD_VAL args from first to 4th arg'
        assert arg.max() < ARGS_DIM and arg.min() >= -1, \
            'args should be in range [-1, ARGS_DIM)'
        assert cmd.max() < len(ALL_COMMANDS) and cmd.min() >= 0, \
            'command should be in range [0, len(ALL_COMMANDS))'

def augment(command, args, cfg):
    # command, args -> (command1, args1, command2, args2)
    # Note that all input-output are NON-BATCHED NUMPY ARRAYS
    if cfg.augment_method == 'manual':
        augment1 = AUG2FN[cfg.cl_augment1]
        augment2 = AUG2FN[cfg.cl_augment2]
        
    elif cfg.augment_method == 'randaug':
        augment1 = RandAugment(cfg)
        augment2 = RandAugment(cfg)
    
    elif cfg.augment_method == 'randaug_oneside':
        augment1 = identity
        augment2 = RandAugment(cfg)
        
    if augment1 == corrupt_v3:
        augment1 = partial(augment1, p=cfg.corruption_prob)
    if augment2 == corrupt_v3:
        augment2 = partial(augment2, p=cfg.corruption_prob)
    if augment1 == random_augment:
        augment1 = partial(augment1, cfg=cfg)
    if augment2 == random_augment:
        augment2 = partial(augment2, cfg=cfg)
    if augment1 == delete:
        augment1 = partial(augment1, p=cfg.deletion_prob)
    if augment2 == delete:
        augment2 = partial(augment2, p=cfg.deletion_prob)
        
    command1, args1 = augment1(command, args)
    command2, args2 = augment2(command, args)
    verify([command1, args1], [command2, args2])
    return command1, args1, command2, args2

class RandAugment:
    def __init__(self, cfg):
        self.n = cfg.n_augment
        self.m = cfg.m_augment
        self.p = cfg.corruption_prob  # for corrupt_v3
        self.augment_list = list(AUG2FN.values())[2:-1]  # exclude identity, absorb and random

    def __call__(self, command, args):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            # print(op.__name__)
            if op == corrupt_v3:
                op = partial(op, p=self.p)
            command, args = op(command, args)
        # print('--------')

        return command, args
    
def identity(command, args):
    return command, args

def random_augment(command, args, cfg):
    # raise NotImplementedError
    augmenter = random.choice(list(AUG2FN.values())[:-1])
    if augmenter == corrupt_v3:
        augmenter = partial(augmenter, p=cfg.corruption_prob)
    return augmenter(command, args)

def corrupt(command, args):
    """
    Add random 'noises' to command parameters
    """ 
    command, args = command.copy(), args.copy()
    seq_len = command.shape[0]
    n_commands_to_corrupt = np.random.randint(1, seq_len)
    command_to_corrupt_idx = np.random.choice(range(0, seq_len), size=n_commands_to_corrupt, replace=False)
    for idx in command_to_corrupt_idx:
        if command[idx] in [SOL_IDX, EOS_IDX]:
            continue
        n_args_to_corrupt = np.random.randint(1, N_ARGS)
        args_to_corrupt_idx = np.random.choice(range(0, N_ARGS), size=n_args_to_corrupt, replace=False)
        for j in args_to_corrupt_idx:
            if args[idx,j] == -1:
                continue
            if command[idx] == EXT_IDX:
                if j == 15:  # Extrude type
                    args[idx,j] = np.random.randint(0, len(EXTENT_TYPE))
                elif j == 14:  # boolean operation
                    args[idx,j] = np.random.randint(0, len(EXTRUDE_OPERATIONS))
            elif command[idx] == ARC_IDX and j == 3:  # CCW flag {0,1}
                args[idx,j] = np.random.randint(0, 2)
            else:
                args[idx,j] = np.random.randint(0, ARGS_DIM)
    return command, args

def corrupt_v2(command, args):
    """
    Add random 'noises' to command parameters (all the args for a `command_to_corrupt` will be corrupted)
    """ 
    command, args = command.copy(), args.copy()
    seq_len = command.shape[0]
    n_commands_to_corrupt = np.random.randint(1, seq_len)
    command_to_corrupt_idx = np.random.choice(range(0, seq_len), size=n_commands_to_corrupt, replace=False)
    for idx in command_to_corrupt_idx:
        if command[idx] in [SOL_IDX, EOS_IDX]:
            continue
        corruptions = np.random.randint(0, ARGS_DIM, size=(N_ARGS))  # corrupt all parameter values
        corruptions[3] = np.random.randint(0, 2)  # set range of arc CCW flag {0,1}
        corruptions[14] = np.random.randint(0, len(EXTRUDE_OPERATIONS))  # as well ext boolean operation
        corruptions[15] = np.random.randint(0, len(EXTENT_TYPE))  # ext type
        corrupted = np.ma.array(corruptions, mask=get_parameter_mask(command[idx]))
        args[idx] = corrupted.filled(fill_value=PAD_VAL)

    return command, args

def corrupt_v3(command, args, p):
    """
    corrupt_v2 but with corruption probability `p`
    """
    command, args = command.copy(), args.copy()
    seq_len = command.shape[0]
    command_to_corrupt_idx = np.random.choice(range(0, seq_len), size=int(p * seq_len), replace=False)
    for idx in command_to_corrupt_idx:
        if command[idx] in [SOL_IDX, EOS_IDX]:
            continue
        corruptions = np.random.randint(0, ARGS_DIM, size=(N_ARGS))  # corrupt all parameter values
        corruptions[3] = np.random.randint(0, 2)  # set range of arc CCW flag {0,1}
        corruptions[14] = np.random.randint(0, len(EXTRUDE_OPERATIONS))  # as well ext boolean operation
        corruptions[15] = np.random.randint(0, len(EXTENT_TYPE))  # ext type
        corrupted = np.ma.array(corruptions, mask=get_parameter_mask(command[idx]))
        args[idx] = corrupted.filled(fill_value=PAD_VAL)
    
    return command, args

    

def absorb(command, args):
    """
    Eliminate `extrude` command in order to make 2D sketch shape out of 3D cad
    """
    ext_indices = np.where(command != EXT_IDX)[0]
    command_aug = command[ext_indices]
    args_aug = args[ext_indices]
    assert command_aug.shape[0] == args_aug.shape[0]
    command_aug = np.pad(command_aug, (0, len(command) - len(command_aug)), 'constant', constant_values=EOS_IDX)
    args_aug = np.pad(args_aug, ((0, len(args) - len(args_aug)), (0, 0)), 'constant', constant_values=PAD_VAL)
    assert command_aug.shape[0] == args_aug.shape[0] and args_aug.shape[1] == N_ARGS
    return command_aug, args_aug

def re_extrude(command, args):
    """
    Transform `extrude` parameters; e.g. extrude type-distance,
    Order: ..., e1, e2, boolean, type
    the extrusion type can be either `one-sided`, `symmetric`, or `two-sided` with respect to the profile’s sketch plane
    boolean operations: either creating a `new` body, or `joining`, `cutting` or `intersecting`
    """
    command, args = command.copy(), args.copy()
    ext_indices = np.where(command == EXT_IDX)[0]
    for ext_index in ext_indices:
        args[ext_index][-4:-2] = np.random.randint(0, ARGS_DIM, (2,))
        args[ext_index][-1] = np.random.randint(0, len(EXTENT_TYPE), (1,))
    return command, args

# def re_extrude_single(command, args):
#     ext_indices = torch.where(command == EXT_IDX)[0]
#     for ext_index in ext_indices:
#         args[ext_index][-4:-2] = torch.randint(0, ARGS_DIM, (2,))
#         args[ext_index][-1] = torch.randint(0, 3, (1,))
#     return command, args


def replace_extrusion(command, args):
    """
    If len(extrude) > 1, then replace one of the extrusions to another model's one
    If len == 1, then run `re_extrude`
    """
    ext_indices = np.where(command == EXT_IDX)[0]
    if len(ext_indices) > 1:
        command_aug, args_aug = _replace(command, args, ext_indices)
    else:
        command_aug, args_aug = re_extrude(command, args)
    return command_aug, args_aug

with open('../datasets/cad_data/train_val_test_split.json', 'r') as fp:
    all_data = json.load(fp)['train']
def _replace(command: np.ndarray, args: np.ndarray, ext_indices: np.ndarray):
    cad_vec = np.hstack([command[:,np.newaxis], args])
    ext_vec1 = np.split(cad_vec, ext_indices + 1, axis=0)[:-1]

    data_id2 = all_data[random.randint(0, len(all_data) - 1)]
    h5_path2 = os.path.join('../datasets/cad_data/cad_vec', data_id2 + ".h5")
    with h5py.File(h5_path2, "r") as fp:
        cad_vec2 = fp["vec"][:]
    command2 = cad_vec2[:, 0]
    ext_indices2 = np.where(command2 == EXT_IDX)[0]
    ext_vec2 = np.split(cad_vec2, ext_indices2 + 1, axis=0)[:-1]

    n_replace = random.randint(1, min(len(ext_vec1) - 1, len(ext_vec2)))
    old_idx = sorted(random.sample(list(range(len(ext_vec1))), n_replace))
    new_idx = sorted(random.sample(list(range(len(ext_vec2))), n_replace))
    for i in range(len(old_idx)):
        ext_vec1[old_idx[i]] = ext_vec2[new_idx[i]]

    sum_len = 0
    new_vec = []
    for i in range(len(ext_vec1)):
        sum_len += len(ext_vec1[i])
        if sum_len > MAX_TOTAL_LEN:
            break
        new_vec.append(ext_vec1[i])
    cad_vec = np.concatenate(new_vec, axis=0)
    pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
    cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    command, args = cad_vec[:,0], cad_vec[:,1:]
    return command, args

def get_parameter_mask(cmd):
    mask = np.zeros((N_ARGS,))
    if cmd in [LINE_IDX, ARC_IDX, CIRCLE_IDX]:
        mask[:2] += 1
        if cmd == ARC_IDX:
            mask[2:4] += 1
        elif cmd == CIRCLE_IDX:
            mask[4] += 1
    elif cmd == EXT_IDX:
        mask[5:] += 1
    return (~mask.astype(bool)).astype(int)

def attach(command, args):
    max_total_len = command.shape[0]
    command_org, args_org = command.copy(), args.copy()
    cad_vec = np.hstack([command[:,np.newaxis], args])
    ext_indices = np.where(command == EXT_IDX)[0]
    ext_vec = np.split(cad_vec, ext_indices + 1, axis=0)[:-1]
    
    data_id2 = all_data[random.randint(0, len(all_data) - 1)]
    h5_path2 = os.path.join('../datasets/cad_data/cad_vec', data_id2 + ".h5")
    with h5py.File(h5_path2, "r") as fp:
        cad_vec2 = fp["vec"][:]
    command2 = cad_vec2[:, 0]
    ext_indices2 = np.where(command2 == EXT_IDX)[0]
    ext_vec2 = np.split(cad_vec2, ext_indices2 + 1, axis=0)[:-1]

    # randomly select a sketch & extrusion
    rand_index = random.randint(0, len(ext_vec2) - 1)
    new_vec = ext_vec2[rand_index]

    # attach new_vec to the end of cad_vec and EOS_VEC
    cad_vec = np.concatenate(ext_vec, axis=0)
    cad_vec = np.concatenate((cad_vec, new_vec), axis=0)

    pad_len = max_total_len - cad_vec.shape[0]
    if pad_len <= 0:
        return command_org, args_org
    cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    return cad_vec[:,0], cad_vec[:,1:]

def translate(command, args):
    cad_vec = np.hstack([command[:, np.newaxis], args])
    special_token_mask = (cad_vec[:, 0] == SOL_IDX) + (cad_vec[:, 0] == EXT_IDX) + (cad_vec[:, 0] == EOS_IDX)
    min_trans = 0 - cad_vec[~special_token_mask, 1:3].min()
    max_trans = ARGS_DIM - cad_vec[:, 1:3].max()
    translate_scale = np.random.randint(min_trans, max_trans, size=(2,))
    cad_vec[:, 1:3] += translate_scale
    cad_vec[special_token_mask, 1:3] = np.ones_like(cad_vec[special_token_mask, 1:3]) * PAD_VAL
    return cad_vec[:, 0], cad_vec[:, 1:]

def redraw(command, args):
    # Warning: `redraw` for CL and `redraw` for DeepCAD are differently implemented (bc. of EOS_VEC)
    max_total_len = command.shape[0]
    command_org, args_org = command.copy(), args.copy()
    cad_vec = np.hstack([command[:,np.newaxis], args])
    ext_indices = np.where(command == EXT_IDX)[0]
    ext_vec = np.split(cad_vec, ext_indices + 1, axis=0)[:-1]
    # randomly select a setch & extrusion
    rand_index = random.randint(0, len(ext_vec) - 1)
    new_vec = ext_vec[rand_index].copy()  # e.g. [4, 0, 0, ..., 5]
    # translate the sketch
    # 1:-1 => ignore initial SOL and EXT
    # 1:3 => x,y coordinate (starting from 1 bc. 0 is command idex)
    xy_coords = new_vec[1:-1, 1:3]
    sol_mask = new_vec[1:-1, 0] == SOL_IDX  # remove parameters of intermediate SOL commands
    max_trans = ARGS_DIM - xy_coords.max()  # so that the sketch is still in the boundary
    min_trans = 0  - xy_coords[~sol_mask].min()         # so that the sketch is still in the boundary
    translate_scale = np.random.randint(min_trans, max_trans, size=(xy_coords.shape[0], 2))
    xy_coords += translate_scale
    xy_coords[sol_mask] = np.ones_like(xy_coords[sol_mask]) * PAD_VAL  # remove parameters of intermediate SOL commands
    # attach new_vec to the end of cad_vec and EOS_VEC
    cad_vec = np.concatenate(ext_vec, axis=0)
    cad_vec = np.concatenate((cad_vec, new_vec), axis=0)
    pad_len = max_total_len - cad_vec.shape[0]
    if pad_len <= 0:
        return command_org, args_org
    cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    assert not (cad_vec.max() >= ARGS_DIM or cad_vec.min() < -1)
    return cad_vec[:,0], cad_vec[:,1:]

    
# def scale(command, args):
#     command_aug, args_aug = command.copy(), args.copy()
#     ext_mask = (command_aug == EXT_IDX)
#     line_arc_mask = (command_aug == LINE_IDX) + (command_aug == ARC_IDX)
#     circle_mask = (command_aug == CIRCLE_IDX)
#     minmax_mask = ext_mask + line_arc_mask
#     min_scale = 0 - args_aug[minmax_mask, 3:9].min()
#     max_scale = ARGS_DIM - args_aug[minmax_mask, 3:9].max()
#     scale = np.random.randint(min_scale, max_scale)
#     args_aug[ext_mask, 3:9] += scale
#     args_aug[line_arc_mask, 0:2] += scale
#     args_aug[circle_mask, 0:2] += scale
#     args_aug[circle_mask, 4] += scale
#     return command_aug, args_aug

def scale(command, args):
    raise NotImplementedError
    # if s is None:
    #     s = np.random.uniform(0.5, 2)
    #     print('Scaling factor is:', s)
    command_aug, args_aug = command.copy(), args.copy()
    
    # scale line/arc length
    minmax_mask = (command_aug == LINE_IDX) + (command_aug == ARC_IDX) + (command_aug == CIRCLE_IDX)
    # args_aug[line_arc_mask, 0:2] = (args_aug[line_arc_mask, 0:2].astype(float) * s).astype(int)
    # args_aug[line_arc_mask, 0:2] = np.clip(args_aug[line_arc_mask, 0:2], 0, ARGS_DIM - 1)
    min_scale = 0 - args_aug[minmax_mask, 0:2].min()
    max_scale = ARGS_DIM - args_aug[minmax_mask, 0:2].max()
    scale = np.random.randint(min_scale, max_scale)
    args_aug[minmax_mask, 0:2] += scale
    # scale radius of circles
    circle_mask = (command_aug == CIRCLE_IDX)
    # args_aug[circle_mask, 4] = (args_aug[circle_mask, 4].astype(float) * s).astype(int)
    # args_aug[circle_mask, 4] = np.clip(args_aug[circle_mask, 4], 0, ARGS_DIM - 1)
    args_aug[circle_mask, 4] += scale
    # 
    ext_mask = (command_aug == EXT_IDX)
    args_aug[ext_mask, -3:-5] += scale
    print(scale)
    
    return command_aug, args_aug 

def rotate(command, args, deg=60):
    raise NotImplementedError
    command_aug, args_aug = command.copy(), args.copy()
    rot_matrix = np.array([
        [np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg))],
        [np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg))]
    ])
    # multiply with rot matrix with xy coordinates for line,arc,circle
    # FIXME: output should not be negative
    special_token_mask = (command_aug == SOL_IDX) + (command_aug == EXT_IDX) + (command_aug == EOS_IDX)
    args_aug[~special_token_mask, 0:2] = np.matmul(rot_matrix, args_aug[~special_token_mask, 0:2].T).T.astype(int)
    return command_aug, args_aug


def delete(command, args, p=None):  # p: proportion of deletion
    if p is None:
        p = np.random.uniform()
    len_command = command[command != EOS_IDX].shape[0]
    idx_to_keep = sorted(np.random.choice(range(0, len_command), size=int(len_command * (1-p)), replace=False))
    command_aug = command[idx_to_keep].copy()
    args_aug = args[idx_to_keep].copy()
    cad_vec = np.concatenate([command_aug[:,np.newaxis], args_aug], axis=1)
    pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
    cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    return cad_vec[:,0], cad_vec[:,1:]

def reorder(command, args):
    len_command = command[command != EOS_IDX].shape[0]
    new_order = np.random.permutation(len_command)
    command_aug = command[new_order].copy()
    args_aug = args[new_order].copy()
    cad_vec = np.concatenate([command_aug[:,np.newaxis], args_aug], axis=1)
    pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
    cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    return cad_vec[:,0], cad_vec[:,1:]

def insert(command, args, p=None):
    def get_random_token():
        data_id2 = all_data[random.randint(0, len(all_data) - 1)]
        h5_path2 = os.path.join('../datasets/cad_data/cad_vec', data_id2 + ".h5")
        with h5py.File(h5_path2, "r") as fp:
            cad_vec2 = fp["vec"][:]
        command2 = cad_vec2[:, 0]
        len_command2 = command2[command2 != EOS_IDX].shape[0]
        return cad_vec2[random.randint(0, len_command2 - 1)]
    if p is None:
        p = np.random.uniform()
    len_command = command[command != EOS_IDX].shape[0]
    cad_vec = np.hstack([command[:, np.newaxis], args])
    token_list = []
    for i in range(len_command):
        if len(token_list) < MAX_TOTAL_LEN:
            token_list.append(cad_vec[i])
        if np.random.uniform() < p and len(token_list) < MAX_TOTAL_LEN:
            random_token = get_random_token()
            token_list.append(random_token)
    cad_vec = np.stack(token_list, axis=0)
    pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
    cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    return cad_vec[:,0], cad_vec[:,1:]

            
def legacy_subtitute(command, args, p=None):
    if p is None:
        p = np.random.uniform()
    cad_vec = np.hstack([command[:, np.newaxis], args])

    data_id2 = all_data[random.randint(0, len(all_data) - 1)]
    h5_path2 = os.path.join('../datasets/cad_data/cad_vec', data_id2 + ".h5")
    with h5py.File(h5_path2, "r") as fp:
        cad_vec2 = fp["vec"][:]
    
    len_command = command[command != EOS_IDX].shape[0]
    idx_to_change = sorted(np.random.choice(range(0, len_command), size=int(len_command * p), replace=False))
    len_command2 = cad_vec2[:,0][cad_vec2[:,0] != EOS_IDX].shape[0]
    size_to_bring_cadvec2 = int(len_command * p)
    if MAX_TOTAL_LEN - len_command <= size_to_bring_cadvec2:
        size_to_bring_cadvec2 = MAX_TOTAL_LEN - len_command - 1
    idx_to_bring_cadvec2 = sorted(np.random.choice(range(0, len_command2), size=size_to_bring_cadvec2, replace=False))

    cad_vec[idx_to_change] = cad_vec2[idx_to_bring_cadvec2]
    return cad_vec[:,0], cad_vec[:,1:]

def rre(command, args):
    command_aug, args_aug = command.copy(), args.copy()
    ext_indices = np.where(command_aug == EXT_IDX)[0]
    if len(ext_indices) > 1:
        command_aug, args_aug = _replace(command_aug, args_aug, ext_indices)
    command_aug, args_aug = re_extrude(command_aug, args_aug)
    command_aug, args_aug = arc_augment_v2(command_aug, args_aug)
    return command_aug, args_aug
 
AUG2FN = {
    'identity': identity,
    'absorb': absorb,
    're-extrude': re_extrude,
    'replace': replace_extrusion,
    'corrupt': corrupt,
    'corrupt2': corrupt_v2,
    'corrupt3': corrupt_v3,
    'redraw': redraw,
    'attach': attach,
    'translate': translate,
    'delete': delete,  # nlp-like augmentation
    'reorder': reorder,  # nlp-like augmentation
    'insert': insert,  # nlp-like augmentation
    'random': random_augment  # `random` should always be the last one
}

def dataset_augment(cad_vec, augment_type, aug_prob=0.5):
    if random.uniform(0, 1) > aug_prob:
        command, args = cad_vec[:, 0], cad_vec[:, 1:]
        if augment_type == 'default':
            ext_indices = np.where(command == EXT_IDX)[0]
            if len(ext_indices) > 1:
                command, args = _replace(command, args, ext_indices)
        elif augment_type == 're-extrude':
            command, args = re_extrude(command, args)
        elif augment_type == 'replace':
            command, args = replace_extrusion(command, args)
        elif augment_type == 'corrupt2':
            command, args = corrupt_v2(command, args)
        elif augment_type == 'redraw':
            command, args = redraw(command, args)
        elif augment_type == 'replace_arc':
            command, args = replace_extrusion(command, args)
            command, args = arc_augment(command, args)
        elif augment_type == 'replace_p_arc':
            command, args = replace_extrusion(command, args)
            if random.uniform(0, 1) > 0.5:
                command, args = arc_augment(command, args)
        elif augment_type == 'replace_arc2':
            command, args = replace_extrusion(command, args)
            command, args = arc_augment_v2(command, args)
        elif augment_type == 'replace_p_arc2':
            command, args = replace_extrusion(command, args)
            if random.uniform(0, 1) > 0.5:
                command, args = arc_augment_v2(command, args)
        elif augment_type == 'arc':
            command, args = arc_augment(command, args)
        elif augment_type == 'arc2':
            command, args = arc_augment_v2(command, args)
        elif augment_type == 'rre':
            command, args = rre(command, args)
        else:
            raise NotImplementedError
        cad_vec = np.hstack([command[:,np.newaxis], args])
    return cad_vec

def arc_augment(command, args):
    cad_vec = np.hstack([command[:, np.newaxis], args])
    if (cad_vec[:,0] == LINE_IDX).sum() == 0:
        return cad_vec[:, 0], cad_vec[:, 1:]
    target_index = random.choice(np.where(cad_vec[:, 0] == LINE_IDX)[0])
    end_x, end_y = cad_vec[target_index, 1:3]
    angle = random.randint(1, 255)
    flag = random.randint(0, 1)
    arc_token = np.array([ARC_IDX, end_x, end_y, angle, flag,] + [-1]*12)
    cad_vec[target_index] = arc_token
    return cad_vec[:,0], cad_vec[:,1:]

def arc_augment_v2(command, args):
    cad_vec = np.hstack([command[:, np.newaxis], args])
    num_lines = (cad_vec[:,0] == LINE_IDX).sum()
    if num_lines == 0:
        return cad_vec[:, 0], cad_vec[:, 1:]
    num_lines_to_change = random.randint(1, num_lines)
    target_indices_to_change = np.random.choice(np.where(cad_vec[:, 0] == LINE_IDX)[0], size=num_lines_to_change, replace=False)
    for idx in target_indices_to_change:
        end_x, end_y = cad_vec[idx, 1:3]
        angle = random.randint(1, 255)
        flag = random.randint(0, 1)
        arc_token = np.array([ARC_IDX, end_x, end_y, angle, flag,] + [-1]*12)
        cad_vec[idx] = arc_token
    return cad_vec[:,0], cad_vec[:,1:]