import numpy as np
import os
import json
from cadlib.extrude import CADSequence
from cadlib.macro import *

def get_seq_from_json(data_id):
    json_path = os.path.join('../datasets/cad_data/cad_json/', data_id + '.json')
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    cad_seq = CADSequence.from_dict(data)
    return cad_seq

def transform(cfg, data_id):
    if cfg.augment_method == 'scale_transform':
        return scale_transform(cfg, data_id, scale=cfg.scale_factor)
    elif cfg.augment_method == 'flip_sketch':
        return flip_sketch(data_id, axis=cfg.flip_axis)
    elif cfg.augment_method == 'random_transform':
        return random_transform(data_id)
    elif cfg.augment_method == 'random_flip':
        return random_flip(data_id)
    elif cfg.augment_method == 'rotate_transform':
        return rotate_transform(data_id, deg=cfg.rotate_deg)

def random_transform(data_id):
    cad_seq = get_seq_from_json(data_id)
    cad_seq.normalize()
    cad_seq.numericalize()
    cad_vec = cad_seq.to_vector()
    
    cad_seq.random_transform()
    transf_vec = cad_seq.to_vector()

    assert (ARGS_DIM > transf_vec).all(), f'{transf_vec}'
    assert (transf_vec >= -1).all(), f'{transf_vec}'
    assert not np.array_equal(transf_vec, cad_vec)

    pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
    cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    transf_vec = np.concatenate([transf_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    command, args = cad_vec[:,0], cad_vec[:,1:]
    command_transf, args_transf = transf_vec[:,0], transf_vec[:,1:]
    return command, args, command_transf, args_transf

def random_flip(data_id):
    cad_seq = get_seq_from_json(data_id)
    cad_seq.normalize()
    cad_seq.numericalize()
    cad_vec = cad_seq.to_vector()
    
    cad_seq.random_flip_sketch()
    transf_vec = cad_seq.to_vector()

    assert (ARGS_DIM > transf_vec).all(), f'{transf_vec}'
    assert (transf_vec >= -1).all(), f'{transf_vec}'
    assert not np.array_equal(transf_vec, cad_vec)

    pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
    cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    transf_vec = np.concatenate([transf_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    command, args = cad_vec[:,0], cad_vec[:,1:]
    command_transf, args_transf = transf_vec[:,0], transf_vec[:,1:]
    return command, args, command_transf, args_transf

def rotate_transform(data_id, deg=None):
    if deg is None:
        deg = np.random.uniform(-180, 180)

    cad_seq = get_seq_from_json(data_id)
    cad_seq.normalize()
    cad_seq.numericalize()
    cad_vec = cad_seq.to_vector()
    command, args = cad_vec[:,0], cad_vec[:,1:]
    command_aug, args_aug = command.copy(), args.copy()
    
    rot_matrix = np.array([
        [np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg))],
        [np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg))]
    ])
    rot_mat_3d = np.vstack([
        np.hstack([rot_matrix, np.array([[0],[0]])]),
        np.array([[0,0,1]])
    ])
    # multiply with rot matrix with xy coordinates for line,arc,circle
    # FIXME: output should not be negative
    special_token_mask = (command_aug == SOL_IDX) + (command_aug == EXT_IDX) + (command_aug == EOS_IDX)
    args_aug[~special_token_mask, 0:2] = np.matmul(rot_matrix, args_aug[~special_token_mask, 0:2].T).T.astype(int)

    rotated_vec = np.concatenate([command_aug[:,np.newaxis], args_aug], axis=1)
    rotated_seq = CADSequence.from_vector(rotated_vec)
    rotated_seq.bbox = cad_seq.bbox  # np.matmul(rot_mat_3d, cad_seq.bbox.T).T
    rotated_seq.normalize()
    rotated_vec = rotated_seq.to_vector().astype(int)

    return command, args, rotated_vec[:,0], rotated_vec[:,1:]


def flip_sketch(data_id, axis='xy'):
    cad_seq = get_seq_from_json(data_id) 
    cad_seq.normalize()
    cad_seq.numericalize()
    cad_vec = cad_seq.to_vector()
    
    cad_seq.flip_sketch(axis=axis)
    flipped_vec = cad_seq.to_vector()

    assert (ARGS_DIM > flipped_vec).all(), f'{flipped_vec}'
    assert (flipped_vec >= -1).all(), f'{flipped_vec}'

    pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
    cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    flipped_vec = np.concatenate([flipped_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    return cad_vec[:,0], cad_vec[:,1:], flipped_vec[:,0], flipped_vec[:,1:]
    

def scale_transform(cfg, data_id, scale=None):
    assert cfg.scale_transform_seq or cfg.scale_transform_profile
    if scale is None:
        scale = np.random.uniform(0.3, 1.7)
    # process anchor  TODO: room for optimization : torch.util.Dataset.__getitem__ returns only data_id and read data here
    cad_seq = get_seq_from_json(data_id)
    cad_seq.normalize()
    cad_seq.numericalize()
    cad_vec = cad_seq.to_vector()

    if cfg.scale_transform_seq:
        cad_seq.transform(0.0, scale)
    for ext in cad_seq.seq:
        if cfg.scale_transform_profile:
            ext.profile.transform(0.0, scale)
        ext.profile.normalize()
    scaled_vec = cad_seq.to_vector().astype(int)
    assert (ARGS_DIM > scaled_vec).all(), f'{scaled_vec}'
    assert (scaled_vec >= -1).all(), f'{scaled_vec}'
    assert not np.array_equal(scaled_vec, cad_vec)
    
    pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
    cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    scaled_vec = np.concatenate([scaled_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    command, args = cad_vec[:,0], cad_vec[:,1:]
    command_scaled, args_scaled = scaled_vec[:,0], scaled_vec[:,1:]
    return command, args, command_scaled, args_scaled
    
    
    


    
    