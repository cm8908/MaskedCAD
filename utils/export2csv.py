import argparse
import os
from pprint import pprint
from notion_client import Client
import pandas as pd

# from utils.file_utils import ensure_dirs
def read_acc(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    metrics = ['avg_cmd_acc', 'avg_param_acc']
    metrics += [f'{cmd.lower()}_cmd_acc' for cmd in ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']]
    metrics += [f'line_param_acc_{i}' for i in range(2)]
    metrics += [f'arc_param_acc_{i}' for i in range(4)]
    metrics += [f'circle_param_acc_{i}' for i in range(3)]
    metrics += [f'ext_param_acc_{i}' for i in range(11)]

    avg_command_acc = float(lines[0].split(':')[-1].strip())
    avg_param_acc = float(lines[1].split(':')[-1].strip())
    get_list_accs = lambda s: list(map(float, [x for x in s.split(':')[-1].strip()[1:-1].split(' ') if x != '']))
    cmd_accs =  get_list_accs(lines[3]) 
    line_param_accs = get_list_accs(lines[4]) 
    arc_param_accs = get_list_accs(lines[5])
    circle_param_accs = get_list_accs(lines[6])
    ext_param_accs = get_list_accs(lines[9])
    accs = [avg_command_acc]  + [avg_param_acc] + cmd_accs + line_param_accs + arc_param_accs + circle_param_accs + ext_param_accs
    
    assert len(metrics) == len(accs)
    dictionary = {'name': path.split('/')[-3]}
    dictionary.update({
        k: v for k, v in zip(metrics, accs)
    })
    # pprint(dictionary)
    return dictionary

def read_cd(path):
    # raise NotImplementedError
    with open(path, 'r') as f:
        lines = f.readlines()
    inval_rate = float(lines[1][lines[1].find('ratio'):].split(':')[-1])
    cd = float(lines[2][lines[2].find('med'):].split(':')[-1])
    dictionary = {'name': path.split('/')[-3], 'inval_rate': inval_rate, 'cd': cd}
    return dictionary

def read_silhouette(path):
    # raise NotImplementedError
    with open(path, 'r') as f:
        lines = f.readlines()
    dictionary = {'name': path.split('/')[-3]}
    for line in lines:
        category = line.split(']')[0][1:]
        silhouette = float(line.split(':')[-1].strip())
        dictionary.update({f'{category}_silh': silhouette})
    return dictionary

def read_iou(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    iou = float(lines[1].split(':')[-1].strip())
    dictionary = {'name': path.split('/')[-3], 'iou': iou}
    return dictionary

def write_to_csv(csv_path, dictionary):
    if os.path.exists(csv_path):
        target_df = pd.read_csv(csv_path)
    else:
        target_df = pd.DataFrame(columns=list(dictionary.keys()))
    src_df = pd.DataFrame.from_dict([dictionary])
    if dictionary['name'] in target_df['name'].to_list():
        raise Exception('Already exists')
        # target_df = target_df.drop(target_df[target_df['name'] == dictionary['name']].index)
    target_df = target_df.append(src_df)
    target_df.to_csv(csv_path, index=0)


def write_to_excel(excel_path, dictionary, ):
    raise NotImplementedError
    if os.path.exists(excel_path):
        target_df = pd.read_csv(excel_path)
    else:
        target_df = pd.DataFrame(columns=list(dictionary.keys()))
    src_df = pd.DataFrame.from_dict([dictionary])
    if dictionary['name'] in target_df['name'].to_list():
        raise Exception('Already exists')
    target_df = target_df.append(src_df)
    target_df.to_excel(excel_path, index=0)
    

def create_db(notion: Client, dictionary: dict):
    raise NotImplementedError
    properties = {
        k: {
            'id': k,
            'type': 'title',
            'title': k,
        } for k, v in dictionary.items()
    }

    parent_page = {'type': 'page_id', 'page_id': '3caceaa332af44a2ba40d260910414b9'}
    title = {'type': 'title', 'title': 'Accuracies'}
    test_db =notion.databases.query('9c7e34a190bd45f0b2e453b83e44b76d')
    notion.databases.create(parent=parent_page, title=title, properties=properties)
    
def save_to_csv(args):
    read_fn = {
        'acc': read_acc,
        'pc': read_cd,
        'silhouette': read_silhouette,
        'ioubbox': read_iou,
    }[args.stat]

    exp_name = args.exp_name
    if args.stat in ['acc', 'pc', 'ioubbox']:
        filename = 'test' if args.test else 'reconstructed'
        if args.suffix:
            filename += f'_{args.suffix}'
        if args.ckpt:
            filename += f'_{args.ckpt}'
        filename += f'_{args.stat}_stat.txt'
    elif args.stat == 'silhouette':
        filename = f'all_zs_ckpt{args.ckpt}_{args.stat}.txt'
    
    result_path = os.path.join(args.proj_dir, exp_name, 'results', filename)
    if args.stat in ['acc', 'pc', 'ioubbox']:
        csv_path = f'../{filename.split(".")[0][:-5]}s.csv'
    else:
        csv_path = f'../all_zs_{args.stat}s.csv'
    dic = read_fn(result_path)
    write_to_csv(csv_path, dic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_dir', type=str, default='../proj_log')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--stat', type=str, choices=['acc', 'pc', 'silhouette', 'ioubbox'])
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    save_to_csv(args)




