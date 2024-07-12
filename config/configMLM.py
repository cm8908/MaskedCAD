import argparse
import os
from cadlib.macro import MAX_TOTAL_LEN
from config import ConfigCL
from trainer.finetuner import CLASSIFIERS


class ConfigMLM(ConfigCL):
    def __init__(self, phase):
        super(ConfigMLM, self).__init__(phase)
        # self.args_dim += 1
        # self.n_commands += 1
        self.loss_weights.update(
            {
                'loss_cmd_weight_mlm': 1.0,
                'loss_args_weight_mlm': 2.0
            }
        )
        self.only_mlm = True

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--max_total_len', type=int, default=MAX_TOTAL_LEN)

        parser.add_argument('--proj_dir', type=str, default="proj_log", help="path to project folder where models and logs will be saved")
        parser.add_argument('--data_root', type=str, default="../datasets/cad_data", help="path to source data folder")
        parser.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1], help="name of this experiment")
        parser.add_argument('-g', '--gpu_ids', type=str, default='0', help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

        parser.add_argument('--batch_size', type=int, default=512, help="batch size")
        parser.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

        parser.add_argument('--nr_epochs', type=int, default=100, help="total number of epochs to train")
        parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
        parser.add_argument('--grad_clip', type=float, default=1.0, help="initial learning rate")
        parser.add_argument('--warmup_step', type=int, default=2000, help="step size for learning rate warm up")
        parser.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
        parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
        parser.add_argument('--save_frequency', type=int, default=500, help="save models every x epochs")
        parser.add_argument('--val_frequency', type=int, default=10, help="run validation every x iterations")
        parser.add_argument('--vis_frequency', type=int, default=2000, help="visualize output every x iterations")
        parser.add_argument('--augment', action='store_true', help="use random data augmentation")
        parser.add_argument('-t', '--tag', type=str, default=None)

        parser.add_argument('--fp32', dest='fp16', action='store_false', default=True)

        parser.add_argument('--keep_seq_len', action='store_true', help="keep z variable's length as original sequence length. (1 if set False)")

        parser.add_argument('--mask_ratio', type=float, default=0.15)
        # parser.add_argument('--encoder_type', type=str, default='DeepCAD', choices=['DeepCAD', 'SkexGen'])
        parser.add_argument('--include_eos', action='store_true', default=False)
        
        if self.is_test:
            parser.add_argument('-m', '--mode', type=str, choices=['rec', 'enc', 'dec'])
            parser.add_argument('-o', '--outputs', type=str, default=None)
            parser.add_argument('--z_path', type=str, default=None)
        if self.is_finetune:
            parser.add_argument('--ft_lr', type=float, default=1e-4)
            parser.add_argument('--ft_nr_epochs', type=int, default=100)
        if self.is_finetune or self.is_test:
            parser.add_argument('--classifier_type', type=str, choices=list(CLASSIFIERS.keys()), default='transformer_decoder')
            
        args = parser.parse_args()
        return parser, args