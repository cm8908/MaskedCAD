# MaskedCAD - A Masked Language Model Learning CAD
##  Train
For basic MLM (without contrastive learning or autoencoding), run:
```bash
# `mask_ratio` controls how many tokens to be masked
# `include_eos` decides whether or not to include <EOS> token in learning sequences
python train_mlm.py --proj_dir my_mlm --data_root /path/to/your/data --exp_name my_exp --mask_ratio 0.15 --include_eos
```

For MLM with contrastive learning (CL), run:
```bash
python train_cl_mlm.py --proj_dir my_cl_mlm --data_root /path/to/your/data --exp_name my_exp
```

For MLM with dropout-based CL (such as in SimCSE), run:
```bash
python train_dropout_cl_mlm.py --proj_dir my_cl_mlm --data_root /path/to/your/data --exp_name my_exp
```