from config import ConfigMLM
from dataset.mlm_dataset import get_dataloader
from trainer import TrainerMLM
from utils.file_utils import cycle
from tqdm import tqdm
from collections import OrderedDict


def main():
    cfg = ConfigMLM('train')

    tr_agent = TrainerMLM(cfg)

    if cfg.cont:
        tr_agent.load_ckpt(cfg.ckpt)

    train_loader = get_dataloader('train', cfg)
    val_loader = get_dataloader('validation', cfg)
    val_loader = cycle(val_loader)
    
    clock = tr_agent.clock
    
    for e in range(clock.epoch, cfg.nr_epochs):
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            outputs, losses = tr_agent.train_func(data)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            if clock.step % cfg.val_frequency == 0:
                data = next(val_loader)
                outputs, losses = tr_agent.val_func(data)

            clock.tick()

            tr_agent.update_learning_rate()
        
        clock.tock()
        
        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()
        
        tr_agent.save_ckpt('latest')
        
if __name__ == '__main__':
    main()