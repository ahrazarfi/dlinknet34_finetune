
import argparse, json, os
from finetune_configs import get_config
import train_finetune as train

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--preset', choices=['conservative','aggressive','gradual','decoder_only','full'],
                   default='gradual')
    p.add_argument('--weights', default='weights/log01_dink34.th')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--project', default='dink-rural-roads')
    args = p.parse_args()

    cfg = get_config(args.preset)
    cfg.update({
        'data_root':args.data,
        'pretrained':args.weights,
        'exp':cfg['experiment_name'],
        'weights_dir':'weights',
        'epochs':args.epochs or cfg['total_epochs'],
        'batch_per_gpu':4,
        'wandb':args.wandb,
        'wandb_project':args.project,
        'val_frac':0.2,
        'img_log_freq':5
    })
    print(json.dumps(cfg, indent=2))
    train.run(cfg)

if __name__=='__main__':
    main()
