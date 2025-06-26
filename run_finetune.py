# === run_finetune.py =========================================================
"""
Wrapper / experiment runner for progressive-unfreezing finetune script.
Focuses on validation + preset handling.
"""
import argparse, os, sys
from finetune_configs import (
    get_finetune_config,
    print_available_configs,
    create_custom_config
)
from train_finetune_minimal import train_with_progressive_unfreezing

def validate_setup(cfg) -> bool:
    print("Validating setup …")
    d = cfg['data_root']
    if not os.path.isdir(d):
        print(f"❌ data_root does not exist: {d}")
        return False
    sats  = [f for f in os.listdir(d) if '_sat'  in f]
    masks = [f for f in os.listdir(d) if '_mask' in f]
    if not sats:
        print("❌ no *_sat.* images found")
        return False
    if not masks:
        print("❌ no *_mask.* images found")
        return False
    if cfg.get('pretrained_weights') and \
       not os.path.exists(cfg['pretrained_weights']):
        print("⚠ pretrained weights not found → will train from scratch")
    os.makedirs(cfg.get('log_dir', 'logs'),     exist_ok=True)
    os.makedirs(cfg.get('weights_dir', 'weights'), exist_ok=True)
    print(f"✅ {len(sats)} images | {len(masks)} masks")
    return True


def main():
    pa = argparse.ArgumentParser(
        description="Progressive unfreeze experiment runner")
    pa.add_argument('--data',  required=True)
    pa.add_argument('--weights', default='weights/log01_dink34.th')

    pa.add_argument('--config', choices=[
        'conservative','aggressive','gradual','decoder_only','full_unfreeze'])
    pa.add_argument('--custom', action='store_true')
    pa.add_argument('--baseline-experiments', action='store_true')

    pa.add_argument('--experiment-name', default='custom_finetune')
    pa.add_argument('--learning-rate', type=float, default=2e-4)
    pa.add_argument('--epochs', type=int,  default=50)
    pa.add_argument('--unfreeze-epochs', type=int, nargs=3,
                    default=[5,15,25])

    pa.add_argument('--wandb', action='store_true')
    pa.add_argument('--wandb-project', default='dlink-rural-roads')

    pa.add_argument('--list-configs', action='store_true')
    pa.add_argument('--validate-only', action='store_true')
    args = pa.parse_args()

    if args.list_configs:
        print_available_configs(); return

    # choose / build config ----------------------------------------------------
    if args.config:
        cfg = get_finetune_config(args.config,
                                  data_root=args.data,
                                  pretrained_weights=args.weights,
                                  use_wandb=args.wandb and WANDB_AVAILABLE,
                                  wandb_project=args.wandb_project)
    elif args.custom:
        cfg = create_custom_config(
            experiment_name=args.experiment_name,
            data_root=args.data,
            pretrained_weights=args.weights,
            learning_rate=args.learning_rate,
            total_epochs=args.epochs,
            unfreeze_epochs=args.unfreeze_epochs,
            use_wandb=args.wandb and WANDB_AVAILABLE,
            wandb_project=args.wandb_project)
    else:  # default
        cfg = get_finetune_config('gradual',
                                  data_root=args.data,
                                  pretrained_weights=args.weights,
                                  use_wandb=args.wandb and WANDB_AVAILABLE,
                                  wandb_project=args.wandb_project)

    # add LR ladder override
    cfg['lr_ratios'] = {'decoder':1.0,'encoder4':0.1,'encoder3':0.03,'encoder2':0.01}

    # validate ----------------------------------------------------------------
    if not validate_setup(cfg):
        sys.exit("Setup invalid → abort")

    if args.validate_only:
        print("Validation only ✓");  return

    train_with_progressive_unfreezing(cfg)


if __name__ == '__main__':
    main()
