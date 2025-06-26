
""" Finetuning presets (unchanged) """
from pathlib import Path

CONSERVATIVE = dict(
    experiment_name='conservative',
    learning_rate=1e-4,
    total_epochs=50,
    unfreeze_schedule={10:1,20:2,35:3},
)
AGGRESSIVE = dict(
    experiment_name='aggressive',
    learning_rate=2e-4,
    total_epochs=40,
    unfreeze_schedule={3:1,8:2,15:3},
)
GRADUAL = dict(
    experiment_name='gradual',
    learning_rate=2e-4,
    total_epochs=60,
    unfreeze_schedule={5:1,15:2,25:3},
)
DECODER_ONLY = dict(
    experiment_name='decoder_only',
    learning_rate=2e-4,
    total_epochs=30,
    unfreeze_schedule={},
)
FULL = dict(
    experiment_name='full_unfreeze',
    learning_rate=1e-4,
    total_epochs=30,
    unfreeze_schedule={1:3},
)

PRESETS = {'conservative':CONSERVATIVE,'aggressive':AGGRESSIVE,
           'gradual':GRADUAL,'decoder_only':DECODER_ONLY,'full':FULL}

def get_config(name:str): return PRESETS[name].copy()
