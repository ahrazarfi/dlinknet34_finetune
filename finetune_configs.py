"""
Simple configuration presets for progressive finetuning
Focuses only on the progressive unfreezing strategy
"""

# Conservative unfreezing - good for small datasets
CONSERVATIVE_CONFIG = {
    'experiment_name': 'conservative_unfreeze',
    'learning_rate': 1e-4,  # Lower LR for stability
    'batch_size_per_gpu': 4,
    'total_epochs': 50,
    'unfreeze_schedule': {
        10: 1,  # Unfreeze encoder4 after 10 epochs
        20: 2,  # Unfreeze encoder3 after 20 epochs
        35: 3   # Unfreeze encoder2 after 35 epochs (if needed)
    },
    'description': 'Conservative unfreezing for small datasets or challenging domains'
}

# Aggressive unfreezing - good for larger datasets
AGGRESSIVE_CONFIG = {
    'experiment_name': 'aggressive_unfreeze',
    'learning_rate': 2e-4,  # Original LR
    'batch_size_per_gpu': 4,
    'total_epochs': 40,
    'unfreeze_schedule': {
        3: 1,   # Unfreeze encoder4 quickly
        8: 2,   # Unfreeze encoder3
        15: 3   # Unfreeze encoder2
    },
    'description': 'Faster unfreezing for larger datasets with good road coverage'
}

# Gradual unfreezing - balanced approach
GRADUAL_CONFIG = {
    'experiment_name': 'gradual_unfreeze',
    'learning_rate': 2e-4,
    'batch_size_per_gpu': 4,
    'total_epochs': 60,
    'unfreeze_schedule': {
        5: 1,   # Unfreeze encoder4
        15: 2,  # Unfreeze encoder3
        25: 3   # Unfreeze encoder2
    },
    'description': 'Balanced unfreezing schedule for most use cases'
}

# Decoder only - baseline for comparison
DECODER_ONLY_CONFIG = {
    'experiment_name': 'decoder_only',
    'learning_rate': 2e-4,
    'batch_size_per_gpu': 4,
    'total_epochs': 30,
    'unfreeze_schedule': {},  # Never unfreeze encoder
    'description': 'Only train decoder layers - baseline for comparison'
}

# Full unfreezing - for comparison
FULL_UNFREEZE_CONFIG = {
    'experiment_name': 'full_unfreeze',
    'learning_rate': 1e-4,  # Lower LR since more parameters
    'batch_size_per_gpu': 4,
    'total_epochs': 30,
    'unfreeze_schedule': {
        1: 3  # Unfreeze everything immediately
    },
    'description': 'Full unfreezing from start - for comparison'
}

# All available configs
PRESET_CONFIGS = {
    'conservative': CONSERVATIVE_CONFIG,
    'aggressive': AGGRESSIVE_CONFIG,
    'gradual': GRADUAL_CONFIG,
    'decoder_only': DECODER_ONLY_CONFIG,
    'full_unfreeze': FULL_UNFREEZE_CONFIG
}

def get_finetune_config(preset_name, **overrides):
    """
    Get a finetuning configuration by preset name
    
    Args:
        preset_name: One of 'conservative', 'aggressive', 'gradual', 'decoder_only', 'full_unfreeze'
        **overrides: Any configuration values to override
    
    Returns:
        Dictionary with complete configuration
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    config = PRESET_CONFIGS[preset_name].copy()
    
    # Apply any overrides
    config.update(overrides)
    
    # Add default values that are always needed
    defaults = {
        'log_dir': 'logs',
        'weights_dir': 'weights',
        'use_wandb': False,
        'wandb_project': 'dlink-rural-roads'
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config

def print_available_configs():
    """Print information about available configuration presets"""
    print("Available Finetuning Configurations:")
    print("=" * 50)
    
    for name, config in PRESET_CONFIGS.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Total Epochs: {config['total_epochs']}")
        print(f"  Unfreeze Schedule: {config['unfreeze_schedule']}")

def create_custom_config(experiment_name, data_root, pretrained_weights,
                        learning_rate=2e-4, total_epochs=50, 
                        unfreeze_epochs=None, **kwargs):
    """
    Create a custom configuration
    
    Args:
        experiment_name: Name for the experiment
        data_root: Path to training data
        pretrained_weights: Path to pretrained model weights
        learning_rate: Learning rate for training
        total_epochs: Total number of epochs
        unfreeze_epochs: List of epochs to unfreeze layers [encoder4_epoch, encoder3_epoch, encoder2_epoch]
        **kwargs: Additional configuration options
    
    Returns:
        Complete configuration dictionary
    """
    if unfreeze_epochs is None:
        unfreeze_epochs = [5, 15, 25]  # Default gradual schedule
    
    # Create unfreeze schedule
    unfreeze_schedule = {}
    for i, epoch in enumerate(unfreeze_epochs):
        if epoch > 0:  # Skip if epoch is 0 or negative
            unfreeze_schedule[epoch] = i + 1
    
    config = {
        'experiment_name': experiment_name,
        'data_root': data_root,
        'pretrained_weights': pretrained_weights,
        'learning_rate': learning_rate,
        'batch_size_per_gpu': 4,
        'total_epochs': total_epochs,
        'unfreeze_schedule': unfreeze_schedule,
        'log_dir': 'logs',
        'weights_dir': 'weights',
        'use_wandb': False,
        'wandb_project': 'dlink-rural-roads'
    }
    
    # Add any additional options
    config.update(kwargs)
    
    return config

if __name__ == '__main__':
    print_available_configs()
    
    print("\n" + "=" * 50)
    print("USAGE EXAMPLES:")
    print("=" * 50)
    
    print("\n1. Using preset configuration:")
    print("   python train_finetune_minimal.py --data_root dataset/custom/train \\")
    print("          --config conservative")
    
    print("\n2. Custom configuration in Python:")
    print("   config = create_custom_config(")
    print("       experiment_name='my_experiment',")
    print("       data_root='dataset/custom/train',")
    print("       pretrained_weights='weights/log01_dink34.th',")
    print("       unfreeze_epochs=[3, 10, 20],")
    print("       learning_rate=1e-4")
    print("   )")
    
    print("\n3. Available unfreeze strategies:")
    print("   - Conservative: Good for small datasets")
    print("   - Aggressive: Good for large datasets") 
    print("   - Gradual: Balanced approach")
    print("   - Decoder only: Baseline comparison")
    print("   - Full unfreeze: Traditional finetuning")