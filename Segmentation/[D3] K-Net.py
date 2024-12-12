import os
from mmengine import Config

# Change to the MMSegmentation project directory
os.chdir('mmsegmentation')
print("Current Directory:", os.getcwd())

# Load base configurations
cfg = Config.fromfile('./configs/knet/knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-512x512.py')
dataset_cfg = Config.fromfile('./configs/pipeline.py')

# Merge dataset-specific configurations into the main config
cfg.merge_from_dict(dataset_cfg)

# Define number of classes for the segmentation task
NUM_CLASS = 5
cfg.model.decode_head.kernel_generate_head.num_classes = NUM_CLASS
cfg.model.auxiliary_head.num_classes = NUM_CLASS

# Update training settings
cfg.train_dataloader.batch_size = 4  # Batch size for training
cfg.train_cfg.max_iters = 100000  # Maximum number of iterations
cfg.train_cfg.val_interval = 500  # Interval for validation

# Configure logging and checkpoint settings
cfg.default_hooks.logger.interval = 100  # Log interval
cfg.default_hooks.checkpoint.interval = 2500  # Checkpoint save interval
cfg.default_hooks.checkpoint.max_keep_ckpts = 2  # Maximum number of checkpoints to keep
cfg.default_hooks.checkpoint.save_best = 'mIoU'  # Save the best model based on mIoU

# Set normalization settings
cfg.norm_cfg = dict(type='BN', requires_grad=True)

# Set crop size (image size) for the model's data preprocessor
cfg.model.data_preprocessor.size = cfg.crop_size

# Specify working directory for saving outputs
cfg.work_dir = './work_dirs/KNet'

# Set random seed for reproducibility
cfg['randomness'] = dict(seed=0)

# Save the updated configuration to a new file
cfg.dump('KNet.py')
print("Updated configuration saved as 'KNet.py'")

# Start the training process
os.system('python tools/train.py KNet.py')
