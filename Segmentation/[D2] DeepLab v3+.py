import os
from mmengine import Config

# Change directory to the correct path only once
os.chdir('mmsegmentation')

# Load the DeepLabV3+ base configuration and dataset pipeline
cfg = Config.fromfile('./configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py')
dataset_cfg = Config.fromfile('./configs/pipeline.py')

# Merge dataset pipeline into the main configuration
cfg.merge_from_dict(dataset_cfg)

# Update configuration parameters
NUM_CLASS = 5
cfg.crop_size = (512, 512)
cfg.model.data_preprocessor.size = cfg.crop_size

# Update normalization configuration
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

# Update number of classes for the head
cfg.model.decode_head.num_classes = NUM_CLASS
cfg.model.auxiliary_head.num_classes = NUM_CLASS

# Update training settings
cfg.train_dataloader.batch_size = 4
cfg.work_dir = './work_dirs/DeepLabV3plus'
cfg.train_cfg.max_iters = 100000
cfg.train_cfg.val_interval = 500

# Update hooks and checkpoints
cfg.default_hooks.logger.interval = 100
cfg.default_hooks.checkpoint.interval = 2500
cfg.default_hooks.checkpoint.max_keep_ckpts = 1
cfg.default_hooks.checkpoint.save_best = 'mIoU'

# Set randomness for reproducibility
cfg['randomness'] = dict(seed=0)

# Save the updated configuration
cfg.dump('DeepLabV3plus.py')

# Run the training script with the correct configuration file
os.system('python tools/train.py DeepLabV3plus.py')
