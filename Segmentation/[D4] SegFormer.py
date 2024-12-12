import os
from mmengine import Config

# Set the working directory to 'mmsegmentation' (if needed)
os.chdir('mmsegmentation')

# Check the current working directory (ensure it is correct)
print("Current working directory:", os.getcwd())

# Load the configuration files
cfg = Config.fromfile('./configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py')
dataset_cfg = Config.fromfile('./configs/pipeline.py')

# Merge the dataset configuration into the main config
cfg.merge_from_dict(dataset_cfg)

# Set custom values for the configuration
NUM_CLASS = 5
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.num_classes = NUM_CLASS
cfg.train_dataloader.batch_size = 4
cfg.work_dir = './work_dirs/Segformer'
cfg.train_cfg.max_iters = 100000
cfg.train_cfg.val_interval = 500
cfg.default_hooks.logger.interval = 100
cfg.default_hooks.checkpoint.interval = 2500
cfg.default_hooks.checkpoint.max_keep_ckpts = 2
cfg.default_hooks.checkpoint.save_best = 'mIoU'

# Set the seed for randomness
cfg['randomness'] = dict(seed=0)

# Save the modified configuration to a new file
cfg.dump('./configs/Segformer.py')
print("Updated configuration saved as 'Segformer.py'")

# Start the training process
os.system('python tools/train.py Segformer.py')

