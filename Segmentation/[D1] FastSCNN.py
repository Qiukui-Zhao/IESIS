import os
from mmengine import Config

# Change directory to mmsegmentation
os.chdir('mmsegmentation')

# Load and update the Fast-SCNN config
cfg = Config.fromfile('configs/fastscnn/fast_scnn_8xb4-160k_cityscapes-512x1024.py')

# Modify the dataset pipeline
dataset_cfg = Config.fromfile('./configs/pipeline.py')
cfg.merge_from_dict(dataset_cfg)

# Set number of classes
NUM_CLASS = 5
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head[0].norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head[1].norm_cfg = cfg.norm_cfg
cfg.model.decode_head.num_classes = NUM_CLASS
cfg.model.auxiliary_head[0]['num_classes'] = NUM_CLASS
cfg.model.auxiliary_head[1]['num_classes'] = NUM_CLASS

# Update training parameters
cfg.train_dataloader.batch_size = 4
cfg.test_dataloader = cfg.val_dataloader
cfg.work_dir = './work_dirs/FastSCNN'
cfg.train_cfg.max_iters = 100000
cfg.train_cfg.val_interval = 500
cfg.default_hooks.logger.interval = 100
cfg.default_hooks.checkpoint.interval = 2500
cfg.default_hooks.checkpoint.max_keep_ckpts = 2
cfg.default_hooks.checkpoint.save_best = 'mIoU'
cfg['randomness'] = dict(seed=0)

# Save the updated configuration
cfg.dump('FastSCNN.py')

# Run the training script
os.system('python tools/train.py FastSCNN.py')
