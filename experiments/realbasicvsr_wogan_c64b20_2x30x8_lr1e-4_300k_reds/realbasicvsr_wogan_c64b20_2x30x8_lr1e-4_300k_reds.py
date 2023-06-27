exp_name = 'realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds'
scale = 2
model = dict(
    type='RealBasicVSR',
    generator=dict(
        type='RealBasicVSRNet',
        mid_channels=64,
        num_propagation_blocks=20,
        num_cleaning_blocks=20,
        dynamic_refine_thres=255,
        spynet_pretrained=
        'https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth',
        is_fix_cleaning=False,
        is_sequential_cleaning=False),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    cleaning_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    is_use_sharpened_gt_in_pixel=True,
    is_use_ema=True)
train_cfg = dict()
test_cfg = dict(metrics=['PSNR'], crop_border=0)
train_dataset_type = 'SRFolderMultipleGTDataset'
val_dataset_type = 'SRFolderMultipleGTDataset'
train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='FixedCrop', keys=['gt'], crop_size=(128, 128)),
    dict(type='RescaleToZeroOne', keys=['gt']),
    dict(type='Flip', keys=['gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['gt'], transpose_ratio=0.5),
    dict(type='MirrorSequence', keys=['gt']),
    dict(
        type='UnsharpMasking',
        keys=['gt'],
        kernel_size=51,
        sigma=0,
        weight=0.5,
        threshold=10),
    dict(type='CopyValues', src_keys=['gt_unsharp'], dst_keys=['lq']),
    dict(
        type='RandomBlur',
        params=dict(
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 3],
            sigma_y=[0.2, 3],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2],
            sigma_x_step=0.02,
            sigma_y_step=0.02,
            rotate_angle_step=0.31416,
            beta_gaussian_step=0.05,
            beta_plateau_step=0.1,
            omega_step=0.0628),
        keys=['lq']),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0.2, 0.7, 0.1],
            resize_scale=[0.15, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[
                0.3333333333333333, 0.3333333333333333, 0.3333333333333333
            ],
            resize_step=0.015,
            is_size_even=True),
        keys=['lq']),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 30],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 3],
            poisson_gray_noise_prob=0.4,
            gaussian_sigma_step=0.1,
            poisson_scale_step=0.005),
        keys=['lq']),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[30, 95], quality_step=3),
        keys=['lq']),
    dict(
        type='RandomVideoCompression',
        params=dict(
            codec=['libx264', 'h264', 'mpeg4'],
            codec_prob=[
                0.3333333333333333, 0.3333333333333333, 0.3333333333333333
            ],
            bitrate=[10000.0, 100000.0]),
        keys=['lq']),
    dict(
        type='RandomBlur',
        params=dict(
            prob=0.8,
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 1.5],
            sigma_y=[0.2, 1.5],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2],
            sigma_x_step=0.02,
            sigma_y_step=0.02,
            rotate_angle_step=0.31416,
            beta_gaussian_step=0.05,
            beta_plateau_step=0.1,
            omega_step=0.0628),
        keys=['lq']),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0.3, 0.4, 0.3],
            resize_scale=[0.3, 1.2],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[
                0.3333333333333333, 0.3333333333333333, 0.3333333333333333
            ],
            resize_step=0.03,
            is_size_even=True),
        keys=['lq']),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 25],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 2.5],
            poisson_gray_noise_prob=0.4,
            gaussian_sigma_step=0.1,
            poisson_scale_step=0.005),
        keys=['lq']),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[30, 95], quality_step=3),
        keys=['lq']),
    dict(
        type='DegradationsWithShuffle',
        degradations=[{
            'type': 'RandomVideoCompression',
            'params': {
                'codec': ['libx264', 'h264', 'mpeg4'],
                'codec_prob':
                [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                'bitrate': [10000.0, 100000.0]
            },
            'keys': ['lq']
        },
                      [{
                          'type': 'RandomResize',
                          'params': {
                              'target_size': (64, 64),
                              'resize_opt': ['bilinear', 'area', 'bicubic'],
                              'resize_prob': [
                                  0.3333333333333333, 0.3333333333333333,
                                  0.3333333333333333
                              ]
                          }
                      }, {
                          'type': 'RandomBlur',
                          'params': {
                              'prob': 0.8,
                              'kernel_size': [7, 9, 11, 13, 15, 17, 19, 21],
                              'kernel_list': ['sinc'],
                              'kernel_prob': [1],
                              'omega': [1.0472, 3.1416],
                              'omega_step': 0.0628
                          }
                      }]],
        keys=['lq']),
    dict(type='Quantize', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq', 'gt', 'gt_unsharp']),
    dict(
        type='Collect', keys=['lq', 'gt', 'gt_unsharp'], meta_keys=['gt_path'])
]
val_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]
test_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:08d}.png'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]
data = dict(
    workers_per_gpu=10,
    train_dataloader=dict(
        samples_per_gpu=2, drop_last=True, persistent_workers=False),
    val_dataloader=dict(samples_per_gpu=1, persistent_workers=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=150,
        dataset=dict(
            type='SRFolderMultipleGTDataset',
            lq_folder='./data/train_sharp_sub',
            gt_folder='./data/train_sharp_sub',
            num_input_frames=15,
            pipeline=[
                dict(type='GenerateSegmentIndices', interval_list=[1]),
                dict(
                    type='LoadImageFromFileList',
                    io_backend='disk',
                    key='gt',
                    channel_order='rgb'),
                dict(type='FixedCrop', keys=['gt'], crop_size=(128, 128)),
                dict(type='RescaleToZeroOne', keys=['gt']),
                dict(
                    type='Flip',
                    keys=['gt'],
                    flip_ratio=0.5,
                    direction='horizontal'),
                dict(
                    type='Flip',
                    keys=['gt'],
                    flip_ratio=0.5,
                    direction='vertical'),
                dict(
                    type='RandomTransposeHW', keys=['gt'],
                    transpose_ratio=0.5),
                dict(type='MirrorSequence', keys=['gt']),
                dict(
                    type='UnsharpMasking',
                    keys=['gt'],
                    kernel_size=51,
                    sigma=0,
                    weight=0.5,
                    threshold=10),
                dict(
                    type='CopyValues',
                    src_keys=['gt_unsharp'],
                    dst_keys=['lq']),
                dict(
                    type='RandomBlur',
                    params=dict(
                        kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
                        kernel_list=[
                            'iso', 'aniso', 'generalized_iso',
                            'generalized_aniso', 'plateau_iso',
                            'plateau_aniso', 'sinc'
                        ],
                        kernel_prob=[
                            0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1
                        ],
                        sigma_x=[0.2, 3],
                        sigma_y=[0.2, 3],
                        rotate_angle=[-3.1416, 3.1416],
                        beta_gaussian=[0.5, 4],
                        beta_plateau=[1, 2],
                        sigma_x_step=0.02,
                        sigma_y_step=0.02,
                        rotate_angle_step=0.31416,
                        beta_gaussian_step=0.05,
                        beta_plateau_step=0.1,
                        omega_step=0.0628),
                    keys=['lq']),
                dict(
                    type='RandomResize',
                    params=dict(
                        resize_mode_prob=[0.2, 0.7, 0.1],
                        resize_scale=[0.15, 1.5],
                        resize_opt=['bilinear', 'area', 'bicubic'],
                        resize_prob=[
                            0.3333333333333333, 0.3333333333333333,
                            0.3333333333333333
                        ],
                        resize_step=0.015,
                        is_size_even=True),
                    keys=['lq']),
                dict(
                    type='RandomNoise',
                    params=dict(
                        noise_type=['gaussian', 'poisson'],
                        noise_prob=[0.5, 0.5],
                        gaussian_sigma=[1, 30],
                        gaussian_gray_noise_prob=0.4,
                        poisson_scale=[0.05, 3],
                        poisson_gray_noise_prob=0.4,
                        gaussian_sigma_step=0.1,
                        poisson_scale_step=0.005),
                    keys=['lq']),
                dict(
                    type='RandomJPEGCompression',
                    params=dict(quality=[30, 95], quality_step=3),
                    keys=['lq']),
                dict(
                    type='RandomVideoCompression',
                    params=dict(
                        codec=['libx264', 'h264', 'mpeg4'],
                        codec_prob=[
                            0.3333333333333333, 0.3333333333333333,
                            0.3333333333333333
                        ],
                        bitrate=[10000.0, 100000.0]),
                    keys=['lq']),
                dict(
                    type='RandomBlur',
                    params=dict(
                        prob=0.8,
                        kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
                        kernel_list=[
                            'iso', 'aniso', 'generalized_iso',
                            'generalized_aniso', 'plateau_iso',
                            'plateau_aniso', 'sinc'
                        ],
                        kernel_prob=[
                            0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1
                        ],
                        sigma_x=[0.2, 1.5],
                        sigma_y=[0.2, 1.5],
                        rotate_angle=[-3.1416, 3.1416],
                        beta_gaussian=[0.5, 4],
                        beta_plateau=[1, 2],
                        sigma_x_step=0.02,
                        sigma_y_step=0.02,
                        rotate_angle_step=0.31416,
                        beta_gaussian_step=0.05,
                        beta_plateau_step=0.1,
                        omega_step=0.0628),
                    keys=['lq']),
                dict(
                    type='RandomResize',
                    params=dict(
                        resize_mode_prob=[0.3, 0.4, 0.3],
                        resize_scale=[0.3, 1.2],
                        resize_opt=['bilinear', 'area', 'bicubic'],
                        resize_prob=[
                            0.3333333333333333, 0.3333333333333333,
                            0.3333333333333333
                        ],
                        resize_step=0.03,
                        is_size_even=True),
                    keys=['lq']),
                dict(
                    type='RandomNoise',
                    params=dict(
                        noise_type=['gaussian', 'poisson'],
                        noise_prob=[0.5, 0.5],
                        gaussian_sigma=[1, 25],
                        gaussian_gray_noise_prob=0.4,
                        poisson_scale=[0.05, 2.5],
                        poisson_gray_noise_prob=0.4,
                        gaussian_sigma_step=0.1,
                        poisson_scale_step=0.005),
                    keys=['lq']),
                dict(
                    type='RandomJPEGCompression',
                    params=dict(quality=[30, 95], quality_step=3),
                    keys=['lq']),
                dict(
                    type='DegradationsWithShuffle',
                    degradations=[{
                        'type': 'RandomVideoCompression',
                        'params': {
                            'codec': ['libx264', 'h264', 'mpeg4'],
                            'codec_prob': [
                                0.3333333333333333, 0.3333333333333333,
                                0.3333333333333333
                            ],
                            'bitrate': [10000.0, 100000.0]
                        },
                        'keys': ['lq']
                    },
                                  [{
                                      'type': 'RandomResize',
                                      'params': {
                                          'target_size': (64, 64),
                                          'resize_opt':
                                          ['bilinear', 'area', 'bicubic'],
                                          'resize_prob': [
                                              0.3333333333333333,
                                              0.3333333333333333,
                                              0.3333333333333333
                                          ]
                                      }
                                  }, {
                                      'type': 'RandomBlur',
                                      'params': {
                                          'prob':
                                          0.8,
                                          'kernel_size':
                                          [7, 9, 11, 13, 15, 17, 19, 21],
                                          'kernel_list': ['sinc'],
                                          'kernel_prob': [1],
                                          'omega': [1.0472, 3.1416],
                                          'omega_step':
                                          0.0628
                                      }
                                  }]],
                    keys=['lq']),
                dict(type='Quantize', keys=['lq']),
                dict(type='FramesToTensor', keys=['lq', 'gt', 'gt_unsharp']),
                dict(
                    type='Collect',
                    keys=['lq', 'gt', 'gt_unsharp'],
                    meta_keys=['gt_path'])
            ],
            scale=2,
            test_mode=False)),
    val=dict(
        type='SRFolderMultipleGTDataset',
        lq_folder='./data/UDM10/BIx4',
        gt_folder='./data/UDM10/GT',
        pipeline=[
            dict(type='GenerateSegmentIndices', interval_list=[1]),
            dict(
                type='LoadImageFromFileList',
                io_backend='disk',
                key='lq',
                channel_order='rgb'),
            dict(
                type='LoadImageFromFileList',
                io_backend='disk',
                key='gt',
                channel_order='rgb'),
            dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
            dict(type='FramesToTensor', keys=['lq', 'gt']),
            dict(
                type='Collect',
                keys=['lq', 'gt'],
                meta_keys=['lq_path', 'gt_path', 'key'])
        ],
        scale=2,
        test_mode=True),
    test=dict(
        type='SRFolderMultipleGTDataset',
        lq_folder='./data/VideoLQ',
        gt_folder='./data/VideoLQ',
        pipeline=[
            dict(
                type='GenerateSegmentIndices',
                interval_list=[1],
                filename_tmpl='{:08d}.png'),
            dict(
                type='LoadImageFromFileList',
                io_backend='disk',
                key='lq',
                channel_order='rgb'),
            dict(type='RescaleToZeroOne', keys=['lq']),
            dict(type='FramesToTensor', keys=['lq']),
            dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
        ],
        scale=2,
        test_mode=True))
optimizers = dict(generator=dict(type='Adam', lr=0.0001, betas=(0.9, 0.99)))
total_iters = 1000
lr_config = dict(policy='Step', by_epoch=False, step=[400000], gamma=1)
checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
visual_config = None
custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.999))
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './experiments/realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpus = 1
