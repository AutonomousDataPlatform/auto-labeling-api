net = dict(type='Detector', )

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

num_points = 72
max_lanes = 10
sample_y = range(1500, 150, -10) # SDLane

heads = dict(type='CLRHead',
             num_priors=192,
             refine_layers=3,
             fc_hidden_dim=64,
             sample_points=36)

iou_loss_weight = 2.
cls_loss_weight = 6.
xyt_loss_weight = 0.5
seg_loss_weight = 1.0

img_h = 320
img_w = 800

dataset_number = 3
work_dirs = "work_dirs/clr/r18_total"
# tusimple: 0, culane: 1, sdlane: 2
# tusimple: 0
ori_img_w_0 = 1280
ori_img_h_0 = 720
cut_height_0 = 160
# culane: 1
ori_img_w_1 = 1640
ori_img_h_1 = 590
cut_height_1 = 270
# sdlane: 2
ori_img_w_2 = 1920
ori_img_h_2 = 1208
cut_height_2 = 550

# cut_height = 0 # KITTI
# ori_img_w = 1242 # KITTI
# ori_img_h = 375 # KITTI

neck = dict(type='FPN',
            in_channels=[128, 256, 512],
            out_channels=64,
            num_outs=3,
            attention=False)

test_parameters = dict(conf_threshold=0.40, nms_thres=50, nms_topk=max_lanes)

epochs = 10
batch_size = 5

optimizer = dict(type='AdamW', lr=1.0e-3)  # 3e-4 for batchsize 8
total_iter = (3616 // batch_size + 1) * epochs * 10000
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_ep = 1
save_ep = epochs

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),
]

# tusimple: 0, culane: 1, sdlane: 2
# tusimple: 0
dataset_path_0 = './data/tusimple'
dataset_type_0 = 'TuSimple'
test_json_file_0 = 'data/tusimple/test_label.json'
dataset_0 = dict(train=dict(
    type=dataset_type_0,
    data_root=dataset_path_0,
    split='trainval',
    processes=train_process,
    ori_img_h=ori_img_h_0,
    ori_img_w=ori_img_w_0,
    cut_height=cut_height_0,
),
val=dict(
    type=dataset_type_0,
    data_root=dataset_path_0,
    split='test',
    processes=val_process,
    ori_img_h=ori_img_h_0,
    ori_img_w=ori_img_w_0,
    cut_height=cut_height_0,
),
test=dict(
    type=dataset_type_0,
    data_root=dataset_path_0,
    split='test',
    processes=val_process,
    ori_img_h=ori_img_h_0,
    ori_img_w=ori_img_w_0,
    cut_height=cut_height_0,
))
# culane: 1
dataset_path_1 = './data/CULane'
dataset_type_1 = 'CULane'
dataset_1 = dict(train=dict(
    type=dataset_type_1,
    data_root=dataset_path_1,
    split='train',
    processes=train_process,
    ori_img_h=ori_img_h_1,
    ori_img_w=ori_img_w_1,
    cut_height=cut_height_1,
),
val=dict(
    type=dataset_type_1,
    data_root=dataset_path_1,
    split='test',
    processes=val_process,
    ori_img_h=ori_img_h_1,
    ori_img_w=ori_img_w_1,
    cut_height=cut_height_1,
),
test=dict(
    type=dataset_type_1,
    data_root=dataset_path_1,
    split='test',
    processes=val_process,
    ori_img_h=ori_img_h_1,
    ori_img_w=ori_img_w_1,
    cut_height=cut_height_1,
))
# sdlane: 2
dataset_path_2 = '/home/ailab/AILabDataset/01_Open_Dataset/35_SDLane/SDLane/train'
dataset_type_2 = 'SDLane'
test_json_file_2 = 'data/sdlane/test_label.json'
dataset_2 = dict(train=dict(
    type=dataset_type_2,
    data_root=dataset_path_2,
    split='trainval',
    processes=train_process,
    ori_img_h=ori_img_h_2,
    ori_img_w=ori_img_w_2,
    cut_height=cut_height_2,
),
val=dict(
    type=dataset_type_2,
    data_root=dataset_path_2,
    split='test',
    processes=val_process,
    ori_img_h=ori_img_h_2,
    ori_img_w=ori_img_w_2,
    cut_height=cut_height_2,
),
test=dict(
    type=dataset_type_2,
    data_root=dataset_path_2,
    split='test',
    processes=val_process,
    ori_img_h=ori_img_h_2,
    ori_img_w=ori_img_w_2,
    cut_height=cut_height_2,
))

datasets = [dataset_0, dataset_1, dataset_2]

workers = 10
log_interval = 100
# seed = 0
num_classes = 6 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
