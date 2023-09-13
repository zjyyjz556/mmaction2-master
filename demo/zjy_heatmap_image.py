#%%
import os
import cv2
import os.path as osp
import decord
import numpy as np
import matplotlib.pyplot as plt
import urllib
import moviepy.editor as mpy
import random as rd
from mmpose.apis import vis_pose_result
from mmpose.models import TopDown
from mmcv import load, dump


from PIL import Image

import pickle

# We assume the annotation is already prepared
gym_train_ann_file = '../data/skeleton/gym_train.pkl'
gym_val_ann_file = '../data/skeleton/gym_val.pkl'
#ntu60_xsub_train_ann_file = '../data/skeleton/ntu60_xsub_train.pkl'
ntu60_xsub_train_ann_file = '/home/zeng_jy/mmaction2-master/data/skeleton/ntu60_xsub_train.pkl'
#ntu60_xsub_val_ann_file = '../data/skeleton/ntu60_xsub_val.pkl'
ntu60_xsub_val_ann_file = '/home/zeng_jy/mmaction2-master/data/skeleton/ntu60_xsub_val.pkl'



FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.6
FONTCOLOR = (255, 255, 255)
BGBLUE = (0, 119, 182)
THICKNESS = 1
LINETYPE = 1




def add_label(frame, label, BGCOLOR=BGBLUE):
    threshold = 30

    def split_label(label):
        label = label.split()
        lines, cline = [], ''
        for word in label:
            if len(cline) + len(word) < threshold:
                cline = cline + ' ' + word
            else:
                lines.append(cline)
                cline = word
        if cline != '':
            lines += [cline]
        return lines

    if len(label) > 30:
        label = split_label(label)
    else:
        label = [label]
    label = ['Action: '] + label

    sizes = []
    for line in label:
        sizes.append(cv2.getTextSize(line, FONTFACE, FONTSCALE, THICKNESS)[0])
    box_width = max([x[0] for x in sizes]) + 10
    text_height = sizes[0][1]
    box_height = len(sizes) * (text_height + 6)

    cv2.rectangle(frame, (0, 0), (box_width, box_height), BGCOLOR, -1)
    for i, line in enumerate(label):
        location = (5, (text_height + 6) * i + text_height + 3)
        cv2.putText(frame, line, location, FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
    return frame


def vis_skeleton(vid_path, anno, category_name=None, ratio=0.5):
    vid = decord.VideoReader(vid_path)
    frames = [x.asnumpy() for x in vid]

    h, w, _ = frames[0].shape
    new_shape = (int(w * ratio), int(h * ratio))
    frames = [cv2.resize(f, new_shape) for f in frames]

    assert len(frames) == anno['total_frames']
    # The shape is N x T x K x 3
    kps = np.concatenate([anno['keypoint'], anno['keypoint_score'][..., None]], axis=-1)
    kps[..., :2] *= ratio
    # Convert to T x N x K x 3
    kps = kps.transpose([1, 0, 2, 3])
    vis_frames = []

    # we need an instance of TopDown model, so build a minimal one
    model = TopDown(backbone=dict(type='ShuffleNetV1'))

    for f, kp in zip(frames, kps):
        bbox = np.zeros([0, 4], dtype=np.float32)
        result = [dict(bbox=bbox, keypoints=k) for k in kp]
        vis_frame = vis_pose_result(model, f, result)

        if category_name is not None:
            vis_frame = add_label(vis_frame, category_name)

        vis_frames.append(vis_frame)
    return vis_frames




keypoint_pipeline = [
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(type='GeneratePoseTarget', sigma=0.6, use_score=True, with_kp=True, with_limb=False)
]

limb_pipeline = [
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(type='GeneratePoseTarget', sigma=0.6, use_score=True, with_kp=False, with_limb=True)
]

from mmaction.datasets.pipelines import Compose


def get_pseudo_heatmap(anno, flag='keypoint'):
    assert flag in ['keypoint', 'limb']
    pipeline = Compose(keypoint_pipeline if flag == 'keypoint' else limb_pipeline)
    return pipeline(anno)['imgs']


def vis_heatmaps(heatmaps, channel=-1, ratio=8):
    # if channel is -1, draw all keypoints / limbs on the same map
    import matplotlib.cm as cm
    h, w, _ = heatmaps[0].shape
    newh, neww = int(h * ratio), int(w * ratio)

    if channel == -1:
        heatmaps = [np.max(x, axis=-1) for x in heatmaps]
    cmap = cm.viridis
    heatmaps = [(cmap(x)[..., :3] * 255).astype(np.uint8) for x in heatmaps]
    heatmaps = [cv2.resize(x, (neww, newh)) for x in heatmaps]
    return heatmaps

# The name list of
ntu_categories = ['drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup',
                  'throw', 'sitting down', 'standing up (from sitting position)', 'clapping', 'reading',
                  'writing', 'tear up paper', 'wear jacket', 'take off jacket', 'wear a shoe',
                  'take off a shoe', 'wear on glasses', 'take off glasses', 'put on a hat/cap',
                  'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something',
                  'reach into pocket', 'hopping (one foot jumping)', 'jump up',
                  'make a phone call/answer phone', 'playing with phone/tablet', 'typing on a keyboard',
                  'pointing to something with finger', 'taking a selfie', 'check time (from watch)',
                  'rub two hands together', 'nod head/bow', 'shake head', 'wipe face', 'salute',
                  'put the palms together', 'cross hands in front (say stop)', 'sneeze/cough',
                  'staggering', 'falling', 'touch head (headache)', 'touch chest (stomachache/heart pain)',
                  'touch back (backache)', 'touch neck (neckache)', 'nausea or vomiting condition',
                  'use a fan (with hand or paper)/feeling warm', 'punching/slapping other person',
                  'kicking other person', 'pushing other person', 'pat on back of other person',
                  'point finger at the other person', 'hugging other person',
                  'giving something to other person', "touch other person's pocket", 'handshaking',
                  'walking towards each other', 'walking apart from each other']
ntu_annos = load(ntu60_xsub_train_ann_file) + load(ntu60_xsub_val_ann_file)

'''
ntu_root = 'ntu_samples/'
#ntu_root = '/home/zeng_jy/mmaction2-master/demo/ntu_sample.avi'
#ntu_root = '/home/zeng_jy/mmaction2-master/demo/ntu_samples'
ntu_vids = os.listdir(ntu_root)
# visualize pose of which video? index in 0 - 50.
idx = 20
vid = ntu_vids[idx]

frame_dir = vid.split('.')[0]
vid_path = osp.join(ntu_root, vid)
anno = [x for x in ntu_annos if x['frame_dir'] == frame_dir.split('_')[0]][0]

keypoint_heatmap = get_pseudo_heatmap(anno)
keypoint_mapvis = vis_heatmaps(keypoint_heatmap)
'''

path_heatmap = r'/home/zeng_jy/mmaction2-master/data/posec3d/ntu60_xsub_val.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
val_pkl = open(path_heatmap, 'rb')
data_pkl = pickle.load(val_pkl)

for index in data_pkl:
    read_heatmap_data = np.load(r"/home/zeng_jy/mmaction2-master/heatmap4/{}results.npy".format(index['frame_dir']),
                                allow_pickle=True).item()
    read_heatmap = read_heatmap_data['imgs'].squeeze()
    read_heatmap = np.transpose(read_heatmap, (1, 2, 3, 0))
    keypoint_mapvis = vis_heatmaps(read_heatmap)
    path = r"/home/zeng_jy/mmaction2-master/demo/heatmap_image/{}".format(index['frame_dir'])  # 定义一个变量储存要指定的文件夹目录
    if not os.path.exists("/home/zeng_jy/mmaction2-master/demo/heatmap_image/{}".format(index['frame_dir'])):  # 没有这个文件目录则新建一个
        os.mkdir(path)  # 创建G盘文件名为 hello 的文件夹
        for i in range(len(keypoint_mapvis)):
            im = Image.fromarray(keypoint_mapvis[i])
            im.save("/home/zeng_jy/mmaction2-master/demo/heatmap_image/{}/{}.jpeg".format(index['frame_dir'], i))
print("val_done")

path_heatmap = r'/home/zeng_jy/mmaction2-master/data/posec3d/ntu60_xsub_train.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
train_pkl = open(path_heatmap, 'rb')
data_pkl = pickle.load(train_pkl)

for index in data_pkl:
    read_heatmap_data = np.load(r"/home/zeng_jy/mmaction2-master/heatmap4/{}results.npy".format(index['frame_dir']),
                                allow_pickle=True).item()
    read_heatmap = read_heatmap_data['imgs'].squeeze()
    read_heatmap = np.transpose(read_heatmap, (1, 2, 3, 0))
    keypoint_mapvis = vis_heatmaps(read_heatmap)
    path = r"/home/zeng_jy/mmaction2-master/demo/heatmap_image/{}".format(index['frame_dir'])  # 定义一个变量储存要指定的文件夹目录
    if not os.path.exists("/home/zeng_jy/mmaction2-master/demo/heatmap_image/{}".format(index['frame_dir'])):  # 没有这个文件目录则新建一个
        os.mkdir(path)  # 创建G盘文件名为 hello 的文件夹
        for i in range(len(keypoint_mapvis)):
            im = Image.fromarray(keypoint_mapvis[i])
            im.save("/home/zeng_jy/mmaction2-master/demo/heatmap_image/{}/{}.jpeg".format(index['frame_dir'], i))
print("train_done")

'''
read_heatmap_data = np.load(r"/home/zeng_jy/mmaction2-master/heatmap4/S008C002P007R002A047results.npy", allow_pickle=True).item()
#read_heatmap_data = np.load(r"/home/zeng_jy/mmaction2-master/heatmap4/S006C003P022R001A049results.npy", allow_pickle=True).item()
read_heatmap = read_heatmap_data['imgs'].squeeze()
#read_heatmap = read_heatmap_data['imgs'].reshape(17,640,64,64)
read_heatmap = np.transpose(read_heatmap, (1, 2, 3, 0))
keypoint_mapvis = vis_heatmaps(read_heatmap)


im = Image.fromarray(keypoint_mapvis[0])
im.save("/home/zeng_jy/mmaction2-master/demo/val/heatval1.jpeg")
im = Image.fromarray(keypoint_mapvis[20])
im.save("/home/zeng_jy/mmaction2-master/demo/val/heatval11.jpeg")
im = Image.fromarray(keypoint_mapvis[30])
im.save("/home/zeng_jy/mmaction2-master/demo/val/heatval111.jpeg")
#im = Image.fromarray(keypoint_mapvis[47])
#im.save("heatval111.jpeg")
#keypoint_mapvis = [add_label(f, gym_categories[anno['label']]) for f in keypoint_mapvis]
keypoint_mapvis = [add_label(f, ntu_categories[read_heatmap_data['label']]) for f in keypoint_mapvis]
im = Image.fromarray(keypoint_mapvis[0])
im.save("/home/zeng_jy/mmaction2-master/demo/val/heatval2.jpeg")
im = Image.fromarray(keypoint_mapvis[20])
im.save("/home/zeng_jy/mmaction2-master/demo/val/heatval22.jpeg")
im = Image.fromarray(keypoint_mapvis[30])
im.save("/home/zeng_jy/mmaction2-master/demo/val/heatval222.jpeg")
#im = Image.fromarray(keypoint_mapvis[47])
#im.save("heatval222.jpeg")

vid = mpy.ImageSequenceClip(keypoint_mapvis, fps=24)
vid.ipython_display()
'''