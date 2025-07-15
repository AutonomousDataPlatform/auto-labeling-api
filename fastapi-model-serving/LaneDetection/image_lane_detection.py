import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from clrnet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm
import json
from clrnet.engine.runner_total import Runner
from PIL import Image
import io
import sys
sys.path.append("~/AILabDataset/03_Shared_Repository/jonghyun/Project/iitp_bigdata/bigdata-auto-labeling/fastapi-model-serving/LaneDetection")
def get_lane_detector(device):
    cfg_path = './configs/clrnet/clr_resnet18_total.py'
    load_from = './weights/tusimple_r18.pth'
    cfg = Config.fromfile(cfg_path)
    model = build_net(cfg)
    model = torch.nn.parallel.DataParallel(model, device_ids = range(1)).cuda()
    model = load_network(model, load_from)
    model.eval()
    
    return model

def get_lane_detections(model, file, device):
    cfg_path = './configs/clrnet/clr_resnet18_total.py'
    cfg = Config.fromfile(cfg_path)
    processes = Process(cfg.val_process, cfg)
    
    ori_img = Image.open(io.BytesIO(file)).convert("RGB")
    ori_img = np.array(ori_img)
    rgb_img = ori_img.copy()
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
    
    ori_h, ori_w, _ = ori_img.shape
    cfg.ori_img_h, cfg.ori_img_w = ori_h, ori_w
    # print(ori_img.shape)
    if cfg.ori_img_w == 1280 and cfg.ori_img_h == 720:
        cfg.cut_height = 160
    elif cfg.ori_img_w == 1640 and cfg.ori_img_h == 590:
        cfg.cut_height = 270
    elif cfg.ori_img_w == 1920 and cfg.ori_img_h == 1208:
        cfg.cut_height = 550
    else:
        ori_img = resize_image(ori_img) # 1640x590
        cfg.cut_height = 270
        cfg.ori_img_h, cfg.ori_img_w, _ = ori_img.shape
    sx = cfg.ori_img_w / ori_w
    sy = cfg.ori_img_h / ori_h
            
    img = ori_img[cfg.cut_height:, :, :].astype(np.float32)
    data = {'img': img, 'lanes': []}
    data = processes(data)
    data['img'] = data['img'].unsqueeze(0)
    data.update({'img_path':"", 'ori_img':ori_img})
    
    # update model parameters
    model.module.heads.cfg.ori_img_w = cfg.ori_img_w
    model.module.heads.cfg.ori_img_h = cfg.ori_img_h
    model.module.heads.cfg.cut_height = cfg.cut_height
    
    with torch.no_grad():
        ret = model(data)
        lane_ret = model.module.heads.get_lanes(ret)
    data['lanes'] = lane_ret[0]
    
    lanes = [lane.to_array(cfg) for lane in data['lanes']]        
    lanes = [lane for lane in lanes if lane.size > 0]
    
    # 원본 이미지에 맞춰서 역 스케일
    lanes_ori = [np.column_stack([lane[:, 0] / sx, lane[:, 1] / sy]) for lane in lanes]
        
    image = imshow_lanes(rgb_img, lanes_ori)
    
    image = Image.fromarray(image)
    
    return image, lanes_ori

## sub function
def resize_keep_ratio(ori_img, target_w, target_h):
    h, w = ori_img.shape[:2]
    # 각 축마다 필요 스케일
    scale_w = target_w / w
    scale_h = target_h / h
    # 둘 중 작은 스케일에 맞춘다
    scale = min(scale_w, scale_h)

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(ori_img, (new_w, new_h))
    return resized

def resize_image(ori_img, color=(114,114,114)):
    resized_img = resize_keep_ratio(ori_img, 1640, 590)
    resized_h, resized_w = resized_img.shape[:2]

    pad_top = (590 - resized_h)
    pad_bottom = 0
    pad_left = (1640 - resized_w) // 2
    pad_right = (1640 - resized_w) - pad_left

    padded_img = cv2.copyMakeBorder(resized_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)

    return padded_img

def preprocess(file):
    img_h = 320
    img_w = 800
    
    ori_img = Image.open(io.BytesIO(file)).convert("RGB")
    ori_img = np.array(ori_img)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
    ori_img_w, ori_img_h, _ = ori_img.shape
    
    if ori_img_w == 1280 and ori_img_h == 720: # tuSimple
        cut_height = 160
    elif ori_img_w == 1640 and ori_img_h == 590: # culane
        cut_height = 270
    elif ori_img_w == 1920 and ori_img_h == 1208: # sdlane
        cut_height = 550
    else:
        cut_height = 0
    img = ori_img[cut_height:, :, :].astype(np.float32)
    
    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("/home/ailab/AILabDataset/03_Shared_Repository/jonghyun/Project/iitp_bigdata/bigdata-auto-labeling/fastapi-model-serving/LaneDetection/tmp/test.jpg", img)
    # img = torch.from_numpy(img).permute(2, 0, 1).float()
    print(img.shape)
    img = torch.from_numpy(img)
    
    data = {'img': img, 'lanes': []}
    data['img'] = data['img'].unsqueeze(0)
    data.update({'img_path':"", 'ori_img':ori_img})
    
    return data, ori_img_w, ori_img_h, cut_height



