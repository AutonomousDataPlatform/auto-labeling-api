import os.path as osp
import numpy as np
import cv2
import os
import json
import torchvision
from .base_dataset import BaseDataset
from clrnet.utils.tusimple_metric import LaneEval
from .registry import DATASETS
import logging
import random
import torch

@DATASETS.register_module
class SDLane(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None, transforms=None, ori_img_h=1208, ori_img_w=1920, cut_height=550):
        super().__init__(data_root, split, processes, cfg)
        self.data_root = data_root
        self.datalist_path = osp.join(data_root, 'train_list.txt')
        self.load_annotations()
        self.h_samples = None  # y 좌표의 가변 간격 처리
        self.num_classes = cfg.get('num_classes', 2)
        self.transforms = transforms
        self.ori_img_h = ori_img_h
        self.ori_img_w = ori_img_w
        self.cut_height = cut_height

    def load_datalist(self):
        with open(self.datalist_path) as f:
            datalist = [line.rstrip('\n') for line in f]
        return datalist

    def get_label(self, datalist, idx):
        """
        returns the corresponding label path for each image path
        """
        image_path = datalist[idx]
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.json')
        return image_path, label_path

    def load_json(self, label_path):
        with open(label_path, "r") as f:
            annotation = json.load(f)
        return annotation

    def load_annotations(self):
        self.logger.info('Loading SDLane annotations...')
        self.data_infos = []
        max_lanes = 0
        datalist = self.load_datalist()

        for idx in range(len(datalist)):
            img_rel_path, label_rel_path = self.get_label(datalist, idx)
            img_path = osp.join(self.data_root, img_rel_path)
            label_path = osp.join(self.data_root, label_rel_path)
            mask_rel_path = img_rel_path.replace('images', 'masks').replace('.jpg', '.png')
            mask_path = osp.join(self.data_root, mask_rel_path)

            annotation = self.load_json(label_path)
            lanes= []
            for lane in annotation['geometry']:
                y_samples = [point[1] for point in lane]
                gt_lanes = [point[0] for point in lane]
                lane_points = [(x, y) for x, y in zip(gt_lanes, y_samples) if x >= 0]
                if lane_points:
                    lanes.append(lane_points)
            max_lanes = max(max_lanes, len(lanes))
            self.data_infos.append({
                'img_path': img_path,
                'img_name': img_rel_path,
                'label_path': label_path,
                'lanes': lanes,
                'mask_path': mask_path
            })

        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes

    def pred2lanes(self, pred):
        lanes = []
        for lane in pred:
            xs = lane[0]
            ys = lane[1]
            # lane_points = [(int(x * self.cfg.ori_img_w), int(y * self.cfg.ori_img_h)) for x, y in zip(xs, ys) if x >= 0]
            lane_points = [(int(x * self.ori_img_w), int(y * self.ori_img_h)) for x, y in zip(xs, ys) if x >= 0]
            lanes.append(lane_points)
        return lanes

    def pred2sdlaneformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_sdlane_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes)):
            line = self.pred2sdlaneformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate(self, predictions, output_basedir, runtimes=None):
        pred_filename = os.path.join(output_basedir, 'sdlane_predictions.json')
        self.save_sdlane_predictions(predictions, pred_filename, runtimes)
        result, acc = LaneEval.bench_one_submit(pred_filename, self.cfg.test_json_file)
        self.logger.info(result)
        return acc
    