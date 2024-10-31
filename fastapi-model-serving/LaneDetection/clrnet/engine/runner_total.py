import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Subset, DataLoader

from clrnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from clrnet.datasets import build_dataloader
from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel
from mmcv.parallel import DataContainer
from functools import partial
from mmcv.parallel import collate


class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net,
                                  device_ids=range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss'].sum()
            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = self.build_combined_dataloader('train')

        self.recorder.logger.info('Start training...')
        start_epoch = 0
        if self.cfg.resume_from:
            start_epoch = resume_network(self.cfg.resume_from, self.net,
                                         self.optimizer, self.scheduler,
                                         self.recorder)
        for epoch in range(start_epoch, self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch +
                    1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch +
                    1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()


    def test(self):
        if not self.test_loader:
            self.test_loader = self.build_combined_dataloader('test')

        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                self.cfg.ori_img = Image.open(Path(data['meta'].data[0][0]['full_img_path']))
                self.cfg.ori_img_w, self.cfg.ori_img_h = self.cfg.ori_img.size
                if self.cfg.ori_img_w == 1280 and self.cfg.ori_img_h == 720:
                    self.cfg.cut_height = 160
                elif self.cfg.ori_img_w == 1640 and self.cfg.ori_img_h == 590:
                    self.cfg.cut_height = 270
                elif self.cfg.ori_img_w == 1920 and self.cfg.ori_img_h == 1208:
                    self.cfg.cut_height = 550
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.test_loader.dataset.view(output, data['meta'])

        metric = self.test_loader.dataset.evaluate(predictions,
                                                   self.cfg.work_dir)
        if metric is not None:
            self.recorder.logger.info('metric: ' + str(metric))

    def validate(self):
        if not self.val_loader:
            self.val_loader = self.build_combined_dataloader('val')

        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                # dataset_idx = self.check_dataset(data['meta'].data[0][0]['full_img_path'])
                self.cfg.ori_img = Image.open(Path(data['meta'].data[0][0]['full_img_path']))
                self.cfg.ori_img_w, self.cfg.ori_img_h = self.cfg.ori_img.size
                if self.cfg.ori_img_w == 1280 and self.cfg.ori_img_h == 720:
                    self.cfg.cut_height = 160
                elif self.cfg.ori_img_w == 1640 and self.cfg.ori_img_h == 590:
                    self.cfg.cut_height = 270
                elif self.cfg.ori_img_w == 1920 and self.cfg.ori_img_h == 1208:
                    self.cfg.cut_height = 550
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        metric = self.val_loader.dataset.evaluate(predictions,
                                                  self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))

    def save_ckpt(self, is_best=True):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder,
                   is_best)
        
    def build_combined_dataloader(self, split):
        # 데이터셋을 합쳐서 사용할 때 사용
        dataloaders = []
        for dataset_cfg in self.cfg.datasets:
            dataloader = build_dataloader(dataset_cfg[split], self.cfg, is_train=(split == 'train'))
            dataloaders.append(dataloader)
        
        combined_dataset = torch.utils.data.ConcatDataset(dataloader.dataset for dataloader in dataloaders)
        combined_loader = torch.utils.data.DataLoader(combined_dataset,
                                                      batch_size=self.cfg.batch_size,
                                                      shuffle=(split == 'train'),
                                                      num_workers=self.cfg.workers,
                                                      pin_memory=False,
                                                      drop_last=False,
                                                      collate_fn=partial(collate, samples_per_gpu=self.cfg.batch_size // self.cfg.gpus))

        return combined_loader
    
    def check_dataset(path):
        idx = 0
        if 'tusimple' in path.lower():
            idx = 0
        elif 'culane' in path.lower():
            idx = 1
        elif 'sdlane' in path.lower():
            idx = 2
        else:
            raise ValueError('Unknown dataset')
        
        return idx