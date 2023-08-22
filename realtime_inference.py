import argparse
import glob
import os

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmedit.core import tensor2img

from realbasicvsr.models.builder import build_model

import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference script of RealBasicVSR')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
#    parser.add_argument('input_dir', help='directory of the input video')
#    parser.add_argument('output_dir', help='directory of the output video')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=None,
        help='maximum sequence length to be processed')
    parser.add_argument(
        '--is_save_as_png',
        type=bool,
        default=True,
        help='whether to save as png')
    parser.add_argument(
        '--fps', type=float, default=25, help='FPS of the output video')
    args = parser.parse_args()
    return args


def init_model(config, checkpoint=None):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
    model.cfg = config
    model.eval()
    return model


def main():
    args = parse_args()

    model = init_model(args.config, args.checkpoint)

    video = cv2.VideoCapture(0)

    video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    print("video start!")
    
    cnt = 0
    inputs = []
    while True:
        t0 = time.time()
        ret_val, img = video.read()
        if cnt == 0:
            img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
#            img = torch.from_numpy(img / 255.).permute(0, 1, 2).float()
            inputs.append(img.unsqueeze(0))
            cnt = 1
        elif cnt == 1:
            img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
#            img = torch.from_numpy(img / 255.).permute(0, 1, 2).float()
            inputs.append(img.unsqueeze(0))
            cnt = 2
        elif cnt == 2:
            img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
#            img = torch.from_numpy(img / 255.).permute(0, 1, 2).float()
            inputs.append(img.unsqueeze(0))
            cnt = 3
        elif cnt == 3:
            inputs[0] = inputs[1]
            inputs[1] = inputs[2]
            temp_img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
#            temp_img = torch.from_numpy(img / 255.).permute(0, 1, 2).float()
            inputs[2] = temp_img.unsqueeze(0)
            algo_input = torch.stack(inputs, dim=1)
            
            cuda_flag = False
            if torch.cuda.is_available():
                model = model.cuda()
                cuda_flag = True
         
            with torch.no_grad():
                if cuda_flag:
                    algo_input = algo_input.cuda()
                outputs = model(algo_input, test_mode=True)['output'].cpu()
                torch.cuda.synchronize()
            outputs = outputs.squeeze(0)
            result_img = outputs[2,:,:,:]
            result_img = result_img.permute(1, 2, 0)
            result_img = result_img.numpy()
#            result_img = inputs[2].squeeze(0)
#            result_img = result_img.numpy()
            cv2.imshow("img", result_img)
            cv2.waitKey(1)
            t1 = time.time()
            print("===> Timer: %.4f sec." % (t1 - t0))

if __name__ == '__main__':
    main()
    
