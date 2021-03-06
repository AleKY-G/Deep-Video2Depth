import sys
sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import argparse
import cv2
import os
import time
import glob
import random

from core import config
from deepv2d import DeepV2D


def load_test_sequence(path, n_frames=-1):
    """ loads images and intrinsics from demo folder """

    # Order 1 2 0 3 4
    images = []
    for imfile in sorted(glob.glob(os.path.join(path, "*.png"))):
        img = cv2.imread(imfile)
        images.append(img)

    inds = np.arange(1, len(images))
    if n_frames > 0:
        inds = np.random.choice(inds, n_frames, replace=False)

    inds = [0] + inds.tolist() # put keyframe image first
    images = [images[i] for i in inds]

    images = np.stack(images).astype(np.float32)
    intrinsics = np.loadtxt(os.path.join(path, 'intrinsics.txt'))

    return images, intrinsics

def unpack(intrinsics):
    intrinsics44 = np.eye(4)
    intrinsics44[0, 0] = intrinsics[0]
    intrinsics44[1, 1] = intrinsics[1]
    intrinsics44[0, 2] = intrinsics[2]
    intrinsics44[1, 2] = intrinsics[3]
    return intrinsics44

def sample_seq(depths, images, intrinsics, poses):
    intrinsics44 = unpack(intrinsics)
    depth = depths[0]
    _, h, w, _ = images.shape
    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
    xxf = xx.flatten()
    yyf = yy.flatten()
    for k in range(len(poses)):
        pose = poses[k] @ np.linalg.inv(poses[0])

        depthf = depth.flatten()

        pts3d = np.stack([xxf * depthf, yyf * depthf, depthf, np.ones_like(depthf)], axis=0)
        pts3d = np.linalg.inv(intrinsics44) @ pts3d
        pts2doview = intrinsics44 @ pose @ pts3d
        pts2doview[0, :] = pts2doview[0, :] / pts2doview[2, :]
        pts2doview[1, :] = pts2doview[1, :] / pts2doview[2, :]
        pts2doview = pts2doview[0:2, :]

def main(args):

    if args.cfg is None:
        if 'nyu' in args.model:
            args.cfg = 'cfgs/nyu.yaml'
        elif 'scannet' in args.model:
            args.cfg = 'cfgs/scannet.yaml'
        elif 'kitti' in args.model:
            args.cfg = 'cfgs/kitti.yaml'
        else:
            args.cfg = 'cfgs/nyu.yaml'
        
    cfg = config.cfg_from_file(args.cfg)
    is_calibrated = not args.uncalibrated

    # build the DeepV2D graph
    deepv2d = DeepV2D(cfg, args.model, use_fcrn=args.fcrn, is_calibrated=is_calibrated, mode=args.mode)

    with tf.Session() as sess:
        deepv2d.set_session(sess)

        # call deepv2d on a video sequence
        images, intrinsics = load_test_sequence(args.sequence)

        depths, poses = deepv2d(images, intrinsics, viz=False, iters=args.n_iters)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='config file used to train the model')
    parser.add_argument('--model', default='models/nyu.ckpt', help='path to model checkpoint')
    
    parser.add_argument('--mode', default='keyframe', help='keyframe or global pose optimization')
    parser.add_argument('--fcrn', action="store_true", help='use fcrn for initialization')
    parser.add_argument('--n_iters', type=int, default=5, help='number of iterations to use')
    parser.add_argument('--uncalibrated', action="store_true", help='use fcrn for initialization')
    parser.add_argument('--sequence', help='path to sequence folder')
    args = parser.parse_args()

    main(args)
