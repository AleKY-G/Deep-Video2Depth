import sys

sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf

import cv2
import os
import time
import argparse
import glob
import vis
import pickle

from core import config
from deepv2d import DeepV2D
from data_stream.kitti import KittiRaw
from PIL import Image

def estimate_runningtime(args):
    """ Run inference over the test images """
    np.random.seed(1234)
    cfg = config.cfg_from_file(args.cfg)

    db = KittiRaw(args.dataset_dir)

    deepv2d = DeepV2D(cfg, args.model, use_fcrn=False, mode='keyframe')

    st = time.time()
    count = 0

    with tf.Session() as sess:
        deepv2d.set_session(sess)
        for (images, intrinsics, test_frame) in db.eigen_set_iterator():
            depth_predictions, poses = deepv2d(images, intrinsics, iters=args.n_iters)

            count += 1
            dr = time.time() - st
            print("Img idx:%d, ave time: %f s" % (count, dr / count))
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/kitti.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/kitti.ckpt', help='path to model checkpoint')
    parser.add_argument('--dataset_dir', default='data/kitti/raw', help='config file used to train the model')
    parser.add_argument('--viz', action="store_true", help='display depth maps during inference')
    parser.add_argument('--n_iters', type=int, default=5, help='number of video frames to use for reconstruction')
    parser.add_argument('--odomoutput', type=str, default='/media/shengjie/disk1/Prediction/Deepv2d_odom')
    args = parser.parse_args()

    # run inference on the test images
    estimate_runningtime(args)