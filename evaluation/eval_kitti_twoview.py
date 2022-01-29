import sys

sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import cv2
import vis
import glob
import time
import pickle
import tqdm
import argparse

import eval_utils
from PIL import Image
from core import config
from deepv2d import DeepV2D
from data_stream.kitti import KittiRaw


def process_for_evaluation(depth, scale, crop):
    """ During training ground truth depths are scaled and cropped, we need to
        undo this for evaluation """
    depth = (1.0 / scale) * np.pad(depth, [[crop, 0], [0, 0]], 'mean')
    return depth

def make_predictions(args):
    """ Run inference over the test images """

    semidensegtroot = '/home/shengjie/Documents/Data/Kitti/semidense_gt'
    svpath = '/media/shengjie/disk1/Prediction/Deepv2d_eigen_twoview'
    np.random.seed(1234)
    cfg = config.cfg_from_file(args.cfg)

    db = KittiRaw(args.dataset_dir)
    scale = db.args['scale']
    crop = db.args['crop']

    os.makedirs(svpath, exist_ok=True)

    deepv2d = DeepV2D(cfg, args.model, use_fcrn=False, mode='keyframe')

    count = 0
    with tf.Session() as sess:
        deepv2d.set_session(sess)

        predictions = []
        for (images, intrinsics, test_frame) in db.test_set_iterator_twoview():
            gt_path = os.path.join(semidensegtroot, test_frame.replace('/data', ''))
            if not os.path.exists(gt_path):
                continue

            depth_predictions, _ = deepv2d(images, intrinsics, iters=args.n_iters)

            keyframe_depth = depth_predictions[0]
            keyframe_image = images[0]

            pred = process_for_evaluation(keyframe_depth, scale, crop)

            gt = Image.open(gt_path)
            gtw, gth = gt.size

            svname = test_frame.split('/')[1] + '_' + test_frame.split('/')[-1]
            predresized = cv2.resize(pred, (gtw, gth))
            Image.fromarray((np.array(predresized).astype(np.float32) * 256.0).astype(np.uint16)).save(os.path.join(svpath, svname))

            print("Finish prediciton image: %d" % count)
            count += 1
        return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/kitti.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/kitti.ckpt', help='path to model checkpoint')
    parser.add_argument('--dataset_dir', default='data/kitti/raw', help='config file used to train the model')
    parser.add_argument('--n_iters', type=int, default=5, help='number of video frames to use for reconstruction')
    args = parser.parse_args()

    # run inference on the test images
    predictions = make_predictions(args)