import sys

sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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


def process_for_evaluation(depth, scale, crop):
    """ During training ground truth depths are scaled and cropped, we need to
        undo this for evaluation """
    depth = (1.0 / scale) * np.pad(depth, [[crop, 0], [0, 0]], 'mean')
    return depth


def make_predictions(args):
    """ Run inference over the test images """
    kittiroot = '/media/shengjie/disk1/data/Kitti'
    odomoutput = '/media/shengjie/disk1/Prediction/Deepv2d_odom'

    np.random.seed(1234)
    cfg = config.cfg_from_file(args.cfg)

    db = KittiRaw(args.dataset_dir)
    scale = db.args['scale']
    crop = db.args['crop']

    deepv2d = DeepV2D(cfg, args.model, use_fcrn=False, mode='keyframe')

    count = 0
    with tf.Session() as sess:
        deepv2d.set_session(sess)

        predictions = []
        for (images, intrinsics, test_frame) in db.odom_set_iterator():
            if not os.path.exists(os.path.join(kittiroot, test_frame)):
                print("skip %s" % test_frame)
                continue

            depth_predictions, poses = deepv2d(images, intrinsics, iters=args.n_iters)

            keyframe_depth = depth_predictions[0]

            pred = process_for_evaluation(keyframe_depth, scale, crop)

            rgb = Image.open(os.path.join(kittiroot, test_frame))
            gtw, gth = rgb.size

            predresized = cv2.resize(pred, (gtw, gth))

            depthsvfolder = os.path.join(odomoutput, test_frame.split('/')[0], test_frame.split('/')[1], 'depthpred')
            posesvfolder = os.path.join(odomoutput, test_frame.split('/')[0], test_frame.split('/')[1], 'posepred')
            os.makedirs(depthsvfolder, exist_ok=True)
            os.makedirs(posesvfolder, exist_ok=True)

            depthsvname = test_frame.split('/')[-1]
            Image.fromarray((np.array(predresized).astype(np.float32) * 256.0).astype(np.uint16)).save(os.path.join(depthsvfolder, depthsvname))

            posesvname = os.path.join(posesvfolder, "{}.txt".format(depthsvname.split('.')[0]))
            with open(posesvname, "w") as text_file:
                for pose in poses:
                    for num in list(pose.flatten().tolist()):
                        text_file.write("{} ".format(str(num)))
                    text_file.write('\n')

            print("Finish prediciton image: %d" % count)
            count += 1
        return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/kitti.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/kitti.ckpt', help='path to model checkpoint')
    parser.add_argument('--dataset_dir', default='data/kitti/raw', help='config file used to train the model')
    parser.add_argument('--viz', action="store_true", help='display depth maps during inference')
    parser.add_argument('--n_iters', type=int, default=5, help='number of video frames to use for reconstruction')
    parser.add_argument('--kittiroot', type=int, default=5, help='number of video frames to use for reconstruction')
    parser.add_argument('--odomoutput', type=int, default=5, help='number of video frames to use for reconstruction')
    args = parser.parse_args()

    # run inference on the test images
    predictions = make_predictions(args)
    groundtruth = pickle.load(open('./data/kitti/kitti_groundtruth.pickle', 'rb'))