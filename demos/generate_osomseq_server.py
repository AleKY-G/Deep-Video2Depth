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


def process_for_evaluation(depth, scale, crop):
    """ During training ground truth depths are scaled and cropped, we need to
        undo this for evaluation """
    depth = (1.0 / scale) * np.pad(depth, [[crop, 0], [0, 0]], 'mean')
    return depth

def get_imgnum():
    seqs = [
        ['2011_10_03/2011_10_03_drive_0027', '000000', '004540'],
        ['2011_09_30/2011_09_30_drive_0016', '000000', '000270'],
        ['2011_09_30/2011_09_30_drive_0018', '000000', '002760'],
        ['2011_09_30/2011_09_30_drive_0027', '000000', '001100']
    ]
    test_list = list()
    for seqname, sidx, eid in seqs:
        sidx = int(sidx)
        eidx = int(eid)
        for idx in range(sidx, eidx):
            test_list.append("{}_sync/image_02/data/{}.png".format(seqname, str(idx).zfill(10)))
    return len(test_list)

def make_predictions(args):
    """ Run inference over the test images """
    odomoutput = args.odomoutput

    np.random.seed(1234)
    cfg = config.cfg_from_file(args.cfg)

    db = KittiRaw(args.dataset_dir)
    scale = db.args['scale']
    crop = db.args['crop']

    deepv2d = DeepV2D(cfg, args.model, use_fcrn=False, mode='keyframe')

    totnum = get_imgnum()
    st = time.time()
    count = 0

    addconfig = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=addconfig) as sess:
        deepv2d.set_session(sess)

        for (images, intrinsics, test_frame) in db.odom_evalset_iterator():
            if not os.path.exists(os.path.join(args.dataset_dir, test_frame)):
                print("skip %s" % test_frame)
                break

            depthsvfolder = os.path.join(odomoutput, test_frame.split('/')[0], test_frame.split('/')[1], 'depthpred')
            posesvfolder = os.path.join(odomoutput, test_frame.split('/')[0], test_frame.split('/')[1], 'posepred')
            os.makedirs(depthsvfolder, exist_ok=True)
            os.makedirs(posesvfolder, exist_ok=True)

            depthsvname = test_frame.split('/')[-1]
            posesvname = os.path.join(posesvfolder, "{}.txt".format(depthsvname.split('.')[0]))

            if os.path.exists(posesvname):
                print("%s exists" % test_frame)
                totnum = totnum - 1
                continue

            depth_predictions, poses = deepv2d(images, intrinsics, iters=args.n_iters)
            pred = process_for_evaluation(depth_predictions[0], scale, crop)

            rgb = Image.open(os.path.join(args.dataset_dir, test_frame))
            gtw, gth = rgb.size
            predresized = cv2.resize(pred, (gtw, gth))

            Image.fromarray((np.array(predresized).astype(np.float32) * 256.0).astype(np.uint16)).save(os.path.join(depthsvfolder, depthsvname))

            with open(posesvname, "w") as text_file:
                for pose in poses:
                    for num in list(pose.flatten().tolist()):
                        text_file.write("{} ".format(str(num)))
                    text_file.write('\n')

            count += 1
            dr = time.time() - st
            lefth = dr / count / 60 / 60 * (totnum - count)
            print("Img idx:%d, left hours: %f" % (count, lefth))
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
    make_predictions(args)