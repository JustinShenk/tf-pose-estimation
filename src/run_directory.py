import argparse
import logging
import time
import glob
import ast
import os
import dill

import common
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

logger = logging.getLogger('TfPoseEstimator')
# logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='tf-pose-estimation run by folder')
    parser.add_argument('--folder', type=str, default='./images/')
    parser.add_argument(
        '--resolution',
        type=str,
        default='432x368',
        help='network input resolution. default=432x368')
    parser.add_argument(
        '--model',
        type=str,
        default='mobilenet_thin',
        help='cmu / mobilenet_thin')
    parser.add_argument(
        '--scales',
        type=str,
        default='[None]',
        help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    types = ('*.png', '*.jpg')
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(os.path.join(args.folder, files)))
    all_humans = dict()
    if not os.path.exists('output'):
        os.mkdir('output')
    for i, file in enumerate(files_grabbed):
        # estimate human poses from a single image !
        image = common.read_imgfile(file, None, None)
        t = time.time()
        humans = e.inference(image, scales=scales)
        elapsed = time.time() - t

        logger.info('inference image #%d: %s in %.4f seconds.' % (i, file,
                                                                  elapsed))

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        # cv2.imshow('tf-pose-estimation result', image)
        output_filepath = os.path.join('output', 'pose_{}.png'.format(i))
        cv2.imwrite(output_filepath, image)
        logger.info('image saved: {}'.format(output_filepath))
        # cv2.waitKey(5000)

        all_humans[file.replace(args.folder, '')] = humans

    with open(os.path.join(args.folder, 'pose.dil'), 'wb') as f:
        dill.dump(all_humans, f, protocol=dill.HIGHEST_PROTOCOL)
