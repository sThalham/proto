import sys
import os
import subprocess
import yaml
import cv2
import numpy as np
import json
from scipy import ndimage, signal
import math
import datetime
import copy
import transforms3d as tf3d
import time
import itertools
import random
#import pcl
#from pcl import pcl_visualization

import OpenEXR, Imath
from pathlib import Path

#kin_res_x = 640
#kin_res_y = 480
resX = 640
resY = 480
# fov = 1.0088002681732178
fov = 57.8
fxkin = 579.68  # blender calculated
fykin = 542.31  # blender calculated
cxkin = 320
cykin = 240
depthCut = 1.8

np.set_printoptions(threshold=np.nan)


def manipulate_depth(fn_gt, fn_depth, fn_part):

    with open(fn_gt, 'r') as stream:
        query = yaml.load(stream)
        bboxes = np.zeros((len(query), 5), np.int)
        poses = np.zeros((len(query), 7), np.float32)
        mask_ids = np.zeros((len(query)), np.int)
        for j in range(len(query)-1): # skip cam pose
            qr = query[j]
            class_id = qr['class_id']
            bbox = qr['bbox']
            mask_ids[j] = int(qr['mask_id'])
            pose = np.array(qr['pose']).reshape(4, 4)
            bboxes[j, 0] = class_id
            bboxes[j, 1:5] = np.array(bbox)
            q_pose = tf3d.quaternions.mat2quat(pose[:3, :3])
            poses[j, :4] = np.array(q_pose)
            poses[j, 4:7] = np.array([pose[0, 3], pose[1, 3], pose[2, 3]])

    if bboxes.shape[0] < 2:
        print('invalid train image, no bboxes in fov')
        return None, None

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    golden = OpenEXR.InputFile(fn_depth)
    dw = golden.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = golden.channel('R', pt)
    depth = np.fromstring(redstr, dtype=np.float32)
    depth.shape = (size[1], size[0])

    centerX = depth.shape[1] / 2.0
    centerY = depth.shape[0] / 2.0

    uv_table = np.zeros((depth.shape[0], depth.shape[1], 2), dtype=np.int16)
    column = np.arange(0, depth.shape[0])
    uv_table[:, :, 1] = np.arange(0, depth.shape[1]) - centerX
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY
    uv_table = np.abs(uv_table)

    depth = depth * np.cos(np.radians(fov / depth.shape[1] * np.abs(uv_table[:, :, 1]))) * np.cos(
        np.radians(fov / depth.shape[1] * uv_table[:, :, 0]))

    depthTrue = cv2.resize(depth, (resX, resY))

    if np.nanmean(depth) < 0.5 or np.nanmean(depth) > 2.0:
        print('invalid train image; range is wrong')
        return None, None

    partmask = cv2.imread(fn_part, 0)

    return depthTrue, partmask


def augmentDepth(depth, mask_ori, shadowClK, shadowMK, blurK, blurS, depthNoise):
    depth = cv2.resize(depth, (resX, resY))
    depthOri = depth
    # erode and blur mask to get more realistic appearance

    partmask = cv2.resize(mask_ori, (resX, resY))
    partmask = partmask.astype(np.float32)
    mask = partmask > (np.median(partmask)*0.4)
    partmask = np.where(mask, 255.0, 0.0)

    #apply shadow
    kernel = np.ones((shadowClK, shadowClK))
    partmask = cv2.morphologyEx(partmask, cv2.MORPH_OPEN, kernel)
    partmask = signal.medfilt2d(partmask, kernel_size=shadowMK)
    partmask = partmask.astype(np.uint8)
    mask = partmask > 20
    depth = np.where(mask, depth, 0.0)

    depth = cv2.resize(depth, None, fx=1/2, fy=1/2)

    res = (((depth / 1000.0) * 1.41421356) ** 2) * 1000.0

    depthFinal = cv2.GaussianBlur(depth, (blurK, blurK), blurS, blurS)

    # discretize to resolution and apply gaussian
    dNonVar = np.divide(depthFinal, res, out=np.zeros_like(depth), where=res != 0)
    dNonVar = np.round(dNonVar)
    dNonVar = np.multiply(dNonVar, res)

    noise = np.multiply(dNonVar, depthNoise)
    depthFinal = np.random.normal(loc=dNonVar, scale=noise, size=dNonVar.shape)

    depthFinal = cv2.resize(depthFinal, (resX, resY))

    return depthFinal


def get_normal(depth_refine, fx=-1, fy=-1, cx=-1, cy=-1, for_vis=True):
    res_y = depth_refine.shape[0]
    res_x = depth_refine.shape[1]

    # inpainting
    scaleOri = np.amax(depth_refine)
    inPaiMa = np.where(depth_refine == 0.0, 255, 0)
    inPaiMa = inPaiMa.astype(np.uint8)
    inPaiDia = 5.0
    depth_refine = depth_refine.astype(np.float32)
    depPaint = cv2.inpaint(depth_refine, inPaiMa, inPaiDia, cv2.INPAINT_NS)

    depNorm = depPaint - np.amin(depPaint)
    rangeD = np.amax(depNorm)
    depNorm = np.divide(depNorm, rangeD)
    depth_refine = np.multiply(depNorm, scaleOri)

    centerX = cx
    centerY = cy

    constant = 1 / fx
    uv_table = np.zeros((res_y, res_x, 2), dtype=np.int16)
    column = np.arange(0, res_y)

    uv_table[:, :, 1] = np.arange(0, res_x) - centerX  # x-c_x (u)
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY  # y-c_y (v)
    uv_table_sign = np.copy(uv_table)
    uv_table = np.abs(uv_table)

    # kernel = np.ones((5, 5), np.uint8)
    # depth_refine = cv2.dilate(depth_refine, kernel, iterations=1)
    # depth_refine = cv2.medianBlur(depth_refine, 5 )
    depth_refine = ndimage.gaussian_filter(depth_refine, 2)  # sigma=3)
    # depth_refine = ndimage.uniform_filter(depth_refine, size=11)

    # very_blurred = ndimage.gaussian_filter(face, sigma=5)
    v_x = np.zeros((res_y, res_x, 3))
    v_y = np.zeros((res_y, res_x, 3))
    normals = np.zeros((res_y, res_x, 3))

    dig = np.gradient(depth_refine, 2, edge_order=2)
    v_y[:, :, 0] = uv_table_sign[:, :, 1] * constant * dig[0]
    v_y[:, :, 1] = depth_refine * constant + (uv_table_sign[:, :, 0] * constant) * dig[0]
    v_y[:, :, 2] = dig[0]

    v_x[:, :, 0] = depth_refine * constant + uv_table_sign[:, :, 1] * constant * dig[1]
    v_x[:, :, 1] = uv_table_sign[:, :, 0] * constant * dig[1]
    v_x[:, :, 2] = dig[1]

    cross = np.cross(v_x.reshape(-1, 3), v_y.reshape(-1, 3))
    norm = np.expand_dims(np.linalg.norm(cross, axis=1), axis=1)
    # norm[norm == 0] = 1

    cross = cross / norm
    cross = cross.reshape(res_y, res_x, 3)
    cross = np.abs(cross)
    cross = np.nan_to_num(cross)

    #cross[depth_refine <= 0.3] = 0  # 0 and near range cut
    cross[depth_refine > depthCut] = 0  # far range cut
    if not for_vis:
        scaDep = 1.0 / np.nanmax(depth_refine)
        depth_refine = np.multiply(depth_refine, scaDep)
        cross[:, :, 0] = cross[:, :, 0] * (1 - (depth_refine - 0.5))  # nearer has higher intensity
        cross[:, :, 1] = cross[:, :, 1] * (1 - (depth_refine - 0.5))
        cross[:, :, 2] = cross[:, :, 2] * (1 - (depth_refine - 0.5))
        scaCro = 255.0 / np.nanmax(cross)
        cross = np.multiply(cross, scaCro)
        cross = cross.astype(np.uint8)

    return cross, depth_refine


##########################
#         MAIN           #
##########################
if __name__ == "__main__":

    root = '/home/sthalham/data/renderings/linemod_nBG/linemod_data/patches18062018'  # path to train samples
    #root = "/home/sthalham/patches18062018"

    now = datetime.datetime.now()
    dateT = str(now)

    dict = {"info": {
        "description": "tless",
        "url": "cmp.felk.cvut.cz/t-less/",
        "version": "1.0",
        "year": 2018,
        "contributor": "Stefan Thalhammer",
        "date_created": dateT
    },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    dictVal = copy.deepcopy(dict)

    annoID = 0
    gloCo = 0
    all = 10000
    times = []

    trainN = 1
    testN = 1
    valN = 1

    depPath = root + "/depth/"
    partPath = root + "/part/"
    gtPath = root
    maskPath = root + "/mask/"
    excludedImgs = []

    for fileInd in os.listdir(root):
        if fileInd.endswith(".yaml"):

            print('Processing next image')

            start_time = time.time()
            gloCo = gloCo + 1

            redname = fileInd[:-8]
            if int(redname) > 10130:
                continue

            gtfile = gtPath + '/' + fileInd
            depfile = depPath + redname + "_depth.exr"
            partfile = partPath + redname + "_part.png"

            # depth_refine, bboxes, poses, mask_ids = get_a_scene(gtfile, depfile, disfile)
            depthTrue, mask = manipulate_depth(gtfile, depfile, partfile)

            if depthTrue is None:
                excludedImgs.append(int(redname))
                continue

            rows, cols = depthTrue.shape

            for i in range(1,6):

                drawN = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3]
                freq = np.bincount(drawN)
                rnd = np.random.choice(np.arange(len(freq)), 1, p=freq / len(drawN), replace=False)

                fn = '/home/sthalham/git/Keras-GAN/pix2pix/datasets/aug2synth/waste.jpg'

                if rnd == 1:
                    fn = '/home/sthalham/git/Keras-GAN/pix2pix/datasets/aug2synth/train/' + str(trainN) + '.jpg'
                    trainN += 1
                elif rnd == 2:
                    fn = '/home/sthalham/git/Keras-GAN/pix2pix/datasets/aug2synth/val/' + str(valN) + '.jpg'
                    valN += 1
                if rnd == 3:
                    fn = '/home/sthalham/git/Keras-GAN/pix2pix/datasets/aug2synth/test/' + str(testN) + '.jpg'
                    testN += 1

                true_xyz, depth_refine_true = get_normal(depthTrue, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin, for_vis=False)

                drawKern = [3, 5, 7]
                freqKern = np.bincount(drawKern)
                kShadow = np.random.choice(np.arange(len(freqKern)), 1, p=freqKern / len(drawKern), replace=False)
                kMed = np.random.choice(np.arange(len(freqKern)), 1, p=freqKern / len(drawKern), replace=False)
                kBlur = np.random.choice(np.arange(len(freqKern)), 1, p=freqKern / len(drawKern), replace=False)
                sBlur = random.uniform(0.25, 3.5)
                sDep = random.uniform(0.001, 0.005)
                kShadow.astype(int)
                kMed.astype(int)
                kBlur.astype(int)
                kShadow = kShadow[0]
                kMed = kMed[0]
                kBlur = kBlur[0]
                depthAug = augmentDepth(depthTrue, mask, kShadow, kMed, kBlur, sBlur, sDep)

                aug_xyz, depth_refine_aug = get_normal(depthAug, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin, for_vis=False)

                sca = 255.0 / np.amax(true_xyz)
                scaled = np.multiply(true_xyz, sca)
                true_xyz = scaled.astype(np.uint8)

                sca = 255.0 / np.amax(aug_xyz)
                scaled = np.multiply(aug_xyz, sca)
                aug_xyz = scaled.astype(np.uint8)

                image = np.concatenate((true_xyz, aug_xyz), axis=1)

                cv2.imwrite(fn, image)

                print('stop')

            gloCo += 1

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            meantime = sum(times)/len(times)
            eta = ((all - gloCo) * meantime) / 60
            if gloCo % 100 == 0:
                print('eta: ', eta, ' min')
                times = []

    print('Chill for once in your life... everything\'s done')
