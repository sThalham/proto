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
from noise import pnoise2, snoise2
import pyfastnoisesimd as fns

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
depthCut = 1800

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
        return None, None, None, None, None

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

    if np.nanmean(depth) < 0.5 or np.nanmean(depth) > 2.0:
        print('invalid train image; range is wrong')
        return None, None, None, None, None

    depth = cv2.resize(depth, (resX, resY))
    partmask = cv2.imread(fn_part, 0)

    return depth, partmask, bboxes, poses, mask_ids


def augmentDepth(depth, obj_mask, mask_ori, shadowClK, shadowMK, blurK, blurS, depthNoise):
    # erode and blur mask to get more realistic appearance
    partmask = cv2.resize(mask_ori, (resX, resY))
    partmask = partmask.astype(np.float32)
    mask = partmask > (np.median(partmask) * 0.4)
    partmask = np.where(mask, 255.0, 0.0)

    # apply shadow
    kernel = np.ones((shadowClK, shadowClK))
    partmask = cv2.morphologyEx(partmask, cv2.MORPH_OPEN, kernel)
    partmask = signal.medfilt2d(partmask, kernel_size=shadowMK)
    partmask = partmask.astype(np.uint8)
    mask = partmask > 20
    depth = np.where(mask, depth, 0.0)

    # augment foreground
    obj_mask = obj_mask.astype(np.uint8)
    objmask = cv2.resize(obj_mask, dsize=(resX, resY), interpolation=cv2.INTER_LINEAR)
    fg = np.where(objmask > 0, depth, 0.0)

    shape = [resY, resX]

    # fast perlin noise
    seed = np.random.randint(2 ** 31)
    N_threads = 4
    perlin = fns.Noise(seed=seed, numWorkers=N_threads)
    perlin.frequency = 0.02
    perlin.noiseType = fns.NoiseType.SimplexFractal
    perlin.fractal.fractalType = fns.FractalType.FBM
    perlin.fractal.octaves = 8
    perlin.fractal.lacunarity = 2.1
    perlin.fractal.gain = 0.45
    perlin.perturb.perturbType = fns.PerturbType.NoPerturb

    # noise according to keep it unreal
    noiseX = np.random.uniform(0.0001, 0.1, resX * resY)
    noiseY = np.random.uniform(0.001, 0.1, resX * resY)
    noiseZ = np.random.uniform(0.01, 0.1, resX * resY)
    Wxy = np.random.randint(0, 10)
    Wz = np.random.uniform(0, 0.005)

    X, Y = np.meshgrid(np.arange(resX), np.arange(resY))
    coords0 = fns.emptyCoords(resX * resY)
    coords1 = fns.emptyCoords(resX * resY)
    coords2 = fns.emptyCoords(resX * resY)

    coords0[0, :] = noiseX.ravel()
    coords0[1, :] = Y.ravel()
    coords0[2, :] = X.ravel()
    VecF0 = perlin.genFromCoords(coords0)
    VecF0 = VecF0.reshape((resY, resX))

    coords1[0, :] = noiseY.ravel()
    coords1[1, :] = Y.ravel()
    coords1[2, :] = X.ravel()
    VecF1 = perlin.genFromCoords(coords1)
    VecF1 = VecF1.reshape((resY, resX))

    coords2[0, :] = noiseZ.ravel()
    coords2[1, :] = Y.ravel()
    coords2[2, :] = X.ravel()
    VecF2 = perlin.genFromCoords(coords2)
    VecF2 = VecF2.reshape((resY, resX))

    x = np.arange(resX, dtype=np.uint16)
    x = x[np.newaxis, :].repeat(resY, axis=0)
    y = np.arange(resY, dtype=np.uint16)
    y = y[:, np.newaxis].repeat(resX, axis=1)

    fx = x + Wxy * VecF0
    fy = y + Wxy * VecF1
    fx = np.where(fx < 0, 0, fx)
    fx = np.where(fx >= resX, resX - 1, fx)
    fy = np.where(fy < 0, 0, fy)
    fy = np.where(fy >= resY, resY - 1, fy)
    fx = fx.astype(dtype=np.uint16)
    fy = fy.astype(dtype=np.uint16)
    depDis = depth[fy, fx] + Wz * VecF2

    del perlin
    depth = np.where(depDis > 0, depDis, 0.0)
    print(np.amax(depth))
    print(np.amin(depth))

    depth = cv2.resize(depth, None, fx=1 / 2, fy=1 / 2)

    res = (((depth / 1000.0) * 1.41421356) ** 2)

    depthFinal = cv2.GaussianBlur(depth, (blurK, blurK), blurS, blurS)

    # discretize to resolution and apply gaussian
    dNonVar = np.divide(depthFinal, res, out=np.zeros_like(depth), where=res != 0)
    dNonVar = np.round(dNonVar)
    dNonVar = np.multiply(dNonVar, res)

    noise = np.multiply(dNonVar, depthNoise)
    print(dNonVar.shape)
    print(noise.shape)

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

    cross[depth_refine <= 300] = 0  # 0 and near range cut
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

    root = '/home/sthalham/data/renderings/linemod_BG/patches31052018/patches'  # path to train samples
    target = '/home/sthalham/data/prepro/augm_fastSimplex_std/'

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
            maskfile = maskPath + redname + "_mask.npy"

            depth_refine, mask, bboxes, poses, mask_ids = manipulate_depth(gtfile, depfile, partfile)
            obj_mask = np.load(maskfile)
            obj_mask = obj_mask.astype(np.int8)
            obj_mask = np.where(obj_mask > 0, 255, 0)

            if bboxes is None:
                excludedImgs.append(int(redname))
                continue

            depth_refine = np.multiply(depth_refine, 1000.0)  # to millimeters
            rows, cols = depth_refine.shape

            newredname = redname[1:]

            fileName = target + "coco_train2014/" + newredname + '.jpg'
            myFile = Path(fileName)
            if myFile.exists():
                print('File exists, skip encoding and safing.')

            else:
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
                depthAug = augmentDepth(depth_refine, obj_mask, mask, kShadow, kMed, kBlur, sBlur, sDep)

                aug_xyz, depth_refine_aug = get_normal(depthAug, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin,
                                                         for_vis=False)
                cv2.imwrite(fileName, aug_xyz)

            imgID = int(newredname)
            imgName = newredname + '.jpg'

            # bb scaling because of image scaling
            bbsca = 640.0 / 720.0
            for bbox in bboxes:
                objID = np.asscalar(bbox[0]) + 1
                bbox = (bbox * bbsca).astype(int)
                x1 = np.asscalar(bbox[2])
                y1 = np.asscalar(bbox[1])
                x2 = np.asscalar(bbox[4])
                y2 = np.asscalar(bbox[3])
                nx1 = bbox[2]
                ny1 = bbox[1]
                nx2 = bbox[4]
                ny2 = bbox[3]
                w = (x2 - x1)
                h = (y2 - y1)
                bb = [x1, y1, w, h]
                area = w * h
                npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
                seg = npseg.tolist()

                annoID = annoID + 1
                tempTA = {
                    "id": annoID,
                    "image_id": imgID,
                    "category_id": objID,
                    "bbox": bb,
                    "segmentation": [seg],
                    "area": area,
                    "iscrowd": 0
                }

                dict["annotations"].append(tempTA)
                '''
                    cv2.rectangle(depI, (x1, y1), (x2, y2), ( 255, 255, 0), 2, 1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (x1, y1)
                    fontScale = 1
                    fontColor = (255, 255, 0)
                    fontthickness = 1
                    lineType = 2
                    gtText = str(objID)
                    cv2.putText(depI, gtText,
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    fontthickness,
                                    lineType)
                '''

            tempTL = {
                "url": "cmp.felk.cvut.cz/t-less/",
                "id": imgID,
                "name": imgName
            }
            dict["licenses"].append(tempTL)

            tempTV = {
                "license": 2,
                "url": "cmp.felk.cvut.cz/t-less/",
                "file_name": imgName,
                "height": rows,
                "width": cols,
                "date_captured": dateT,
                "id": imgID
            }
            dict["images"].append(tempTV)

            gloCo += 1

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            meantime = sum(times)/len(times)
            eta = ((all - gloCo) * meantime) / 60
            if gloCo % 100 == 0:
                print('eta: ', eta, ' min')
                times = []

    catsInt = range(1, 16)

    for s in catsInt:
        objName = str(s)
        tempC = {
            "id": s,
            "name": objName,
            "supercategory": "object"
        }
        dict["categories"].append(tempC)

    traAnno = target + "annotations/instances_train2014.json"

    with open(traAnno, 'w') as fpT:
        json.dump(dict, fpT)

    excludedImgs.sort()
    print('excluded images: ')
    for ex in excludedImgs:
        print(ex)
    print('Chill for once in your life... everything\'s done')
