#!/usr/bin/env python

import sys
import os
import yaml
import cv2
import numpy as np
import json
from scipy import ndimage
import math
import datetime
import copy
import time

from pcl import pcl_visualization
import pandas as pd
import tensorflow as tf

np.set_printoptions(threshold=np.nan)

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

    # cross_ref = np.copy(cross)
    # cross[cross_ref==[0,0,0]]=0 #set zero for nan values

    # cam_angle = np.arccos(cross[:, :, 2])
    # cross[np.abs(cam_angle) > math.radians(75)] = 0  # high normal cut
    # cross[depth_refine <= 300] = 0  # 0 and near range cut
    cross[depth_refine > 1500] = 0  # far range cut
    if not for_vis:
        cross[:, :, 0] = cross[:, :, 0] * (1 - (depth_refine - 0.5))  # nearer has higher intensity
        cross[:, :, 1] = cross[:, :, 1] * (1 - (depth_refine - 0.5))
        cross[:, :, 2] = cross[:, :, 2] * (1 - (depth_refine - 0.5))

    return cross


def create_BB(rgb):

    imgray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 25, 255, cv2.THRESH_BINARY)
    im2, contours, hier = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]

    tup = np.nonzero(thresh)
    x = tup[1]
    y = tup[0]
    x1 = min(x)
    x2 = max(x)
    y1 = min(y)
    y2 = max(y)

    w = x2 - x1
    h = y2 - y1
    area = w * h
    area = float(area)

    bb = [int(x1),int(y1),int(w),int(h)]

    return cnt, bb, area, thresh


if __name__ == "__main__":

    root = "/home/sthalham/data/t-less_v2/train_kinect/"  # path to train samples, depth + rgb
    safe = "/home/sthalham/data/T-less_Detectron/tless_train/"
    # print(root)

    sub = os.listdir(root)

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
    all = 1296 * 29

    for s in sub:
        rgbPath = root + s + "/rgb/"
        depPath = root + s + "/depth/"
        gtPath = root + s + "/gt.yml"
        infoPath = root + s + "/info.yml"

        with open(infoPath, 'r') as stream:
            opYML = yaml.load(stream)

        subsub = os.listdir(rgbPath)

        counter = 0
        for ss in subsub:
            start_time = time.time()
            gloCo = gloCo + 1
            # print(gloCo, '/', all)
            imgname = ss
            rgbImgPath = rgbPath + ss
            depImgPath = depPath + ss
            # print(rgbImgPath)

            if ss.startswith('000'):
                ss = ss[3:]
            elif ss.startswith('00'):
                ss = ss[2:]
            elif ss.startswith('0'):
                ss = ss[1:]
            ss = ss[:-4]

            calib = opYML[int(ss)]
            K = calib["cam_K"]
            depSca = calib["depth_scale"]
            fxkin = K[0]
            # print(fx)
            fykin = K[4]
            # print(fy)
            cxx = K[2]
            # print(cx)
            cyy = K[5]
            # print(cy)

            #########################
            # Prepare the stuff
            #########################

            # read images and create mask
            rgbImg = cv2.imread(rgbImgPath)
            depImg = cv2.imread(depImgPath, cv2.IMREAD_UNCHANGED)
            depImg = np.multiply(depImg, 0.1)

            normImg = get_normal(depImg, fx=fxkin, fy=fykin, cx=cxx, cy=cyy, for_vis=True)

            normImg = np.multiply(normImg, 255.0)
            imgI = normImg.astype(np.uint8)

            cnt, bb, area, mask = create_BB(rgbImg)

            # normals image
            # kernel = np.ones((11, 11), np.uint8)
            # blur = cv2.dilate(depImg, kernel, iterations=1)
            # blur = cv2.medianBlur(blur, 5, )

            rows, cols = depImg.shape

            # create image number and name
            template = '00000'
            s = int(s)
            ss = int(ss) + 1
            pre = (s-1) * 1296
            id = pre + ss
            tempSS = template[:-len(str(id))]
            ssStr = str(id)
            imgNum = str(id)
            imgNam = tempSS + imgNum + '.jpg'
            iname = str(imgNam)

            # cnt = cnt.ravel()
            # cont = cnt.tolist()

            nx1 = bb[0]
            ny1 = bb[1]
            nx2 = nx1 + bb[2]
            ny2 = ny1 + bb[3]
            npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
            cont = npseg.tolist()

            #drawN = [1, 1, 1, 1, 2]
            drawN = [1]
            freq = np.bincount(drawN)

            rnd = np.random.choice(np.arange(len(freq)), 1, p=freq / len(drawN), replace=False)

            # change drawN if you want a data split
            # print("storage choice: ", rnd)
            rnd = 1
            if rnd == 1:

                # depthName = '/home/sthalham/data/T-less_Detectron/tless_all/trainD/' + imgNam
                # cv2.imwrite(rgbName, imgI)
                depthName = safe + 'train/' + imgNam
                cv2.imwrite(depthName, imgI)

                #print("TRAIN")
                #print("storing in train: ", rgbName)

                # create dictionaries for json
                tempTl = {
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "id": id,
                    "name": iname
                }
                dict["licenses"].append(tempTl)

                tempTi = {
                    "license": 2,
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "file_name": iname,
                    "height": rows,
                    "width": cols,
                    "date_captured": dateT,
                    "id": id
                }
                dict["images"].append(tempTi)

                annoID = annoID + 1
                tempTa = {
                    "id": id,
                    "image_id": id,
                    "category_id": s,
                    "bbox": bb,
                    "segmentation": [cont],
                    "area": area,
                    "iscrowd": 0
                }
                dict["annotations"].append(tempTa)

                counter = counter + 1
                # if counter > 1:
                #     trainAnno = "/home/sthalham/test.json"

                #     with open(trainAnno, 'w') as fp:
                #         json.dump(dict, fp)

                #     sys.exit()

            else:
                # depthName = '/home/sthalham/data/T-less_Detectron/tlessSplit/valD/' + imgNam
                # cv2.imwrite(depthName, imgI)
                #rgbName = '/home/sthalham/data/T-less_Detectron/tlessC_split/val/' + imgNam
                cv2.imwrite(rgbName, rgbImg)

                # print("VAL")
                print("storing in val: ", imgNam)

                # create dictionaries for json
                tempVl = {
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "id": id,
                    "name": iname
                }
                dictVal["licenses"].append(tempVl)

                tempVi = {
                    "license": 2,
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "file_name": iname,
                    "height": rows,
                    "width": cols,
                    "date_captured": dateT,
                    "id": id
                }
                dictVal["images"].append(tempVi)

                annoID = annoID + 1
                tempVa = {
                    "id": id,
                    "image_id": id,
                    "category_id": s,
                    "bbox": bb,
                    "segmentation": [cont],
                    "area": area,
                    "iscrowd": 0
                }
                dictVal["annotations"].append(tempVa)

            elapsed_time = time.time() - start_time
            eta = ((all-gloCo) * elapsed_time) / 60
            if gloCo % 100 == 0:
                print('eta: ', eta, ' min')

        category = str(s)
        tempC = {
            "id": s,
            "name": category,
            "supercategory": "object"
        }
        dict["categories"].append(tempC)
        dictVal["categories"].append(tempC)

    # valAnno = "/home/sthalham/data/T-less_Detectron/tlessC_split/annotations/instances_val_tless.json"
    trainAnno = safe + "annotations/instances_train_tless.json"

    with open(trainAnno, 'w') as fpT:
        json.dump(dict, fpT)

    # with open(valAnno, 'w') as fpV:
    #     json.dump(dictVal, fpV)

    print('everythings done')


