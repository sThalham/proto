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


depSca = 1.0
widthKin = 640
heightKin = 480
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899


def HHA_encoding(depth, normals, fx, fy, cx, cy, ds, R_w2c, T_w2c):

    rows, cols, channels = normals.shape

    # calculate disparity
    depthFloor = 100.0
    depthCeil = 1000.0

    disparity = np.ones((depth.shape), dtype=np.float32)
    disparity = np.divide(disparity, depth)
    disparity = disparity - (1 / depthCeil)
    denom = (1 / depthFloor) - (1 / depthCeil)
    disparity = np.divide(disparity, denom)
    disparity = np.where(np.isinf(disparity), 0.0, disparity)
    dispSca = disparity - np.nanmin(disparity)
    maxV = 255.0 / np.nanmax(dispSca)
    scatemp = np.multiply(dispSca, maxV)
    disp_final = scatemp.astype(np.uint8)

    # compute height
    depRe = depth.reshape(rows * cols)
    zP = np.multiply(depRe, ds)
    x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1), indexing='xy')
    yP = y.reshape(rows * cols) - cy
    xP = x.reshape(rows * cols) - cx
    yP = np.multiply(yP, zP)
    xP = np.multiply(xP, zP)
    yP = np.divide(yP, fy)
    xP = np.divide(xP, fx)

    cloud = np.transpose(np.array((xP, yP, zP)))

    zFloor = 0.0
    zCeil = 1000

    R_cam = np.asarray(R_w2c).reshape(3, 3)
    camPoints = np.transpose(np.matmul(R_cam, np.transpose(cloud))) + np.tile(T_w2c, cloud.shape[0]).reshape(
        cloud.shape[0], 3)
    height = camPoints[:, 2] - zFloor
    denom = zCeil - zFloor
    height = np.divide(height, denom)
    height = height.reshape(rows, cols)
    height = height - np.nanmin(height)
    scatemp = np.multiply(height, 255.0)
    height_final = scatemp.astype(np.uint8)
    height_final = np.where(height_final <= 0.0, 0.0, height_final)

    # compute gravity vector deviation
    angEst = np.zeros(normals.shape, dtype=np.float32)
    angEst[:, :, 0] = R_cam[0, 2]
    angEst[:, :, 1] = R_cam[1, 2]
    angEst[:, :, 2] = -R_cam[2, 2]

    angtemp = np.einsum('ijk,ijk->ij', normals, angEst)
    angEstNorm = np.linalg.norm(angEst, axis=2)
    normalsNorm = np.linalg.norm(normals, axis=2)
    normalize = np.multiply(normalsNorm, angEstNorm)
    angDif = np.divide(angtemp, normalize)

    np.where(angDif < 0.0, angDif + 1.0, angDif)
    angDif = np.arccos(angDif)
    angDif = np.multiply(angDif, (180 / math.pi))

    angDifSca = angDif - np.nanmin(angDif)
    maxV = 255.0 / np.nanmax(angDifSca)
    scatemp = np.multiply(angDifSca, maxV)
    grav_final = scatemp.astype(np.uint8)
    grav_final[grav_final is np.NaN] = 0

    # encode
    encoded = np.zeros((normals.shape), dtype=np.uint8)
    encoded[:, :, 0] = disp_final
    encoded[:, :, 1] = height_final
    encoded[:, :, 2] = grav_final

    return encoded


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
        scaDep = 1.0 / np.nanmax(depth_refine)
        depth_refine = np.multiply(depth_refine, scaDep)
        cross[:, :, 0] = cross[:, :, 0] * (1 - (depth_refine - 0.5))  # nearer has higher intensity
        cross[:, :, 1] = cross[:, :, 1] * (1 - (depth_refine - 0.5))
        cross[:, :, 2] = cross[:, :, 2] * (1 - (depth_refine - 0.5))

    return cross, depth_refine


def create_BB(rgb):

    imgray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    mask = imgray > 25

    oneA = np.ones(imgray.shape)
    masked = np.where(mask, oneA, 0)

    kernel = np.ones((9, 9), np.uint8)
    mask_dil = cv2.dilate(masked, kernel, iterations=1)

    im2, contours, hier = cv2.findContours(np.uint8(mask_dil), 1, 2)

    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    area = cv2.contourArea(box)

    # cv2.drawContours(rgb, [box], -1, (170, 160, 0), 2)
    # cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
    bb = [int(x),int(y),int(w),int(h)]

    return cnt, bb, area, mask_dil


if __name__ == "__main__":

    root = "/home/sthalham/data/LINEMOD/test/"  # path to train samples, depth + rgb
    # safe = sys.argv[2] + "/"
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
    allCo = 504 * 20
    times = []

    for s in sub:
        rgbPath = root + s + "/rgb/"
        depPath = root + s + "/depth/"
        gtPath = root + s + "/gt.yml"
        infoPath = root + s + "/info.yml"

        with open(infoPath, 'r') as stream:
            opYML = yaml.load(stream)

        with open(gtPath, 'r') as streamGT:
            gtYML = yaml.load(streamGT)

        subsub = os.listdir(rgbPath)

        counter = 0
        for ss in subsub:

            start_time = time.time()
            gloCo = gloCo + 1

            imgname = ss
            rgbImgPath = rgbPath + ss
            depImgPath = depPath + ss
            #print(rgbImgPath)

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
            #fxkin = K[0]
            #fykin = K[4]
            #cxx = K[2]
            #cyy = K[5]
            #cam_R = calib["cam_R_w2c"]
            #cam_T = calib["cam_t_w2c"]

            #########################
            # Prepare the stuff
            #########################

            # read images and create mask
            rgbImg = cv2.imread(rgbImgPath)
            depImg = cv2.imread(depImgPath, cv2.IMREAD_UNCHANGED)
            depImg = np.multiply(depImg, 0.1)
            rows, cols = depImg.shape

            normImg, depth_inpaint = get_normal(depImg, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin, for_vis=True)

            #encoded = HHA_encoding(depth_inpaint, normImg, fxkin, fykin, cxkin, cykin, depSca, cam_R, cam_T)
            scaNorm = 255.0 / np.nanmax(normImg)
            normImg = np.multiply(normImg, scaNorm)
            imgI = normImg.astype(np.uint8)

            #cv2.imwrite('/home/sthalham/disparity.png', imgI)
            #cv2.imwrite('/home/sthalham/height.png', encoded[:, :, 1])
            #cv2.imwrite('/home/sthalham/gravity.png', encoded[:, :, 2])
            #cv2.imwrite('/home/sthalham/HHA.png', encoded)

            # create image number and name
            template = '00000'
            s = int(s)
            ssm = int(ss) + 1
            pre = (s-1) * 1296
            img_id = pre + ssm
            tempSS = template[:-len(str(img_id))]

            imgNum = str(img_id)
            imgNam = tempSS + imgNum + '.jpg'
            iname = str(imgNam)

            # THIS for Training
            # cnt, bb, area, mask = create_BB(rgbImg)

            # THAT for Testing
            # print(s)
            # print(ss)
            # print(gtYML[int(ss)])
            gtImg = gtYML[int(ss)]

            drawN = [1, 1, 1, 1, 2]
            freq = np.bincount(drawN)
            rnd = np.random.choice(np.arange(len(freq)), 1, p=freq / len(drawN), replace=False)

            # change drawN if you want a data split
            # print("storage choice: ", rnd)
            rnd = 1
            if rnd == 1:

                for i in range(len(gtImg)):

                    curlist = gtImg[i]
                    obj_id = curlist["obj_id"]
                    obj_bb = curlist["obj_bb"]
                    #print(type(obj_id))

                    area = obj_bb[2] * obj_bb[3]

                    nx1 = obj_bb[0]
                    ny1 = obj_bb[1]
                    nx2 = nx1 + obj_bb[2]
                    ny2 = ny1 + obj_bb[3]
                    npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
                    cont = npseg.tolist()

                    annoID = annoID + 1
                    tempVa = {
                        "id": annoID,
                        "image_id": img_id,
                        "category_id": obj_id,
                        "bbox": obj_bb,
                        "segmentation": [cont],
                        "area": area,
                        "iscrowd": 0
                    }
                    dictVal["annotations"].append(tempVa)

                # cnt = cnt.ravel()
                # cont = cnt.tolist()

                depthName = '/home/sthalham/data/T-less_Detectron/linemodTest/val/' + imgNam
                cv2.imwrite(depthName, imgI)
                # cv2.imwrite(rgbName, rgbImg)

                #print("storing in test: ", imgNam)

                # create dictionaries for json
                tempVl = {
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "id": img_id,
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
                    "id": img_id
                }
                dictVal["images"].append(tempVi)
            else:

                for i in range(len(gtImg)):

                    curlist = gtImg[i]
                    obj_id = curlist["obj_id"]
                    obj_bb = curlist["obj_bb"]
                    #print(type(obj_id))

                    area = obj_bb[2] * obj_bb[3]

                    nx1 = obj_bb[0]
                    ny1 = obj_bb[1]
                    nx2 = nx1 + obj_bb[2]
                    ny2 = ny1 + obj_bb[3]
                    npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
                    cont = npseg.tolist()

                    annoID = annoID + 1
                    tempVa = {
                        "id": annoID,
                        "image_id": img_id,
                        "category_id": obj_id,
                        "bbox": obj_bb,
                        "segmentation": [cont],
                        "area": area,
                        "iscrowd": 0
                    }
                    dict["annotations"].append(tempVa)

                # cnt = cnt.ravel()
                # cont = cnt.tolist()

                depthName = '/home/sthalham/data/T-less_Detectron/linemodTest_HHA/train/' + imgNam
                # rgbName = '/home/sthalham/data/T-less_Detectron/tlessC_all/val/' + imgNam
                #cv2.imwrite(depthName, imgI)
                # cv2.imwrite(rgbName, rgbImg)

                #print("storing in test: ", imgNam)

                # create dictionaries for json
                tempVl = {
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "id": img_id,
                    "name": iname
                }
                dict["licenses"].append(tempVl)

                tempVi = {
                    "license": 2,
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "file_name": iname,
                    "height": rows,
                    "width": cols,
                    "date_captured": dateT,
                    "id": img_id
                }
                dict["images"].append(tempVi)

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            meantime = sum(times) / len(times)
            eta = ((allCo - gloCo) * meantime) / 60
            if gloCo % 100 == 0:
                print('eta: ', eta, ' min')

    catsInt = range(1, 31)

    for s in catsInt:
        objName = str(s)
        tempC = {
            "id": s,
            "name": objName,
            "supercategory": "object"
        }
        dict["categories"].append(tempC)
        dictVal["categories"].append(tempC)

    valAnno = "/home/sthalham/data/T-less_Detectron/linemodTest/annotations/instances_val_tless.json"
    #trainAnno = "/home/sthalham/data/T-less_Detectron/linemodTest_HHA/annotations/instances_train_tless.json"

    with open(valAnno, 'w') as fpV:
        json.dump(dictVal, fpV)

    #with open(trainAnno, 'w') as fpT:
    #    json.dump(dict, fpT)

    print('everythings done')


