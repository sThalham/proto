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
import pickle
import time

# mAP
# Precision = True positive / (True positive + False positive)
# Recall = True positive / (True positive + False negative)

def IntersectUnion(bbA, bbB):
    xA = max(bbA[0], bbB[0])
    yA = max(bbA[1], bbB[1])
    xB = min(bbA[2], bbB[2])
    yB = min(bbA[3], bbB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (bbA[2] - bbA[0] + 1) * (bbA[3] - bbA[1] + 1)
    boxBArea = (bbB[2] - bbB[0] + 1) * (bbB[3] - bbB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def boxoverlap(a, b):
    # Compute the symmetric intersection over union overlap between a set of
    # bounding boxes in a and a single bounding box in b.

    # a  a matrix where each row specifies a bounding box
    # b  a single bounding box

    # AUTORIGHTS
    # -------------------------------------------------------
    # Copyright (C) 2011-2012 Ross Girshick
    # Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick

    # This file is part of the voc-releaseX code
    # (http://people.cs.uchicago.edu/~rbg/latent/)
    # and is available under the terms of an MIT-like license
    # provided in COPYING. Please retain this notice and
    # COPYING if you use this file (or a portion of it) in
    # your project.
    # -------------------------------------------------------

    x1 = np.amax(np.array([a[0], b[0]]))
    y1 = np.amax(np.array([a[1], b[1]]))
    x2 = np.amin(np.array([a[2], b[2]]))
    y2 = np.amin(np.array([a[3], b[3]]))

    wid = x2-x1+1
    hei = y2-y1+1
    inter = wid * hei
    aarea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    # intersection over union overlap
    ovlap = inter / (aarea + barea - inter)
    # set invalid entries to 0 overlap
    maskwid = wid <= 0
    maskhei = hei <= 0
    np.where(ovlap, maskwid, 0)
    np.where(ovlap, maskhei, 0)

    return ovlap


if __name__ == "__main__":

    root = sys.argv[1] +'/'  # path to train samples, depth + rgb
    print('file: ', root)

    imgroot = '/home/sthalham/data/t-less_v2/train_kinect/'
    annoroot = '/home/sthalham/data/T-less_Detectron/tlessC_split/annotations/'
    jsons = root + 'bbox_coco_2015_test_results.json'
    pkls = root + 'detection_results.pkl'

    json_data = open(jsons).read()
    data = json.loads(json_data)
    print(len(data))

    annotations = annoroot + 'instances_val_tless.json'
    anno_data = open(annotations).read()
    annodata = json.loads(anno_data)

    ob2det = 0
    obcorr = 0
    catLst = [0] * 31

    gtLst = [0] * 31

    maxInd = 20 * 1296
    picT = '0000'
    repT = '00'

    toRM = 0
    for coOut, preRM in enumerate(data):
        if preRM is None:
            continue
        for count, prm in enumerate(data):
            if prm is None:
                continue

            if preRM['image_id'] == prm['image_id']:
                if preRM['score'] > prm['score']:
                    data[count] = None

    data = list(filter(None, data))
    print(len(data))

    for detect in data:

        cat = detect['category_id']
        score = detect['score']
        Iid = detect['image_id']
        bb = detect['bbox']

        if Iid < 1297:
            Isep = Iid - 1
            strID = str(Isep)
            picN = picT[:-len(strID)]
            picN = picN + strID + '.png'
            repo = '01'
        else:
            Isep = Iid - 1
            Isep = (Isep % 1296)
            strID = str(Isep)
            picN = picT[:-len(strID)]
            picN = picN + strID + '.png'
            rep = (Iid // 1296)+1
            strRE = str(rep)
            repo = repT[:-len(strRE)]
            repo = repo + strRE

        imageCO = repo + '/rgb/' + picN
        imgName = imgroot + imageCO
        imgName = imgroot + imageCO
        img = cv2.imread(imgName, -1)

        bb = np.array(bb, dtype=np.int16)

        testgt = annodata['annotations']
        for y in testgt:
            if y['image_id'] == Iid:
                daBox = y['bbox']
                gtCat = y['category_id']

        gtBox = np.array([daBox[0], daBox[1], daBox[0] + daBox[2], daBox[1] + daBox[3]])
        detBox = np.array([bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]])

        IoU = boxoverlap(gtBox, detBox)


        x = bb[0]
        y = bb[1]
        w = bb[2]
        h = bb[3]

        '''
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x, y)
        fontScale = 0.5
        fontColor = (0, 255, 0)
        lineType = 1
        imgText = 'cat: ' + str(cat) + '; score: ' + str(score)
        cv2.putText(img, imgText,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        
        cv2.rectangle(img, (daBox[0], daBox[1]), (daBox[0] + daBox[2], daBox[1] + daBox[3]), (75, 125, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (daBox[0], daBox[1] + daBox[3])
        fontScale = 0.5
        fontColor = (75, 125, 0)
        lineType = 1
        gtText = 'gtcat: ' + str(gtCat) + '; IoU: ' + str(IoU)
        cv2.putText(img, gtText,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imwrite('/home/sthalham/detect.jpg', img)
        '''

        if IoU > 0.5 and cat == gtCat:
            obcorr = obcorr + 1
            catLst[gtCat] = catLst[gtCat] + 1
            ob2det = ob2det + 1
            gtLst[gtCat] = gtLst[gtCat] + 1

        else:
            ob2det = ob2det + 1
            gtLst[gtCat] = gtLst[gtCat] + 1

    print(obcorr)
    print(ob2det)
    print(gtLst)
    print(catLst)

    overallPOS = obcorr/ob2det

    for i, e in enumerate(catLst):
        if i > 0:
            thisCat = i
            catAcc = e/gtLst[i]
            print('accuracy object ', i, ': ', catAcc)
            print('intDet: ', e)
            print('intgt: ', gtLst[i])

    print('accuracy overall: ', overallPOS)


