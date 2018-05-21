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

# mAP
# Precision = True positive / (True positive + False positive)
# Recall = True positive / (True positive + False negative)


def listDiff(first, second):
    # second = set(second)
    return [item for item in first if item not in second]


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
    a = np.array([a[0], a[1], a[0] + a[2], a[1] + a[3]])
    b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])

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

    root = '/home/sthalham/data/T-less_Detectron/output/linemodArti18052018/test/coco_2014_val/generalized_rcnn/'  # path to train samples, depth + rgb
    # imgroot = '/home/sthalham/data/t-less_v2/test_kinect/'
    # annoroot = '/home/sthalham/data/T-less_Detectron/tlessArti30042018/annotations/'
    jsons = root + 'bbox_coco_2014_val_results.json'
    # pkls = root + 'detection_results.pkl'

    json_data = open(jsons).read()
    data = json.loads(json_data)
    # print(len(data))

    # annotations = annoroot + 'instances_val_tless.json'
    # anno_data = open(annotations).read()
    # annodata = json.loads(anno_data)

    testData = '/home/sthalham/data/LINEMOD/test/'

    sub = os.listdir(testData)


    absObjs = 0
    gtCatLst = [0] * 16
    detCatLst = [0] * 16
    falsePosLst = [0] * 16
    falseNegLst = [0] * 16

    for s in sub:
        rgbPath = testData + s + "/rgb/"
        depPath = testData + s + "/depth/"
        gtPath = testData + s + "/gt.yml"
        infoPath = testData + s + "/info.yml"

        with open(infoPath, 'r') as stream:
            opYML = yaml.load(stream)

        with open(gtPath, 'r') as streamGT:
            gtYML = yaml.load(streamGT)

        subsub = os.listdir(rgbPath)

        counter = 0
        for ss in subsub:
            imgname = ss
            rgbImgPath = rgbPath + ss
            depImgPath = depPath + ss
            # print('processing image: ', rgbImgPath)

            if ss.startswith('000'):
                ss = ss[3:]
            elif ss.startswith('00'):
                ss = ss[2:]
            elif ss.startswith('0'):
                ss = ss[1:]
            ss = ss[:-4]

            template = '00000'
            s = int(s)
            ssm = int(ss) + 1
            pre = (s - 1) * 1296
            img_id = pre + ssm
            tempSS = template[:-len(str(img_id))]

            imgNum = str(img_id)
            imgNam = tempSS + imgNum + '.jpg'
            iname = str(imgNam)

            gtImg = gtYML[int(ss)]
            gtBoxes = []
            gtCats = []
            for gt in gtImg:
                gtBoxes.append(gt['obj_bb'])
                gtCats.append(gt['obj_id'])
            absObjs = absObjs + len(gtCats)  # increment all

            detBoxes = []
            detCats = []
            detSco = []
            for det in data:
                if det['image_id'] == img_id:
                    detBoxes.append(det['bbox'])
                    detCats.append(det['category_id'])
                    detSco.append(det['score'])

            if len(detBoxes) < 1:
                for i in gtCats:
                    gtCatLst[i] = gtCatLst[i] + 1
            else:
                '''
                boxCleaned = copy.deepcopy(detBoxes)
                scoCleaned = copy.deepcopy(detSco)
                catCleaned = copy.deepcopy(detCats)
                for i, box in enumerate(detBoxes):
                    for j, box2 in enumerate(detBoxes):
                        if i == j:
                            continue
                        b1 = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                        b2 = np.array([box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]])
                        IoU = boxoverlap(b1, b2)
                        if IoU > 0.5 and detSco[i] > detSco[j]:
                            boxCleaned[j] = float('nan')
                            scoCleaned[j] = float('nan')
                            catCleaned[j] = float('nan')

                detBoxes = [x for x in boxCleaned if str(x) != 'nan']
                detSco = [x for x in scoCleaned if str(x) != 'nan']
                detCats = [x for x in catCleaned if str(x) != 'nan']
                '''
                boxCleaned = copy.deepcopy(detBoxes)
                scoCleaned = copy.deepcopy(detSco)
                catCleaned = copy.deepcopy(detCats)
                for i, cat in enumerate(detCats):
                    if cat is not gtCats[0]:
                        boxCleaned[i] = float('nan')
                        scoCleaned[i] = float('nan')
                        catCleaned[i] = float('nan')

                detBoxes = [x for x in boxCleaned if str(x) != 'nan']
                detSco = [x for x in scoCleaned if str(x) != 'nan']
                detCats = [x for x in catCleaned if str(x) != 'nan']

                fp = listDiff(detCats, gtCats)
                fn = listDiff(gtCats, detCats)
                tp = listDiff(gtCats, fn)

                for i in gtCats:
                    gtCatLst[i] = gtCatLst[i] + 1
                if tp:
                    for i in tp:
                        detCatLst[i] = detCatLst[i] + 1
                if fp:
                    for i in fp:
                        falsePosLst[i] = falsePosLst[i] + 1
                if fn:
                    for i in fn:
                        falseNegLst[i] = falseNegLst[i] + 1

                # VISUALIZATION

                '''
                img = cv2.imread(rgbImgPath, -1)
                for i, bb in enumerate(detBoxes):
                    cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0]) + int(bb[2]), int(bb[1]) + int(bb[3])), (255, 0, 0),2)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (int(bb[0]), int(bb[1]))
                    fontScale = 1
                    fontColor = (255, 0, 0)
                    lineType = 2
                    gtText = 'cat: ' + str(detCats[i])
                    cv2.putText(img, gtText,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)

                cv2.imwrite('/home/sthalham/visTests/detect.jpg', img)
                '''

                #print('STOP')

    # Precision = True positive / (True positive + False positive)
    # Recall = True positive / (True positive + False negative)
    detAcc = [0] * 16
    detPre = [0] * 16
    detRec = [0] * 16
    for ind, cat in enumerate(gtCatLst):
        if ind == 0:
            continue
        detAcc[ind] = detCatLst[ind] / cat
        print('accuracy category ', ind, ': ', detAcc[ind])

        if (detCatLst[ind] + falsePosLst[ind]) == 0:
            detPre[ind] = 0.0
        else:
            detPre[ind] = detCatLst[ind] / (detCatLst[ind] + falsePosLst[ind])
        if (detCatLst[ind] + falseNegLst[ind]) == 0:
            detRec[ind] = 0.0
        else:
            detRec[ind] = detCatLst[ind] / (detCatLst[ind] + falseNegLst[ind])

        print('precision category ', ind, ': ', detPre[ind])
        print('recall category ', ind, ': ', detRec[ind])

    print('accuracy overall: ', sum(detAcc)/len(detAcc))
    print('mAP: ', sum(detPre) / len(detPre))
    print('mAR: ', sum(detRec) / len(detRec))


