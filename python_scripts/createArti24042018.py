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
import transforms3d as tf3d
import time
import itertools

import OpenEXR, Imath

kin_res_x = 640
kin_res_y = 480
# fov = 1.0088002681732178
fov = 57.8

np.set_printoptions(threshold=np.nan)


def roundPartial(value, resolution):
    if resolution == 0:
        return 0
    return round(value / resolution) * resolution


def manipulate_depth(fn_gt, fn_depth, fn_part):

    with open(fn_gt, 'r') as stream:
        query = yaml.load(stream)
        bboxes = np.zeros((len(query), 5), np.int)
        poses = np.zeros((len(query), 7), np.float32)
        mask_ids = np.zeros((len(query)), np.int)
        for j in range(len(query)):
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

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    golden = OpenEXR.InputFile(fn_depth)
    dw = golden.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = golden.channel('R', pt)
    depth = np.fromstring(redstr, dtype=np.float32)
    depth.shape = (size[1], size[0])

    centerX = kin_res_x / 2.0
    centerY = kin_res_y / 2.0

    uv_table = np.zeros((kin_res_y, kin_res_x, 2), dtype=np.int16)
    column = np.arange(0, kin_res_y)
    uv_table[:, :, 1] = np.arange(0, kin_res_x) - centerX
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY
    uv_table = np.abs(uv_table)

    depth = depth * np.cos(np.radians(fov / kin_res_x * np.abs(uv_table[:, :, 1]))) * np.cos(
        np.radians(fov / kin_res_x * uv_table[:, :, 0]))

    # erode and blur mask to get more realistic appearance
    partmask = cv2.imread(fn_part, -1)
    # partmask = cv2.dilate(partmask, (7, 7))
    partmask = cv2.medianBlur(partmask, 5)
    partmask = partmask.astype(np.uint8)

    mask = partmask > 25
    mask = np.where(mask, 255, 0.0)
    depth = np.where(mask, depth, 0.0)
    depth = cv2.GaussianBlur(depth, (7, 7), 0)

    # round to depth resolution
    res = (depth / 1000 * 1.41421356) ** 2

    # discretize to resolution and apply gaussian
    # discretize to resolution steps
    dNonVar = np.divide(depth, res, out=np.zeros_like(depth), where=res!=0)
    dNonVar = np.round(dNonVar)
    dNonVar = np.multiply(dNonVar, res)

    # determine elementwise noise
    noise = np.multiply(dNonVar, 0.01)

    # gaussian draw
    dMan = np.random.normal(loc=dNonVar, scale=noise, size=dNonVar.shape)

    '''
    dRes = np.empty(depth.shape)
    for y in range(len(depth)):
        for x in range(len(depth[y])):
            dNonVar = roundPartial(depth[y, x], res[y, x])
            noise = dNonVar * 0.015
            dRes[y, x] = np.random.normal(loc=dNonVar, scale=noise)
    '''

    return dMan, bboxes, poses, mask_ids


def get_normal(depth_refine, fx=-1, fy=-1, cx=-1, cy=-1, for_vis=True):
    res_y = depth_refine.shape[0]
    res_x = depth_refine.shape[1]

    centerX = cx
    centerY = cy

    constant = 1 / fx
    uv_table = np.zeros((res_y, res_x, 2), dtype=np.int16)
    column = np.arange(0, res_y)

    uv_table[:, :, 1] = np.arange(0, res_x) - centerX  # x-c_x (u)
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY  # y-c_y (v)
    uv_table_sign = np.copy(uv_table)

    depth_refine = ndimage.gaussian_filter(depth_refine, 2)  # sigma=3)

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

    cross = cross / norm
    cross = cross.reshape(res_y, res_x, 3)
    cross = np.abs(cross)
    cross = np.nan_to_num(cross)

    cam_angle = np.arccos(cross[:, :, 2])
    cross[np.abs(cam_angle) > math.radians(75)] = 0  # high normal cut
    cross[depth_refine <= 300] = 0  # 0 and near range cut
    cross[depth_refine > 3000] = 0  # far range cut
    if not for_vis:
        cross[:, :, 0] = cross[:, :, 0] * (1 - (depth_refine - 0.5))  # nearer has higher intensity
        cross[:, :, 1] = cross[:, :, 1] * (1 - (depth_refine - 0.5))
        cross[:, :, 2] = cross[:, :, 2] * (1 - (depth_refine - 0.5))

    return cross


##########################
#         MAIN           #
##########################
if __name__ == "__main__":

    root = '/home/sthalham/data/t-less_mani/artificialScenes/renderedScenes24042018'  # path to train samples

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
    all = 10724
    times = []

    depPath = root + "/depth/"
    partPath = root + "/part/"
    gtPath = root
    maskPath = root + "/mask/"

    for fileInd in os.listdir(root):
        if fileInd.endswith(".yaml"):

            start_time = time.time()
            gloCo = gloCo + 1

            redname = fileInd[:-8]
            gtfile = gtPath + '/' + fileInd
            depfile = depPath + redname + "_depth.exr"
            partfile = partPath + redname + "_part.png"

            # depth_refine, bboxes, poses, mask_ids = get_a_scene(gtfile, depfile, disfile)
            depth_refine, bboxes, poses, mask_ids = manipulate_depth(gtfile, depfile, partfile)
            depth_refine = np.multiply(depth_refine, 1000.0)  # to millimeters

            rows, cols = depth_refine.shape

            # focalL = 1008.80026817 # blender focal length
            # similar to t-less' focal length
            focalL = 580.0  # according to microsoft
            normImg = get_normal(depth_refine, fx=focalL, cx=(kin_res_x * 0.5), cy=(kin_res_y * 0.5), for_vis=True)

            normImg = np.multiply(normImg, 255.0)
            imgI = normImg.astype(np.uint8)

            drawN = [1, 1, 1, 1, 2]
            freq = np.bincount(drawN)
            rnd = np.random.choice(np.arange(len(freq)), 1, p=freq / len(drawN), replace=False)

            # change drawN if you want a data split
            # print("storage choice: ", rnd)
            if rnd == 1:

                imgPath = '/home/sthalham/data/T-less_Detectron/tlessArti24042018_split/train/' + redname + '.jpg'
                imgID = int(redname)
                imgName = redname + '.jpg'

                for bbox in bboxes:
                    objID = np.asscalar(bbox[0]) + 1
                    x1 = np.asscalar(bbox[2])
                    y1 = np.asscalar(bbox[1])
                    x2 = np.asscalar(bbox[4])
                    y2 = np.asscalar(bbox[3])
                    nx1 = bbox[2]
                    ny1 = bbox[1]
                    nx2 = bbox[4]
                    ny2 = bbox[3]
                    w = (x2-x1)
                    h = (y2-y1)
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
                    cv2.rectangle(imgI, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (int(bb[0]), int(bb[1]))
                    fontScale = 1
                    fontColor = (255, 0, 0)
                    lineType = 2
                    gtText = 'cat: ' + str(objID)
                    cv2.putText(imgI, gtText,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)
                    '''

                cv2.imwrite(imgPath, imgI)
                # cv2.imwrite('/home/sthalham/artitest.jpg', imgI)

                # print("storing in test: ", imgName)

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

            else:
                imgPath = '/home/sthalham/data/T-less_Detectron/tlessArti24042018_split/val/' + redname + '.jpg'
                imgID = int(redname)
                imgName = redname + '.jpg'

                # create dictionaries for json
                for bbox in bboxes:
                    objID = np.asscalar(bbox[0]) + 1
                    x1 = np.asscalar(bbox[2])
                    y1 = np.asscalar(bbox[1])
                    x2 = np.asscalar(bbox[4])
                    y2 = np.asscalar(bbox[3])
                    nx1 = bbox[2]
                    ny1 = bbox[1]
                    nx2 = bbox[4]
                    ny2 = bbox[3]
                    w = (x2-x1)
                    h = (y2-y1)
                    bb = [x1, y1, w, h]
                    area = w * h
                    npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
                    seg = npseg.tolist()

                    annoID = annoID + 1
                    tempVA = {
                        "id": annoID,
                        "image_id": imgID,
                        "category_id": objID,
                        "bbox": bb,
                        "segmentation": [seg],
                        "area": area,
                        "iscrowd": 0
                    }
                    dictVal["annotations"].append(tempVA)

                cv2.imwrite(imgPath, imgI)
                # cv2.imwrite('/home/sthalham/artitest.jpg', imgI)

                # print("storing in test: ", imgName)

                tempVL = {
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "id": imgID,
                    "name": imgName
                }
                dictVal["licenses"].append(tempVL)

                tempVI = {
                    "license": 2,
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "file_name": imgName,
                    "height": rows,
                    "width": cols,
                    "date_captured": dateT,
                    "id": imgID
                }
                dictVal["images"].append(tempVI)

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            meantime = sum(times)/len(times)
            eta = ((all - gloCo) * meantime) / 60
            if gloCo % 100 == 0:
                print('eta: ', eta, ' min')

        # if counter > 1:
        #     catsInt = range(1, 30)
        #     for s in catsInt:
        #         objName = str(s)
        #         print(s)
        #         print(objName)
        #         tempC = {
        #             "id": s,
        #             "name": objName,
        #             "supercategory": "object"
        #         }
        #         dict["categories"].append(tempC)
        #     trainAnno = "/home/sthalham/test.json"
        #     with open(trainAnno, 'w') as fp:
        #         json.dump(dict, fp)

        #     sys.exit()

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

    traAnno = "/home/sthalham/data/T-less_Detectron/tlessArti24042018_split/annotations/instances_train_tless.json"
    valAnno = "/home/sthalham/data/T-less_Detectron/tlessArti24042018_split/annotations/instances_val_tless.json"

    with open(traAnno, 'w') as fpT:
        json.dump(dict, fpT)

    with open(valAnno, 'w') as fpV:
        json.dump(dictVal, fpV)

    print('Chill for once in your life... everything\'s done')
