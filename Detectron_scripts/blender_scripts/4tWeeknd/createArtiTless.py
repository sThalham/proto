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
import itertools

import OpenEXR, Imath

kin_res_x = 640
kin_res_y = 480
# fov = 1.0088002681732178
fov = 57.8

np.set_printoptions(threshold=np.nan)


def get_a_scene(fn_gt, fn_depth, fn_disp):
    # fn_gt = self.gt_list[index]
    # fn_depth = self.scene_path + "/depth/" + fn_gt.replace("_gt.yaml", "_depth.exr")
    # fn_disp = self.scene_path + "/disp/" + fn_gt.replace("_gt.yaml", "_disp.npy")
    disp = np.load(fn_disp)

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
    depth = np.fromstring(redstr, dtype=np.float32) * 3
    depth.shape = (size[1], size[0])  # Numpy arrays are (row, col)

    # depth = np.load(fn_depth)
    constant_disp = np.nanmean(depth[220:260, 300:340] * disp[220:260, 300:340])

    depth_est = constant_disp / disp

    centerX = kin_res_x / 2.0
    centerY = kin_res_y / 2.0

    points = np.zeros((kin_res_y, kin_res_x, 3), dtype=np.float32)
    constant = np.tan(np.radians(fov / 2)) / (kin_res_x / 2)
    uv_table = np.zeros((kin_res_y, kin_res_x, 2), dtype=np.int16)
    column = np.arange(0, kin_res_y)
    uv_table[:, :, 1] = np.arange(0, kin_res_x) - centerX
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY
    uv_table = np.abs(uv_table)

    depth = depth * np.cos(np.radians(fov / kin_res_x * np.abs(uv_table[:, :, 1]))) * np.cos(
        np.radians(fov / kin_res_x * uv_table[:, :, 0]))

    num_disp = 112 + 1
    depth_est[:, :num_disp] = 0  # block area in the left will be filled out with depth
    ratio = np.random.rand(kin_res_y, kin_res_x) * 0.1 + 0.9  # 0.3~0.6
    ratio[:, :num_disp] = 1.0  # use pure depth value for blocked area
    ratio = np.minimum(ratio, 1)

    depth_refine = ratio * depth + (1 - ratio)  # * depth_est
    depth_refine[depth_est > 3.0] = 0  # remove large disp
    depth_refine[:, 0:num_disp] = depth[:, :num_disp] + np.random.rand(kin_res_y, num_disp) * 0.002 - 0.001
    depth_refine[depth == 0] = 0  # post processing, keep Nan points with 0
    depth_refine[depth >= 3] = 0  # remove larger than 3 pixels with 0
    # set 0 -> can be a best

    return depth_refine, bboxes, poses, mask_ids

def manipulate_depth(fn_gt, fn_depth, fn_disp):
    # fn_gt = self.gt_list[index]
    # fn_depth = self.scene_path + "/depth/" + fn_gt.replace("_gt.yaml", "_depth.exr")
    # fn_disp = self.scene_path + "/disp/" + fn_gt.replace("_gt.yaml", "_disp.npy")
    disp = np.load(fn_disp)

    cv2.imwrite('/home/sthalham/depth.jpg', disp)

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
    depth = np.fromstring(redstr, dtype=np.float32) * 3



    depth.shape = (size[1], size[0])  # Numpy arrays are (row, col)

    # depth = np.load(fn_depth)
    constant_disp = np.nanmean(depth[220:260, 300:340] * disp[220:260, 300:340])

    depth_est = constant_disp / disp

    centerX = kin_res_x / 2.0
    centerY = kin_res_y / 2.0

    points = np.zeros((kin_res_y, kin_res_x, 3), dtype=np.float32)
    constant = np.tan(np.radians(fov / 2)) / (kin_res_x / 2)
    uv_table = np.zeros((kin_res_y, kin_res_x, 2), dtype=np.int16)
    column = np.arange(0, kin_res_y)
    uv_table[:, :, 1] = np.arange(0, kin_res_x) - centerX
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY
    uv_table = np.abs(uv_table)

    depth = depth * np.cos(np.radians(fov / kin_res_x * np.abs(uv_table[:, :, 1]))) * np.cos(
        np.radians(fov / kin_res_x * uv_table[:, :, 0]))

    cv2.imwrite('/home/sthalham/depth.jpg', depth)

    num_disp = 112 + 1
    depth_est[:, :num_disp] = 0  # block area in the left will be filled out with depth
    ratio = np.random.rand(kin_res_y, kin_res_x) * 0.1 + 0.9  # 0.3~0.6
    ratio[:, :num_disp] = 1.0  # use pure depth value for blocked area
    ratio = np.minimum(ratio, 1)

    depth_refine = ratio * depth + (1 - ratio)  # * depth_est
    depth_refine[depth_est > 3.0] = 0  # remove large disp
    depth_refine[:, 0:num_disp] = depth[:, :num_disp] + np.random.rand(kin_res_y, num_disp) * 0.002 - 0.001
    depth_refine[depth == 0] = 0  # post processing, keep Nan points with 0
    depth_refine[depth >= 3] = 0  # remove larger than 3 pixels with 0
    # set 0 -> can be a best

    return depth_refine, bboxes, poses, mask_ids

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

    root = sys.argv[1]  # path to train samples
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

    annoID = 0
    counter = 0

    depPath = root + "/depth/"
    disPath = root + "/disp/"
    gtPath = root
    maskPath = root + "/mask/"

    for fileInd in os.listdir(root):
        if fileInd.endswith(".yaml"):
            redname = fileInd[:-8]
            gtfile = gtPath + '/' + fileInd
            depfile = depPath + redname + "_depth.exr"
            disfile = disPath + redname + "_disp.npy"
            maskfile = maskPath + redname + "_mask.npy"

            # depth_refine, bboxes, poses, mask_ids = get_a_scene(gtfile, depfile, disfile)
            depth_refine, bboxes, poses, mask_ids = manipulate_depth(gtfile, depfile, disfile)
            depth_refine = np.multiply(depth_refine, 1000.0)  # to millimeters

            rows, cols = depth_refine.shape

            # focalL = 1008.80026817 # blender focal length
            # similar to t-less' focal length
            focalL = 580.0  # according to microsoft
            normImg = get_normal(depth_refine, fx=focalL, cx=(kin_res_x * 0.5), cy=(kin_res_y * 0.5), for_vis=True)

            normImg = np.multiply(normImg, 255.0)
            imgI = normImg.astype(np.uint8)

            imgPath = '/home/sthalham/data/T-less_Detectron/tlessDArtiCloser/trainD/' + redname + '.jpg'
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
                tempVa = {
                    "id": annoID,
                    "image_id": imgID,
                    "category_id": objID,
                    "bbox": bb,
                    "segmentation": [seg],
                    "area": area,
                    "iscrowd": 0
                }
                dict["annotations"].append(tempVa)

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

            # cv2.imwrite(imgPath, imgI)
            cv2.imwrite('/home/sthalham/artitest.jpg', imgI)

            print("storing in test: ", imgName)

            tempVl = {
                "url": "cmp.felk.cvut.cz/t-less/",
                "id": imgID,
                "name": imgName
            }
            dict["licenses"].append(tempVl)

            tempVi = {
                "license": 2,
                "url": "cmp.felk.cvut.cz/t-less/",
                "file_name": imgName,
                "height": rows,
                "width": cols,
                "date_captured": dateT,
                "id": imgID
            }
            dict["images"].append(tempVi)

            counter = counter + 1
            print(counter)
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
        print(s)
        print(objName)
        tempC = {
            "id": s,
            "name": objName,
            "supercategory": "object"
        }
        dict["categories"].append(tempC)

    valAnno = "/home/sthalham/data/T-less_Detectron/tlessDArtiCloser/annotations/instances_train_tless.json"

    # with open(valAnno, 'w') as fpV:
    #     json.dump(dict, fpV)

    print('Chill for once in your life... everything\'s done')
