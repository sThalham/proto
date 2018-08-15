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
depthCut = 1800.0

np.set_printoptions(threshold=np.nan)


def encodeImage(depth):
    img = np.zeros((resY, resX, 3), dtype=np.uint8)

    normImg, depImg = get_normal(depth, fxkin, fykin, cxkin, cykin, for_vis=True)
    img[:, :, 0] = compute_disparity(depImg)
    img[:, :, 1] = encode_area(depImg)
    img[:, :, 2] = compute_angle2gravity(normImg, depImg)

    return img


def create_point_cloud(depth, fx, fy, cx, cy, ds):

    rows, cols = depth.shape

    depRe = depth.reshape(rows * cols)
    zP = np.multiply(depRe, ds)

    x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1), indexing='xy')
    yP = y.reshape(rows * cols) - cy
    xP = x.reshape(rows * cols) - cx
    yP = np.multiply(yP, zP)
    xP = np.multiply(xP, zP)
    yP = np.divide(yP, fy)
    xP = np.divide(xP, fx)

    cloud_final = np.transpose(np.array((xP, yP, zP)))

    return cloud_final


def encode_area(depth):

    onethird = cv2.resize(depth, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
    pc = create_point_cloud(onethird, fxkin, fykin, cxkin, cykin, 1.0)
    cloud = pcl.PointCloud(np.array(pc, dtype=np.float32))
    CPPInput = "/home/sthalham/workspace/proto/python_scripts/CPP_workaround/bin/tempCloud.pcd"
    CPPOutput = "/home/sthalham/workspace/proto/python_scripts/CPP_workaround/bin/output.pcd"
    pcl.save(cloud, CPPInput)

    args = ("/home/sthalham/workspace/proto/python_scripts/CPP_workaround/bin/conditional_euclidean_clustering", CPPInput, CPPOutput)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    cloudNew = pcl.load_XYZI(CPPOutput)
    pcColor = cloudNew.to_array()
    inten = pcColor[:, 3]
    inten = np.reshape(inten, (int(resY/3), int(resX/3)))

    clusters, surf = np.unique(inten, return_counts=True)

    flat = surf.flatten()
    flat.sort()
    area_ref = np.mean(flat[0:-1])
    area_max = np.nanmax(surf)
    areaCol = np.ones(inten.shape, dtype=np.uint8)

    for i, cl in enumerate(clusters):
        if surf[i] < area_ref:
            mask = np.where(inten == cl, True, False)
            val = 255.0 - ((surf[i] / area_ref) * 127.5)  # prob.: 255 - ...
            val = val.astype(dtype=np.uint8)
            areaCol = np.where(mask, val, areaCol)

        else:
            mask = np.where(inten == cl, True, False)
            val = 127.5 - ((surf[i] / area_max) * 126.5)  # prob.: 255 - ...
            val = val.astype(dtype=np.uint8)
            areaCol = np.where(mask, val, areaCol)

    areaCol = cv2.resize(areaCol, (resX, resY), interpolation=cv2.INTER_NEAREST)

    areaCol = np.where(depth > depthCut, 0, areaCol)

    return areaCol


def compute_disparity(depth):
    # calculate disparity
    depthFloor = 100.0
    depthCeil = depthCut

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

    return disp_final


def compute_angle2gravity(normals, depth):
    r, c, p = normals.shape
    mask = depth < depthCut
    normals[:, :, 0] = np.where(mask, normals[:, :, 0], np.NaN)
    normals[:, :, 1] = np.where(mask, normals[:, :, 1], np.NaN)
    normals[:, :, 2] = np.where(mask, normals[:, :, 2], np.NaN)

    angEst = np.zeros(normals.shape, dtype=np.float32)
    angEst[:, :, 2] = 1.0
    ang = (45.0, 45.0, 45.0, 45.0, 45.0, 15.0, 15.0, 15.0, 15.0, 15.0, 5.0, 5.0)
    for th in ang:
        angtemp = np.einsum('ijk,ijk->ij', normals, angEst)
        angEstNorm = np.linalg.norm(angEst, axis=2)
        normalsNorm = np.linalg.norm(normals, axis=2)
        normalize = np.multiply(normalsNorm, angEstNorm)
        angDif = np.divide(angtemp, normalize)

        np.where(angDif < 0.0, angDif + 1.0, angDif)
        angDif = np.arccos(angDif)
        angDif = np.multiply(angDif, (180 / math.pi))

        cond1 = (angDif < th)
        cond1_ = (angDif > (180.0 - th))
        cond2 = (angDif > (90.0 - th)) & (angDif < (90.0 + th))
        cond1 = np.repeat(cond1[:, :, np.newaxis], 3, axis=2)
        cond1_ = np.repeat(cond1_[:, :, np.newaxis], 3, axis=2)
        cond2 = np.repeat(cond2[:, :, np.newaxis], 3, axis=2)

        NyPar1 = np.extract(cond1, normals)
        NyPar2 = np.extract(cond1_, normals)
        NyPar = np.concatenate((NyPar1, NyPar2))
        npdim = (NyPar.shape[0] / 3)
        NyPar = np.reshape(NyPar, (int(npdim), 3))
        NyOrt = np.extract(cond2, normals)
        nodim = (NyOrt.shape[0] / 3)
        NyOrt = np.reshape(NyOrt, (int(nodim), 3))

        cov = (np.transpose(NyOrt)).dot(NyOrt) - (np.transpose(NyPar)).dot(NyPar)
        u, s, vh = np.linalg.svd(cov)
        angEst = np.tile(u[:, 2], r * c).reshape((r, c, 3))

    angDifSca = angDif - np.nanmin(angDif)
    maxV = 255.0 / np.nanmax(angDifSca)
    scatemp = np.multiply(angDifSca, maxV)
    gImg = scatemp.astype(np.uint8)
    gImg[gImg is np.NaN] = 0

    return gImg


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
        return None, None, None, None

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
        return None, None, None, None

    depth = cv2.resize(depth, (resX, resY))

    # erode and blur mask to get more realistic appearance
    partmask = cv2.imread(fn_part, 0)
    partmask = cv2.resize(partmask, (resX, resY))
    partmask = partmask.astype(np.float32)
    mask = partmask > (np.median(partmask)*0.4)
    partmask = np.where(mask, 255.0, 0.0)
    ###################################################
    # DON'T REMOVE, best normal map up to now !!! tested without lateral noise !!!
    kernel = np.ones((3, 3))
    partmask = signal.medfilt2d(partmask, kernel_size=3)
    partmask = cv2.morphologyEx(partmask, cv2.MORPH_OPEN, kernel)
    #partmask = signal.medfilt2d(partmask, kernel_size=3)
    ###################################################
    partmask = partmask.astype(np.uint8)

    mask = partmask > 20
    depth = np.where(mask, depth, 0.0)

    depth = cv2.resize(depth, None, fx=1/2, fy=1/2)

    res = (((depth / 1000) * 1.41421356) ** 2) * 1000

    depthFinal = cv2.GaussianBlur(depth, (5, 5), 0.5, 0.5)

    # discretize to resolution and apply gaussian
    dNonVar = np.divide(depthFinal, res, out=np.zeros_like(depth), where=res != 0)
    dNonVar = np.round(dNonVar)
    dNonVar = np.multiply(dNonVar, res)

    '''
    according to Khoshelham et al.
    std-dev(mm) calib:   x = y = 10, z = 18
    std-dev(mm) uncalib: x = 14, y = 15, z = 18
    '''
    noise = np.multiply(dNonVar, 0.003)
    depthFinal = np.random.normal(loc=dNonVar, scale=noise, size=dNonVar.shape)

    # INTER_NEAREST - a nearest-neighbor interpolation
    # INTER_LINEAR - a bilinear interpolation (used by default)
    # INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
    # INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
    # INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
    #depthFinal = cv2.resize(depthFinal, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    depthFinal = cv2.resize(depthFinal, (resX, resY))

    ### standard procedure
    '''
    partmask = cv2.imread(fn_part, 0)
    partmask = cv2.resize(partmask, (resX, resY))
    partmask = partmask.astype(np.float32)
    mask = partmask > (np.median(partmask)*0.4)
    partmask = np.where(mask, 255.0, 0.0)
    kernel = np.ones((7, 7))
    partmask = cv2.morphologyEx(partmask, cv2.MORPH_OPEN, kernel)
    partmask = signal.medfilt2d(partmask, kernel_size=5)
    partmask = partmask.astype(np.uint8)
    mask = partmask > 20
    depth = np.where(mask, depth, 0.0)
    depth = cv2.resize(depth, None, fx=1/2, fy=1/2)
    res = (((depth / 1000) * 1.41421356) ** 2) * 1000
    depth = cv2.GaussianBlur(depth, (5, 5), 0.5, 0.5)
    dNonVar = np.divide(depth, res, out=np.zeros_like(depth), where=res != 0)
    dNonVar = np.round(dNonVar)
    dNonVar = np.multiply(dNonVar, res)
    noise = np.multiply(dNonVar, 0.003)
    depthFinal = np.random.normal(loc=dNonVar, scale=noise, size=dNonVar.shape)
    depthFinal = cv2.resize(depthFinal, (resX, resY))
    '''

    return depthFinal, bboxes, poses, mask_ids


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
    target = '/home/sthalham/data/prepro/lm_std_m3o3/'

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
    all = 10130
    times = []

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
            if int(redname) > 10000:
                continue

            gtfile = gtPath + '/' + fileInd
            depfile = depPath + redname + "_depth.exr"
            partfile = partPath + redname + "_part.png"

            # depth_refine, bboxes, poses, mask_ids = get_a_scene(gtfile, depfile, disfile)
            depth_refine, bboxes, poses, mask_ids = manipulate_depth(gtfile, depfile, partfile)

            if bboxes is None:
                excludedImgs.append(int(redname))
                continue

            depth_refine = np.multiply(depth_refine, 1000.0)  # to millimeters
            rows, cols = depth_refine.shape

            newredname = redname

            fileName = target + 'coco_train2014/' + newredname + '.jpg'

            myFile = Path(fileName)
            if myFile.exists():
                print('File exists, skip encoding and safing.')

            else:
                imgI, depth_refine = get_normal(depth_refine, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin, for_vis=False)
                #imgI = encodeImage(depth_refine)
                cv2.imwrite(fileName, imgI)

            imgID = int(newredname)
            imgName = newredname + '.jpg'

            # bb scaling because of image scaling
            bbsca = 640.0/720.0
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
