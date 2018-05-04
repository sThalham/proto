import sys
import os
import yaml
import cv2
import numpy as np
import json
from scipy import ndimage, signal
import math
import datetime
import copy
import transforms3d as tf3d
import itertools

import OpenEXR, Imath

#kin_res_x = 640
#kin_res_y = 480
kin_res_x = 720
kin_res_y = 540
fov = 57.8
focalLx = 579.68  # blender calculated
focalLy = 542.31  # blender calculated

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
    partmask = partmask.astype(np.float32)
    ###################################################
    # DON'T REMOVE, best normal map up to now !!! tested without lateral noise !!!
    kernel = np.ones((7, 7))
    partmask = signal.medfilt2d(partmask, kernel_size=7)
    partmask = cv2.morphologyEx(partmask, cv2.MORPH_CLOSE, kernel)
    ###################################################
    partmask = partmask.astype(np.uint8)

    cv2.imwrite('/home/sthalham/mask.png', partmask)

    mask = partmask > 20
    depth = np.where(mask, depth, 0.0)

    # round to depth resolution
    res = (((depth / 1000) * 1.41421356) ** 2) * 1000

    # discretize to resolution and apply gaussian
    # discretize to resolution steps
    dNonVar = np.divide(depth, res, out=np.zeros_like(depth), where=res!=0)
    dNonVar = np.round(dNonVar)
    dNonVar = np.multiply(dNonVar, res)

    # determine elementwise noise
    '''
    according to Khoshelham et al.
    std-dev(mm) calib:   x = y = 10, z = 18
    std-dev(mm) uncalib: x = 14, y = 15, z = 18
    '''
    noise = np.multiply(dNonVar, 0.018)
    dMan = np.random.normal(loc=dNonVar, scale=noise, size=dNonVar.shape)

    # nguyen et al. sig_lat ~ 0.85 pixel
    noisex = np.zeros((depth.shape), dtype=np.float32)
    noisey = np.zeros((depth.shape), dtype=np.float32)
    noisex = np.random.normal(loc=noisex, scale=1.0)
    noisey = np.random.normal(loc=noisey, scale=1.0)
    noisex = np.round(noisex)
    noisey = np.round(noisey)

    # apply lateral noise by sampling which pixels to choose
    # vectorize?
    depthFinal = np.zeros((depth.shape), dtype=np.float32)
    for (y, x), pixel in np.ndenumerate(depth):
        indX = x+noisex[y, x]
        indY = y+noisey[y, x]
        indX = indX.astype(int)
        indY = indY.astype(int)
        if indX < 0:
            indX = 0
        if indX > (kin_res_x - 1):
            indX = (kin_res_x - 1)
        if indY < 0:
            indY = 0
        if indY > (kin_res_y - 1):
            indY = (kin_res_y - 1)

        depthFinal[y, x] = dMan[indY, indX]

    '''
    # inflate missing pixel values
    dep0 = np.where(depthFinal < 0.01)
    dep0 = np.asarray(dep0)
    xy, multi = dep0.shape
    rows, cols = depth.shape
    for i in range(0, multi):
        y = dep0[0, i]
        x = dep0[1, i]
        if y < 1 or x < 1:
            continue
        if y > (rows-1) or x > (cols-1):
            continue
        depthFinal[(y-1):(y+1), (x-1):(x+1)] = 0.0
    '''

    # depthFinal = cv2.GaussianBlur(depthFinal, (9, 9), 2.0, 2.0)  # only god knows why

    return depthFinal, bboxes, poses, mask_ids


def get_normal(depth_refine, fx=-1.0, fy=-1.0, cx=-1, cy=-1, for_vis=True):
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
    #cross[np.abs(cam_angle) > math.radians(75)] = 0  # high normal cut
    #cross[depth_refine <= 100] = 0  # 0 and near range cut
    cross[depth_refine > 1500] = 0  # far range cut
    if not for_vis:
        cross[:, :, 0] = cross[:, :, 0] * (1 - (depth_refine - 0.5))  # nearer has higher intensity
        cross[:, :, 1] = cross[:, :, 1] * (1 - (depth_refine - 0.5))
        cross[:, :, 2] = cross[:, :, 2] * (1 - (depth_refine - 0.5))

    return cross


##########################
#         MAIN           #
##########################
if __name__ == "__main__":

    root = '/home/sthalham/data/t-less_mani/artificialScenes/renderedScenes02052018/patches'  # path to train samples
    model = '/home/sthalham/workspace/python/src/model.yml.gz'

    compare = cv2.imread('/home/sthalham/data/t-less_v2/test_kinect/19/depth/0412.png', -1)
    compare1 = cv2.imread('/home/sthalham/data/t-less_v2/test_kinect/09/depth/0117.png', -1)
    compare2 = cv2.imread('/home/sthalham/data/t-less_v2/test_kinect/13/depth/0079.png', -1)
    compare3 = cv2.imread('/home/sthalham/data/t-less_v2/test_kinect/04/depth/0291.png', -1)
    compare4 = cv2.imread('/home/sthalham/data/t-less_v2/test_kinect/16/depth/0205.png', -1)

    rowcom, colcom = compare.shape
    compare = np.multiply(compare, 0.1)
    compare1 = np.multiply(compare1, 0.1)
    compare2 = np.multiply(compare2, 0.1)
    compare3 = np.multiply(compare3, 0.1)
    compare4 = np.multiply(compare4, 0.1)

    fkinx = 1076.74
    fkiny = 1075.18
    normImg = get_normal(compare, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    normImg = np.multiply(normImg, 255.0)
    imgI = normImg.astype(np.uint8)
    cv2.imwrite('/home/sthalham/normkin_19_0412.png', imgI)

    normImg = get_normal(compare1, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    normImg = np.multiply(normImg, 255.0)
    imgI = normImg.astype(np.uint8)
    cv2.imwrite('/home/sthalham/normkin_9_0117.png', imgI)

    normImg = get_normal(compare2, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    normImg = np.multiply(normImg, 255.0)
    imgI = normImg.astype(np.uint8)
    cv2.imwrite('/home/sthalham/normkin_13_0079.png', imgI)

    normImg = get_normal(compare3, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    normImg = np.multiply(normImg, 255.0)
    imgI = normImg.astype(np.uint8)
    cv2.imwrite('/home/sthalham/normkin_4_0291.png', imgI)

    normImg = get_normal(compare4, fx=fkinx, fy=fkiny, cx=(colcom * 0.5), cy=(rowcom * 0.5), for_vis=True)
    normImg = np.multiply(normImg, 255.0)
    imgI = normImg.astype(np.uint8)
    cv2.imwrite('/home/sthalham/normkin_16_0205.png', imgI)

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
    partPath = root + "/part/"
    gtPath = root
    maskPath = root + "/mask/"

    for fileInd in os.listdir(root):
        if fileInd.endswith(".yaml"):
            redname = fileInd[:-8]
            gtfile = gtPath + '/' + fileInd
            depfile = depPath + redname + "_depth.exr"
            partfile = partPath + redname + "_part.png"

            depth_refine, bboxes, poses, mask_ids = manipulate_depth(gtfile, depfile, partfile)
            depth_refine = np.multiply(depth_refine, 1000.0)  # to millimeters

            #sca = 255.0 / np.amax(depth_refine)
            #depth_scaled = np.multiply(depth_refine, sca)
            #depth_int8 = depth_scaled.astype(np.uint8)
            #cv2.imwrite('/home/sthalham/depth_refine.png', depth_int8)

            rows, cols = depth_refine.shape

            '''
            depthcanny = cv2.Canny(depth_int8, 100, 200)
            cv2.imwrite('/home/sthalham/cannydepth.png', depthcanny)
            '''
            depth_blur = cv2.GaussianBlur(depth_refine, (9, 9), 2.5, 2.5)

            focalL = 580        # according to microsoft
            focalLx = 579.68    # blender calculated
            focalLy = 542.31    # blender calculated
            normImg = get_normal(depth_blur, fx=focalLx, fy=focalLy, cx=(kin_res_x * 0.5), cy=(kin_res_y * 0.5), for_vis=True)
            normImg = np.multiply(normImg, 255.0)
            imgI = normImg.astype(np.uint8)
            cv2.imwrite('/home/sthalham/normarti992525.png', imgI)

            depth_blur = cv2.GaussianBlur(depth_refine, (9, 9), 2, 2)

            normImg = get_normal(depth_blur, fx=focalLx, fy=focalLy, cx=(kin_res_x * 0.5), cy=(kin_res_y * 0.5),
                                 for_vis=True)
            normImg = np.multiply(normImg, 255.0)
            imgI = normImg.astype(np.uint8)
            cv2.imwrite('/home/sthalham/normarti.png', imgI)



            '''
            edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)

            depth_im = cv2.cvtColor(imgI, cv2.COLOR_BGR2RGB)
            #rgb_im = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)

            #edges = edge_detection.detectEdges(np.float32(depth_im) / 255.0)
            edgesdep = edge_detection.detectEdges(np.float32(depth_im) / 255.0)
            orimapdep = edge_detection.computeOrientation(edgesdep)
            edgesdep = edge_detection.edgesNms(edgesdep, orimapdep)
            sca = 255.0 / np.amax(edgesdep)
            edgesdep = np.multiply(edgesdep, sca)
            edgesdep = edgesdep.astype(np.uint8)
            cv2.imwrite("/home/sthalham/strucedgedepth.png", edgesdep)

            
            edgesrgb = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
            orimaprgb = edge_detection.computeOrientation(edgesrgb)
            edgesrgb = edge_detection.edgesNms(edgesrgb, orimaprgb)

            edge_boxes = cv2.ximgproc.createEdgeBoxes()
            edge_boxes.setMaxBoxes(30)
            boxes = edge_boxes.getBoundingBoxes(edges, orimap)

            for b in boxes:
                x, y, w, h = b
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1, cv.LINE_AA)

            sca = 255.0 / np.amax(edgesrgb)
            edgesrgb = np.multiply(edgesrgb, sca)
            edgesrgb = edgesrgb.astype(np.uint8)
            cv2.imwrite("/home/sthalham/strucedgecompare.png", edgesrgb)

            
            #cv2.imshow("edgeboxes", im)
            '''

            focalL = 1008.80026817 # blender focal length; just some line to place breakpoint
            # similar to t-less' focal length
            # focalL = 580.0  # according to microsoft
            # normImg = get_normal(depth_refine, fx=focalL, cx=(kin_res_x * 0.5), cy=(kin_res_y * 0.5), for_vis=True)