# author: Stefan Thalhammer
# date: 28.5.2018

import string, sys
import numpy as np
import segment_normals as seg
import cv2


class HHAEncoding:

    def __init__(self, depth, fx, fy, cx, cy, depCut=1500.0):

        self.depth = depth
        self.resX = depth.shape[0]
        self.resY = depth.shape[1]
        self.focalLx = fx
        self.focalLy = fy
        self.centerX = cx
        self.centerY = cy
        self.depthCut = depCut

    def encodeImage(self):

        img = np.zeros((self.resY, self.resX, 3), dtype=np.uint8)

        normImg, depImg = self.get_normal(self.depth, for_vis=True)
        img[:, :, 0] = self.compute_disparity(depImg)
        img[:, :, 1] = self.encode_area(depImg)
        img[:, :, 2] = self.compute_angle2gravity(normImg, depImg)

        return img

    def create_point_cloud(self, depth):

        zP = depth.reshape(self.resY * self.resX)
        x, y = np.meshgrid(np.arange(0, self.resX, 1), np.arange(0, self.resY, 1), indexing='xy')
        yP = y.reshape(self.resX * self.resY) - self.centerY
        xP = x.reshape(self.resX * self.resY) - self.centerX
        yP = np.multiply(yP, zP)
        xP = np.multiply(xP, zP)
        yP = np.divide(yP, self.focalLy)
        xP = np.divide(xP, self.focalLx)
        cloud_final = np.transpose(np.array((xP, yP, zP)))

        return cloud_final

    def get_normal(self, depth_refine, for_vis=True):

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

        constant = 1 / self.focalLx
        uv_table = np.zeros((self.resY, slef.resX, 2), dtype=np.int16)
        column = np.arange(0, self.resY)

        uv_table[:, :, 1] = np.arange(0, self.resX) - self.centerX  # x-c_x (u)
        uv_table[:, :, 0] = column[:, np.newaxis] - self.centerY  # y-c_y (v)
        uv_table_sign = np.copy(uv_table)

        depth_refine = ndimage.gaussian_filter(depth_refine, 2)  # sigma=3)

        v_x = np.zeros((self.resY, self.resX, 3))
        v_y = np.zeros((self.resY, self.resX, 3))
        normals = np.zeros((self.resY, self.resX, 3))

        dig = np.gradient(depth_refine, 3, edge_order=2)
        v_y[:, :, 0] = uv_table_sign[:, :, 1] * constant * dig[0]
        v_y[:, :, 1] = depth_refine * constant + (uv_table_sign[:, :, 0] * constant) * dig[0]
        v_y[:, :, 2] = dig[0]

        v_x[:, :, 0] = depth_refine * constant + uv_table_sign[:, :, 1] * constant * dig[1]
        v_x[:, :, 1] = uv_table_sign[:, :, 0] * constant * dig[1]
        v_x[:, :, 2] = dig[1]

        cross = np.cross(v_x.reshape(-1, 3), v_y.reshape(-1, 3))
        norm = np.expand_dims(np.linalg.norm(cross, axis=1), axis=1)

        cross = cross / norm
        cross = cross.reshape(self.resY, self.resX, 3)
        cross = np.abs(cross)
        cross = np.nan_to_num(cross)

        cam_angle = np.arccos(cross[:, :, 2])
        # cross[np.abs(cam_angle) > math.radians(75)] = 0  # high normal cut
        # cross[depth_refine <= 100] = 0  # 0 and near range cut
        cross[depth_refine > self.depthCut] = np.NaN  # far range cut
        if not for_vis:
            scaDep = 1.0 / np.nanmax(depth_refine)
            depth_norm = np.multiply(depth_refine, scaDep)
            cross[:, :, 0] = cross[:, :, 0] * (1 - (depth_norm - 0.5))  # nearer has higher intensity
            cross[:, :, 1] = cross[:, :, 1] * (1 - (depth_norm - 0.5))
            cross[:, :, 2] = cross[:, :, 2] * (1 - (depth_norm - 0.5))

        return cross, depth_refine

    def encode_area(self, depth, k=5, p_thresh=0.35, area_ref=2500):
        # pass depth in mm

        areaCol = np.ones((self.resY, self.resX), dtype=np.uint8)
        offset = int((k - 1) * 0.5)

        pcA = create_point_cloud(depth, self.focalLx, focalLy, kin_res_x * 0.5, kin_res_y * 0.5, 1.0)

        label = 0
        E = seg.UnionFind()  # equivalence class object to use with seq labeling algorithm

        I = seg.Image(pcA, self.resX, self.resY)

        # SEQUENTIAL LABELING ALGORITHM
        for row in range(offset, I.rows - offset):
            for col in range(offset, I.cols - offset):
                if depth[row, col] > depthCut:
                    continue

                P = I.get_kxk_neighborhood(row, col, k)

                if seg.are_locally_coplanar(P, p_thresh):

                    I.type[row, col] = 'planar'  # it's locally coplanar with its k-neighborhood, so it's a planar point

                    N = I.eclass_label[row - 1, col]
                    W = I.eclass_label[row, col - 1]
                    NW = I.eclass_label[row - 1, col - 1]

                    if NW != -1:
                        I.eclass_label[row, col] = NW
                    elif N != -1 and W != -1:
                        I.eclass_label[row, col] = N
                        E.add(N, W)

                    elif N != -1:
                        I.eclass_label[row, col] = N

                    elif W != -1:
                        I.eclass_label[row, col] = W
                    else:
                        label += 1
                        I.eclass_label[row, col] = label
                        E.make_new(label)
                else:
                    I.is_boundary[row, col] = True
                    I.type[row, col] = 'nonplanar'

        classCounter = []
        clusterInd = np.ones((rows, cols), dtype=np.uint32) * -1
        for row in range(0, I.rows):
            for col in range(0, I.cols):
                if I.coords[row, col].any() and I.eclass_label[row, col] != -1:
                    eclass = E.leader[I.eclass_label[row, col]]
                    classCounter.append(eclass)
                    clusterInd[row, col] = eclass

        clusters, surf = np.unique(clusterInd, return_counts=True)

        for i, cl in enumerate(clusters):

            if cl == -1:
                mask = np.where(clusterInd == cl, True, False)
                val = int(255)
                areaCol = np.where(mask, val, areaCol)
            elif surf[i] > area_ref:
                continue
            else:
                mask = np.where(clusterInd == cl, True, False)
                val = 255 - ((surf[i] / area_ref) * 255.0)
                val = val.astype(dtype=np.uint8)
                areaCol = np.where(mask, val, areaCol)
        areaCol = np.where(depth > self.depthCut, 0, areaCol)

        return areaCol

    def compute_disparity(self, depth):

        # calculate disparity
        depthFloor = 100.0
        depthCeil = self.depthCut

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
        disp_final = np.where(mask, disp_final, 0)

        return disp_final

    def compute_angle2gravity(self, normals, depth):

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
# end
