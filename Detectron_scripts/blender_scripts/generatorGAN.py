import os,sys
import bpy
import numpy as np
from random import randint
from random import random
from random import gauss
from random import uniform
import cv2
import yaml
import itertools
from math import radians,degrees,tan,cos, sin, atan, pi
from numpy.linalg import inv


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

pcd_header = '''VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 400
HEIGHT 400
VIEWPOINT 0 0 0 1 0 0 0
POINTS 307200
DATA ascii
'''

#Height : 480 for kinect, 512 for ensenso
#Points : 307200 for kinect, 327680 for ensenso
# tless: 720x540

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def write_pcd(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 1)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write(pcd_header.encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %1.7e ')

def on_projector(on):
            ir_projection = bpy.data.objects["Spot"]
            emission = bpy.data.objects["Plane.Emission"]
            #on=True -> spot lamp on

            #spot lamp off
            #emission on
            bpy.ops.object.select_all(action='DESELECT')

            ir_projection.select = True
            emission.select = False

            for sel in bpy.context.selected_objects:
                sel.hide_render = not(on)

            ir_projection.select = False
            emission.select = True

            for sel in bpy.context.selected_objects:
                sel.hide_render = on

def getVisibleBoundingBox(objectPassIndex):

    S = bpy.context.scene
    width  = int( S.render.resolution_x * S.render.resolution_percentage / 100 )
    height = int( S.render.resolution_y * S.render.resolution_percentage / 100 )
    depth  = 4

    pixels = np.array( bpy.data.images['Render Result'].pixels[:] ).reshape( [height, width, depth] )
    # Keep only one value for each pixel (white pixels have 1 in all RGBA channels anyway), thus converting the image to black and white
    pixels = np.array( [ [ pixel[0] for pixel in row ] for row in pixels ] )

    bbox = np.argwhere( pixels == objectPassIndex )
    (ystart, xstart), (ystop, xstop) = bbox.min(0), bbox.max(0) + 1
    bb = (xstart, xstart, height - ystart, height - ystop)
    return bb, bbox


# 23.4.2018
# render image 350 of object 23
# cam_K: [1076.74064739, 0.0, 215.98264967, 0.0, 1075.17825536, 204.59181836, 0.0, 0.0, 1.0]
# depth_scale: 0.1
# elev: 45
# mode: 0

# cam_R_m2c: [0.62268218, -0.78164004, -0.03612308, -0.56354950, -0.41595975, -0.71371609, 0.54284357, 0.46477486, -0.69950372]
# cam_t_m2c: [-9.10674129, -2.47862668, 634.81667094]
# obj_bb: [120, 121, 197, 190]
# obj_id: 23

# f = 580
# b = 0.0075

model_dir = "/home/sthalham/data/t-less_v2/t-less_scaled(ST)"
train_dir = "/home/sthalham/data/t-less_v2/train_kinect"

sample_dir = '/home/sthalham/data/renderings/GAN_training' #directory for temporary files (cam_L, cam_R, masks..~)
target_dir = '/home/sthalham/data/renderings/GAN_training/patches'
index=0
isfile=True
while isfile:
    prefix='{:08}_'.format(index)
    if(os.path.exists(os.path.join(target_dir,prefix+'gt.yaml'))):
        index+=1
    else:
        isfile=False


#create dir if not exist
#if not(os.path.exists(target_dir+"/disp")):
#    os.makedirs(target_dir+"/disp")

if not(os.path.exists(target_dir+"/depth")):
    os.makedirs(target_dir+"/depth")

if not(os.path.exists(target_dir+"/mask")):
    os.makedirs(target_dir+"/mask")

if not(os.path.exists(target_dir+"/part")):
    os.makedirs(target_dir+"/part")


sub = os.listdir(train_dir)

for s in sub:
        rgbPath = train_dir + '/' + s + "/rgb/"
        depPath = train_dir + '/' + s + "/depth/"
        gtPath = train_dir + '/' + s + "/gt.yml"
        infoPath = train_dir + '/' + s + "/info.yml"

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
            fykin = K[4]
            cxx = K[2]
            cyy = K[5]
            
            annot = gtYML[int(ss)]
            annot = annot[0]
            cam_R = annot['cam_R_m2c']
            cam_T = annot['cam_t_m2c']
            obj_id = annot['obj_id']

            #########################
            # Prepare the stuff
            #########################
            
            
            bpy.ops.object.select_all(action='DESELECT')
            scene = bpy.context.scene
            scene.objects.active = bpy.data.objects["template"]
            for obj in scene.objects:
                if obj.type == 'MESH':
                    if obj.name == 'template':
                        obj.select = False          
                    elif obj.name[0:5] == 'Plane':
                        obj.select = False
                    elif obj.name == 'Plane':
                        obj.select = False
                    elif obj.name == 'InvisibleCube':
                        obj.select = False
                    else:
                        obj.select = True

            bpy.ops.object.delete()
            bpy.ops.object.select_all(action='DESELECT')
            obj_object = bpy.data.objects["template"]
            obj_object.pass_index = 1
            mat = obj_object.active_material
            
            file_model = model_dir + '/obj_' + s + '.stl'
            print(file_model)
            solo_model = 'obj_' + s + '.stl'
            imported_object = bpy.ops.import_mesh.stl(filepath=file_model, filter_glob="*.stl", files=[{"name":solo_model, "name":solo_model}], directory=model_dir)

            obj_object = bpy.context.selected_objects[0]
            obj_object.active_material = mat
            obj_object.pass_index = 0 +2 # don't add?
            offX = obj_object.dimensions[0] * 0.5
            offY = obj_object.dimensions[1] * 0.5
            offZ = obj_object.dimensions[2] * 0.5
            
            scene.frame_set(0)
            for obj in scene.objects:
                if obj.type == 'MESH':
                    obj_object= bpy.data.objects[obj.name]
            
                if obj_object.pass_index>1:
                    
                    obj_object.location.x = 0.0
                    obj_object.location.y = 0.0
                    obj_object.location.z = 0.0
                 
                    # assign different color
                    rand_color = (random(), random(), random())
                    obj_object.active_material.diffuse_color = rand_color
                    if obj_object.pass_index > (1):
                        obj_object.pass_index = 0
                 
                if obj.name == 'InvisibleCube':
                    #obj_object.rotation_euler.x = 0.0 
                    #obj_object.rotation_euler.y = 0.0
                    #obj_object.rotation_euler.z = 0.0
                    obj_object.matrix_world[0][0:3] = cam_R[0:3]
                    obj_object.matrix_world[1][0:3] = cam_R[3:6]
                    obj_object.matrix_world[2][0:3] = cam_R[6:9]
                    #obj_object.matrix_world = cam_R

                if obj.type == 'CAMERA' and  obj.name=='cam_L':
                    obj_object = bpy.data.objects[obj.name]
                    obj_object.location.x= - cam_T[0] * 0.001
                    obj_object.location.y= - cam_T[1] * 0.001
                    obj_object.location.z = cam_T[2] * 0.001
                    
                if obj.name =='Plane.Ground':
                    if bpy.data.objects['cam_L'].matrix_world[2][3] < 0.0:
                        bpy.data.objects['Plane.Ground'].location.z = - offZ - (( bpy.data.objects['Plane.Ground'].dimensions[2] * bpy.data.objects['Plane.Ground'].scale[2])*0.5)
                    else:
                        bpy.data.objects['Plane.Ground'].location.z = offZ + (( bpy.data.objects['Plane.Ground'].dimensions[2] * bpy.data.objects['Plane.Ground'].scale[2])*0.5) 

                if obj.name =='Plane.Room.001':
                    gegk = bpy.data.objects['cam_L'].matrix_world[1][3]
                    anka = bpy.data.objects['cam_L'].matrix_world[0][3]
                    ang = (gegk/anka)
                    ang = atan(ang)
                    
                    if offX > offY:
                        obShift = offX * 0.5
                    else:
                        obShift = offY * 0.5
                        
                    obj_object.location.x = (- obShift * sin(ang))
                    obj_object.location.y = (- obShift * cos(ang))
                    obj_object.rotation_euler.z= ang * (180/pi)
                    
                        

            tree = bpy.context.scene.node_tree
            nodes = tree.nodes

            #prefix='{:08}_'.format(index)
            prefix = s + ss

            maskfile = os.path.join(target_dir+'/mask' , 'mask.png')
            depthfile = os.path.join(target_dir+'/depth', prefix+'depth.exr')
            partfile= os.path.join(target_dir+"/part", prefix+'part.png')

            for ob in scene.objects:
                if ob.type == 'CAMERA':          
                    if ob.name=='cam_L': #ob.name =='mask':
                        #Render IR image and Mask
                        bpy.context.scene.camera = ob
                        print('Set camera %s for IR' % ob.name )
                        file_L = os.path.join(sample_dir , ob.name )
                        auto_file = os.path.join(sample_dir, ob.name+'0000.png')
                        node= nodes['maskout']
                        node.file_slots[0].path = ob.name
                        node_mix = nodes['ColorRamp']
                        link_mask= tree.links.new(node_mix.outputs["Image"], node.inputs[0])
                        node.base_path=sample_dir                  
                  
                        auto_file_depth = os.path.join(sample_dir+'/temp/', ob.name+'0000.exr')
                        node= nodes['depthout']
                        node.file_slots[0].path = ob.name
                        node_mix = nodes['Render Layers']
                        link_depth = tree.links.new(node_mix.outputs["Z"], node.inputs[0])
                        node.base_path=sample_dir+'/temp/'
                    
                  
                        auto_file_part = os.path.join(sample_dir+'/temp/', ob.name+'0000.png')
                        node= nodes['rgbout']
                        node.file_slots[0].path = ob.name
                        node_mix = nodes['Render Layers']
                        link_part = tree.links.new(node_mix.outputs["Diffuse Color"], node.inputs[0])
                        link_part = tree.links.new(node_mix.outputs["Image"], node.inputs[0])
                        node.base_path=sample_dir+'/temp/'
                  
                        scene.render.filepath = file_L
                        bpy.ops.render.render( write_still=True )
                        tree.links.remove(link_mask)
                        tree.links.remove(link_depth)
                        tree.links.remove(link_part)
                  
                        os.rename(auto_file, maskfile)
                        os.rename(auto_file_depth, depthfile)
                        os.rename(auto_file_part, partfile)
                        
        break

 