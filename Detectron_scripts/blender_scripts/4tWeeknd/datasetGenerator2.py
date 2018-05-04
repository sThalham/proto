import os,sys
import bpy
import numpy as np
from random import randint
from random import random
from random import gauss
import cv2
import yaml
import itertools
from math import radians,degrees,tan,cos
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
WIDTH 640
HEIGHT 480
VIEWPOINT 0 0 0 1 0 0 0
POINTS 307200
DATA ascii
'''

#Height : 480 for kinect, 512 for ensenso
#Points : 307200 for kinect, 327680 for ensenso

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


base_dir = "/home/sthalham/data/t-less_mani/t-less_scaled"
total_set = 40000 #10000 set of scenes, each set has identical objects with varied poses to anchor pose (+-15)
pair_set = 20 #number of pair scene for each set, 10
sample_dir = '/home/sthalham/data/t-less_mani/artificialScenes' #directory for temporary files (cam_L, cam_R, masks..~)
target_dir = '/home/sthalham/data/t-less_mani/artificialScenes/singlePatches'
index=0
isfile=True
while isfile:
    prefix='{:08}_'.format(index)
    if(os.path.exists(os.path.join(target_dir,prefix+'gt.yaml'))):
        index+=1
    else:
        isfile=False


#create dir if not exist
if not(os.path.exists(target_dir+"/disp")):
    os.makedirs(target_dir+"/disp")

if not(os.path.exists(target_dir+"/depth")):
    os.makedirs(target_dir+"/depth")

if not(os.path.exists(target_dir+"/mask")):
    os.makedirs(target_dir+"/mask")

if not(os.path.exists(target_dir+"/part")):
    os.makedirs(target_dir+"/part")


model_file=[]
model_solo=[]
for root, dirs, files in os.walk(base_dir):
    for file in sorted(files):
        if file.endswith(".stl"):
             temp_fn =os.path.join(root, file)
             model_file.append(temp_fn)
             model_solo.append(file)
             print(len(model_file),temp_fn)


for num_set in np.arange(total_set):
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

  drawAmo = list(range(1,2))
  freqAmo = np.bincount(drawAmo)
  AmoDraw = np.random.choice(np.arange(len(freqAmo)), 1, p=freqAmo / len(drawAmo), replace=False)
  drawObj = list(range(1,len(model_file)))
  freqObj = np.bincount(drawObj)
  ObjDraw = np.random.choice(np.arange(len(freqObj)), AmoDraw, p=freqObj / len(drawObj), replace=True)

  # num_object = np.asscalar(AmoDraw)
  num_object = 1
  print(num_object)
  object_label =[]
  anchor_pose = np.zeros((num_object,6)) #location x,y,z, euler x,y,z

  for i in np.arange(num_object):
      file_idx = randint(0,len(model_file)-1)
      file_model = model_file[file_idx]
      solo_model = model_solo[file_idx]
      imported_object = bpy.ops.import_mesh.stl(filepath=file_model, filter_glob="*.stl", files=[{"name":solo_model, "name":solo_model}], directory=root)
      object_label.append(file_idx)
      obj_object = bpy.context.selected_objects[0]
      obj_object.active_material = mat
      obj_object.pass_index = i+2
      anchor_pose[i,0] = random()*0.2-0.1
      anchor_pose[i,1] = random()*0.2-0.1
      anchor_pose[i,2] =0.4+0.2*float(i)
      anchor_pose[i,3] =radians(random()*360.0) #0-360 degree
      anchor_pose[i,4] =radians(random()*360.0)
      anchor_pose[i,5] =radians(random()*360.0)

	    #define anchor pos&rotation


	#Set object physics
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

  bpy.ops.rigidbody.object_settings_copy()

    #Define Object position&rotation
  for iii in np.arange(pair_set):

      scene.frame_set(0)
      for obj in scene.objects:
        if obj.type == 'MESH':
           obj_object= bpy.data.objects[obj.name]
           if obj_object.pass_index>1:
                 idx = obj_object.pass_index -2
                 obj_object.location.x=anchor_pose[idx,0]
                 obj_object.location.y=anchor_pose[idx,1]
                 obj_object.location.z=anchor_pose[idx,2]
                 obj_object.rotation_euler.x= radians(random()*360.0) #anchor_pose[idx,3] + radians(random()*30.0-15.0)
                 obj_object.rotation_euler.y= radians(random()*360.0) #anchor_pose[idx,4] + radians(random()*30.0-15.0)
                 obj_object.rotation_euler.z= radians(random()*360.0)
                 
                 # assign different color
                 rand_color = (random(), random(), random())
                 obj_object.active_material.diffuse_color = rand_color
                 
           if obj.name == 'InvisibleCube':
               obj_object.rotation_euler.x=radians(random()*70.0) #0~70
               obj_object.rotation_euler.y=radians(random()*60.0-30.0) #-30-30
               obj_object.rotation_euler.z=radians(random()*360.0) #0-360

        if obj.type == 'CAMERA' and  obj.name=='cam_L':
            obj_object = bpy.data.objects[obj.name]
            obj_object.location.z = 0.53  #1.0-2.5
            # obj_object.location.z = random()*0.5+0.5  #1.0-2.5

	#Run physics

      count = 60
      scene.frame_start = 1
      scene.frame_end = count + 1
      for f in range(1,scene.frame_end+1):
          scene.frame_set(f)
          if f <= 1:
               continue

      tree = bpy.context.scene.node_tree
      nodes = tree.nodes
	#When Rander cam_L, render mask together

      prefix='{:08}_'.format(index)
      index+=1
      if(index>10000):
         break

      maskfile = os.path.join(sample_dir , 'mask.png')
      depthfile = os.path.join(target_dir+'/depth', prefix+'depth.exr')
      #partfile= os.path.join(target_dir+"/part", prefix+'part.png')

      for ob in scene.objects:
          if ob.type == 'CAMERA':
              if ob.name=='cam_R':
                  bpy.context.scene.camera = ob
                  print('Set camera %s' % ob.name )
                  file_R = os.path.join(sample_dir , ob.name )
                  scene.render.filepath = file_R
                  bpy.ops.render.render( write_still=True )
                  
              if ob.name=='cam_L': #ob.name =='mask':
                  #Render IR image and Mask
                  bpy.context.scene.camera = ob
                  print('Set camera %s for IR' % ob.name )
                  file_L = os.path.join(sample_dir , ob.name )
                  auto_file = os.path.join(sample_dir, ob.name+'0061.png')
                  node= nodes['maskout']
                  node.file_slots[0].path = ob.name
                  node_mix = nodes['ColorRamp']
                  link_mask= tree.links.new(node_mix.outputs["Image"], node.inputs[0])
                  node.base_path=sample_dir                  
                  
                  auto_file_depth = os.path.join(sample_dir+'/temp/', ob.name+'0061.exr')
                  node= nodes['depthout']
                  node.file_slots[0].path = ob.name
                  node_mix = nodes['depth_lamp']
                  link_depth = tree.links.new(node_mix.outputs["Image"], node.inputs[0])
                  node.base_path=sample_dir+'/temp/'
                  
                  
                  #auto_file_part = os.path.join(sample_dir+'/temp/', ob.name+'0061.png')
                  #node= nodes['part_out']
                  #node.file_slots[0].path = ob.name
                  #node_mix = nodes['Render Layers']
                  #link_part = tree.links.new(node_mix.outputs["Diffuse Color"], node.inputs[0])
                  #node.base_path=sample_dir+'/temp/'
                  
                  scene.render.filepath = file_L
                  bpy.ops.render.render( write_still=True )
                  tree.links.remove(link_mask)
                  tree.links.remove(link_depth)
                  #tree.links.remove(link_part)
                  
                  print(auto_file)
                  print(auto_file_depth)
                  os.rename(auto_file, maskfile)
                  os.rename(auto_file_depth, depthfile)
                  #os.rename(auto_file_part, partfile)


      imgL = cv2.imread(file_L+".png")
      imgR = cv2.imread(file_R+".png")
      mask = cv2.imread(maskfile)

      dist =112
      window_size=7 #7
      min_disp =16
      num_disp = dist-min_disp
      fov = 57.8 #degree

	#stereo = cv2.StereoSGBM_create(numDisparities=160, blockSize=5) #21

      stereo = cv2.StereoSGBM_create(
	          minDisparity = min_disp,
      	    numDisparities = num_disp,
	          blockSize = window_size,
	          uniquenessRatio = 1,
	          speckleWindowSize = 100,
	          speckleRange = 5,
	          disp12MaxDiff = 1,
	          P1 = 8*window_size**2,
	          P2 = 32*window_size**2,
      	)

      disparity = stereo.compute(imgL,imgR).astype(np.float32)
      np.save(target_dir+"/disp/"+prefix+"disp.npy",disparity)

      minmax_vu = np.zeros((num_object+5,4),dtype=np.int) #min v, min u, max v, max u
      label_vu = np.zeros((mask.shape[0],mask.shape[1]),dtype=np.int8) #min v, min u, max v, max u
      colors = np.zeros((num_object+5,3),dtype=mask.dtype)

      n_label=0

      color_index=np.array([  [  0, 0,   0],
						[  0, 100,   0],
						[  0, 139,   0],
						[  0, 167,   0],
						[  0, 190,   0],
						[  0, 210,   0],
						[  0, 228,   0],
						[  0, 244,   0],
						[  0, 252,  50],
						[  0, 236, 112],
						[  0, 220, 147],
						[  0, 201, 173],
						[  0, 179, 196],
						[  0, 154, 215],
						[  0, 122, 232],
						[  0,  72, 248],
						[ 72,   0, 248],
						[122,   0, 232],
						[154,   0, 215],
						[179,   0, 196],
						[201,   0, 173],
						[220,   0, 147],
						[236,   0, 112],
						[252,   0,  50],
						[255,  87,  87],
						[255, 131, 131],
						[255, 161, 161],
						[255, 185, 185],
						[255, 206, 206],
						[255, 224, 224],
						[255, 240, 240],
						[255, 255, 255]])


      for v in np.arange(mask.shape[0]):
        for u in np.arange(mask.shape[1]):
           has_color = False
           if not(mask[v,u,0] ==0 and mask[v,u,1] ==0 and mask[v,u,2] ==0):
                  for ob_index in np.arange(n_label):
                     if colors[ob_index,0]== mask[v,u,0] and colors[ob_index,1]== mask[v,u,1] and colors[ob_index,2]== mask[v,u,2]:
                        has_color = True
                        minmax_vu[ob_index,0] = min(minmax_vu[ob_index,0], v)
                        minmax_vu[ob_index,1] = min(minmax_vu[ob_index,1], u)
                        minmax_vu[ob_index,2] = max(minmax_vu[ob_index,2], v)
                        minmax_vu[ob_index,3] = max(minmax_vu[ob_index,3], u)
                        label_vu[v,u]=ob_index+1
                        continue
                  if has_color ==False: #new label
                   colors[n_label] = mask[v,u]
                   label_vu[v,u]=n_label+1 #identical to object_index in blender
                   minmax_vu[n_label,0] = v
                   minmax_vu[n_label,1] = u
                   minmax_vu[n_label,2] = v
                   minmax_vu[n_label,3] = u
                   n_label=n_label+1
           else:
	       	     label_vu[v,u]=0


      bbox_refined = mask

      color_map=np.zeros(n_label)

      for k in np.arange(n_label)  :
            for i in np.arange(color_index.shape[0]):
                if(color_index[i,0] == colors[k,0] and color_index[i,1] == colors[k,1] and color_index[i,2] == colors[k,2] ):
                 color_map[k]=i
                 continue

      object_no=[]
      refined=[]

      for ob_index in np.arange(n_label): #np.arange(n_label):
        min_v=minmax_vu[ob_index,0]
        min_u=minmax_vu[ob_index,1]
        max_v=minmax_vu[ob_index,2]
        max_u=minmax_vu[ob_index,3]
        bbox = label_vu[min_v:max_v,min_u:max_u]
        bbox=bbox.reshape(-1)
        counts = np.bincount(bbox)
        print(colors[ob_index])
        if(counts.shape[0]>1):
           if(np.argmax(counts[1:]) ==(ob_index)): #(mask.shape[0],mask.shape[1]
               if(min_v>30 and min_u>30 and max_v < (mask.shape[0]-30) and max_u < (mask.shape[1]-30) ):
    	         #cv2.rectangle(bbox_refined,(min_u,min_v),(max_u,max_v),(0,255,0),1)
                 refined.append(ob_index)
                 object_no.append(color_map[ob_index])
    		     #print(color_map[ob_index])

	#cv2.imwrite(os.path.join(target_dir,prefix+'bbox_refined.png'),bbox_refined)

      bbox_refined = minmax_vu[refined]
      poses =np.zeros((len(object_no),4,4),dtype=np.float)
      camera_rot =np.zeros((4,4),dtype=np.float)
      for obj in scene.objects:
        if obj.type == 'MESH':
          if obj.pass_index in object_no:
            idx = object_no.index(obj.pass_index)
            poses[idx]=obj.matrix_world
          if obj.name=='InvisibleCube':
            camera_rot[:,:] = obj.matrix_world
            camera_rot = camera_rot[:3,:3] #only rotation (z was recorded seprately)
            init_rot = np.zeros((3,3))
            init_rot[0,0]=1
            init_rot[1,1]=-1
            init_rot[2,2]=-1
            fin_rot =np.matmul(camera_rot,init_rot)
            fin_rot = inv(fin_rot)
            world_rot=np.zeros((4,4))
            world_rot[:3,:3] = fin_rot
            world_rot[3,3]=1

            #camera_rot=camera_rot.reshape(-1)
        #camera_ext =np.zeros((len(object_no),4,4),dtype=np.float)
        if obj.type == 'CAMERA' and  obj.name=='cam_L':
            obj_object = bpy.data.objects[obj.name]
            camera_z = obj_object.location.z
            #camera_ext[:,:] = obj_object.matrix_world
            #camera_ext = camera_ext.reshape(-1)

      np.save(target_dir+"/mask/"+prefix+"mask.npy",label_vu)
      cam_trans = -np.matmul(camera_rot,np.array([0,0,camera_z]))
      world_trans =np.zeros((4,4))
      world_trans[0,0]=1
      world_trans[1,1]=1
      world_trans[2,2]=1
      world_trans[3,3]=1
      world_trans[:3,3] = cam_trans

      #camera_tran = np.matmul(camera_rot,np.array([0,0,camera_z]))
      #camera_rot = inv(camera_rot)

      with open(os.path.join(target_dir,prefix+'gt.yaml'),'w') as f:
        for i in np.arange(len(object_no)):
          pose = poses[i]
          pose = np.matmul(world_trans,pose)
          pose = np.matmul(world_rot,pose)
          pose_list=pose.reshape(-1)
          id = int(object_label[int(object_no[i]-2)])
          mask_id = int(refined[i]+1)
          gt={int(i):{'bbox':bbox_refined[i].tolist(),'class_id':id,'mask_id':mask_id,'pose':pose_list.tolist()}} #,'camera_z':camera_z,'camera_rot':camera_rot.tolist()
          yaml.dump(gt,f)
