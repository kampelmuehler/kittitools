"""
    Plot velodyne points on image of select camera
    
    this is implemented borrowing from the matlab devkit provided
    for KITTI
"""

import pykitti # pip install pykitti (v0.3.0)
import numpy as np
import matplotlib.pyplot as plt 
from os.path import expanduser

__author__ = "Moritz Kampelmuehler"

# file selection
basedir = expanduser('~/KITTI') # KITTI dataset directory
date = '2011_09_26'
drive = '0028'
frame = 176
camera = 2 # must be in range(4)

# load raw data
data = pykitti.raw(basedir, date, drive, frames=[frame], imtype='jpg')
camera_imgs = {0:data.cam0, 1:data.cam1, 2:data.cam2, 3:data.cam3}
P_rect = {0:data.calib.P_rect_00, 1:data.calib.P_rect_10, 2:data.calib.P_rect_20, 3:data.calib.P_rect_30}

cam_img = next(camera_imgs[camera])
height = cam_img.height
width = cam_img.width

# calculate transformation matrix for velodyne to camera
R_cam_to_rect = data.calib.R_rect_00
Tr_velo_to_cam = data.calib.T_cam0_velo_unrect
P_velo_to_img = P_rect[camera].dot(R_cam_to_rect.dot(Tr_velo_to_cam))

# load velodyne points
pts3d_raw = next(data.velo)
pts3d = pts3d_raw[pts3d_raw[:, 0] >= 1 ,:] # only points in front of vehicle (approx.)
pts3d[:,3] = 1

# project points
dim_norm = P_velo_to_img.shape[0]
dim_proj = P_velo_to_img.shape[1]
p2_in = np.copy(pts3d)

p2_out = (P_velo_to_img.dot(p2_in.T)).T

# normalize homogeneous coordinates:
p_out = p2_out[:,:2]/np.tile(p2_out[:,2].reshape(p2_out.shape[0],1),2)

# add depth to array
p_out = np.append(p_out, pts3d[:,0].reshape((-1,1)), axis=1)

# map to regular grid of size of image (height, width, depth)
p_out[:,0] = np.floor(p_out[:,0])
p_out[:,1] = np.floor(p_out[:,1])
p_out = p_out[np.where(np.logical_and(p_out[:,0] >= 0, p_out[:,0] < width))]
p_out = p_out[np.where(np.logical_and(p_out[:,1] >= 0, p_out[:,1] < height))]
out_quantized = np.empty((width, height, 1))
out_quantized.fill(np.nan) # set nan where no data is available
for i in range(p_out.shape[0]):
    out_quantized[p_out[i,0].astype(int),p_out[i,1].astype(int)] = p_out[i,2]
print(p_out[np.argmax(p_out[:,2])])

# plot whole quantized grid over image
plt.imshow(np.array(cam_img.getdata()).reshape(height, width, 3))
plt.xlim(0, width)
plt.ylim(0, height)
plt.gca().invert_yaxis()
ind = np.indices(out_quantized.shape)
print(ind[0].shape)
plt.scatter(ind[0], ind[1], c=out_quantized[ind[0],ind[1],:].reshape(ind[0].shape),
            marker='o', s=5, alpha=0.7, cmap='plasma') # adjust params as needed
plt.show()
