# Testing homoraphy calculation and related topics
from numpy import array, pi, arctan
from numpy.linalg import inv
from utils import *

# parameters
pix_x = 1600.0      # number horizontal pixels
pix_y = 1200.0      # number vertical pixels
f = 5.9             # approximate focal length (mm)
ox = pix_x/2.0      # camera image frame origin offset in x
oy = pix_y/2.0      # camera image frame origin offset in y
sx = 450.0          # scale factor in x (e.g. pixels/mm)
sy = sx;            # scale factor in y (e.g. pixels/mm)
K = array([[ f*sx,  0.0,  ox],  # intrinsic camera matrix
           [  0.0, f*sy,  oy],  
           [  0.0,  0.0, 1.0]])

# inertial point locations (NED)
pts_I = array([[  0.0,  0.0,  0.0,  0.0 ],
               [  1.0,  1.0, -1.0, -1.0 ],
               [  1.0, -1.0, -1.0,  1.0 ]])

# camera positions (NED)
cam1 =        array([-10.0, 0.0, 0.0])
cam2 = cam1 + array([  0.0, 0.5, 0.0])
cam3 = cam2 + array([  0.0, 0.5, 0.0])

# camera attitudes (3-2-1 Euler)
# force each camera to look at the origin with some roll added
roll1 = 0.0
pitch1 = 0.0
yaw1 = 0.0

roll2 = 0.1
pitch2 = arctan(cam2[2]/cam2[0])
yaw2 = -arctan(cam2[1]/cam2[0])

roll3 = 0.2
pitch3 = arctan(cam3[2]/cam3[0])
yaw3 = -arctan(cam3[1]/cam3[0])

R1 = getRi2c(roll1, pitch1, yaw1)
R2 = getRi2c(roll2, pitch2, yaw2)
R3 = getRi2c(roll3, pitch3, yaw3)

# project points into each camera
pts1 = project2image(pts_I, cam1, R1, K)
pts2 = project2image(pts_I, cam2, R2, K)
pts3 = project2image(pts_I, cam3, R3, K)

# compute homography matrix
H12 = homographyFromPoints(pts1, pts2)
H23 = homographyFromPoints(pts2, pts3)
H13 = homographyFromPoints(pts1, pts3)
H23p = H13.dot(inv(H12))
H23p /= det(H23p)**(1./3.)

print("H23 = ")
print(H23)
print("H23p = ")
print(H23p)
print("pts3 = ")
print(pts3)
print("pts3p = ")
print(warpPoints(pts2, H23p))

# # plot 3 sets of points
# figure(1); clf; hold on; grid on;
# plot(ptsA(1,:),ptsA(2,:),'b*')
# plot(ptsB(1,:),ptsB(2,:),'ro')
# plot(ptsA_H(1,:),ptsA_H(2,:),'m+')
# plot(ptsB_H(1,:),ptsB_H(2,:),'cx')
# axis ij
# xlim([0 1600])
# ylim([0 1200])
# legend('current','previous','projected prev2curr','projected curr2prev','location','northeast')
# title('Points Projected into Both Cameras')
