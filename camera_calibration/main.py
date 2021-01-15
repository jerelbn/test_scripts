# Testing a camera calibration solution.
# A good reference: https://www.mathworks.com/help/vision/ug/camera-calibration.html
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid, array, ones, zeros, vstack, hstack, sqrt
from numpy.linalg import pinv, norm
from utils import *

# Intrinsic parameters
width  = 640.0      # number horizontal pixels
height = 480.0      # number vertical pixels
fx = 400.0          # horizontal focal length (pixels)
fy = 400.0          # vertical focal length (pixels)
cx = width/2.0      # camera image frame origin offset in x
cy = height/2.0     # camera image frame origin offset in y
K = array([[   fx,  0.0,  cx],  # intrinsic camera matrix
           [  0.0,   fy,  cy],  
           [  0.0,  0.0, 1.0]])

# Distortion parameters
k1 = -0.15
k2 = 0.05
k3 = -0.01
p1 = -0.009
p2 = 0.008

# points in 3D camera coordinates (right-down-forward)
M = 11
x = linspace(-1,1,M)
y = linspace(-1,1,M)
xx,yy = meshgrid(x,y)
pts_3d = vstack([xx.flatten(), yy.flatten(), ones(M*M)])
N = pts_3d.shape[1] # number of points

# apply distortion model prior to projection
pts_3d_dist = ones(pts_3d.shape)
for i in range(0,N):
    x = pts_3d[0,i]/pts_3d[2,i]
    y = pts_3d[1,i]/pts_3d[2,i]
    r = sqrt(x**2 + y**2)
    pts_3d_dist[0,i] = x*(1 + k1*r**2 + k2*r**4 + k3*r**6) + 2*p1*x*y + p2*(r**2 + 2*x**2)
    pts_3d_dist[1,i] = y*(1 + k1*r**2 + k2*r**4 + k3*r**6) + p1*(r**2 + 2*y**2) + 2*p2*x*y

# project points into image
pts_undist = project2image(pts_3d, K)
pts_dist = project2image(pts_3d_dist, K)

# using the distorted points and 3d camera points, solve for the camera intrinsic and distortion parameters
# NOTE: this is an iterative solution, i.e. solve for instrinsics then distortion and repeat
x1 = zeros(4)
x2 = zeros(5)
x1_prev = 999.*ones(x1.shape)
x2_prev = 999.*ones(x2.shape)
err = 999.
err_prev = 0.
pts_undist2 = pts_dist.copy()
while abs(err - err_prev) > 1e-6:
    # undistort the points after obtaining initial guess of intrinsic parameters
    if (all(x1) > 0):
        pts_undist2 = undistortPoints(pts_dist, x1, x2)

    # solve for intrinsics
    x1 = solveIntrinsics(pts_3d, pts_undist2)

    # solve for distortion coefficients
    x2 = solveDistortion(pts_3d, pts_dist, x1)

    # compute change in solution
    err_prev = err
    err = norm(hstack([x1 - x1_prev,x2 - x2_prev]))

print('fx: ' + f'{x1[0]:10.3f}' + ', err: ' + f'{fx - x1[0]:10.3f}')
print('fy: ' + f'{x1[1]:10.3f}' + ', err: ' + f'{fy - x1[1]:10.3f}')
print('cx: ' + f'{x1[2]:10.3f}' + ', err: ' + f'{cx - x1[2]:10.3f}')
print('cy: ' + f'{x1[3]:10.3f}' + ', err: ' + f'{cy - x1[3]:10.3f}')
print('k1: ' + f'{x2[0]:10.6f}' + ', err: ' + f'{k1 - x2[0]:10.6f}')
print('k2: ' + f'{x2[1]:10.6f}' + ', err: ' + f'{k2 - x2[1]:10.6f}')
print('k3: ' + f'{x2[2]:10.6f}' + ', err: ' + f'{k3 - x2[2]:10.6f}')
print('p1: ' + f'{x2[3]:10.6f}' + ', err: ' + f'{p1 - x2[3]:10.6f}')
print('p2: ' + f'{x2[4]:10.6f}' + ', err: ' + f'{p2 - x2[4]:10.6f}')

# plot image points
plt.figure()
plt.plot(pts_undist[0,:], pts_undist[1,:], color='b', marker='.', linestyle='None', label='Undistorted Points')
plt.plot(pts_dist[0,:], pts_dist[1,:], color='r', marker='.', linestyle='None', label='Distorted Points')
plt.plot(pts_undist2[0,:], pts_undist2[1,:], color='g', marker='x', linestyle='None', label='Undistorted Points Solution')
plt.xlabel('u')
plt.ylabel('v')
plt.grid(True)
plt.legend(loc=1)
plt.gca().invert_yaxis()
plt.show()
