# Utility type functions
from numpy import array, zeros, ones, sqrt
from numpy.linalg import pinv

### This function takes points and projects them onto a pixel image
# NOTE: camera frame is the standard right-down-forward reference frame
# pts : point positions (3xN matrix for N points in the 3D camera frame)
# K   : intrinsic camera matrix
# return : 2xN matrix of pixel image points
def project2image(pts, K):

    # get depth along optical axis of each point
    zs = pts[2,:]

    # project points into image
    pix = K[0:2,:].dot(pts)

    # divide each pixel by depth
    return pix / zs


### This function solves for camera intrinsic parameters assuming known 3D points in the camera frame
# NOTE: camera frame is the standard right-down-forward reference frame
# pts_3d     : point positions (3xN matrix for N points in the 3D camera frame)
# pts_undist : undistorted image points (2xN matrix for N points)
# return     : 4 element array [fx,fy,cx,cy] 
# NOTE: fx,fy are focal lengths and cx,cy are image centers
def solveIntrinsics(pts_3d, pts_undist):
    N = pts_3d.shape[1]
    A = zeros([2*N,4])
    b = zeros([2*N,1])
    for i in range(0,N):
        x = pts_3d[0,i]/pts_3d[2,i]
        y = pts_3d[1,i]/pts_3d[2,i]
        u = pts_undist[0,i]
        v = pts_undist[1,i]
        A[2*i,0] = x
        A[2*i,2] = 1.0
        A[2*i+1,1] = y
        A[2*i+1,3] = 1.0
        b[2*i] = u
        b[2*i+1] = v
    x = pinv(A.T.dot(A)).dot(A.T.dot(b))
    return x.flatten()


### This function solves for camera distortion parameters assuming known intrinsics and 3D points in the camera frame
# NOTE: camera frame is the standard right-down-forward reference frame
# pts_3d   : point positions (3xN matrix for N points in the 3D camera frame)
# pts_dist : distorted image points (2xN matrix for N points)
# params   : intrinsic parameters (4 element array [fx,fy,cx,cy])
# return   : 5 element array [k1,k2,k3,p1,p2] 
# NOTE: k1,k2,k3 are for radial distortion and p1,p2 are for tangential distortion
def solveDistortion(pts_3d, pts_dist, params):
    N = pts_3d.shape[1]
    A = zeros([2*N,5])
    b = zeros([2*N,1])
    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]
    for i in range(0,N):
        x = pts_3d[0,i]/pts_3d[2,i]
        y = pts_3d[1,i]/pts_3d[2,i]
        u = pts_dist[0,i]
        v = pts_dist[1,i]
        r = sqrt(x**2 + y**2)
        A[2*i,0] = x*r**2
        A[2*i,1] = x*r**4
        A[2*i,2] = x*r**6
        A[2*i,3] = 2*x*y
        A[2*i,4] = r**2 + 2*x**2
        A[2*i+1,0] = y*r**2
        A[2*i+1,1] = y*r**4
        A[2*i+1,2] = y*r**6
        A[2*i+1,3] = r**2 + 2*y**2
        A[2*i+1,4] = 2*x*y
        b[2*i] = (u - cx)/fx - x
        b[2*i+1] = (v - cy)/fy - y
    x = pinv(A.T.dot(A)).dot(A.T.dot(b))
    return x.flatten()


### This function undistorts image points given camera intrinsic and distortion parameters
# pts_dist   : distorted image points (2xN matrix for N points)
# intrinsics : camera intrinsic parameters (4 element array [fx,fy,cx,cy])
# distcoeffs : distortion parameters (5 element array [k1,k2,k3,p1,p2])
# tol        : exit tolerance
# return     : undistorted points (2xN matrix for N points)
# NOTE: fx,fy are focal lengths and cx,cy are image centers
# NOTE: k1,k2,k3 are for radial distortion and p1,p2 are for tangential distortion
def undistortPoints(pts_dist, intrinsics, distcoeffs, tol=1e-6):
    N = pts_dist.shape[1]
    fx = intrinsics[0]
    fy = intrinsics[1]
    cx = intrinsics[2]
    cy = intrinsics[3]
    K = array([[fx,  0, cx],
               [ 0, fy, cy],
               [ 0,  0,  1]])
    k1 = distcoeffs[0]
    k2 = distcoeffs[1]
    k3 = distcoeffs[2]
    p1 = distcoeffs[3]
    p2 = distcoeffs[4]
    pts_3d = ones([3,N])
    max_iters = 1000
    for i in range(0,N):
        xd = (pts_dist[0,i] - cx)/fx
        yd = (pts_dist[1,i] - cy)/fy
        x = xd
        y = yd
        x_prev = 1e9
        y_prev = 1e9
        # an iterative solution is required to solve these equations for x,y
        iters = 0
        while abs(x - x_prev) > tol and abs(y - y_prev) > tol and iters < max_iters:
            x_prev = x
            y_prev = y
            r = sqrt(x**2 + y**2)
            x = (xd - 2*p1*x*y - p2*(r**2 + 2*x**2))/(1.0 + k1*r**2 + k2*r**4 + k3*r**6)
            y = (yd - p1*(r**2 + 2*y**2) - 2*p2*x*y)/(1.0 + k1*r**2 + k2*r**4 + k3*r**6)
            iters += 1
        pts_3d[0,i] = x
        pts_3d[1,i] = y
    return project2image(pts_3d, K)