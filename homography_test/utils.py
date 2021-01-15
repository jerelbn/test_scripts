# Utility type functions

### Compute rotation matrices from 3-2-1 Euler angles
from numpy import array, sin, cos, zeros, ones, vstack
from numpy.linalg import svd, det

def getR3(yaw):
    return array([[  cos(yaw),  sin(yaw),  0.0  ],
                  [ -sin(yaw),  cos(yaw),  0.0  ],
                  [       0.0,       0.0,  1.0  ]])

def getR2(pitch):
    return array([[  cos(pitch),  0.0, -sin(pitch)  ],
                  [         0.0,  1.0,         0.0  ],
                  [  sin(pitch),  0.0,  cos(pitch)  ]])

def getR1(roll):
    return array([[  1.0,        0.0,        0.0  ],
                  [  0.0,  cos(roll),  sin(roll)  ],
                  [  0.0, -sin(roll),  cos(roll)  ]])

def getRcb2c():
    return array([[  0.0,  1.0,  0.0  ],
                  [  0.0,  0.0,  1.0  ],
                  [  1.0,  0.0,  0.0  ]])

def getRi2b(roll, pitch, yaw):
    return getR3(yaw).dot(getR2(pitch).dot(getR1(roll)))

def getRi2c(roll, pitch, yaw):
    return getRcb2c().dot(getRi2b(roll,pitch,yaw))


### This function takes points and projects them onto a pixel image
# NOTE: camera frame is the standard right-down-forward reference frame

# pts : inertial points positions (3xN matrix for N points)
# cam : inertial camera position
# R   : rotation from inertial to camera
# K   : intrinsic camera matrix

# return : 2xN matrix of pixel image points

def project2image(pts, cam, R, K):
    
    # compute point positions relative to camera in camera frame
    pts_c = R.dot(pts - cam[:,None])

    # get depth along optical axis of each point
    zs = pts_c[2,:]

    # project points into image
    pix = K[0:2,:].dot(pts_c)

    # divide each pixel by depth
    return pix / zs


### Compute homography by points

# pts1 : 2xN matrix of N points in first camera frame
# pts2 : 2xN matrix of N points in second camera frame

# return : 3x3 homography matrix mapping from first to second frame

def homographyFromPoints(pts1, pts2):
    N = pts1.shape[1] # number of points
    A = zeros([2*N,9]) # solution matrix

    # populate solution matrix
    for i in range(0,N):
        idx1 = 2*i
        idx2 = 2*i+1
        u1 = pts1[0,i]
        u2 = pts2[0,i]
        v1 = pts1[1,i]
        v2 = pts2[1,i]
        A[idx1,0] = -u1
        A[idx1,1] = -v1
        A[idx1,2] = -1.0
        A[idx1,6] = u1*u2
        A[idx1,7] = v1*u2
        A[idx1,8] = u2
        A[idx2,3] = -u1
        A[idx2,4] = -v1
        A[idx2,5] = -1.0
        A[idx2,6] = u1*v2
        A[idx2,7] = v1*v2
        A[idx2,8] = v2

    # singular value decomposition of A
    u, s, h = svd(A)

    # collect homography components
    H = h[-1,:].reshape(3,3)
    detH = det(H)
    if detH < 0:
        detH *= -1
        H *= -1
    return H/detH**(1.0/3.0)


### Warp points from one image to another

# pts : 2xN matrix of N points
# H   : homography matrix

def warpPoints(pts, H):
    pts = vstack([pts, ones([1,pts.shape[1]])])
    pts = H.dot(pts)
    pts /= pts[2,:]
    return pts[0:2,:]