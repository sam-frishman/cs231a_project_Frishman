import sys
import numpy as np
import os
import scipy.misc
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

random.seed(10)
np.random.seed(10)
'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
#   print('linear estiamte 3d point')
    M = image_points.shape[0]
    A = np.zeros((M*2,4))
    for i in range(0,M):
      A[i*2,:] = image_points[i,0]*camera_matrices[i,2,:]-camera_matrices[i,0,:]
      A[i*2+1,:] = image_points[i,1]*camera_matrices[i,2,:]-camera_matrices[i,1,:]

    #print(A)
    U, s, V_T = np.linalg.svd(A)
    V = V_T.T
    V_last_col = V[:, V.shape[1]-1]
    point_3d = V_last_col[:3]/V_last_col[3]
    #print(point_3d.shape)
    addNoise = True
    percent = 0.20
    if addNoise:
      for i in range(0, point_3d.shape[0]):
        val = point_3d[i]
        delta = abs(val)*percent
        #new_val = random.uniform(val-delta,val+delta)
        new_val = np.asscalar(np.random.normal(val,2,1))
        point_3d[i] = new_val
        #print(new_val)
    return point_3d
'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # TODO: Implement this method!
#print('reprojection_error')
    M = image_points.shape[0]
    point_3d = np.append(point_3d, [1]).reshape(4,1)

    p = np.zeros((M,2))
    for i in range(0,M):
      y = np.dot(camera_matrices[i,:,:],point_3d).reshape(1,3)
      p[i,:] = y[0,:2]/y[0,2]

    error = p-image_points
    error  = error.reshape(2*M,1)
    #error  = error.reshape(1,2*M)
    return error

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # TODO: Implement this method!
#    print('jacobian')
    M = camera_matrices.shape[0]
    point_3d = np.append(point_3d, [1]).reshape(4,1)

    jacobian = np.zeros((2*M,3))
    for i in range(0,M):
      numerator1 = np.dot(camera_matrices[i,0,:4],point_3d)
      numerator2 = np.dot(camera_matrices[i,1,:4],point_3d)
      den = np.dot(camera_matrices[i,2,:], point_3d)

      jacobian[i*2,0] = (camera_matrices[i,0,0]*den-numerator1*camera_matrices[i,2,0])/den**2
      jacobian[i*2,1] = (camera_matrices[i,0,1]*den-numerator1*camera_matrices[i,2,1])/den**2
      jacobian[i*2,2] = (camera_matrices[i,0,2]*den-numerator1*camera_matrices[i,2,2])/den**2
      jacobian[i*2+1,0] = (camera_matrices[i,1,0]*den-numerator2*camera_matrices[i,2,0])/den**2
      jacobian[i*2+1,1] = (camera_matrices[i,1,1]*den-numerator2*camera_matrices[i,2,1])/den**2
      jacobian[i*2+1,2] = (camera_matrices[i,1,2]*den-numerator2*camera_matrices[i,2,2])/den**2

    #print(jacobian.shape)
    return jacobian

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    #print('nonlinear estimate 3d point')

    P_hat = linear_estimate_3d_point(image_points, camera_matrices)
    for i in range(0,10):
      J = jacobian(P_hat, camera_matrices)
      e = reprojection_error(P_hat, image_points, camera_matrices)
      delta = np.dot(np.dot(np.linalg.inv(np.dot(J.T,J)),J.T),e)
      P_hat = P_hat.reshape(3,1) - delta

    #print(P_hat.shape)
    #print(P_hat)
    return P_hat.reshape(3,)

def nonlinear_estimate_3d_point_LM(image_points, camera_matrices):
    # TODO: Implement this method!
    #print('nonlinear estimate 3d point')
    factor = 10
    P_hat = linear_estimate_3d_point(image_points, camera_matrices)
    lambda_ = 0.
    #print('LM------------')
    for i in range(0,10):
      error_reduced = False
      count = 0
      J = jacobian(P_hat, camera_matrices)
      e = reprojection_error(P_hat, image_points, camera_matrices)
      if i == 0:
        N = np.dot(J.T,J)
        lambda_ = (10**-3)*np.mean(np.diagonal(N))
      while not error_reduced:
        count += 1
        #print('----------------------------------')
        #print(count)
        delta = np.dot(np.dot(np.linalg.inv(np.dot(J.T,J)+lambda_*np.identity(3)),J.T),e)
        #print(delta)
        P_hat_test = P_hat.reshape(3,1) - delta
        e_new = reprojection_error(P_hat_test, image_points, camera_matrices)
        #print(e_new)
        #print(P_hat)
        #print(e)
        #print(lambda_)
        #if np.all(e_new <= e):
        #if np.mean(abs(e_new)) <= np.mean(abs(e)):
        if np.linalg.norm(e_new) <= np.linalg.norm(e):
          P_hat = P_hat_test
          lambda_ = lambda_/factor
          error_reduced = True
        else:
          #print('else')
          lambda_ = lambda_*factor
          #print(lambda_)
      

    #print(P_hat.shape)
    #print(P_hat)
    return P_hat.reshape(3,)

#read in camera matrices
matrices = []
for i in range(0, 10):
  cam_mat = np.zeros((3,4))
  with open('3D/house.00' + str(i) + '.P','r') as fp:
    count = 0
    for line in fp:
      row = np.fromstring(line, dtype=float, sep=' ')
      cam_mat[count, :] = row
      count += 1
      print(row)
    matrices.append(cam_mat)

camera_matrices_all = np.asarray(matrices)

#read in 2d points
points = []
for i in range(0, 10):
  points2d = np.zeros((1,2))
  with open('2D/house.00' + str(i) + '.corners','r') as fp:
    firstRow = True
    for line in fp:
      row = np.fromstring(line, dtype=float, sep=' ').reshape(1,2)
      if firstRow:
        points2d[0,:] = row
        firstRow = False
      else:
        points2d = np.append(points2d, row, axis = 0)
      print(row)
    points.append(points2d)

points2d_all = np.asarray(points)

#read in match rules
matches = []
with open('2D/house.nview-corners','r') as fp:
  for line in fp:
    row = np.array(line.split())
    matches.append(row.astype(np.float))
    print(row.astype(np.float))
matches_all = np.asarray(matches)

points3d = []
error = []
for i in range(0, matches_all.shape[0]):
  match_row = matches_all[i]
  point_matches = []
  cam_matrices = []
  for j in range(0, match_row.shape[0]):
    view_j_index = match_row[j]
    if view_j_index != -1:
      points2d_j = points2d_all[j][view_j_index,:].reshape(1,2)
      point_matches.append(points2d_j)
      cam_matrices.append(camera_matrices_all[j])
  point_matches_np = np.asarray(point_matches)
  point_matches_np = point_matches_np.reshape(point_matches_np.shape[0], point_matches_np.shape[2])
  cam_matrices_np = np.asarray(cam_matrices)
  print(point_matches_np.shape)
  print(cam_matrices_np.shape)
  points3d.append(nonlinear_estimate_3d_point_LM(point_matches_np, cam_matrices_np))
  error.append(np.linalg.norm(reprojection_error(points3d[i],point_matches_np, cam_matrices_np)))
  print('point added')

points3d_all = np.asarray(points3d)
error_all = np.asarray(error)
print(np.linalg.norm(error_all))
print(np.mean(error_all))
