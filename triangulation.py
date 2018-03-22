import sys
import numpy as np
import os
import scipy.misc
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # TODO: Implement this method!
    print('estimate initial RT')
    U, D, V_T = np.linalg.svd(E)
    D = np.diag(D)

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    Q1 = np.dot(np.dot(U,W),V_T)
    Q2 = np.dot(np.dot(U,W.T),V_T)

    R1 = np.dot(np.linalg.det(Q1),Q1)
    R2 = np.dot(np.linalg.det(Q2),Q2)

    T1 = U[:,2][:,np.newaxis]
    T2 = -T1

    RT0 = np.concatenate((R1,T1), axis = 1)
    RT1 = np.concatenate((R1,T2), axis = 1)
    RT2 = np.concatenate((R2,T1), axis = 1)
    RT3 = np.concatenate((R2,T2), axis = 1)

    RT = np.zeros((4,3,4))
    RT[0] = RT0;
    RT[1] = RT1;
    RT[2] = RT2;
    RT[3] = RT3;

    return RT

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
'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # TODO: Implement this method!
    #print('estimate_RT_from_E')
    N = image_points.shape[0]

    RT_initial = estimate_initial_RT(E)

    M1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    M1 = np.dot(K,M1)

    M = np.zeros((2,3,4))
    M[0,:,:] = M1
    num = RT_initial.shape[0]
    counts = np.zeros((num,1))

    for i in range(0,num):
      RT_test = RT_initial[i,:,:]

      M2 = np.dot(K,RT_test);
      M[1,:,:] = M2

      for j in range(0,N):
        P1_test = nonlinear_estimate_3d_point_LM(image_points[j], M)
        P2_test = np.dot(RT_test,np.append(P1_test, [1]).reshape(4,1))
        if P1_test[2] > 0 and P2_test[2] > 0:
          counts[i] = counts[i] + 1

    index = np.argmax(counts)
    RT = RT_initial[index,:,:]
    return RT

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'))[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches_lin_noise30.npy'))
    #dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches_lin_noise.npy'))
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'))[0,:]
#    fundamental_matrices = np.load(os.path.join(image_data_dir,
#        'fundamental_matrices_lin_noise.npy'))[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print '-' * 80
    print "Part A: Check your matrices against the example R,T"
    print '-' * 80
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = scipy.misc.imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print "Example RT:\n", example_RT
    estimated_RT = estimate_initial_RT(E)
    print
    print "Estimated RT:\n", estimated_RT

    # Part B: Determining the best linear estimate of a 3D point
    print '-' * 80
    print 'Part B: Check that the difference from expected point '
    print 'is near zero'
    print '-' * 80
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print "Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum()

    # Part C: Calculating the reprojection error and its Jacobian
    print '-' * 80
    print 'Part C: Check that the difference from expected error/Jacobian '
    print 'is near zero'
    print '-' * 80
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631)).reshape(4,1) #edited 
    print "Error Difference: ", np.fabs(estimated_error - expected_error).sum()
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print "Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum()

    # Part D: Determining the best nonlinear estimate of a 3D point
    print '-' * 80
    print 'Part D: Check that the reprojection error from nonlinear method'
    print 'is lower than linear method'
    print '-' * 80
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point_LM(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print "Linear method error:", np.linalg.norm(error_linear)
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print "Nonlinear method error:", np.linalg.norm(error_nonlinear)

    # Part E: Determining the correct R, T from Essential Matrix
    print '-' * 80
    print "Part E: Check your matrix against the example R,T"
    print '-' * 80
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print "Example RT:\n", example_RT
    print
    print "Estimated RT:\n", estimated_RT

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print '-' * 80
    print 'Part F: Run the entire SFM pipeline'
    print '-' * 80
    frames = [0] * (len(image_paths) - 1)
    for i in xrange(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    err_vec = np.zeros((1,0))
    for i in xrange(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in xrange(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point_LM(
                matches[:,j].reshape((2,2)), camera_matrices)
            err = np.linalg.norm(reprojection_error(points_3d[j,:], matches[:,j].reshape(2,2),camera_matrices))
            err_vec = np.append(err_vec, err)
            #print(err)
            #print(err_vec.shape)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))
    err_vec[0] = np.linalg.norm(err_vec)
    print(err_vec[0])
    print(np.mean(err_vec))
    np.savetxt('error_GN_linNoise30.csv', err_vec)
    np.savetxt('structure_GN_linNoise30.csv', dense_structure, delimiter = ',')
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        c='k', depthshade=True, s=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    ax.view_init(-100, 90)

    plt.show()
