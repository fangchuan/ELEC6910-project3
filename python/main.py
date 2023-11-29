import os
import sys
import cv2
import numpy as np

from AprilGrid import AprilGrid
from homography.computeH_ransac import computeH_ransac

if __name__ == '__main__':
    TAG_SIZE = 0.088
    TAG_SPACE = 0.3
    ROWS, COLS = 6, 6
    image_folderpath = '../data/'
    img_path_lst = [f for f in os.listdir(image_folderpath) if f.endswith('.jpeg')]
    img_path_lst = sorted(img_path_lst, key=lambda x: int(x.split('_')[0]))

    # log all results to a file
    output_filepath = 'results.txt'
    sys.stdout = open(output_filepath, 'w')
    # vislizations folder
    if not os.path.exists('../extrinsic_results'):
        os.makedirs('../extrinsic_results')
        
    grid_detector = AprilGrid(rows=ROWS, columns=COLS, size=TAG_SIZE, spacing=TAG_SPACE)

    # homography matrix list
    H_lst = {}
    # compuute homography
    for image_path in img_path_lst:
        image_path = os.path.join(image_folderpath, image_path)
        if not os.path.isfile(image_path):
            print('WARNING: Image path is not valid: {}'.format(image_path))
            continue
        img_name = os.path.basename(image_path)[: -5]
        rgb_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        results = grid_detector.compute_observation(img)

        if not results.success or len(results.ids) < 10:
            print('WARNING: Not enough tags detected: {}'.format(image_path))
            continue
        
        image_pts = np.array(results.image_points)
        # print(f'Image points: {image_pts.shape}')
        target_pts = np.array(results.target_points)
        # print(f'Target points: {target_pts.shape}')

        H_target_to_image, is_inliers = computeH_ransac(image_pts, target_pts, inlier_dist_thresh=5)
        print(f'Homography inlier num: {len(is_inliers.nonzero()[0])}')
        # H_lst.append(H_target_to_image)
        H_lst[img_name] = H_target_to_image

    print(f'Number of images are valid: {len(H_lst)}')

    # compute cameraa intrinsic matrix
    V = np.zeros((2*len(H_lst), 6))
    # construct geometric constrains
    for i, H_target_to_image in enumerate(H_lst.values()):
        h1 = H_target_to_image[:, 0]
        h2 = H_target_to_image[:, 1]
        h_11, h_12, h_13 = h1[0], h1[1], h1[2]
        h_21, h_22, h_23 = h2[0], h2[1], h2[2]
        V[2*i, :] = np.array([h_11*h_21,
                              h_11*h_22 + h_12*h_21,
                              h_12*h_22,
                              h_13*h_21 + h_11*h_23,
                              h_13*h_22 + h_12*h_23,
                              h_13*h_23])
        V[2*i+1, :] = np.array([h_11**2 - h_21**2,
                                2*(h_11*h_12 - h_21*h_22),
                                h_12**2 - h_22**2,
                                2*(h_11*h_13 - h_21*h_23),
                                2*(h_12*h_13 - h_22*h_23),
                                h_13**2 - h_23**2])
    # print(V)
    # compute SVD of V
    U, S, V = np.linalg.svd(V)
    # print(f'Singular values: {S}')
    # 6 vector Image of Absolute Conic
    b = V[-1]
    # print(f'b: {b}')
    B = np.array([[b[0], b[1], b[3]],
                    [b[1], b[2], b[4]],
                    [b[3], b[4], b[5]]])
    # recover intrinsic matrix
    v0 = (B[0, 1]*B[0, 2] - B[0, 0]*B[1, 2]) / (B[0, 0]*B[1, 1] - B[0, 1]**2)
    lamda = B[2, 2] - (B[0, 2]**2 + v0*(B[0, 1]*B[0, 2] - B[0, 0]*B[1, 2])) / B[0, 0]
    alpha = np.sqrt(lamda / B[0, 0])
    beta = np.sqrt(lamda*B[0, 0] / (B[0, 0]*B[1, 1] - B[0, 1]**2))
    gamma = -B[0, 1]*alpha**2*beta / lamda
    u0 = gamma*v0 / beta - B[0, 2]*alpha**2 / lamda
    K = np.array([[alpha, 0, u0],
                    [0, beta, v0],
                    [0, 0, 1]])
    print(f'Intrinsic matrix K: \n {K}')

    # distortion equations for all valid images
    D_lst = []
    d_lst = []

    # compute extrinsic matrix for each image
    for image_name, H_target_to_image in H_lst.items():
        h1 = H_target_to_image[:, 0]
        h2 = H_target_to_image[:, 1]
        h3 = H_target_to_image[:, 2]
        lamda = 1 / np.linalg.norm(np.dot(np.linalg.inv(K), h1))
        r1 = lamda * np.dot(np.linalg.inv(K), h1)
        r2 = lamda * np.dot(np.linalg.inv(K), h2)
        r3 = np.cross(r1, r2)
        t = lamda * np.dot(np.linalg.inv(K), h3)
        # current Rotation matrix is not orthogonal
        R_appro = np.array([r1, r2, r3]).T
        # print(f'approximate Rotation matrix: \n{R_appro}')
        # print(f'Translation vector: {t}')

        # estimate orthogonal rotation matrix
        U, S, V = np.linalg.svd(R_appro)
        R = np.dot(U, V)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        print(f'Image {image_name} camera extrinsic matrix:\n {T}')

        # compute projection matrix, world to image
        P = np.dot(K, np.concatenate((R, t.reshape((3, 1))), axis=1))
        # print(f'Projection matrix: \n{P}')
        # compute reprojection error
        image_path = os.path.join(image_folderpath, image_name + '.jpeg')
        rgb_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        results = grid_detector.compute_observation(img)
        image_pts = np.array(results.image_points)
        target_pts = np.array(results.target_points)
        
        target_pts = np.concatenate((target_pts, np.zeros((target_pts.shape[0], 1)), np.ones((target_pts.shape[0], 1))), axis=1)
        # print(f'Target points: {target_pts.shape}')

        target_pts_in_img = np.dot(P, target_pts.T).T
        target_pts_in_img = target_pts_in_img[:,:2] / target_pts_in_img[:, 2:]
        # draw detected points: blue
        for i in range(image_pts.shape[0]):
            x, y = int(image_pts[i, 0]), int(image_pts[i, 1])
            cv2.circle(rgb_img, (x, y), 5, (255, 0, 0), -1)
        # draw reprojection points: red
        for i in range(target_pts_in_img.shape[0]):
            x, y = int(target_pts_in_img[i, 0]), int(target_pts_in_img[i, 1])
            cv2.circle(rgb_img, (x, y), 5, (0, 0, 255), -1)

        # reprojection error
        error = np.linalg.norm(image_pts - target_pts_in_img, axis=1)
        # print(f'Reprojection error: {error}')
        print(f'Image {image_name} Average reprojection error: {np.mean(error)}')
        cv2.imwrite(f'../extrinsic_results/{image_name}_reprojection.jpg', rgb_img)

        # estimate distortion coefficients k1, k2

        def radial_distortion_func(obs_points:np.ndarray, projected_points:np.ndarray, K:np.ndarray):
            """ construct radial distortion function

            Args:
                obs_points (np.ndarray): (m, 2) observed points in image
                projected_points (np.ndarray): (m, 2) projected points on image plane
                K (np.ndarray): (3, 3) intrinsic matrix
            """
            # back-project to normalized image plane
            u0, v0 = K[0, 2], K[1, 2]
            fu, fv = K[0,0], K[1, 1]
            normalized_x = (projected_points[:, 0] - u0)/fu
            normalized_y = (projected_points[:, 1] - v0)/fv
            r2 = normalized_x**2 + normalized_y**2
            r4 = r2**2
            # construct the distortion matrix on image plane
            D = np.zeros((2 * obs_points.shape[0], 2))
            d = np.zeros((2 * obs_points.shape[0], 1))
            for i in range(projected_points.shape[0]):
                # projected coord without considering distortion
                u, v = projected_points[i, 0], projected_points[i, 1]
                obs_u, obs_v = obs_points[i, 0], obs_points[i, 1]
                D[2*i, :] = np.array([(u - u0) * r2[i], (u - u0)*r4[i]])
                D[2*i+1, :] = np.array([(v - v0)*r4[i], (v - v0)*r2[i]])

                d[2*i, :] = np.array([obs_u - u])
                d[2*i+1, :] = np.array([obs_v - v])

            return D, d

        D, d = radial_distortion_func(image_pts, target_pts_in_img, K)
        D_lst.append(D)
        d_lst.append(d)

    D = np.concatenate(D_lst, axis=0)
    d = np.concatenate(d_lst, axis=0)
    print(f'D: {D.shape}, d: {d.shape}')
    # compute distortion coefficients
    k1, k2 = np.linalg.lstsq(D, d, rcond=None)[0]
    print(f'Distortion coefficients: k1: {k1}, k2: {k2}')

