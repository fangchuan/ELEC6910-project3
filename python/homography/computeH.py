# Q2.3
import sys
sys.path.append('./')
sys.path.append('../')

import numpy as np
import cv2 


def computeH(x1:np.ndarray, x2:np.ndarray):
    """ compute the homography matrix using direct linear transformation

    Args:
        x1 (np.ndarray): pixcel points in image 1, shape (num_kpt, 2)
        x2 (np.ndarray): pixel points in image 2, shape (num_kpt, 2)

    Returns:
        np.ndarray: homography matrix, shape (3, 3), warp image 2 to image 1
    """
    assert x1.shape[0] == x2.shape[0]

    num_kpt = x1.shape[0]
    A = np.zeros((2*num_kpt, 9))
    for i in range(num_kpt):
        x, y = x1[i, 0], x1[i, 1]
        u, v = x2[i, 0], x2[i, 1]
        A[2*i, :] = np.array([-u, -v, -1, 0, 0, 0, x*u, x*v, x])
        A[2*i+1, :] = np.array([0, 0, 0, -u, -v, -1, y*u, y*v, y])
    # print(A)
    # Compute SVD of A
    U, S, V = np.linalg.svd(A)
    # print(f'Singular values: {S}')
    # Extract homography
    H2to1 = V[-1].reshape((3, 3))
    return H2to1

# from matchPics import matchPics

# if __name__=="__main__":
#     # Load images
#     cv_img = cv2.imread('../data/cv_cover.jpg')
#     desk_img = cv2.imread('../data/cv_desk.png')

#     # Extract features and match
#     locs1, locs2 = matchPics(cv_img, desk_img)

#     # Compute homography using RANSAC
#     use_kpt_num = 100
#     kpts1 = locs1[:use_kpt_num]
#     kpts2 = locs2[:use_kpt_num]
#     H2to1 = computeH(kpts1, kpts2)

#     H1to2 = np.linalg.inv(H2to1)

#     # randonmly select 10 points, vis warpped points
#     num_kpt = locs1.shape[0]
#     assert num_kpt >= 10, 'Number of keypoints should be larger than 10'
#     sample_idx = np.random.choice(num_kpt, 10)
#     sample_kpts1 = locs1[sample_idx]
#     warpped_kpts1 = np.dot(H1to2, np.concatenate((sample_kpts1, np.ones((sample_kpts1.shape[0], 1))), axis=1).T).T
#     warpped_kpts1 = warpped_kpts1[:, :2] / warpped_kpts1[:, 2:]

#     matches = [cv2.DMatch(i, i, 0) for i in range(10)]
#     # vis matches
#     kpts1 = [cv2.KeyPoint(x=kpt[0], y=kpt[1], size=1) for kpt in sample_kpts1]
#     kpts2 = [cv2.KeyPoint(x=kpt[0], y=kpt[1], size=1) for kpt in warpped_kpts1]
#     match_img = cv2.drawMatches(cv_img, kpts1, desk_img, kpts2, matches, None, flags=2)
#     cv2.imwrite('./results/compute_H_matches.jpg', match_img)

#     # visualize the warped image
#     warped_img = cv2.warpPerspective(cv_img, H1to2, (desk_img.shape[1], desk_img.shape[0]))
#     cv2.imwrite('./results/compute_H_warped_img.jpg', warped_img)


