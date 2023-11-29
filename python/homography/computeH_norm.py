# Q2.4
import sys
sys.path.append('./')
sys.path.append('../')

import numpy as np
from .computeH import computeH
# !!! YOU CAN USE np.linalg.svd()

def computeH_norm(x1:np.ndarray, x2:np.ndarray):
    # Compute centroids of the points
    centroid1 = np.mean(x1, axis=0)
    centroid2 = np.mean(x2, axis=0)

    # Shift the origin of the points to the centroid
    x1_centered = x1 - centroid1
    x2_centered = x2 - centroid2

    # Normalize the points so that the average distance from the origin is equal to sqrt(2)
    avg_dist1 = np.mean(np.linalg.norm(x1_centered, axis=1))
    avg_dist2 = np.mean(np.linalg.norm(x2_centered, axis=1))
    scale1 = np.sqrt(2) / (avg_dist1 + 1e-6)
    scale2 = np.sqrt(2) / (avg_dist2 + 1e-6)

    # Similarity transform 1
    T1 = np.array([[scale1, 0, -scale1*centroid1[0]], [0, scale1, -scale1*centroid1[1]], [0, 0, 1]])
    T2 = np.array([[scale2, 0, -scale2*centroid2[0]], [0, scale2, -scale2*centroid2[1]], [0, 0, 1]])

    # Compute Homography
    x1_normalized = np.dot(T1, np.concatenate((x1, np.ones((x1.shape[0], 1))), axis=1).T).T
    x2_normalized = np.dot(T2, np.concatenate((x2, np.ones((x2.shape[0], 1))), axis=1).T).T
    H2to1 = computeH(x1_normalized[:, :2], x2_normalized[:, :2])

    # Denormalization
    H2to1 = np.dot(np.dot(np.linalg.inv(T1), H2to1), T2)

    return H2to1

# from matchPics import matchPics
# import cv2
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
#     H2to1 = computeH_norm(kpts1, kpts2)

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
#     cv2.imwrite('./results/compute_H_norm_matches.jpg', match_img)

#     # visualize the warped image
#     warped_img = cv2.warpPerspective(cv_img, H1to2, (desk_img.shape[1], desk_img.shape[0]))
#     cv2.imwrite('./results/compute_H_norm_warped_img.jpg', warped_img)