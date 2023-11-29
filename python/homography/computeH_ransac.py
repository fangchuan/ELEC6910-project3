# Q2.5
import sys
sys.path.append('./')
sys.path.append('../')

import numpy as np
import random

from .computeH_norm import computeH_norm

def computeIterationsNumberAdaptively(inlier_prob, sample_size):
    """ compute the number of iterations for RANSAC adaptively

    Args:
        inlier_prob (float): inlier probability
        sample_size (int): sample size

    Returns:
        int: number of iterations
    """
    if inlier_prob < 0.01:
        return 1000000
    
    unstable_p = 1 - 0.99
    unstable_p_log = np.log10(unstable_p)

    outlier_pro = 1 - inlier_prob ** sample_size
    N = unstable_p_log / np.log10(outlier_pro)
    return N

def computeH_ransac(locs1, locs2, max_iter=10000, sample_size=4, inlier_dist_thresh=10, verbose=False):
    assert locs1.shape == locs2.shape
    num_kpt = locs1.shape[0]
    assert num_kpt >= 4, 'Number of keypoints should be larger than 4'

    # ransac parameters
    max_iter = 10000
    sample_size = 4
    best_inliner_num = 0
    inlier_prob = 0.0
    inlier_dist_thresh = 10
    best_H2to1 = None
    is_inliers_lst = np.zeros(num_kpt)

    itr = 0
    # compute homography in a ransac manner
    while itr < max_iter:
        itr += 1
        # randomly sample 4 keypoints
        sample_idx = random.sample(range(num_kpt), sample_size)
        sample_kpts1 = locs1[sample_idx]
        sample_kpts2 = locs2[sample_idx]

        # compute homography
        H2to1 = computeH_norm(sample_kpts1, sample_kpts2)

        # compute inliner number
        inliner_num = 0
        for j in range(num_kpt):
            x1 = np.concatenate((locs1[j], np.array([1])))
            x2 = np.concatenate((locs2[j], np.array([1])))
            x1_pred = np.dot(H2to1, x2)
            x1_pred /= x1_pred[-1]
            if np.linalg.norm(x1 - x1_pred) < inlier_dist_thresh:
                inliner_num += 1
                is_inliers_lst[j] = 1

        # update best homography
        if inliner_num > best_inliner_num:
            best_inliner_num = inliner_num
            best_H2to1 = H2to1
            inlier_prob = inliner_num / num_kpt
            max_iter = itr + computeIterationsNumberAdaptively(inlier_prob, sample_size)
            if verbose:
                print(f'Iteration {itr}: inliner number: {inliner_num}, inliner probability: {inlier_prob}, max iteration: {max_iter}')

    return best_H2to1, is_inliers_lst

# from matchPics import matchPics
# import cv2
# if __name__=="__main__":
#     # Load images
#     cv_img = cv2.imread('../data/cv_cover.jpg')
#     desk_img = cv2.imread('../data/cv_desk.png')

#     # Extract features and match
#     locs1, locs2 = matchPics(cv_img, desk_img)

#     # Compute homography using RANSAC
#     H2to1, is_inliers_lst = computeH_ransac(locs1, locs2)

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
#     cv2.imwrite('./results/compute_H_ransac_matches.jpg', match_img)

#     # visualize the warped image
#     warped_img = cv2.warpPerspective(cv_img, H1to2, (desk_img.shape[1], desk_img.shape[0]))
#     cv2.imwrite('./results/compute_H_ransac_warped_img.jpg', warped_img)