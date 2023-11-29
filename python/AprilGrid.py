from collections import namedtuple
from aprilgrid import Detector
from aprilgrid.detection import Detection

# NOTE: all of this ultimately only works with t36h11, you have been warned

DetectionResult = namedtuple('DetectionResult',
                             ['success', 'image_points',
                              'target_points', 'ids'])


class AprilGrid(object):
    def __init__(self, rows, columns, size, spacing, family='t36h11'):  # spacing is a fraction
        assert size != 0.0
        assert spacing != 0.0
        self.rows = rows
        self.columns = columns
        self.size = size
        self.spacing = spacing
        self.detector = Detector(family)

    def is_detection_valid(self, detection:Detection, image):
        h, w = image.shape[0:2]
        for corner in detection.corners:
            cx, cy = np.round(corner[0]).astype(np.int32)
            if cx < 0 or cx > w:
                return False
            if cy < 0 or cy > h:
                return False
        # if not d.good:
        #     return False
        if detection.tag_id >= self.rows * self.columns:  # original code divides this by 4????
            return False

        return True

    def get_tag_corners_for_id(self, tag_id):
        # order is lower left, lower right, upper right, upper left
        # Note: tag_id of lower left tag is 0, not 1
        a = self.size  # https://user-images.githubusercontent.com/5337083/41458381-be379c6e-7086-11e8-9291-352445140e88.png
        b = self.spacing * a
        tag_row = (tag_id) // self.columns
        tag_col = (tag_id) % self.columns
        left = bottom = lambda i: i*(a + b)
        right = top = lambda i: (i + 1) * a + (i) * b
        return [
            (left(tag_col), bottom(tag_row)),
            (right(tag_col), bottom(tag_row)),
            (right(tag_col), top(tag_row)),
            (left(tag_col), top(tag_row))
        ]

    def compute_observation(self, image):
        # return imagepoints and the coordinates of the corners
        # 1. remove non good tags
        detections = self.detector.detect(image)

        # Duplicate ID search
        ids = {}
        for d in detections:
            if d.tag_id in ids:
                raise Exception(
                    "There may be two physical instances of the same tag in the image")
            # ids[d] = True

        filtered = [d for d in detections if self.is_detection_valid(d, image)]

        image_points = []
        target_points = []
        ids = []

        filtered.sort(key=lambda x: x.tag_id)

        # TODO: subpix refinement?
        for f in filtered:
            target_points.extend(self.get_tag_corners_for_id(f.tag_id))
            image_points.extend([(c[0][0], c[0][1]) for c in f.corners])
            ids.extend([f.tag_id, f.tag_id, f.tag_id, f.tag_id])

        success = True if len(filtered) > 0 else False

        return DetectionResult(success, image_points, target_points, ids)

import os
import sys
import cv2
import numpy as np

if __name__ == '__main__':
    test_image = '../data/12_CG12206A001.jpeg'
    rgb_img = cv2.imread(test_image, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    
    aprilgrid = AprilGrid(6, 6, 0.088, 0.3)
    result = aprilgrid.compute_observation(img)
    for image_point, tgt_point, ids in zip(result.image_points, result.target_points, result.ids):
        x = int(image_point[0])
        y = int(image_point[1])
        tx = tgt_point[0]
        ty = tgt_point[1]
        print(tgt_point)
        cv2.circle(rgb_img, (x, y), 5, (255 - tx/0.1205*200, 255, ty/0.1406*200), -1)


    img_color = cv2.resize(rgb_img, None, None, 0.5, 0.5)
    cv2.imshow("Image", img_color)
    cv2.waitKey(0)
