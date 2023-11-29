# PROJECT3:Camera calibration

1. Pre-requirement: to run the main.py, you need to install the apriltag-grid detection library.

```
 pip install aprilgrid
```

![image-20231119020445099](/Users/fc/Desktop/学习/ELEC6910/projects/project3/assets/image-20231119020445099.png)

Note: I notice that the detection algorithm offered by this library is not robust and accurate, the `apriltags2_eth` provides superior detection results, but it require build extra third-party libraries and complex running environment, which might bring your some inconvenience.

2. `cd python`, and run `python main.py` will give you the camera intrinsic matrix `K` , distortion coefficients `[k1, k2]` for the single camera,  and camera pose `T` for each image. You can find these parameters in `results.txt`.
3. In folder `extrinsic_results/`  , You can find the visualizations of reprojections  when we calculate the camera pose for each valid image, and find the concrete reprojection error for these images in `results.txt`.  My current algorithm does not optimize the camera intrinsic, extrinsic, and distortion coeff at the final step, the noisy observations affect the estimated distortion coefficients, I admit that more efforts should be taken to get better calibration results.

## Reference
1. [apriltag-grid](https://github.com/powei-lin/aprilgrid)
2. [apriltags2_eth](https://github.com/safijari/apriltags2_ethz)
3. [Camera Calibration with OpenCV](https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html)
4. a flexible new technique for camera calibration. Zhengyou Zhang. Microsoft Research. One Microsoft Way. Redmond, WA 98052.



