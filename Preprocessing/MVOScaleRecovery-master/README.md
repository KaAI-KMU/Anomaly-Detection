# MVOScaleRecovery
Recover the scale of monocular visual odometry

# RUN
modify the `src/param.py` based on your dataset 

#path.txt 생성하는 코드

   python src/main.py {이미지 경로}
   
#visualize

   python script/plot_path.py {생성 path} {GT path}
   
   python script/test.py {ego 파일}
   

# Note
this is a scale recoery for a simple monocular VO, the accuracy is degraded. Current error of KITTI 00 by  [KITTI benchmark](https://github.com/TimingSpace/EvaluateVisualOdometryKITTI) is 2.17% (ave every 800m)

## Current result
1. KITTI 00
![kitti_00](result/kitti_00_path_filter_10.png)
![kitti_00](result/kitti_00_x_filter_10.png)
![kitti_00](result/kitti_00_z_filter_10.png)
![kitti_00](result/kitti_00_y_filter_10.png)
![kitti_00_scale](result/kitti_00_scale_filter_10.png)

2. KITTI 02
![kitti_02](result/kitti_02_path_remove_outlier_with_gt.pdf)
![kitti_02_scale](result/kitti_02_scale_remove_outlier_with_gt.pdf)

3. Initial Triangles before rance
![triangles](result/before_reject.png)
![triangles_o](result/after_reject.png)
4. depth and reconstruct
![triangles](result/depth.png)
![triangles_o](result/pcl.png)

