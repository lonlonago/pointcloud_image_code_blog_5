Paper: https://arxiv.org/pdf/2006.11275.pdf CenterPoint Code: https://github.com/tianweiy/CenterPoint OpenPCDet Code: https://github.com/open-mmlab/OpenPCDet/ 

#  brief introduction 

 Anchor-based detectors have difficulty enumerating all directions or fitting axis-aligned bounding boxes to a rotating object. CenterPoint proposes a center-based 3D object detection and tracking framework based on a lidar point cloud. First, the center of the object is detected using a keypoint detector, and then other properties are regressed, including 3D size, 3D orientation, and velocity. In the second phase, it uses additional point features on the target to improve these estimates. CenterPoint is simple and near real-time, achieving state-of-the-art performance in Waymo and nuScenes benchmarks. 

 ![avatar]( de57afaa488f49aa83fe612d34ded20a.png) 

 CenterPoint uses a standard Lidar-based backbone, namely VoxelNet or PointPillars, to build a representation of the input point cloud. It then tiles this representation into a BEV view and uses a standard image-based keypoint detector to find the target center. For each detected center, it regresses all other target attributes, such as 3D size, orientation, and velocity, from the point feature at the center position. Additionally, improve the target position with a lightweight second stage.  

>  CenterPoint proposes a center-based framework for representing, detecting, and tracking objects. Previous anchor-based methods used aligning anchors with respect to the vehicle's own coordinate axes. Both the anchor-based method and our center-based method were able to accurately detect objects when the vehicle was traveling on a straight road. However, during a left turn (bottom), the anchor-based method had difficulty fitting an axis-aligned bounding box to a rotating object. Our center-based model accurately detected objects by rotating invariant points. 

 Center-based notation has several key advantages: 

 We tested our model on two popular large datasets: Waymo Open and nuScenes. We found that a simple switch from a box representation to a center-based representation can add 3-4 mAP under different trunks. Stage 2 refinement further results in an additional 2 mAP boost with minimal computational overhead (< 10%). 

#  CenterPoint 

 ![avatar]( e7d029f1834845c8ab0dd0dcd9f817db.png) 

 Figure 2 shows the overall framework of the CenterPoint model. The first stage first extracts the BEV features of the LIDAR point cloud using backbone_3D (using the form of voxel or pillar). Then, backbone_2D the detection head to find the object center and use the center feature to return the complete 3D bounding box (center, length, width, height, heading angle, speed). The second stage is to pass the predicted box point features of the first stage to the MLP, de-refining the confidence level score and 3D box 

##  Based on voxel 

###  MeanVFE 

 The voxel voxel features are calculated using the pretreatment stage, and the point features of voxel are averaged MeanVFE 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
###  VoxelResBackBone8x 

 VoxelResBackBone8x 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 For the forward part of the VoxelBackBone8x module, the most important contents in the input dictionary are voxel_features and voxel_coords. They represent valid input features, respectively, and the spatial location of these valid features. voxel_features size is (N, 5) 

 As can be seen from post_act_block, spconv has 3 kinds of 3D sparse convolutions: SubMConv3d, SparseConv3d and SparseInverseConv3d 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 The 3D sparse convolution of spconv is similar to that of ordinary convolution, except that there is an extra indice_key, which is to repurpose the calculated rulebook and hash table under the same condition as the indice, reducing the calculation 

 Take a look at the following line of code: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 Why do you need to add 1 to the Z axis of sparse_shape? 

>  Reference: https://github.com/open-mmlab/mmdetection3d/issues/282 

 SparseEncoder will downsample in the high dimension. Adding 1 allows the height dimension to be downsampled several times without error, ultimately satisfying the CenterPoint implementation. 

 Continue reading Residual Network Block: SparseBasicBlock 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 Focus on the replace_feature in forward, replace_feature functions are located in OpenPCDet/pcdet/utils/spconv_utils.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 The replace_feature method of the class SparseConvTensor in spconv 2.0 will be called, the code is as follows: spconv/pytorch/core.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 The following is a summary of the specific calling process of backbone3d sparse convolution: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
###  HeightCompression 

 The main purpose is to convert the extracted point cloud sparse features encoded_spconv_tensor to the BEV perspective. In fact, this conversion process is very simple and crude. First, the sparse features are converted into the format of voxel features, and then the Z-axis and channels are merged to become 2D features on the BEV perspective. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 Dense () is a method that calls the class SparseConvTensor in spconv, the class SparseConvTensor is located in spconv/__init__ .py, and acts as a torch tensor that converts the output of backbone_3D sparse convolution into the shape of (batch_size, chanels, grid_nums_z, grid_nums_y, grid_nums_x) 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
##  Based on pillar 

###  DynamicPillarVFE 

 Look directly at the code and comments. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
###  PointPillarScatter 

 Turn the point cloud extracted feature to the bev perspective 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
##  BaseBEVBackbone 

 Pillar-based profiles: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 Voxel-based configuration file: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 The following is an example of the voxel parameter: 

 Use a similar (SSD) architecture to build the RPN architecture. The input to the RPN includes feature extraction from backbone3d sparse convolution intermediate spatial_features. The RPN architecture consists of three stages. Each stage starts with a downsampled convolutional layer, followed by several convolutional layers. After each convolutional layer, BatchNorm and ReLU layers are applied. Then the different downsampled features are deconvolved into feature maps of the same size, and these feature maps from different scales are spliced to build high-resolution feature maps for final detection 

 There are two downsampling branch structures in the centerpoint backbone2d part based on voxel, which corresponds to two deconvolution structures: The BEV feature map obtained by HeightCompression is: (batch_size, 128 * 2,180,180) 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 Sort out the overall network structure of backbone2d as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
##  CenterHead 

 In the nuscenes dataset, 10 categories of targets are divided into 6 categories: [['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']], each class in the network is assigned a head, corresponding to the SeparateHead class, that is, each class is assigned an MLP prediction center, center_z, dim, rot, vel, hm 

 Configuration parameters: tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint 

 pcdet/models/dense_heads/center_head.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 In the assign_target_of_single_head function, centerpoint uses Gaussian circle to calculate the label range in the heatmap, first determines the minimum Gaussian radius according to the true value GT and IOU threshold, and then generates the heatmap based on the Gaussian radius 

 How to determine the minimum Gaussian radius? According to the positional relationship between the predicted two corners and the Ground Truth corners, consider three scenarios: 

 参考：https://blog.csdn.net/x550262257/article/details/121289242 pcdet/models/model_utils/centernet_utils.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 pcdet/models/model_utils/centernet_utils.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
##  loss 

 Get the inference result of each head, then calculate the classification and regression loss in combination with the true value: 

 pcdet/models/detectors/centerpoint.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 pcdet/models/dense_heads/center_head.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 pcdet/utils/loss_utils.py 

###  FocalLoss 

 Focal loss core idea: for easy-to-distinguish samples, reduce their loss weight, while for hard-to-distinguish samples, relatively increase their weight; in this way, the model is more inclined to learn those hard-to-distinguish samples during bp, so that the overall learning efficiency is higher, and the learning will not be biased towards positive or negative samples; 

 Where: and is a hyperparameter, N is the number of positive samples in gt 

 In time: 

 In the code, 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
###  RegLoss 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
##  inference 

 generate_predicted_boxes位于pcdet/models/dense_heads/center_head.py下 

 Traverse in 6 heads, decode and output the predicted box, score, lable according to the heat map heartmap 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
 decode_bbox_from_heatmap位于pcdet/models/model_utils/centernet_utils.py下 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573726745
  ```  
#  Two-Stage 

 Use CenterPoint as the first stage. The second stage extracts additional point features from the output of the backbone network. We extract a point feature from the three-dimensional center of each face of the predicted bounding box. Note that the center of the bounding box, the center of the top and the center of the bottom are all projected onto the same point in the map view. Therefore, we only consider the four outward-facing frame faces and the predicted target center. For each point, we extract a feature from the main map view output M using bilinear interpolation. Next, we join the extracted point features and pass them through an MLP. The second stage predicts a class-agnostic confidence level score and refinement of the box on top of the prediction results of the first-level CenterPoint. 

 For the prediction of the score with the confidence level of class-agnostic, we follow and use the score target guided by the 3D IoU of the box and the corresponding ground truth bounding box: where, is the IoU between the first proposed box and the ground truth. The training is supervised by the binary cross entropy loss: where is the confidence level of the prediction, in reasoning, we directly use the single-stage CenterPoint category prediction, and compute the geometric average of the final confidence level, is the confidence level of the final predicted target,, is the confidence level of the first stage and the second stage goal t, respectively. 

 For box regression, model prediction is proposed for improvement in the first stage, and we train the model with loss. Our two-stage CenterPoint simplifies and accelerates the previous two-stage 3D detector using expensive PointNet-based feature extractors and RoIAlign operations. 

##  Architecture 

 All first-level outputs share a first 3 × 3 convolutional layer, Batch Normalization, and ReLU. Each output then uses its own two 3 × 3 convolutional branches separated by batch norm and ReLU. Our second stage uses a shared two-layer MLP, batch norm, ReLU, and dropout with a drop rate of 0.3, followed by a separate three-layer MLP for confidence level prediction and box regression. 

#  Experiments 

 Evaluate CenterPoint on the Waymo Open Dataset and nuScenes Dataset. We implement CenterPoint using two 3D encoders: VoxelNet and PointPillars, called CenterPoint-Voxel and CenterPoint-Pillar, respectively. 

 Waymo Open Dataset. The Waymo Open Dataset contains 798 training sequences and 202 validation sequences for vehicles and pedestrians. The point cloud contains 64 LIDAR channels, corresponding to 180k points per 0.1 s. The official 3D detection evaluation metrics include 3D bounding box average accuracy (mAP) and mAP weighted directional accuracy (mAPH). The mAP and mAPH are 0.7 IoU-based vehicles and 0.5 pedestrians. For 3D tracking, the official metrics are Multi-object tracking accuracy (MOTA) and Multi-object tracking accuracy (MOTP). The official evaluation kit also provides a performance breakdown for two difficulty levels: LEVEL_1 boxes containing more than 5 LIDAR points, LEVEL_2 boxes containing at least 1 LIDAR point. 

 Our Waymo model has a detection range of [-75.2m, 75.2m] for the X and Y axes, and [2m, 4m] for the Z axes. CenterPoint-Voxel uses (0.1m, 0.1m, 0.15m) voxel sizes that follow PV-RCNN, while CenterPoint-Pillar uses grid sizes (0.32m, 0.32m). 

 nuScenes Dataset.nuScenes contains 1000 driver sequences with 700, 150, and 150 sequences for training, validation, and testing, respectively. Each sequence is approximately 20 seconds long and has a lidar frequency of 20 FPS. The dataset provides calibrated vehicle attitude information for each lidar frame, but only box annotations are provided for every 10 frames (0.5s). nuScenes uses 32 channels of lidar and produces approximately 30,000 points per frame. There are 28k, 6k, 6k in total for annotation frames for training, validation, and testing. These annotations include 10 classes with a long tail distribution. The official evaluation metric is the average of the categories. For 3D detection, the main metrics are the mean mean accuracy (mAP) and nuScenes detection score (NDS). 

 Following the submission of our test set, the nuScenes team added a new Neural Programming Metric (PKL). The PKL metric is based on the KL divergence of the planner's route (using 3D detection) and the ground-truth trajectory to measure the impact of 3D object detection on the downstream autonomous driving task. Therefore, we also report PKL metrics for all methods evaluated on the test set. 

 For 3D tracking, nuScenes uses AMOTA, which penalizes ID switches, false positives, and false negatives for exceeding various recall thresholds on average. 

 For the nuScenes experiment, we set the detection range for the X and Y axes to [51.2m, 51.2m], and the Z axis to [5m, 3m]. CenterPoint-Voxel uses (0.1m, 0.1m, 0.2m) voxel sizes, and CenterPoint-Pillars uses (0.2m, 0.2m) grids. 

 Training and Inference. We use the same network design and training plan as the previous work. See Supplement for detailed hyperparameters. During the training of the two-stage CenterPoint, we randomly selected 128 boxes with a 1:1 positive-negative ratio from the predictions of the first stage. A proposal is a positive sample if it overlaps with a ground truth annotation of at least 0.55 IoU. During the inference, we run the second stage on the first 500 predictions after non-maximum suppression (NMS). The inference time is measured on an Intel Core i7 CPU and a Titan RTX GPU. 

##  Main Results 

 3D Detection We first present our 3D detection results on the test set of Waymo and nuScenes. A CenterPoint-Voxel model is used for both results. Tables 1 and 2 summarize our results. On the Waymo test set, our model achieves 71.8 level 2 mAPH of vehicle detection and 66.4 level 2 mAPH of pedestrian detection. The mAPH of vehicles and pedestrians is improved by 7.1% and 10.6% respectively compared to the previous method. On nuScenes (Table 2), our model outperforms last year's champion CBGS by 5.2% mAP and 2.2% NDS in terms of multi-scale input and multi-model integration. As shown later, our model is also much faster. The supplementary material contains a breakdown along the class. Our model showed consistent performance improvements across all categories and more significant improvements in the small category (traffic cone + 5.6 mAP) and the extreme aspect ratio category (bicycle + 6.4 mAP, construction vehicle + 7.0 mAP). More importantly, our model significantly outperformed all other submitted models under the Neural Plane Metric (PKL). After our leaderboard submission. This highlights the generalization ability of our framework. 

 ![avatar]( 83c8eddb9a0643479c4cd21605e1bab2.png) 

>  Table 1: Latest comparison of 3D detection on Waymo test set. We show mAP and mAPH for level 1 and level 2 benchmarks. 

 ![avatar]( 8e77ebb6d417455fa444f3bc02f57b7a.png) 

>  Table 2: The latest comparison of 3D detection on the nuScenes test set. We show the nuScenes detection score (NDS) and average average accuracy (mAP). 

 ![avatar]( f809f68e3a8c4200b19f6b9fec6a07d4.png) 

>  Table 3: Latest comparison of 3D tracking on Waymo test set. We show MOTA and MOTP. $\ uparrow  

          The higher the representation, the better. 

         The higher the representation, the better. 

     The higher the representation, the better, and the lower the\ downarrow $representation, the better. 

 ![avatar]( 7c225654077d47488ea9aa7d26201f52.png) 

>  Table 4: Latest comparison of 3D traces on the nuScenes test set. We show AMOTA, false positives (FP), false negatives (FN), id switches (IDS), and AMOTA for each category.  

          ↑ 

         \uparrow 

     ↑ The higher the representative, the better. 

          ↓ 

         \downarrow 

     The lower the representative, the better. 

 3D Tracking Table 3 shows the tracking performance of CenterPoint on the Waymo test set. The speed-based closest distance matching we describe in Section 4 significantly outperforms the official tracking baseline in the Waymo paper, which uses a Kalman filter-based tracker. We observe a 19.4 and 18.9 improvement in MOTAs for vehicle and pedestrian tracking, respectively. On nuScenes (Table 4), our framework outperformed the winner of the last challenge, Chiu et al., by 8.8 AMOTA. Notably, our tracking does not require a separate motion model, and the run time is negligible, 1 millisecond longer than the detection time. 

##  Ablation studies 

 ![avatar]( ed14752238254eaf8399132ecd13d2a1.png) 

>  Table 5: Comparison of anchor-based and center-based 3D detection methods in the Waymo validation set. We show the average LEVEL of 2 mAPH per class. 

 ![avatar]( 6570e89e45dc4425b8b79d7c1cd87e99.png) 

>  Table 6: Comparison of anchor-based and center-based 3D detection methods in nuScenes validation. We show average accuracy (mAP) and nuScenes detection score (NDS). 

 ![avatar]( 9a498b4775014e6ca119788e009c6d50.png) 

>  Table 7: Comparison of anchor-based and center-based methods for detecting targets at different heading angles. The ranges of rotation angles and their corresponding target parts are listed in the second and third rows. LEVEL 2 mAPH for both methods is shown in the Waymo validation set. 

 ![avatar]( 1d0b9d50ab284c4482186274ef406669.png) 

>  Table 8: The effect of target size on the performance of anchor-based and center-based methods. We show each type of LEVEL 2 mAPH for objects in different size ranges: 33% small, 33% medium, and 33% large. 

 Center-based vs Anchor-based We first compared a center-based single-stage detector with an anchor-based one of its kind. On Waymo, we followed the state-of-the-art PV-RCNN to set anchor hyperparameters: We used two anchors at each position, 0 ° and 90 °, with a positive/negative IoU threshold of 0.55/0 for the vehicle and 0.5/0 for the pedestrian. On nuScenes, we followed the anchor assignment strategy of the previous Challenge Champion CBGS. All other parameters are the same as our CenterPoint model 

 As shown in Table 5, on the Waymo dataset, simply transitioning from anchor to center, the VoxelNet and PointPillars encoders get improvements of 4.3 mAPH and 4.5 mAPH, respectively. On nuScenes (Table 6), CenterPoint boosts 3.8-4 mAP and 1.1-1 NDS through different trunks. To see where the improvements come from, we further show a breakdown of performance based on different subsets of target size and orientation angles on the Waymo validation set 

 We first divide the ground tructh instances into three bars based on their orientation angle: 0 ° to 15 °, 15 ° to 30 °, and 30 ° to 45 °. This division tests the performance of detectors to detect heavily rotated cabinets, which are critical for the safe deployment of autonomous driving. We also divide the dataset into three sections: Small, Medium, and Large, each containing 1/3 of the ground true value box. 

 Tables 7 and 8 summarize the results. Our center-based detector performs much better than the anchor-based baseline when the box rotates or deviates from the average size of the box, demonstrating the model's ability to capture rotation and size invariance when detecting targets. These results convincingly highlight the advantages of using point-based 3D target representations. 

 One-stage vs. Two-stage In Table 9, we show a comparison between single-stage and two-stage CenterPoint models using 2D CNN features in Waymo validation. Two-stage refinement with multiple central features provides a large improvement in accuracy for both 3D encoders with less overhead (6ms-7ms). We also compare with RoIAlign, which densely samples 6 × 6 points in RoI. Our center-based feature aggregation achieves similar performance, but is faster and simpler. 

 ![avatar]( 9c1ad20ee1b549e2a9517a6984a4e874.png) 

>  Table 9: Comparison of 3D LEVEL 2 mAPH for VoxelNet and PointPillars encoders using single-stage, two-stage with 3D-centered features, and two-stage with 3D-centered and surface-centered features in the Waymo validation set. 

 Voxel quantization limits the improvement of two-stage CenterPoint for PointPillars pedestrian detection, as pedestrians typically only stay within 1 pixel in the model input. In our experiments, two-stage refinement did not result in an improvement of the single-stage CenterPoint model on nuScenes. This is due in part to the sparse point cloud in nuScenes. nuScenes uses 32-channel lidar, which produces about 30,000 lidar points per frame, which is about 1/6 the number of points in the Waymo dataset. This limits the information available and the potential for two-stage improvement. Similar results were observed in the PointRCNN and PV-RCNN two-stage methods. 

 ![avatar]( 86a221fefd74478ba0c8fc0e08579b60.png) 

>  Figure 3: Example qualitative results for CenterPoint in the Waymo validation set. We show the original point cloud in blue, the objects we detected in a green bounding box, and the lidar points within the bounding box in red. 

 Effects of different feature components In our two-stage CenterPoint model, we only use features from 2D CNN feature maps. However, previous methods also propose the use of voxel features for second-stage refinement. Here, we compare two voxel feature extraction baselines 

 ![avatar]( 2a5a97df0d434649a6ce6b1f2f0a86fb.png) 

>  Table 10: An ablation study of different feature components of a two-stage refinement module. VSA stands for Voxel Set Abstraction, which is the feature aggregation method used in PV-RCNN. RBF interpolates the 3 nearest neighbors using a radial basis function. We compare bird's-eye view and 3D voxel features using LEVEL 2 mAPH in Waymo validation. 

 ![avatar]( f73d584032dd420e9782b8db217b90be.png) 

>  Table 11: The latest comparison of Waymo's validation centralized 3D detection. 

 3D Tracking. Table 12 shows the 3-D tracking ablation experiment based on nuScenes validation. We compared it with last year's Challenge winner, Chiu et al., who used a Kalman filter based on the Mahalanobis distance to correlate CBGS detection results. We broke down the evaluation into detectors and trackers, making the comparison rigorous. For the same detection targets, using simple velocity-based closest point distance matching performed better than Kalman-based Mahalanobis distance matching 3.7 AMOTA (row 1 vs. 3, row 2 vs. 4). There are two sources of improvement: 

