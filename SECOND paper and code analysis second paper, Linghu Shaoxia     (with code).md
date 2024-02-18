Paper address: https://www.mdpi.com/1424-8220/18/10/3337 Code address: https://github.com/traveller59/second.pytorch OpenPCDdet: https://github.com/open-mmlab/OpenPCDet 

#  brief introduction 

 Most existing 3D object detection methods convert point cloud data into 2D representations, such as BEV and front view representations, thus losing most of the spatial information contained in the original point cloud. In this paper, a new angle loss regression method is introduced, which successfully applies sparse convolution to a lidar-based network, and a new method for data augmentation that takes full advantage of the point cloud is proposed. Experiments on the KITTI dataset show that the proposed network outperforms other state-of-the-art methods. 

 The main contributions are as follows: 

#  network architecture 

>  The structure of the SECOND detector. The detector takes the raw point cloud as input, converts it into voxel features and coordinates, and applies two VFE (voxel feature coding) layers and a linear layer. Then a sparse CNN is applied. Finally, the RPN generates the detection. 

 The similarities and differences between SECOND and VoxelNet network structures 

 ![avatar]( 738e3dcafe5c44a0b0da8346381d561b.png) 

>  Note: The point cloud feature extraction VFE module in VoxelNet has been replaced in the author's latest implementation; because the original VFE operation speed is too slow and unfriendly to video memory. For details, please check this issue: 

###  point cloud grouping 

 The method of turning a point cloud into a Voxel is the same as in VoxelNet. First, pre-allocate buffers according to the specified limit of the number of voxels; then, traverse the point cloud and assign these points to their associated voxels, and save the voxel coordinates and the number of points for each voxel. During the iteration process, check the existence of voxels according to the hash table. If the voxels related to a certain point do not yet exist, set the corresponding value in the hash table; otherwise, add one to the number of voxels, and the iteration process will stop once the number of voxels reaches the specified limit. 

 After grouping the point cloud data, three pieces of data are obtained: 

>  For cars and other objects,  

            z 

            y 

            x 

           z y x 

       The range of the zyx axis point cloud is [0, -40, -3, 70.4, 40, 1] for pedestrian and cyclist detection, 

            z 

            y 

            x 

           z y x 

       The range of the zyx axis point cloud is [0, -20, -3, 70.4, 20, 1] for our smaller model, 

            z 

            y 

            x 

           z y x 

       The range of the point cloud on the zyx axis is [0, -32, -3, 52.8, 32, 1] 

 The cropped area needs to be fine-tuned according to the voxel size to ensure that the size of the generated feature map can be properly downsampled in subsequent networks. Voxel size. The maximum number of points per voxel detected by the car is set to T = 35 and for pedestrians and cyclists to T = 45, since pedestrians and cyclists are relatively small and more points are required for voxel feature extraction. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
###  Voxelwise Feature Extractor 

 The paper uses the Voxelwise Feature Extractor (VFE) layer to extract voxel features, and the VFE module is the same as in VoxelNet. 

 All points in the same voxel will be used as input through the fully connected network FCN (each fully connected layer consists of linear layer, batch normalization (BN) and linear unit (ReLU.) mapped to a high-dimensional feature space. After obtaining the point-wise feature, then use element-wise MaxPooling to obtain the local aggregation features of voxel. Finally, the local aggregation features and point-wise features are spliced to obtain the output feature set. The voxel feature extractor consists of 2 VFE layers and one FCN layer. 

>  The original VFE operation speed is too slow and unfriendly to video memory. In the new implementation, the original Stacked Voxel Feature Encoding is removed, and the average value of each voxel inner point is directly calculated as the feature of this voxel; the calculation speed is greatly improved, and good detection results are also achieved. The dimensional transformation of the voxel feature is (Batch * 16000, 5,4 ) --> ( Batch * 16000, 4) 

 OpenPcdet implementation code: pcdet/models/backbones_3d/vfe/mean_vfe.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
###  Sparse Convolutional Intermediate Extractor 

 Review of Sparse Convolutional Networks 

 Reference. [25] is the first paper to introduce spatial sparse convolution. In this method, no output points are computed if there are no associated input points. This method provides a computational advantage in LiDAR-based detection, as the grouping step of the point cloud in KITTI will generate 5k-8k voxels with a sparsity close to 0.005. As an alternative to normal sparse convolution, submanifold convolution restricts the output position to being active if and only if the corresponding input position is active. This avoids generating an excessive number of active positions, which can lead to a decrease in the speed of subsequent convolutional layers due to an excessive number of active points. 

 sparse convolution algorithm 

 Reference: https://zhuanlan.zhihu.com/p/97367489 Let's first consider the two-dimensional dense convolution algorithm. We use, to represent the filtered element, using, to represent the image element, where, and is the spatial position index, to represent the input channel, to represent the output channel. Given the output position, the function generates the input position that needs to be calculated. Therefore, the convolution output of is given by the following formula: 

 Where: Yes Yes Output spatial index, and, representation, and coordinate kernel-offset. Algorithms based on General Matrix Multiplication (GEMM) (also known as im2col-based algorithms) can be used to get all the data needed to build a matrix, and then execute GEMM itself: 

 Where $W_ {¶, l, m} corresponds to, but in GEMM form. For sparse data, and associated output, the direct calculation algorithm can be written as: 

 Where: is a function to obtain the input index and filter offset, and the subscript k is the 1D kernel offset, corresponding to the "and" subscript in equation (1), corresponding to the "and" in equation (1). The GEMM-based version of equation (3) is given by the following equation: 

 The aggregation matrix of sparse data still contains many that do not require zero computation. To solve this problem, we do not directly convert equation (3) to equation (4) as follows: 

 Where, also known as Rule, is a matrix that specifies the input index i for a given kernel offset k and output index j. The inner sum in equation (5) cannot be computed by GEMM, so we need to take the necessary inputs to construct the matrix, perform GEMM, and then spread the data back out. In practice, we can get the data directly from the raw sparse data by using a pre-built I-O index rule matrix. This increases the speed. Specifically, we construct a table of rule matrices with dimensions of, where K is the kernel size (expressed as volume), is, the number of input features, is the input/output index. Element, which stores the input index for collection, and element, which stores the output index for scattering. The top of Figure 2 shows our proposed algorithm. 

>  Figure 2 The sparse convolution algorithm is depicted in the figure above, and the GPU rule generation algorithm is depicted in the figure below. 

            N 

             i 

             n 

          N_{in} 

      Nin represents the number of input features. 

            N 

             o 

             u 

             t 

          N_{out} 

      Nout represents the number of output features. N is the number of features collected. Rule is the rule matrix, where Rule [i,:,:] is the i-th rule corresponding to the i-th kernel matrix in the convolution kernel. Color boxes except white represent points with sparse data, and white boxes represent empty points. 

 Rule Generation Algorithm 

 The main performance challenge facing the current implementation is related to the rule generation algorithm. CPU-based rule generation algorithms that use hash tables are typically used, but such algorithms are slower and require data to be transferred between the CPU and GPU. A more straightforward approach to rule generation is to traverse the input points to find the output associated with each input point and store the corresponding index into the rule. During the iteration, a table is required to check the presence of each output location to decide whether to use a global output index counter to accumulate data. This is the biggest challenge that hinders algorithms from using parallel computing. 

 We designed a GPU-based rule generation algorithm (Algorithm 1) that runs faster on GPUs. The bottom of Figure 1 shows our proposed algorithm. 

 ![avatar]( 2fc0a07d13744301943e8725dcf2d445.png) 

 Table 1 shows the performance comparison between our implementation and the existing approach. 

>  SparseConvNet is the official implementation of submanifold convolution 

 代码在：pcdet/models/backbones_3d/spconv_backbone.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
 Where blocks are built for sparse convolution: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
 Sparse Convolutional Intermediate Extractor 

 The intermediate extractor is used to learn information about the z-axis and convert sparse 3D data into a 2D BEV image. Figure 3 shows the structure of the intermediate extractor. It consists of two stages of sparse convolution. Each stage contains several submanifold convolution layers and a normal sparse convolution to perform downsampling on the z-axis. After the z-dimension is downsampled to either one or two dimensions, the sparse data is converted into a dense feature map. Then, the data is simply reshaped into 2D data similar to the image. 

 ![avatar]( 67fd2da5340a4de6a2a9153df953d462.png) 

>  The structure of the sparse intermediate feature extractor. The yellow boxes represent sparse convolution, the white boxes represent submanifold convolution, and the red boxes represent sparse to dense layers. The upper half of the graph shows the spatial dimensions of the sparse data. 

 Since the previous tensor obtained by VoxelBackBone8x is sparse, the data is: [batch_size, 128, [2,200,176]] 

 Here, it is necessary to convert the original sparse data into dense data; at the same time, the resulting dense data is stacked in the Z-axis direction, because in the KITTI dataset, no objects will coincide in the Z-axis; the advantages of doing so are: 

 The final BEV characteristic map is: (batch_size, 128 * 2,200,176)  

 代码在pcdet/models/backbones_2d/map_to_bev/height_compression.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
###  RPN 

 The Similar to (SSD) architecture is used to build the RPN architecture. The input to the RPN consists of a feature map from a sparse convolutional intermediate extractor. The RPN architecture consists of three stages. Each stage starts with a downsampled convolutional layer, followed by several convolutional layers. After each convolutional layer, the BatchNorm and ReLU layers are applied. We then upsample the output of each stage as feature maps of the same size and join these into one feature map. Finally, three 1 × 1 convolutions are applied to predict the class, regression offset, and direction. 

 If there are two downsampling branch structures in SECOND, there are two deconvolution structures: The BEV feature map obtained by HeightCompression is: (batch_size, 128 * 2,200,176) 

 The feature map dimension where the structure is finally spliced in the channel dimension: (batch, 256 * 2, 200, 176) 

 代码在：pcdet/models/backbones_2d/base_bev_backbone.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
###  Anchors and Targets 

 Each anchor is assigned a one-hot vector for classification targets, a 7-vector box regression target, and a one-hot vector for directional classification targets. Different classes have different match and mismatch thresholds. For cars, anchors are assigned to ground-truth objects using a cross-parallel join (IoU) threshold of 0.6, and if their IoU is less than 0.45, the anchor is assigned to the background (negative value). Anchors with IoU between 0.45 and 0.6 are ignored during training. 

 Anchor generation 

 With fixed-size anchors, anchors are determined based on the average of the sizes and center positions of all ground truths in the KITTI training set, with two directional angles of 0 and 90 degrees for each class of anchors. 

 pcdet/models/dense_heads/target_assigner/anchor_generator.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
 Target assignment 

 Process the anchors and gt_boxes of all point clouds in a batch of data, classify and calculate whether each anchor belongs to the foreground or the background, assign categories to each foreground anchor and calculate the regression residuals and regression weights of the box, unlike computing iou as a whole in SSD and taking the maximum 

 ![avatar]( 76508364b039434eab13275d4d4b8624.png) 

 For the regression objective, we use the following box encoding function: pcdet/models/dense_heads/target_assigner/axis_aligned_target_assigner.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
 pcdet/utils/box_coder_utils.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
###  Loss 

 The final form of the total loss function is as follows: 

  Where, is the classification loss, is the regression loss of position and dimension, is the new angle loss, is the direction classification loss. = 1.0, = 2.0 and = 0.2 are the constant coefficients of the loss formula, using relatively small values to avoid situations where the network has difficulty recognizing the orientation of objects. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
 Focal Loss for Classification 

 To address the imbalance between foreground and background classes in the sample, the authors introduce focal loss, which takes the form of: 

 Where, is the estimated probability of the model, and α and γ are the parameters of focal loss. Use = 0.25 and = 2 during training. 

 pcdet/models/dense_heads/anchor_head_template.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
 The focal loss class SigmoidFocalClassificationLoss code is in: pcdet/utils/loss_utils.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
>  Previous angular regression methods, including angular coding, direct coding, and vector coding, have generally performed poorly. Corner prediction methods cannot determine the orientation of an object, nor can they be used for pedestrian detection because BEV boxes are nearly square. Vector coding methods retain redundant information, making it difficult to detect distant objects based on LiDAR. VoxelNet directly predicts radian offset, but encounters adversarial example problems in the case of 0 and π radians because the two angles correspond to the same box. The authors introduce a new angle loss regression to solve the problem 

 In VoxelNet, a 3D BBox is modeled as a 7-dimensional vector representation. During the training process, the Smooth L1 loss is used for regression training of these 7 variables. When the prediction direction of the same 3D detection box is exactly the opposite of the true direction, the regression loss of the first 6 variables of the above 7 variables is small, and the regression loss of the last direction will be large, which is actually not conducive to model training. To solve this problem, the author introduces the sine error loss of angular regression, which is defined as follows: 

 Is the predicted direction angle, is the true direction angle. Then, when the difference between and, the loss tends to 0, which is more conducive to model training. 

 In this case, the predicted direction of the model is likely to differ by 180 degrees from the true direction. To solve the problem of treating boxes with opposite directions as the same, a simple direction classifier is added to the output of the RPN, which uses the softmax loss function. Yaw rotation around the z-axis is higher than 0, the result is positive; otherwise it is negative. 

 pcdet/models/dense_heads/anchor_head_template.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
 add_sin_difference function: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
 WeightsSmoothL1 loss function, the code is in: pcdet/utils/loss_utils.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
###  data augmentation 

 The main problem encountered in the training process of Sample Ground Truths from the Database is that there are too few ground truths, which limits the convergence speed and final performance of the network. 

 SECOND proposed sampling and intercepting GT to generate GT Database, which has been used in many subsequent networks. 

 ![avatar]( 3a246837d14542a68d65cd3904186bc3.png) 

 This method speeds up the convergence speed of the network and improves the accuracy of the network. The comparison chart is as follows:  

 Object Noise 

 To account for noise, we adopt the same approach as used in VoxelNet, where each GT and its point clouds are independently and randomly transformed, rather than transforming all point clouds using the same parameters. Second uses random rotations sampled from a uniform distribution ∆θ ∈ [− π/ 2, π/2] and random linear transformations sampled from a Gaussian distribution with a mean of 0 and a standard deviation of 1.0. 

 Global rotation and scaling 

 Second applies global scaling and rotation to the entire point cloud and all real boxes. The scaling noise is taken from the uniform distribution [0.95, 1.05 ]，[− π/ 4, π/4] for global rotation noise. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573794057
  ```  
##  Ablation study 

 ![avatar]( 51d3b1de47ac4ec6bb079a585ce8d719.png) 

 Vector is the implementation of angle encoding in AVOD. 

