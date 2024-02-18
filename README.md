 ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573771443
  ```  
 ![avatar]( 2019091220432046.png) 

 Reference: PCL Classic Code Appreciation 4: Point Cloud Filter PCL Official Website: http://docs.pointclouds.org/trunk/group__filters.html 



--------------------------------------------------------------------------------

Topic: Convert RGB + depth images taken by a given 3-frame (discontinuous) RGB-D camera, and the transformation matrix between them into a point cloud, and fuse the final point cloud output 

 pointCloudFusion.cpp 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573728584
  ```  
 slamBase.cpp 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573728584
  ```  
 slamBase.hpp 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573728584
  ```  


--------------------------------------------------------------------------------

Reference: https://mp.weixin.qq.com/s/FfHkVY-lmlOSf4jKoZqjEA 

#  What is a grid? 

 Mesh is mainly used in computer graphics, there are many kinds of triangle, quadrangular mesh, etc. Most of the mesh processing in computer graphics is based on triangular mesh. Triangular mesh is widely used in graphics and 3D modeling to simulate the surface of complex objects, such as buildings, vehicles, animals, etc. You can see that the rabbit, ball and other models in the picture below are based on triangular mesh 

 Triangles represent meshes, also known as triangulation. It has the following advantages: 

 3. Helps to restore the surface details of the model. 

#  Mesh generation algorithm requirements 

 The grid generation algorithm has the following capabilities: 

 At present, point cloud mesh generation is generally divided into two categories: 

#  Principle of point cloud greedy triangulation 

 This paper mainly introduces a relatively simple greedy triangulation method (corresponding class name: pcl :: GreedyProjectionTriangulation), that is, using the greedy projection triangulation algorithm to triangulate directed point clouds. Advantages: It can be used to deal with scattered point clouds that are scanned from one or more devices and have multiple connections. However, it also has great limitations. It is more suitable for sampling point clouds from continuous smooth surfaces, and the density change of point clouds is relatively uniform 

 The approximate process of greedy projection triangulation is as follows: 

#  Introduction to Delaunay Triangulation 

 For numerical analysis and graphics, triangulation is a very important preprocessing technique. And Delaunay triangulation is a commonly used triangulation method. Many geometric diagrams of point sets are related to Delaunay triangulation, such as Voronoi diagrams. 

 Delaunay triangulation has two advantages: 

 Definition: The Delaunay triangulation of a point set satisfies that any point in any? is not within the circumscribed circle of any triangular facet in?. 

 ![avatar]( 20190913154158519.png) 

 Looking at the graph below is to satisfy the Delaunay condition. Are the vertices of all triangles not within the circumcircle of other triangles? A simpler way. You look at the leftmost graph below and observe the two triangles ABD and BCD with common edges BD. If the sum of angles α and γ is less than or equal to 180 °, then the triangle satisfies the Delaunay condition. According to this standard, the left and both of the lower graphs do not satisfy the Delaunay condition, only the right graph satisfies.  

#  Greedy projection triangulation 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573762899
  ```  
#  practice 

 Operation steps: downsampling and filtering, resampling smoothing, normal calculation, greedy projection gridding 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573762899
  ```  
 ![avatar]( 20190913161515304.png) 



--------------------------------------------------------------------------------

#  Point cloud smoothing through resampling 

 Cases that require smoothing: 

 Point cloud resampling, we actually use a method called "Moving Least Squares" (MLS, Moving Least Squares), the corresponding class name is: pcl :: Moving Least Squares (official website) 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573749440
  ```  
 Kd-Tree is a data structure, a special case of spatial binary tree, which can be easily used for range search. Here, KD-Tree is used to facilitate management and search point clouds. This structure can easily find nearest neighbors. 

#  Estimating the surface normal of a point cloud 

 There are generally two methods for computing the normal of a point cloud: 

 The second method is used to approximate the surface normal of each point in the point cloud. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573749440
  ```  
#  practice 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573749440
  ```  
 ![avatar]( 20190913151616123.png) 

 After Smoothing: Smoothing and Normal Estimation: Reference: https://mp.weixin.qq.com/s/GFDWOudJ08In6jFyrZ7hhg 



--------------------------------------------------------------------------------

This section mainly introduces the theoretical basis of convolution. Combined with the analysis of spconv code, the second section starts to introduce. This section introduces the basic theory of 2D and 3D convolution and the classification of sparse convolution, and then introduces the working principle of 3d sparse convolution in detail. 

#  2D convolution 

 2D convolution: The convolution kernel slides a window in the two-dimensional space of the input image 

 2D single channel convolution 

 ![avatar]( eae6a6d68eda4918a2cbc4bd18084378.gif) 

 For 2-dimensional convolution, a 3 * 3 convolution kernel is used to convolution on a single-channel image, resulting in the following moving image:  

 An image is scanned using a convolutional kernel, resulting in a feature map. The "scanned image" here is a channel, not a color image. 

 2D multichannel convolution 

 In one scan, we input a color image with three channels. For this image, three convolutional kernels with the same size but different specific values are scanned on three channels, resulting in three corresponding "new channels". Since the structure of different channels in the same image must be the same, the size of the convolutional kernels is also the same, so the size of the "new channel" obtained after the convolution operation is also the same. 

 ![avatar]( cce9762432734e0b9c29c4debf30795e.gif) 

 After deriving the three "new channels", we add the elements at the corresponding locations to form a new map. This is the first feature map of the three-color image input by the convolutional layer, as shown in the following figure: 

 ![avatar]( 773ee4daeef645bca1f0537a97beb700.gif) 

 This operation is consistent for three-channel RGB images, four-channel RGBA or CYMK images. However, if it is a four-channel image, there will be four convolutional kernels of the same size but different values scanning 4 channels respectively. 

 Therefore, in a scan, no matter how many channels the image itself has, the convolutional kernel will scan all the channels and add the scan results to a feature map. Therefore, a scan corresponds to a feature map, regardless of the number of channels in the original image, so out_channels is the number of scans, the number of convolutional kernels is equal to the number of channels entered in_channels x the number of scans out_channels. 

 Calculation of 2D convolution: 

 Input layer: (the number of input channels) 

 Hyperparameters: 

 Output layer:, the parameter relationship between the output layer and the input layer: 

 Consider the amount of bias parameters as follows: 

 Consider the amount of bias calculation (number of multiplications): 

 FLOPS (floating-point arithmetic): 

 Consider the bias: 

 Without considering the bias: 

>  Multiplication and addition are separated. The addition operation is subtracted by 1 because n numbers are added. Considering the bias, it is added by 1. 

#  3D convolution 

 3D single channel 

 For 3D convolution, a 333 convolution kernel is convolution on a cube, resulting in the following output: 

 ![avatar]( 1633362371da4841a6aea1411f66844a.gif) 

 Attention: Is there any difference between 3D single channel and 2D multi-channel? 

>  Yes. The parameters of the convolution kernels on different channels of 2D multi-channel convolution are different, and the convolution kernels of 3D convolution are themselves 3D. The parameters of a 3D convolution kernel of a 3D single channel are weighted and shared across the entire image. 3D convolution kernels have one more depth dimension than 2D convolution kernels. This depth may be successive frames on a video or different slices in a stereoscopic image. 

 3D multichannel 

 Similar to 2D multichannel, for example, for 3D multichannel, input an image of size (3, depth, height, width). 

 For this graph, three 3D convolution kernels of the same size but with different specific values will be scanned on each of the three channels, resulting in three corresponding "new channels". After the three "new channels" are obtained, we add the elements at the corresponding locations to form a new graph. 

 ![avatar]( d5d19264d09540ff90036d7883f12736.png) 

 Calculation of 3D convolution: 

 Input layer: (the number of input channels) 

 Hyperparameters: 

 Output layer:, the parameter relationship between the output layer and the input layer: 

 The parameters are: 

#  Sparse Convolutional Classification 

 Two convolutional operations: SC and VSC 

###  SC(Sparse Convolution) 

 The sparse convolution operation, sparse convolution, can be expressed as SC (m, n, f, s), where m is the number of input feature channels for input feature planes; n is the number of output feature channels; f is the filter size; s is the stride. 

 Where f can be a non-square filter. The traditional filter size is generally 3x3, 5x5... Here it can be 1x7... 

 Calculate active site: site can be understood as pixels in the image and points in the point cloud. For the first layer, an active site is a pixel or point with data. For the later layer, if there is an active site in the sensory field of this site, then this site is called an active site. SC convolution computes active site like traditional convolution 

 Calculate the non-active site: Set the ground state of the site (the input value of the first layer) directly to 0, and then the output is also 0, so that the non-active site is rather discarded. The input of each layer of SC convolution is a non-active site, and the output of this layer is still a non-active site. This is different from the traditional convolution operation. 

###  VSC(valid sparse convolution) 

 Valid Sparse Convolution can be expressed as VSC (m, n, f, 1), a variant of the previously mentioned SC (m, n, f, 1). 

>  Convolutional networks built using SV or VSC use components such as activation functions, BN layers, and pooling

For activation functions and BN: operations are only used for active sites. For avg pooling: take the sum of the active input vectors divided by the size of the receptive field, regardless of the non-active site. For max-pooling: take the maximum value in the receptive field 

#  The principle of sparse convolution 

 Convolutional neural networks have been shown to be very efficient for two-dimensional image signal processing. However, for three-dimensional point cloud signals, the extra dimension z increases the computational effort significantly. 

 On the other hand, unlike ordinary images, the voxels of most 3D point clouds are empty, which makes the point cloud data in 3D voxels usually sparse signals. 

###  构建 Input Hash Table 和 Output Hash Table 

 ![avatar]( 3005c825c5424ea89ba7c533bc6907e4.png) 

 The input hash table and output hash table correspond to the Hash_in, and Hash_out in the figure above. For Hash_in: v_in is the subscript, key_ in indicates the position of value in the input matrix. 

 The current input consists of two elements, P1 and P2, with P1 at the (2,1) position of the input matrxi and P2 at the (3,2) position of the input matrix, in YX order. 

 Here only the position of p1 is recorded, regardless of the number represented by p1, the input hash table is named input position hash table. 

 The input hash tabel is built, and then the output hash table is built. Use a kernel to perform convolution operations: 

 ![avatar]( fc0684edefef428095819289d3d25bc6.png) 

  However, not every time the convolutional kernel can just touch P1. So, from the 7th time, the output matrix does not change. Then record the position of each element. The above is just the operation P1, of course P2 is the same operation.  

###  Build GetOffset () 

 ![avatar]( 052549e3f01a4562a28fcb8c2cb69035.png) 

 As shown in the figure below, taking the (0,0) position in the output as an example, the value of this point is obtained by convolution from the window in the upper left corner of the input. In this window, only the right P1 position is non-zero, and the rest of the positions are zero. Then this convolution operation only needs to be calculated by the convolution weight of this position and the input value. The position corresponding to the position in the convolution kernel of P1 is (1,0). We put this (1,0) into the GetOffset () result.  

###  Building Rulebook 

 ![avatar]( 6051e57442524b1b890c616643277c2c.png) 

 Each row of the rulebook is an atomic operation, the first column of the rulebook is an index, the second column is a counter count, v_in and v_ out are the index of the input hash table and the index of the output hash tabel of the atomic operation, respectively. (Yes, so far, it is still index, and no real data is used.) 

 ![avatar]( 92057e1be99e4a4bb86125915f35e436.png) 

 The sparse convolution process is:  

 ![avatar]( d73d334a6a5f4cbcb197a646588f42cd.png) 

 The red arrows handle the first atomic operation in rulebook. From rulebook, we know that this atomic operation has input from P1 at input index (v_in) = 0 (2,1), and output index (v_out) = 5 (2,1). For the (0.1, 0.1, 0.1) represented by p1, the convolution operation is performed with the dark and light kernels respectively, resulting in the output of two channels of dark yellow and light yellow. 

 Traditional convolution is implemented by img2col, and sparse convolution is implemented by Rulebook. Essentially, it is a table. First, by establishing the input and output hash tables, the tensor coordinates of the input and output are mapped to the ordinal numbers. Then the ordinal numbers in the input and output hash tables are connected, so that sparse convolution can be basically implemented, so this is also the key to the implementation of sparse convolution. The step of establishing rulebook in the spconv library code calls the Python function get_indice_pairs, which further calls the function getIndicePairs in the spconv shared module to complete it step by step. 

 Reference: 



--------------------------------------------------------------------------------

This paper analyzes the sparse convolutional source code in the spconv library based on the CeneterPoint algorithm in the OpenPCDet framework: 

 First look at pcdet/models/backbones_3d/spconv_backbone.py under OpenPCDet 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573758524
  ```  
 Continue reading: pcdet/utils/spconv_utils.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573758524
  ```  
 Import spconv is to import the installed spconv package. There is a __init__ .py file in the package directory, and the executable code in __init__ .py will be executed when importing spconv 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573758524
  ```  
 After importing spconv, you can directly use spconv. SubMConv3d, spconv. SparseConv3d, spconv. SparseConvTensor, spconv. SparseSequential and other sub-modules 

>  __init__ is the identifier of package in Python, which defines the properties and methods of the package __

One of the main functions of the __init__ file is to turn the folder into a Python module also known as a package. In the package of each module in Python, there is a __init__ .py file, and this file cannot be deleted, otherwise the folder will no longer be considered a module. The __init__ .py file defines the properties and methods of the package. In fact, it can be nothing defined; it can just be an empty file, but it must exist. If __init__ .py does not exist, the directory is just a directory, not a package, and it cannot be imported or contain other modules and nested packages. 

 The core data structure of 3d sparse convolution is defined in __init__ 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573758524
  ```  
 SparseConvTensor, but not a torch tensor per se, is just an abstraction of sparse Tensor. Its internal members features, indices, and spatial_shape represent valid feature data, valid voxel grid coordinates (i.e. voxel spatial indices), and spatial shape sizes, respectively. 

 At the same time, in __init__, you can load libspconv.so dynamic library through torch.ops.load_library, so that you can call the functions implemented by C++/CUDA through the python interface registered by src/spconv/all.c, which will be described in detail later. 

 This article is based on the nuscenes dataset, CenterPoint configuration parameters are as follows: tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint .yaml, several main configuration parameters are as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573758524
  ```  
 Looking at the code in the Centerpoint BACKBONE_3D section of OpenPCDet, the parameters noted below are derived from the nuscenes dataset: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573758524
  ```  
 Features and indices have shapes of [N, 5] and [N, 4], respectively. Where N represents the number of valid voxels. The 4 of indices represents batch_id, z, y, x, and batch_id represents the index of batch_size, starting from 0. spatial_shape POINT_CLOUD_RANGE and VOXEL_SIZE, and the Z-axis plus 1 is [41, 1440, 1440] 

 Take a look at the following line of code: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573758524
  ```  
 Why do you need to add 1 to the Z axis of sparse_shape? 

>  Reference: https://github.com/open-mmlab/mmdetection3d/issues/282

SparseEncoder will downsample in the high dimension. Therefore, this parameter allows the height dimension to be downsampled several times without error, and ultimately satisfies the implementation of CenterPoint. 

 The SparseSequential-like code is located at: spconv/modules.py. The SparseSequential class is responsible for building sparse convolutional sequences, similar to nn.sequential in pytorch. 

 Class inheritance relationship: nn. Module - > SparseModule - > SparseSequential 

 Next, let's look at the sparse convolution SubMConv3d and SparseConv3d's parent class, SparseConvolution. 

##  SparseConvolution 

 Both SubMConv3d and SparseConv3d in spconv/conv.py inherit from SparseConvolution. SubMConv3d and SparseConv3d are mainly called at initialization, and the forward of SparseConvolution is responsible for scheduling the entire sparse convolution. 

 The following code involves some specific parameters labeled with the input parameters of the first layer convolution conv_input. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573758524
  ```  
 When non-submanifold convolution (ordinary sparse convolution) is not transposed: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573758524
  ```  
 out_shape obtained by get_conv_output_size, the output size is:  

 Look at this line of code: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573758524
  ```  
 indice_key role: 

 find_indice_pair functions are located in spconv/__init__ 

  Input is SparseConvTensor, because submconv3d does not change the input and output position index and output feature map space shape, if the outids, indice_pairs, indice_pair_num, out_spatial_shape of this layer are the same as the previously calculated layer, stored here in a dictionary, used directly later to avoid repeated computation 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573758524
  ```  
 In the first build, the indice_key is empty, only the three blocks in the spconv. SparseSequential stack, the last spconv. SubMConv3d can reuse the indice_key of the second spconv. SubMConv3d, as shown in the following code: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573758524
  ```  
 The forward function input of SparseConvolution must be a custom SparseConvTensor type defined in a spconv. The two most important steps of sparse convolution are completed in forward: 

 3D sparse standard sparse convolution and 3D subpopular sparse convolution are defined by two classes, SparseConv3d and SubMConv3d, respectively. Both classes are derived from SparseConvolution. Its input parameter subm is used to distinguish between standard 3d sparse convolution and 3d subpopular sparse convolution. 



--------------------------------------------------------------------------------

The following describes the implementation of specific sparse convolution calculations based on the built Rulebook, continue to look at the class SparseConvolution, the code is located at: spconv/conv.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573783518
  ```  
  Fsp.indice_subm_conv and Fsp.indice_conv will continue to call functions such as indice_conv in the spconv/ops.py module after passing through the SubMConvFunction and SparseConvFunction objects in the spconv/functional.py. 

 First look at the interface of sub-streamline convolution: indice_subm_conv, code: spconv/functional.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573783518
  ```  
 Calling C++ function directly with the Python interface is not very "beautiful", here torch.autograd. Function is used to encapsulate the low-level call of the operator. 

 The Function class itself represents a differentiable function of PyTorch. As long as an implementation of forward inference and backpropagation is defined for it, we can use it as a normal PyTorch function. PyTorch will automatically schedule the function to perform forward and reverse computations appropriately. 

>  Extension: For model deployment, the Function class has a nice property: if it defines symbolic static methods, the Function can be converted into ONNX operators according to the rules defined in symbolic when executing torch.onnx.export () 

 Apply is a method of torch.autograd. Function, this method completes the scheduling of Function in forward inference or backpropagation, using indice_subm_conv = SubMConvFunction.apply to take a shorter alias indice_subm_conv, later when using indice_subm_conv operator, we can ignore the implementation details of SubMConvFunction and only access the operator through indice_subm_conv this interface. 

 The forward propagation of SubMConvFunction forward calls the indice_conv method of spconv/ops.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573783518
  ```  
 The underlying C ++ APIs are registered in the src/spconv/all.cc file via the OP Register provided by Pytorch 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573783518
  ```  
 Call the indiceConv function in the src/spconv/spconv_ops.cc file by loading the .so file load_library, torch.ops.spconf.indice_conv 

 Let's take a look at the indiceConv function of src/spconv/spconv_ops.cc: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573783518
  ```  
 Ghostwriting!! 



--------------------------------------------------------------------------------

###  Ordinary 3d sparse convolutional RuleBook construction 

 We continue to look at the establishment process of ordinary sparse convolution RuleBook, returning src/spconv/spconv_ops.cc, and look at the ordinary 3D sparse convolution part of the getIndicePairs function 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573790407
  ```  
 Ordinary 3d sparse convolution calls create_conv_indice_pair_p1_cuda and create_conv_indice_pair_p2_cuda, we first look at create_conv_indice_pair_p1_cuda function, located in src/spconv/indice.cu 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573790407
  ```  
 Focus on prepareIndicePairsKernel kernel function 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573790407
  ```  
 getValidOutPos calculates the position of the output hash table and the convolution kernel weight used in the output according to the input point, and returns the number of valid outputs 

 Look directly at the following code, the comments are more detailed 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573790407
  ```  
 Regarding how to obtain the upper and lower limits of the output, the calculation process is as follows: 

 Take 1-dim convolution as an example: the output of a given input point depends on the kernel size k, step size s, expansion d, and padding p. 

 For the input position x, its distance from the boundary of the feature map is: Assuming the minimum value of the output point is n, there is the following relationship: where is the effective kernel size, which depends on the kernel size and bloat: 

 The equation becomes: rearrange, calculate lowers as: Similarly, assuming the maximum value of the output point is n, there is the following relationship: then calculate uppers as: Reference: https://github.com/traveller59/spconv/issues/224 

 For the meaning of the counter variable, please refer to the comment code. If there is any misunderstanding, please point it out. 

 create_conv_index_pair_p2_cuda sentire:src/spconv/indice.cu 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573790407
  ```  
 assignGridAndIndiceOutKernel位于:include/spconv/indice.cu.h 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573790407
  ```  
 rowArrayIdxInv位于：include/tensorview/tensorview.h 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573790407
  ```  
 Keep watching assignIndicePairsKernel: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573790407
  ```  
 The subview is located at: include/tensorview/tensorview.h, which means that the subset should be obtained 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573790407
  ```  


--------------------------------------------------------------------------------

##  Building Rulebook 

 Below is ops.get_indice_pairs, located at: spconv/ops.py 

 Building a Rulebook is done by the get_indice_pairs interface 

 get_indice_pairs function specific implementation: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 The main thing is to complete the checksum preprocessing of some parameters. First, for 3d ordinary sparse convolution, the output shape is calculated according to the input shape size, kernel size, stride and other parameters. Subpopular sparse convolution does not need to be calculated. The output shape is the same size as the input shape 

 After preparing the parameters, enter the core get_indice_pairs function. Because spconv is registered by torch.ops.load_library loading .so file, so here torch.ops.spconf.get_indice_pairs this way to call the function. 

 Operator registration: In the src/spconv/all.cc file, the underlying c ++ api is registered through the OP Register provided by Pytorch, and the c ++ operator can be called in the form of python interface 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
>  Like C++ extension method, OP Register is also a low-level extension operator registration method provided by Pytorch. Registered operators can be called in torch.xxx or tensor.xxx ways, which is also decoupled from the pytorch source code. Adding and modifying operators does not require recompiling the pytorch source code. To register a new operator in this way, the process is very simple: first write C++ relevant operator implementation, and then register the operator through the underlying pytorch registration interface (torch :: Register Operators). 

 Build Rulebook actually calls the getIndicePairs function of src/spconv/spconv_ops.cc file type get_indice_pairs python interface 

 The code is located at: src/spconv/spconv_ops.cc 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 The analysis of getIndicePairs directly locks the center of gravity in the GPU logic part, and the subpopular 3d sparse convolution and normal 3d sparse convolution are discussed separately, and the preferable subpopular 3d sparse convolution. 

 The three most important variables in the code are indicePairs, indiceNum, and gridOut, which are created as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 IndicePairs represents the mapping rules for sparse convolutional input and output, namely Input Hash Table and Output Hash Table. Here the theoretical maximum memory is allocated, its shape is {2, kernelVolume, numAct}, 2 represents the input and output directions, and kernelVolume is the volume size of the convolutional kernel. For example, a 3x3x3 convolutional kernel has a volume size of 27 (3 * 3 * 3). NumAct represents the number of active features input. indiceNum is used to store the total number of calculations at each position of the convolutional kernel, and indiceNum corresponds to count in the picture 

 ![avatar]( 2d52c14b2b4b4b8ba3fecdf745d49cbf.png) 

 Code about gpu to establish rulebook call create_submconv_indice_pair_cuda function to complete, the following specific analysis create_submconv_indice_pair_cuda function 

###  Substreamline sparse convolution 

 Substreamline sparse convolution is called create_submconv_indice_pair_cuda function to build a rulebook 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 In create_submconv_indice_pair_cuda there is no need to delve into the operating principle of the following dynamic distribution mechanism. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 Lock the center of gravity directly to the kernel function: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 prepareSubMGridKernel grid_size and block_size in the kernel function are actually integer variables. block_size is tv :: cuda :: CUDA_NUM_THREADS, defined in include/tensorview/cuda_utils, with a size of 1024. The grid_size size is calculated by tv :: cuda :: get Blocks (numActIn), where numActIn represents the number of active input data. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 prepareSubMGridKernel function: create a hash table between the output tensor coordinates (represented by index) and the output sequence number 

 Õ: include / spconv / indice.cu .h 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 Here, the calculation of index is replaced by a template plus recursive writing method, which seems more complicated. Let: new_indicesIn = indicesIn.data (), you can deduce that index is: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 ArrayIndexRowMajor is located in include/tensorview/tensorview.h, and its recursive call is written as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 Then look at the kernel getSubMIndicePairsKernel3: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 : include / spconv / indice.cu .h 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 Look: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 The above writing method is similar to the writing method of the common loop in our function. For details, you can see include/tensorview/kernel_utils. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 If NumILP is equal to 1 by default, its stride is also gridDim.x * blockDim.x. The maximum index value is less than the thread limit index blockDim.x * gridDim.x of the thread block, and the function is similar to the following code: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573742188
  ```  
 The following section writes about performing specific sparse convolution computations based on the built Rulebook. 

 Reference: https://blog.csdn.net/ChuiGeDaQiQiu/article/details/127680713 



--------------------------------------------------------------------------------

http://ronny.rest/tutorials/module/pointclouds_01/point_cloud_birdseye/ 

#  point cloud data 

 ![avatar]( bee749034f0e4743a4207218f595e0ab.png) 

 Point cloud data can be represented as an array of [N, 3], N number of point clouds, each row represents a point, expressed with at least 3 values, such as (x, y, z)  

#  Image and point cloud coordinates 

 ![avatar]( 736a1b36aa4a4c6083bbfdb0bdf6470c.png) 

 Blue is the image coordinate system, and orange is the point cloud coordinate system, as shown in the following figure:  

 Image coordinate system: 

 Point cloud coordinates: 

#  Create a bird's-eye view of point cloud data 

 ![avatar]( b6010e0c5f4d489e86ee7dcd7c991b63.png) 

  Usually only the region of interest of the point cloud needs to be extracted 

 Since the data is in top view, to convert it to an image, an orientation that is more consistent with the axes of the image is to be used. Below, I specify the range of values I want to focus on relative to the origin. Anything to the left of the origin will be considered negative, while anything to the right will be considered positive. The x-axis of the point cloud will be interpreted as the forward direction (which will be the upward direction of our aerial image). 

 The following code sets the rectangle of interest to span 10m on either side of the origin and 20m in front of it. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957376127
  ```  
 Creates a filter that only retains points that are actually within the specified rectangle. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957376127
  ```  
#  Map point positions to pixel positions 

 At the moment, we have a bunch of points with real values. In order for these values to map to integers. We could naively convert all x and y value types to integers, but could end up losing a lot of resolution. For example, if these points are measured in meters, then each pixel would represent a 1x1 meter rectangle in the point cloud, and we would lose any detail smaller than that. This might work well if you have point clouds that resemble mountain views. But if you want to capture finer details and can identify people, cars, or even smaller objects, then this approach won't work well. 

 However, it is possible to modify the above method slightly so that we obtain the desired resolution level. We can scale the data first and then type convert to integers. For example, if the unit of measurement is meters and we want a resolution of 5 centimeters, we can do the following: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957376127
  ```  
#  Translation origin 

 The x and y values are still negative and cannot be projected onto the image, so the data needs to be translated to minimize the data at the (0,0) position. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957376127
  ```  
 Verify that the data is all positive. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957376127
  ```  
#  pixel value 

 At this point, the point data has been used to specify the x and y positions in the image. Now all that needs to be done is to specify the values that fill these pixel positions. One way to do this is to fill in the height data. There are two things to note: 

 Minimum and maximum height values can be taken from the data and re-scaled to the 0-255 range. Alternatively, set the range of height values we want to focus on, and any values above or below that range are set to the minimum and maximum values. This method is useful because it allows us to get maximum detail from the area of interest. 

 In the code below, set the range to 2 meters below the origin and half a meter above the origin. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957376127
  ```  
 Next, rescales these values to between 0 and 255 and converts the data type to an integer. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957376127
  ```  
#  Create an image array 

 Now that we are ready to actually create the image, we simply initialize an array whose size depends on the range of values we want in the rectangle and the resolution we choose. We then use the x and y point values we convert to pixel positions to specify the indices in the array and assign to those indices the values we chose as pixel values in the previous section. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957376127
  ```  
 visualization 

 Currently, images are stored as numpy arrays. If you want to visualize them, you can convert them to PIL images. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957376127
  ```  
 ![avatar]( 64e90d27c83a4b02ad33acfc20f87d30.png) 

 Humans are not very good at telling the difference between gray and shadow, so spectral color mapping can be used to make it easier to tell the difference. This can be done in matplotlib. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957376127
  ```  
 In fact, the image generated this way has the exact same amount of information as the one drawn by the PIL, so the machine learning algorithm is able to distinguish highly different, even if we humans can't see the difference very clearly 

#  complete code 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 202402030957376127
  ```  


--------------------------------------------------------------------------------

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



--------------------------------------------------------------------------------

#  brief introduction 

 The algorithm in this paper won the championship in the CVPR 2019 Automated Driving Workshop 3D object detection challenge. The authors extract rich semantic features using sparse 3D convolution and feed them into a class-balanced multi-head network for 3D object detection. To solve the severe class imbalance problem that is naturally present in autonomous driving scenarios, the authors design a class-balanced sampling and enhancement strategy to generate a more balanced data distribution. In addition, the authors also propose a balanced grouping network header, which improves the classification performance of categories with similar shapes. The algorithm proposed in this paper significantly outperforms PointPillars on all evaluation metrics, achieving the detection performance of SOTA on the Scnuenes dataset. 

 Source Code: https://github.com/poodarchu/Det3D 

>  Figure 1: The distribution of instances of categories in the nuScenes dataset is long-tailed, showing an extreme imbalance in the number of examples of common and rare object categories. 

#  data augmentation 

##  DS_Sampling 

 Duplicate samples of a particular class based on the proportion of samples of a particular class in all samples. The fewer samples of a class, the more samples of that class are copied to form the final training dataset. To achieve a class-balanced dataset, all classes should have close proportions in the training split, specifically: 

 Overall, DS Sampling can be seen as increasing the average density of rare classes in the training split, which can effectively alleviate the imbalance problem, as shown in the orange column in Figure 1 

 Code reference: https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/nuscenes/nuscenes_dataset.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573797901
  ```  
 OpenPCDet code can be seen after balanced_infos_resampling, the number of each category is the same, and the above figure shows that the number of category instances before and after CBGS sampling does not change significantly 

##  GT-AUG 

 The ground truths are sampled from an offline generated annotation database using the GT-AUG strategy proposed in SECOND, and the sampled frames are placed in another point cloud. 

>  Before placing the object box correctly, the ground plane position of the point cloud sample needs to be calculated. The least squares method and RANSAC are used to estimate the ground plane of each sample 

>  Figure 2: Example of ground plane detection results, which can be expressed as Ax + By + Cz + D = 0. On average, the ground plane is about -1.82 meters along the z-axis. 

#  network architecture 

 The overall network architecture is shown in the figure below, including four parts: input module, 3D feature extraction, region proposal network, and multi-group header network. 

>  Figure 3: Network architecture. The 3D Feature Extractor consists of a submanifold and a regular 3D sparse convolution. The output of the 3D Feature Extractor is a 16-fold scaled-down, flattened along the output axis and fed into a subsequent regional proposal network to generate an 8-fold feature map, which is then generated by the multi-set header network for the final prediction. The number of header groups is set according to the grouping specification 

 Use sparse 3D convolution with jump connections to build a resnet-like architecture for the 3D feature extractor network. For the input tensor of one, the feature extractor outputs a feature map of one, m, n are the reduction factors of z, x, y dimensions respectively, and l is the last layer of the output channel 3D feature extractor. To make this 3D feature map more suitable for the Region Proposal Network and multi-set of headers below, reshape the feature map into, and then use, the region proposal network to perform regular 2D convolution and deconvolution to further aggregate features and obtain higher resolution feature maps. Based on these feature maps, multi-set of header networks are able to efficiently detect different classes of objects. 

##  class balanced grouping 

 An unbalanced data distribution can make the model dominate the majority of categories. As can be seen from Figure 2, the labeling of cars in the dataset accounts for 43.7%, which is 40 times that of bicycles. Therefore, if you train cars and bicycles together, and there is basically no bicycle data in a batch, the network will have a poor classification effect on bicycles. On the other hand, if you train samples of different shapes and sizes together, the regression targets will have greater intra-class differences, causing classes with different shapes to interfere with each other, which is why the network's performance when learning classes of different shapes at the same time is not as good as training separately. Intuitively, similar shapes have similar characteristics, and training together can promote each other... The two main principles are: 

 Manually divide all categories into groups according to some principles. For a specific header in a multi-group header module, it only needs to identify the class and locate the objects of the classes that belong to that group. There are mainly 2 principles guiding the effective division of 10 classes into groups: 

 Manually group 10 categories in the dataset into 6 groups based on certain principles: (Car), (Truck, Construction Vehicle), (Bus, Trailer), (Barrier), (Motorcycle, Bicycle), (Pedestrian, Traffic Cone). According to the ablation study, as shown in Table 4, class balance grouping contributed the most to the final result 

##  Loss function 

 The loss function part refers to SECOND. If the direction classification in SECOND is used, the mAOE will be very high. The direction of many predicted bounding boxes is exactly the opposite of the ground truth. Common problems such as the object forward and reverse problem (facing the opposite direction) are made in this part. Small improvements are made in this part. Additional direction classification targets are added to add offsets to eliminate direction ambiguity. As for speed estimation, regression without normalization can achieve the best performance compared to adding additional normalization operations. 

 In order to reduce the learning difficulty, the anchor mechanism is used, and other settings are similar to SECOND. Focal loss is used for Classification, Smoothl1 regression x, y, z, l, w, h, yaw, vx, vy, and it is worth mentioning that each branch uses Uniform Scaling as the learning weight. 

 In addition to the general classification and bounding box regression branches required for 3D object detection, we also add the directional classification branch proposed in SECOND [28]. It should be noted that according to our statistics, most of the object boxes are parallel or perpendicular to the LiDAR coordinate axis. Therefore, if directional classification is applied in SECOND, it turns out that the mAOE is very high because the direction of many predicted bounding boxes is exactly the opposite of the ground truth. Therefore, we add an offset to the directional classification target to eliminate directional ambiguity. As for velocity estimation, regression without normalization can achieve optimal performance compared to adding additional normalization operations. 

 In order to reduce the learning difficulty, the anchor mechanism is used. Referring to SECOND, different classes of anchors have different height and width configurations, determined by the class mean. A class has 1 size configuration and 2 different directions. For speed, the anchors are set to 0 on both the x and y axes, no need to estimate the speed on the z-axis. 

 In the experiment, multiple sets of headers were treated as a multi-task learning process, using Uniform Scaling to configure the weights of different branches 

#  Experimental results 

>  Table 2: Overall performance. BRAVE and Tolist are the other top three teams. Our approach achieved the best performance on all metrics except the mAAE metric 

>  Table 3: Comparison of mAP with PointPillars by category. Our method shows more competitive and balanced performance on the tail class. For example, Bicycle improved by 14 times. Motorcycles, construction vehicles (Cons. Veh.), trailers, traffic cones (TC) improved by more than 2 times 

>  Table 4: Ablation studies of the different components used in our validation split method. Database Sampling and Re-Encoder contributed the most to mAP 

>  Table 5: GT-AUG ratings for different categories. For each category, magnitude means the number of instances placed in the point cloud sample 

#  result 

 The performance of the proposed algorithm outperforms the baseline algorithm PointPillars by 73.1%, and for each class, especially for classes with fewer samples, the proposed algorithm has smaller errors in translation (mATE), scale (mASE), direction (mAOE), velocity (mAVE) and attributes (mAAE). 

>  Figure 4: Example of detection results in a validation split. Ground truth is annotated in green and detection results are in blue. Detection results are from a model with 51.9% mAP and 62.5% NDS. The markers at the top of each point cloud bird's-eye view image are their corresponding sample data markers. 

#  conclusion 

 One of the differences between the newly released autonomous driving dataset nuScence and kitti is that there are many categories, and the category samples are uneven. This article mainly proposes a new class imbalance solution. After augmenting the dataset, random sampling is used to balance fewer categories, and those with similar shapes are divided into groups. Categories with fewer samples can be improved by more categories with similar shapes 



--------------------------------------------------------------------------------

论文：Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners作者：Xiao Zhang, Wenda Xu, Chiyu Dong and John M. Dolan 

#  abstract 

 Vehicle surrounding environment detection is an important task in the field of autonomous driving and has received extensive attention in recent years. When using lidar sensors, L-shape fitting is a key step in model-based vehicle detection and tracking, which requires in-depth investigation and comprehensive research. In this paper, the L-shape fitting problem is studied as an optimization problem. And an efficient search-based optimization method is proposed. Our method does not rely on the scan sequence information of the lidar, so it can facilitate data fusion from multiple lidars; the method is efficient and involves very few parameters; and the method is flexible to adapt to various fitting requirements under different fitting standards. In road tests, we demonstrated that the method is effective and robust with a product-grade LiDAR scanner. 

 Laser scanners have been widely used to perceive the surrounding environment because of the sensor's ability to measure its distance to the surface of the surrounding object with high accuracy. A typical method of processing range data is to divide the range data into clusters of points from which meaningful features such as line segments, blocks, and rectangles are extracted. They correspond to objects such as walls, bicycles, woods, bushes, vehicles, and pedestrians in the real world. These features are then associated with static maps or tracking targets and used to update the target state through tracking methods such as Multi-Hypothesis Tracking (MHT) or its advanced version (MHT-RBPF) integrated with a Rao-Blackwellized Particle Filter. 

 In this paper, a search-based L-shape fitting method is proposed for vehicle detection from laser data with high computational efficiency. Through road experiments with product-level sensors, the method is proved to be effective and efficient. 

 There are three main contributions to this paper. First, our method does not rely on laser scan sequence information, making it easy to achieve data fusion from multiple sensors, which makes it suitable for applications with multiple production-grade sensors. Second, our method is computationally efficient, involves few parameters, and does not require hands-on experience or parameter adjustment. Third, our method is able to adapt to any specified criteria, which makes the method flexible to adapt to different fitting needs and can be scaled to a variety of applications. 

#  Vehicle Detection Based on L Shape 

 After obtaining the laser data, the data points are first segmented into clusters. In this paper, we are only interested in the L-shaped fitting of vehicles. Suppose an L-shaped vehicle model, i.e. a rectangle. For each segmented cluster of points, the best rectangle direction is first searched according to the pre-specified criteria, and then the fitted rectangle that conforms to that direction and contains all points in the segmentation is obtained. 

##  A. Segmentation 

 The segmentation algorithm is shown in Alg. 1. The basic idea is to divide range points into clusters according to a pre-specified distance threshold. The input to the segmentation is the two-dimensional coordinates of n range scan points. The output of the algorithm is the clustered point cloud clusters (i.e. Euclidean clustering). Each cluster may correspond to an object in the real world. The main process is: For each scan point, use the K-D tree to find the adjacent points within the distance r, and then combine them into a cluster; then we find the points that are at a distance r from any point in the cluster, and put the newly found points into that cluster again; we repeat this process until the cluster no longer grows, and this final cluster serves as a split in the output. The algorithm ensures that we only run a range search once for each point. 

>  Note that the segmentation algorithm is adaptive because the range search threshold r is proportional to the distance between the point and the laser sensor. Additionally, the RangeSearch function can use this information to speed up processing as long as the scan-ordering index of the range points is available. In cases where scan sequences like ours are not available, efficient range searches can be performed in this two-dimensional low-dimensional space using the K-D tree data structure. 

>   

##  B. L-type fitting 

 Under the assumption of the L-Shape rectangle model, for each segmented cluster of points, it is desirable to find a rectangle that best fits these points. The classical standard uses the least squares method, which involves the following optimization problems: 

>   

 Find the optimal disjunctions (P, Q divide m points into two groups) and the optimal parameters to minimize the square error of angular fitting, corresponding to the two vertical lines of the midpoints of P and Q, respectively. The expressions of the two lines are, and. However, due to the combinatorial complexity of the partition problem, the above optimization problem becomes difficult to solve. 

 It is impractical to accurately solve the above optimization problems in real-time applications, and instead rely on search-based algorithms to approximately find the best-fitting rectangle. 

 The basic idea: Traverse all possible directions of the rectangle; in each iteration, one can easily find a rectangle that faces that direction and contains all the scanning points; Therefore, the distances from all the points to the four sides of the rectangle can be obtained, and based on these distances, we can divide the points into P and Q and calculate the objective function of the corresponding squared error; After iterating over all directions and obtaining all the corresponding squared errors, we look for the best direction to achieve the minimum squared error and fit the rectangle according to that direction. Once this fitted rectangle is obtained, the features of the vehicle tracking can be easily extracted. 

 Rectangle Box Fitting Based on Search 

>   

 In Alg2: The input is m points in the division,. The output of this algorithm is the four edge line representation of the fitted rectangle. The possible directions theta of the rectangle range from 0 ° to 90 °, because the two sides of the rectangle are orthogonal, we only care about the single side that falls between 0 ° and 90 °; the other direction is θ + π/ 2. The search space for theta can be reduced if a tracking system or visual scene understanding is supported. If the line representation of the sides is, the parameters of the four sides can be obtained through steps 12 to 15 in Alg2. 

 Calculating criteria can be defined in a number of ways, each with its own advantages and disadvantages. Three criteria were considered for selecting the fitted rectangle, and three criteria for selecting the rectangle, corresponding to Alg3, 4, and 5, respectively: 

 Each of these three functions can be selected to play the role of the Alg2. CalculatecriterionX function. These criterion calculation functions take the inputs of C1 and C2, which are the projections of all range points on two orthogonal edges determined by theta. 

 ![avatar]( ecb32e270893401b9ee8757d065f8557.png) 

>   

 Algorithm 3: Area minimization criterion. Find the smallest rectangle covering all range points 

 Algorithm 4: Emphasize how close these range points are to the two edges of the upper right corner. In the two-dimensional plane of the projection, and, specifies the boundary of all points on the axis, vector, and records the distance of all points to the two boundaries; from these two boundaries we choose a boundary closer to the range point and represent the corresponding distance vector as D1. The definition of the distance vector D2 is similar to the projection axis. The proximity score is defined as, where di is the distance of the i-th point to its nearest edge. In this way, both reducing the distance and increasing the number of scan points will result in a higher score. 

>  Notice that there is a minimum distance threshold  

           d 

           0 

         d_0 

     D0, to avoid dividing points on the boundary by zero, and points very close to the edge with significant voting power. 

 Algorithm 5: Emphasizes the squared error of two orthogonal edges fitted by two sets of disjoint points. contains the distance from the point to the boundary assigned to, while contains the distance from the point to the boundary assigned to,. The variance of is actually equivalent to the squared error of the straight line in the theta direction to the point belonging to,. Since when calculating the squared error, it is essentially computing the variance of these distances, this criterion is also called "variance". By using this variance criterion, it is actually looking for an approximate solution to the optimization problem in (1). 

#  Experimental results 

 The experiments were tested on a CMU autonomous car SRX on local roads. Six IBEOs were installed to provide multi-layer range scanning. Please note that the experiments here do not use scan sequence/sorting information. 

##  A. Computational efficiency 

 The computational efficiency of the algorithm is demonstrated by experiments with approximately 10,000 laser scans collected on local roads. Each laser scan is split into clusters (approximately 25,000 clusters in total) and a search-based fitting algorithm is performed on each cluster. The computational time is shown in Table 1. The algorithm is implemented in MATLAB and is run on a Linux laptop equipped with an Intel Core i7 CPU. The variance minimization standard is the most time-consuming, and the computational effort to calculate the variance is large, taking about 3.84 ms per cluster on average. The computational performance of the algorithm is better if implemented in a more efficient programming language. Please note that the standard deviation of the computation time is very small, indicating a consistent estimate of the computation time for high-level decision-making tasks. 

>  Table 1: Calculation time for rectangle fitting 

##  B. Rectangle fitting 

 An example of a typical rectangle fitting using the three criteria is shown in Figure 2. In Figure 2 (a), the fitted rectangles for the three criteria are nearly identical; the corresponding normalized criteria along the search direction are shown in Figure 2 (b), where the maxima of the three curves are achieved at the nearby theta s. In Figure 2 (c), the area minimization criteria yield an inaccurate rectangular heading, while the other two criteria are nearly identical. A dataset consisting of 145 point clusters, i.e. laser distance points for 145 vehicles, whose heading is manually marked with a resolution of 1 °, is used to test the correctness of the proposed method. The proximity maximization criterion and the variance minimization criterion are both highly accurate, while the area minimization criterion sometimes fails to obtain the correct heading. The performance of the area minimization criterion can only be guaranteed when the distance scan point density in the segmentation is high, such as in cases suitable for small objects or when using more powerful sensors such as Velodyne [19]. The heading error histogram using the three criteria and PCA is shown in Figure 3, where the heading error is defined as: The heading produced by the method minus the heading of the ground truth. 

>  Figure 2: Example of rectangle fitting. In (a) and (c), the gray dots represent the laser distance scan points, and the green, red, and blue rectangles are fitted rectangles using standard area minimization, proximity maximization, and variance minimization. Normalized scores for the three criteria in the search direction are plotted in (b) and (d), respectively. In example (a), the fitting results for the three criteria are very similar, and the maximums for the three curves in (b) are very close (marked with arrows, reached at 88 °, 89 °, and 0 °, respectively). In example (b), the area criterion is fitted differently from the other two, and the maximum value in (d) is far from the other two (achieved at 69 °, 1 °, and 86 °, respectively). 

>  Figure 3: Heading Error Histogram 

 The mean and standard deviation of the actual error (i.e. course error) and the absolute error are listed in Table 2, where the average actual error is an indicator of the estimate bias, while the average absolute error reflects the estimate accuracy (error of the mean magnitude without regard to their direction). As shown in Table 2, both the true error and the absolute error are small for the proximity and variance criteria. 

>  Table 2: Heading Errors in Rectangle Fitting 

>  Figure 4: Segmentation and vehicle fitting results for a typical laser scan period. Camera views of these two periods are shown in (a) and ©. In (b) and (d), segmented laser scan points are represented by markers of different colors and shapes, and vehicle fitting results by standard proximity and variance are shown by red and blue rectangles, respectively 

 Figure 4 shows the results of two typical cycles. 

 Figure 5 shows an example of the rare case where these methods do not achieve good performance. 

>  Figure 5: The case where the L-model assumption does not hold. For the white SUV in (a), the segmented laser scan points and the rectangle fit results in two consecutive cycles are shown in (b) and ©. For the truck in (d), the scan points and fit results in two consecutive cycles are shown in (e) and (f). In (b), the red proximity criteria are endangered by the side mirrors marked by the black arrows. In (d), the blue variance criteria are affected by the rear windows of the truck. Please note that even for these two rare cases, at least one of the two criteria works well, and the fitting result is corrected in the next cycle 

#  conclusion 

 A search-based L-Shape fitting method for laser distance data is presented in this paper. The proposed method does not rely on scan sequence/ordering information, enabling the fusion of raw laser data from multiple laser scanners; The method is computationally efficient and easy to implement, involving few parameters and requiring no hands-on experience or parameter adjustment; It is able to adapt to any specified criteria, which makes the method flexible to meet different assembly needs and scalable to a variety of applications. Three criteria are discussed in this paper and compared in experiments. As demonstrated by road experiments with production-grade sensors, the proposed method is effective and robust even in cases where the L-Shape model assumption does not hold. 



--------------------------------------------------------------------------------

#  CPU 

 First look at the generate_voxels of the CPU version in spconv1.0, spconv1.0 has no GPU version 

 See centerpoint preprocessing pcdet/datasets/processor/data_processor.py 

 Look directly at the code, with detailed comments 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573745274
  ```  
 Focus on the spconv class VoxelGeneratorV2 is located at: spconv/utils/__init__ 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573745274
  ```  
 Then look at points_to_voxel function, located in spconv/utils/__init__ 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573745274
  ```  
 src/utils/all.cc 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573745274
  ```  
 Pybind11 is a lightweight library that only contains header files, which can be used to call C/C++ interfaces. Compared with boost.python, swig, ctypes, etc., pybind11 has the advantage of relatively simple API and good support for C++ 11. 

 PYBIND11_MODULE () creates a function that is called when called from Python. 

 Here is the core C++ template function points_to_voxel_3d_np: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573745274
  ```  
 mutable_unchecked is to check the dimensions, see the code comments for details 

##  spconv2.0 

 Then look at the GPU version in spconv2.0 generate_voxels 

 The code is located at: spconv/pytorch/utils.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573745274
  ```  
 torch_tensor_to_tv位于：spconv/pytorch/cppcore.py 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573745274
  ```  


--------------------------------------------------------------------------------

 article directory 

 + Introduction to Raw Data 

 + kitti2bag 

 + Generate bag to modify kitti2bag

#  Introduction to Raw Data 

 ![avatar]( 826c8ea7850742e2931db38d6d211ff9.png) 

 Raw Data Address: http://www.cvlibs.net/datasets/kitti/raw_data.php 

 Original data source records, sorted by category (City, Residential, Road, Campus, People, Calibration) 

 The dataset contains the following information, captured and synchronized at a frequency of 10 Hz: 

>  unsynced+unrectified refers to the original input frame, where the image is distorted and the frame index does not correspond, and the download data corresponding name 2011_ ** _ ** _drive_0 ** _extract. Zipsynced + rectified refers to the processed data, where the image has been corrected and not distorted, and the data frame number corresponds to the stream across all sensors, and the data corresponding name is 2011_ ** _ * _drive_0 ** _sync .zip. For both settings, a file with timestamp is provided, most people only need the synced + rectified version of the file. 

#  kitti2bag 

##  Generate bag 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573717954
  ```  
 If there is a lack of dependencies, install the corresponding dependencies. In the conda environment I use, the corresponding dependencies have been installed. 

 The unzipped file is placed as follows: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573717954
  ```  
 One level above the 2011_09_30 folder, open end point and type: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573717954
  ```  
 ![avatar]( 414df9eb91b6436b89e52cc4d0054b9b.png) 

  Rviz visualization: It can be seen that the converted bag package lacks point cloud intensity information, which is caused by inconsistent fields. Modify the kitti2bag code below 

 ![avatar]( 306143fd5cde44bb839571f991d4d54e.png) 

##  Modify kitti2bag 

 In the end point type whereis kitti2bag to find the path to the file. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573717954
  ```  
 ![avatar]( ece8b7954941472ebbf405246194f905.png) 

  Modify kitti2bag, corresponding to 189 lines of code: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573717954
  ```  
 ![avatar]( 396774da50f748a4a486f956f1ea49d0.png) 

  After modification, reconvert the bag file to rviz display  



--------------------------------------------------------------------------------

 Numpy Two N * M-dimensional, two-dimensional arrays a and b, in units of behavior, find rows that exist in both a and b, and the same rows do not necessarily appear in the same row position, by iterating over each row to find. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573745758
  ```  


--------------------------------------------------------------------------------

 article directory 

 + Machine model 

 + Upgrade system 

 + NVIDIA driver upgrade 

 + cuda installation 

 + cudnn installation 

 + conda installation 

 + pytorch installation 

 + Upgrade cmake 

 + Install spconv 

 + OpenPCDet installation 

 + Kitti training and testing 

 + Prepare data for data preprocessing training tests

 + nuscenes training and testing 

 + Data preprocessing training test

 + onnx 

#  Machine model 

 System: Upgrade to ubuntu18 notebook graphics card model: GTX1070 cuda: cuda9.2 Upgrade to cuda11.3 

#  upgrade system 

 Upgrade from ubuntu16 to ubuntu18 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
>  After executing the previous command, the system will be automatically upgraded. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
#  NVIDIA driver upgrade 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Disable nouveau 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Add at the end of the article: blacklist nouveau to execute after saving: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Restart, the screen size will be abnormal when restarting, because the graphics processing that comes with ubuntu is disabled. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 If there is no output, disabling nouveau takes effect 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Check the current graphics card model and recommended drivers 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 ![avatar]( 5f85c562eddf4d30abb1ca3ab345e24a.png) 

  Here I did not use the recommended graphics card driver, go directly to the official website to download the latest: https://www.nvidia.cn/geforce/drivers/  

 Click Start Search to download the latest driver installation. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Mount the Nvidia driver: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Check if the driver is installed successfully. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 ![avatar]( 599fd6cafb4740bbb5ee81afeb09dc7b.png) 

  Restart: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
#  cuda installation 

 Uninstall the old cuda first. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Then switch to the directory where CUDA is located: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Delete the CUDA-9.0 directory: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 ![avatar]( c2d744731e1c456a98f99ce6874df87d.png) 

 Check the correspondence between cuda and NVIDIA drivers: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html see that pytorch only supports cuda11.3 latest, here I download cuda11.3 cuda download: 

 ![avatar]( d709f3fcda90498f8c216a6d87146d56.png) 

 Download cuda11.3 to install 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 ![avatar]( 1058b673bbb04b2d951b99037311c4e3.png) 

 Adding environment variables 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Add at the end of the file: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Activate after saving and exiting 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Create a linked file 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Check out the cuda version: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 ![avatar]( d36f89455e484523b5a6291bff134786.png) 

 You can also test: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 ![avatar]( fca484fac65648d48e6ab7ef18bbf594.png) 

#  cudnn installation 

 https://developer.nvidia.com/rdp/cudnn-download 

 Download cudnn-11.3-linux-x64-2.0.53 

 install 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
```bash
cp -r /usr/src/cudnn_samples_v8 ~/

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 cd ~/cudnn_samples_v8/mnistCUDNN/

make clean && make  -j8

./mnistCUDNN

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 bash Anaconda3-2021.11-Linux-x86_64.sh

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 export PATH="/home/xiaohu/anaconda3/bin:$PATH"

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 conda create -n cuda11.3_python3.7 python=3.7

conda activate cuda11.3_python3.7

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
#  exit environment

conda deactivate

#  Rename environment Rename the environment after --clone to the name after -n)

conda create -n python37 --clone python3.7 

#  List virtual environments	 

conda env list

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
#  conda install pytorch torchvision -c pytorch

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 conda install --use-local ~/下载/pytorch-1.10.1-py3.7_cuda11.3_cudnn8.2.0_0.tar.bz2

conda install --use-local ~/下载/torchvision-0.11.2-py37_cu113.tar.bz2

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 tar -xvzf cmake-3.22.4-linux-x86_64.tar.gz

sudo mv cmake-3.22.4-linux-x86_64 /opt/cmake-3.22.4

#  Create a soft link

sudo ln -sf /opt/cmake-3.22.4/bin/*  /usr/bin/ 

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 cmake --version

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 pip install spconv-cu113

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 git clone https://github.com/traveller59/spconv.git --recursive

cd spconv/

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 pip install pccm

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 python setup.py bdist_wheel

cd ./dist

pip install *

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 git clone https://github.com/open-mmlab/OpenPCDet.git

cd OpenPCDet

#  The download will be relatively slow, so replace it with Tsinghua source to quickly install the dependent python package.

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python setup.py develop 

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 python 

import pcdet

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 cd OpenPCDet/data/kitti

ln -s /media/xiaohu/xiaohu/new\ start/数据集/激光/object/training training

ln -s /media/xiaohu/xiaohu/new\ start/数据集/激光/object/testing testing

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 ├── ImageSets

│   ├── test.txt

│   ├── train.txt

│   └── val.txt

< unk > Testing - > /media/xiaohu/xiaohu/new start/dataset/laser/object/testing

Training - > /media/xiaohu/xiaohu/new start/dataset/laser/object/training

#  training

├── calib

├── image_2

├── label_2

└── velodyne

#  testing

├── calib

├── image_2

└── velodyne

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 cd tools

python train.py --cfg_file cfgs/kitti_models/pointpillar.yaml

#  Doka training

CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch --nproc_per_node=3 tools/train.py --cfg_file tools/cfgs/kitti_models/pointpillar.yaml --launcher pytorch

sh scripts/dist_train.sh 8 --cfg_file tools/cfgs/kitti_models/pointpillar.yaml

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 USE_ROAD_PLANE: False

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 cd output/kitti_models/pointpillar/default/

tensorboard --logdir tensorboard/

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 pip install vtk==8.1.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

python3 -m pip install ~/下载/mayavi-4.7.4.tar.gz -i https://pypi.tuna.tsinghua.edu.cn/simple

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 #inside the OpenPCDet project

cd tools

python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 4 --ckpt ../output/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml  --data_path ../data/kitti/testing/velodyne/000099.bin --ckpt ../output/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 OpenPCDet

├── data

│   ├── nuscenes

│   │   │── v1.0-trainval (or v1.0-mini if you use mini)

│   │   │   │── samples

│   │   │   │── sweeps

│   │   │   │── maps

│ │ │ │── v1.0-trainval  

pcdet

├── tools

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \

    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \

    --version v1.0-trainval

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
#  single card training

cd tools

python train.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml

#  Doka training

CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch --nproc_per_node=3 tools/train.py --cfg_file tools/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml --launcher pytorch

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 #inside the OpenPCDet project

cd tools

python test.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml --batch_size 4 --ckpt ../output/nuscenes_models/cbgs_voxel0075_res3d_centerpoint/default/ckpt/checkpoint_epoch_20.pth

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 pip install onnx  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install onnxsim  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install onnxruntime  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install ~/3D/TensorRT-8.2.3.0/graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl

pip install ~/3D/TensorRT-8.2.3.0/onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930


--------------------------------------------------------------------------------

#  The Background of Point Cloud Semantic Segmentation 

 Point cloud semantic segmentation has great application value in 3D object detection, scene recognition and high-precision map automation construction. With the wide application of deep neural networks in the field of computer vision, point cloud semantic segmentation methods based on deep learning have become the mainstream in this field. Although lidar has high ranging accuracy, there are also several problems: 

 How to extract context information from irregularly distributed points in space, whether local or global. From the perspective of methods for aggregating context information, there are two main ways: parameterization and non-parameterization 

 In order to learn the characteristics of point clouds and solve the problem of point cloud disorder: 

 Extracting the two-dimensional feature map allows machine learning using typical image semantic segmentation neural networks, which originated from FCNs. The basic feature is encoding and decoding, that is, the two-dimensional data features are first compressed to establish high-level features, and then enlarged to learn local features. More advanced methods include Deeplab and UNet. Both aim to fuse contextual information at multiple scales. DeepLab and its successors use diluted convolutional filters to increase the receiving field, while Unet adds skip connections to directly connect different levels of semantic features, which are more effective in images with irregular and rough edges, such as medical images 

 There are currently many LiDAR object detection datasets, such as the Waymo Open Dataset and the KITTI 3D detection dataset. In contrast, LiDAR scan semantic segmentation datasets are relatively rare. There are three commonly used semantic segmentation datasets: Audi dataset, Paris-Lille-3D, and Semantic KITTI dataset. 

#  Conclusion 

 The contributions of PolarNet's work are as follows: 

#  PolarNet Approach 

>  Figure 2. An overview of our model. For a given LiDAR point cloud, we first quantify the points into a grid using polar coordinate BEV. For each grid cell, we use a simplified KNN-free PointNet to transform the points in it into a fixed-length representation. This representation is then assigned to its corresponding position in the ring matrix. We feed the matrix into a ring CNN composed of ring convolutional modules. Finally, the CNN outputs a quantized prediction, which we decode into the point domain 

>   

##  Problem Statement 

 The training dataset for a given N-frame lidar scan is the i-th frame scan containing LiDAR points, containing four dimensional information (x, y, z, reflection) 

##  Polar Bird’s-eye-view 

 ![avatar]( 8335bb59ac0848cdae01c7b95d6ed89d.png) 

>  Figure 4. The relationship between the distance from the grid cells of the sensor and the average number of points in the logarithmic interval of each grid cell. A traditional BEV represents distributing most of its grid cells to a farther end with only a few points. 

 The mesh is the basic image representation, but it may not be the best representation of BEV, which is a compromise between performance and accuracy. 

##  Learning the Polar Grid 

 Unlike cnn-Seg, which uses manual features, PolarNet uses a fixed-length representation to capture the distribution of points in each grid. It is generated by a learnable simplified PointNet and a max-pooling. The characteristics of the first grid cell in a scan ring are: 

 ![avatar]( c1c6a58c7a7648909db1e21052637e1a.png) 

 Where w and l are the quantized size. And is the position of the point p in the map. Note that the position and quantized size can be polar or Cartesian coordinates. We do not quantize the input point cloud along the z-axis, our method represents the entire vertical column of the grid 

 If the representation is learned in polar coordinates, then the two sides of the feature matrix will be connected along the azimuth axis in physical space, developing a discrete convolution which we call ring convolution. Assuming that the matrix is connected at both ends of the radius axis, the ring convolution kernel will convolution the matrix. At the same time, the layer located on the opposite side can propagate back to the opposite side through this ring convolution kernel. By replacing ordinary convolution with ring convolution in a 2D network, the network will be able to handle the polar grid end-to-end without neglecting its connectivity. This provides the model with an extended receptive field. Since it is a 2D neural network, the final prediction will also be a polar grid with feature dimensions equal to the product of quantized height channels and class numbers. We can then reshape the prediction into a 4D matrix to derive the segmentation loss based on voxels. 



--------------------------------------------------------------------------------

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



--------------------------------------------------------------------------------

The subtraction of two point clouds a and b is actually to find the duplicate points and non-duplicate points in the two point clouds. Removing the duplicate points in a is to subtract the point cloud after b. It is easy to achieve with Python, and directly use the search function of numpy. C ++ is a bit more troublesome and demanding. 

 Here is the C ++ and Python code 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573717589
  ```  
 C++ code 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573717589
  ```  


--------------------------------------------------------------------------------

