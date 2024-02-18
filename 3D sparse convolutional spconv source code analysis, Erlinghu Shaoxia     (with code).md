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

