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

