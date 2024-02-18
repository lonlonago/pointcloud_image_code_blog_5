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

