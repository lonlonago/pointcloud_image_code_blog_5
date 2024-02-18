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
