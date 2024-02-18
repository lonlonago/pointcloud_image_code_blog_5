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
