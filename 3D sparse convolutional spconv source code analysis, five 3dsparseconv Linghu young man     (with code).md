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

