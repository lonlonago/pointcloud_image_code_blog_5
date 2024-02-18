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

