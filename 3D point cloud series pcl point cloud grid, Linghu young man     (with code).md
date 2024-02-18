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

