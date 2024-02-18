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
