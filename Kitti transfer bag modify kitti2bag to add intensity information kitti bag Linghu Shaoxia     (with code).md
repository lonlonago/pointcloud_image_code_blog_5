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

