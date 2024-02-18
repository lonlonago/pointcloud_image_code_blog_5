论文：Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners作者：Xiao Zhang, Wenda Xu, Chiyu Dong and John M. Dolan 

#  abstract 

 Vehicle surrounding environment detection is an important task in the field of autonomous driving and has received extensive attention in recent years. When using lidar sensors, L-shape fitting is a key step in model-based vehicle detection and tracking, which requires in-depth investigation and comprehensive research. In this paper, the L-shape fitting problem is studied as an optimization problem. And an efficient search-based optimization method is proposed. Our method does not rely on the scan sequence information of the lidar, so it can facilitate data fusion from multiple lidars; the method is efficient and involves very few parameters; and the method is flexible to adapt to various fitting requirements under different fitting standards. In road tests, we demonstrated that the method is effective and robust with a product-grade LiDAR scanner. 

 Laser scanners have been widely used to perceive the surrounding environment because of the sensor's ability to measure its distance to the surface of the surrounding object with high accuracy. A typical method of processing range data is to divide the range data into clusters of points from which meaningful features such as line segments, blocks, and rectangles are extracted. They correspond to objects such as walls, bicycles, woods, bushes, vehicles, and pedestrians in the real world. These features are then associated with static maps or tracking targets and used to update the target state through tracking methods such as Multi-Hypothesis Tracking (MHT) or its advanced version (MHT-RBPF) integrated with a Rao-Blackwellized Particle Filter. 

 In this paper, a search-based L-shape fitting method is proposed for vehicle detection from laser data with high computational efficiency. Through road experiments with product-level sensors, the method is proved to be effective and efficient. 

 There are three main contributions to this paper. First, our method does not rely on laser scan sequence information, making it easy to achieve data fusion from multiple sensors, which makes it suitable for applications with multiple production-grade sensors. Second, our method is computationally efficient, involves few parameters, and does not require hands-on experience or parameter adjustment. Third, our method is able to adapt to any specified criteria, which makes the method flexible to adapt to different fitting needs and can be scaled to a variety of applications. 

#  Vehicle Detection Based on L Shape 

 After obtaining the laser data, the data points are first segmented into clusters. In this paper, we are only interested in the L-shaped fitting of vehicles. Suppose an L-shaped vehicle model, i.e. a rectangle. For each segmented cluster of points, the best rectangle direction is first searched according to the pre-specified criteria, and then the fitted rectangle that conforms to that direction and contains all points in the segmentation is obtained. 

##  A. Segmentation 

 The segmentation algorithm is shown in Alg. 1. The basic idea is to divide range points into clusters according to a pre-specified distance threshold. The input to the segmentation is the two-dimensional coordinates of n range scan points. The output of the algorithm is the clustered point cloud clusters (i.e. Euclidean clustering). Each cluster may correspond to an object in the real world. The main process is: For each scan point, use the K-D tree to find the adjacent points within the distance r, and then combine them into a cluster; then we find the points that are at a distance r from any point in the cluster, and put the newly found points into that cluster again; we repeat this process until the cluster no longer grows, and this final cluster serves as a split in the output. The algorithm ensures that we only run a range search once for each point. 

>  Note that the segmentation algorithm is adaptive because the range search threshold r is proportional to the distance between the point and the laser sensor. Additionally, the RangeSearch function can use this information to speed up processing as long as the scan-ordering index of the range points is available. In cases where scan sequences like ours are not available, efficient range searches can be performed in this two-dimensional low-dimensional space using the K-D tree data structure. 

>   

##  B. L-type fitting 

 Under the assumption of the L-Shape rectangle model, for each segmented cluster of points, it is desirable to find a rectangle that best fits these points. The classical standard uses the least squares method, which involves the following optimization problems: 

>   

 Find the optimal disjunctions (P, Q divide m points into two groups) and the optimal parameters to minimize the square error of angular fitting, corresponding to the two vertical lines of the midpoints of P and Q, respectively. The expressions of the two lines are, and. However, due to the combinatorial complexity of the partition problem, the above optimization problem becomes difficult to solve. 

 It is impractical to accurately solve the above optimization problems in real-time applications, and instead rely on search-based algorithms to approximately find the best-fitting rectangle. 

 The basic idea: Traverse all possible directions of the rectangle; in each iteration, one can easily find a rectangle that faces that direction and contains all the scanning points; Therefore, the distances from all the points to the four sides of the rectangle can be obtained, and based on these distances, we can divide the points into P and Q and calculate the objective function of the corresponding squared error; After iterating over all directions and obtaining all the corresponding squared errors, we look for the best direction to achieve the minimum squared error and fit the rectangle according to that direction. Once this fitted rectangle is obtained, the features of the vehicle tracking can be easily extracted. 

 Rectangle Box Fitting Based on Search 

>   

 In Alg2: The input is m points in the division,. The output of this algorithm is the four edge line representation of the fitted rectangle. The possible directions theta of the rectangle range from 0 ° to 90 °, because the two sides of the rectangle are orthogonal, we only care about the single side that falls between 0 ° and 90 °; the other direction is θ + π/ 2. The search space for theta can be reduced if a tracking system or visual scene understanding is supported. If the line representation of the sides is, the parameters of the four sides can be obtained through steps 12 to 15 in Alg2. 

 Calculating criteria can be defined in a number of ways, each with its own advantages and disadvantages. Three criteria were considered for selecting the fitted rectangle, and three criteria for selecting the rectangle, corresponding to Alg3, 4, and 5, respectively: 

 Each of these three functions can be selected to play the role of the Alg2. CalculatecriterionX function. These criterion calculation functions take the inputs of C1 and C2, which are the projections of all range points on two orthogonal edges determined by theta. 

 ![avatar]( ecb32e270893401b9ee8757d065f8557.png) 

>   

 Algorithm 3: Area minimization criterion. Find the smallest rectangle covering all range points 

 Algorithm 4: Emphasize how close these range points are to the two edges of the upper right corner. In the two-dimensional plane of the projection, and, specifies the boundary of all points on the axis, vector, and records the distance of all points to the two boundaries; from these two boundaries we choose a boundary closer to the range point and represent the corresponding distance vector as D1. The definition of the distance vector D2 is similar to the projection axis. The proximity score is defined as, where di is the distance of the i-th point to its nearest edge. In this way, both reducing the distance and increasing the number of scan points will result in a higher score. 

>  Notice that there is a minimum distance threshold  

           d 

           0 

         d_0 

     D0, to avoid dividing points on the boundary by zero, and points very close to the edge with significant voting power. 

 Algorithm 5: Emphasizes the squared error of two orthogonal edges fitted by two sets of disjoint points. contains the distance from the point to the boundary assigned to, while contains the distance from the point to the boundary assigned to,. The variance of is actually equivalent to the squared error of the straight line in the theta direction to the point belonging to,. Since when calculating the squared error, it is essentially computing the variance of these distances, this criterion is also called "variance". By using this variance criterion, it is actually looking for an approximate solution to the optimization problem in (1). 

#  Experimental results 

 The experiments were tested on a CMU autonomous car SRX on local roads. Six IBEOs were installed to provide multi-layer range scanning. Please note that the experiments here do not use scan sequence/sorting information. 

##  A. Computational efficiency 

 The computational efficiency of the algorithm is demonstrated by experiments with approximately 10,000 laser scans collected on local roads. Each laser scan is split into clusters (approximately 25,000 clusters in total) and a search-based fitting algorithm is performed on each cluster. The computational time is shown in Table 1. The algorithm is implemented in MATLAB and is run on a Linux laptop equipped with an Intel Core i7 CPU. The variance minimization standard is the most time-consuming, and the computational effort to calculate the variance is large, taking about 3.84 ms per cluster on average. The computational performance of the algorithm is better if implemented in a more efficient programming language. Please note that the standard deviation of the computation time is very small, indicating a consistent estimate of the computation time for high-level decision-making tasks. 

>  Table 1: Calculation time for rectangle fitting 

##  B. Rectangle fitting 

 An example of a typical rectangle fitting using the three criteria is shown in Figure 2. In Figure 2 (a), the fitted rectangles for the three criteria are nearly identical; the corresponding normalized criteria along the search direction are shown in Figure 2 (b), where the maxima of the three curves are achieved at the nearby theta s. In Figure 2 (c), the area minimization criteria yield an inaccurate rectangular heading, while the other two criteria are nearly identical. A dataset consisting of 145 point clusters, i.e. laser distance points for 145 vehicles, whose heading is manually marked with a resolution of 1 °, is used to test the correctness of the proposed method. The proximity maximization criterion and the variance minimization criterion are both highly accurate, while the area minimization criterion sometimes fails to obtain the correct heading. The performance of the area minimization criterion can only be guaranteed when the distance scan point density in the segmentation is high, such as in cases suitable for small objects or when using more powerful sensors such as Velodyne [19]. The heading error histogram using the three criteria and PCA is shown in Figure 3, where the heading error is defined as: The heading produced by the method minus the heading of the ground truth. 

>  Figure 2: Example of rectangle fitting. In (a) and (c), the gray dots represent the laser distance scan points, and the green, red, and blue rectangles are fitted rectangles using standard area minimization, proximity maximization, and variance minimization. Normalized scores for the three criteria in the search direction are plotted in (b) and (d), respectively. In example (a), the fitting results for the three criteria are very similar, and the maximums for the three curves in (b) are very close (marked with arrows, reached at 88 °, 89 °, and 0 °, respectively). In example (b), the area criterion is fitted differently from the other two, and the maximum value in (d) is far from the other two (achieved at 69 °, 1 °, and 86 °, respectively). 

>  Figure 3: Heading Error Histogram 

 The mean and standard deviation of the actual error (i.e. course error) and the absolute error are listed in Table 2, where the average actual error is an indicator of the estimate bias, while the average absolute error reflects the estimate accuracy (error of the mean magnitude without regard to their direction). As shown in Table 2, both the true error and the absolute error are small for the proximity and variance criteria. 

>  Table 2: Heading Errors in Rectangle Fitting 

>  Figure 4: Segmentation and vehicle fitting results for a typical laser scan period. Camera views of these two periods are shown in (a) and ©. In (b) and (d), segmented laser scan points are represented by markers of different colors and shapes, and vehicle fitting results by standard proximity and variance are shown by red and blue rectangles, respectively 

 Figure 4 shows the results of two typical cycles. 

 Figure 5 shows an example of the rare case where these methods do not achieve good performance. 

>  Figure 5: The case where the L-model assumption does not hold. For the white SUV in (a), the segmented laser scan points and the rectangle fit results in two consecutive cycles are shown in (b) and ©. For the truck in (d), the scan points and fit results in two consecutive cycles are shown in (e) and (f). In (b), the red proximity criteria are endangered by the side mirrors marked by the black arrows. In (d), the blue variance criteria are affected by the rear windows of the truck. Please note that even for these two rare cases, at least one of the two criteria works well, and the fitting result is corrected in the next cycle 

#  conclusion 

 A search-based L-Shape fitting method for laser distance data is presented in this paper. The proposed method does not rely on scan sequence/ordering information, enabling the fusion of raw laser data from multiple laser scanners; The method is computationally efficient and easy to implement, involving few parameters and requiring no hands-on experience or parameter adjustment; It is able to adapt to any specified criteria, which makes the method flexible to meet different assembly needs and scalable to a variety of applications. Three criteria are discussed in this paper and compared in experiments. As demonstrated by road experiments with production-grade sensors, the proposed method is effective and robust even in cases where the L-Shape model assumption does not hold. 

