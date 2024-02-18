 article directory 

 + Machine model 

 + Upgrade system 

 + NVIDIA driver upgrade 

 + cuda installation 

 + cudnn installation 

 + conda installation 

 + pytorch installation 

 + Upgrade cmake 

 + Install spconv 

 + OpenPCDet installation 

 + Kitti training and testing 

 + Prepare data for data preprocessing training tests

 + nuscenes training and testing 

 + Data preprocessing training test

 + onnx 

#  Machine model 

 System: Upgrade to ubuntu18 notebook graphics card model: GTX1070 cuda: cuda9.2 Upgrade to cuda11.3 

#  upgrade system 

 Upgrade from ubuntu16 to ubuntu18 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
>  After executing the previous command, the system will be automatically upgraded. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
#  NVIDIA driver upgrade 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Disable nouveau 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Add at the end of the article: blacklist nouveau to execute after saving: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Restart, the screen size will be abnormal when restarting, because the graphics processing that comes with ubuntu is disabled. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 If there is no output, disabling nouveau takes effect 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Check the current graphics card model and recommended drivers 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 ![avatar]( 5f85c562eddf4d30abb1ca3ab345e24a.png) 

  Here I did not use the recommended graphics card driver, go directly to the official website to download the latest: https://www.nvidia.cn/geforce/drivers/  

 Click Start Search to download the latest driver installation. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Mount the Nvidia driver: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Check if the driver is installed successfully. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 ![avatar]( 599fd6cafb4740bbb5ee81afeb09dc7b.png) 

  Restart: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
#  cuda installation 

 Uninstall the old cuda first. 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Then switch to the directory where CUDA is located: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Delete the CUDA-9.0 directory: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 ![avatar]( c2d744731e1c456a98f99ce6874df87d.png) 

 Check the correspondence between cuda and NVIDIA drivers: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html see that pytorch only supports cuda11.3 latest, here I download cuda11.3 cuda download: 

 ![avatar]( d709f3fcda90498f8c216a6d87146d56.png) 

 Download cuda11.3 to install 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 ![avatar]( 1058b673bbb04b2d951b99037311c4e3.png) 

 Adding environment variables 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Add at the end of the file: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Activate after saving and exiting 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Create a linked file 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 Check out the cuda version: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 ![avatar]( d36f89455e484523b5a6291bff134786.png) 

 You can also test: 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
 ![avatar]( fca484fac65648d48e6ab7ef18bbf594.png) 

#  cudnn installation 

 https://developer.nvidia.com/rdp/cudnn-download 

 Download cudnn-11.3-linux-x64-2.0.53 

 install 

  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```  
  ```python  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
```bash
cp -r /usr/src/cudnn_samples_v8 ~/

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 cd ~/cudnn_samples_v8/mnistCUDNN/

make clean && make  -j8

./mnistCUDNN

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 bash Anaconda3-2021.11-Linux-x86_64.sh

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 export PATH="/home/xiaohu/anaconda3/bin:$PATH"

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 conda create -n cuda11.3_python3.7 python=3.7

conda activate cuda11.3_python3.7

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
#  exit environment

conda deactivate

#  Rename environment Rename the environment after --clone to the name after -n)

conda create -n python37 --clone python3.7 

#  List virtual environments	 

conda env list

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
#  conda install pytorch torchvision -c pytorch

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 conda install --use-local ~/下载/pytorch-1.10.1-py3.7_cuda11.3_cudnn8.2.0_0.tar.bz2

conda install --use-local ~/下载/torchvision-0.11.2-py37_cu113.tar.bz2

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 tar -xvzf cmake-3.22.4-linux-x86_64.tar.gz

sudo mv cmake-3.22.4-linux-x86_64 /opt/cmake-3.22.4

#  Create a soft link

sudo ln -sf /opt/cmake-3.22.4/bin/*  /usr/bin/ 

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 cmake --version

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 pip install spconv-cu113

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 git clone https://github.com/traveller59/spconv.git --recursive

cd spconv/

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 pip install pccm

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 python setup.py bdist_wheel

cd ./dist

pip install *

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 git clone https://github.com/open-mmlab/OpenPCDet.git

cd OpenPCDet

#  The download will be relatively slow, so replace it with Tsinghua source to quickly install the dependent python package.

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python setup.py develop 

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 python 

import pcdet

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 cd OpenPCDet/data/kitti

ln -s /media/xiaohu/xiaohu/new\ start/数据集/激光/object/training training

ln -s /media/xiaohu/xiaohu/new\ start/数据集/激光/object/testing testing

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 ├── ImageSets

│   ├── test.txt

│   ├── train.txt

│   └── val.txt

< unk > Testing - > /media/xiaohu/xiaohu/new start/dataset/laser/object/testing

Training - > /media/xiaohu/xiaohu/new start/dataset/laser/object/training

#  training

├── calib

├── image_2

├── label_2

└── velodyne

#  testing

├── calib

├── image_2

└── velodyne

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 cd tools

python train.py --cfg_file cfgs/kitti_models/pointpillar.yaml

#  Doka training

CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch --nproc_per_node=3 tools/train.py --cfg_file tools/cfgs/kitti_models/pointpillar.yaml --launcher pytorch

sh scripts/dist_train.sh 8 --cfg_file tools/cfgs/kitti_models/pointpillar.yaml

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 USE_ROAD_PLANE: False

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 cd output/kitti_models/pointpillar/default/

tensorboard --logdir tensorboard/

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 pip install vtk==8.1.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

python3 -m pip install ~/下载/mayavi-4.7.4.tar.gz -i https://pypi.tuna.tsinghua.edu.cn/simple

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 #inside the OpenPCDet project

cd tools

python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 4 --ckpt ../output/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml  --data_path ../data/kitti/testing/velodyne/000099.bin --ckpt ../output/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 OpenPCDet

├── data

│   ├── nuscenes

│   │   │── v1.0-trainval (or v1.0-mini if you use mini)

│   │   │   │── samples

│   │   │   │── sweeps

│   │   │   │── maps

│ │ │ │── v1.0-trainval  

pcdet

├── tools

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \

    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \

    --version v1.0-trainval

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
#  single card training

cd tools

python train.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml

#  Doka training

CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch --nproc_per_node=3 tools/train.py --cfg_file tools/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml --launcher pytorch

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 #inside the OpenPCDet project

cd tools

python test.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml --batch_size 4 --ckpt ../output/nuscenes_models/cbgs_voxel0075_res3d_centerpoint/default/ckpt/checkpoint_epoch_20.pth

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
  ```python  
 pip install onnx  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install onnxsim  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install onnxruntime  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install ~/3D/TensorRT-8.2.3.0/graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl

pip install ~/3D/TensorRT-8.2.3.0/onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl

  ```  
After clicking on the GitHub Sponsor button above, you will obtain access permissions to my private code repository ( https://github.com/slowlon/my_code_bar ) to view this blog code. By searching the code number of this blog, you can find the code you need, code number is: 2024020309573780930
