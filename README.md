# GOReloc
This repository contains the implementation of our [RAL paper](https://ieeexplore.ieee.org/document/10634741): GOReloc: Graph-based Object-Level Data Association for Relocalization. The article introduces a novel method for object-level relocalization of robotic systems. It determines the pose of a camera sensor by robustly associating the object detections in the current frame with 3D objects in a lightweight object-level map. Object graphs, considering semantic uncertainties, are constructed for both the incoming camera frame and the pre-built map. Objects are represented as graph nodes, and each node employs unique semantic descriptors based on our devised graph kernels. We extract a subgraph from the target map graph by identifying potential object associations for each object detection, then refine these associations and pose estimations using a RANSAC-inspired strategy.

The system overview is as follows:

![pipeline](https://github.com/user-attachments/assets/b4631b04-7804-413a-8d92-6c7c9429a530)


## Installation
The dependencies and installation procedures are consistent with those of our previous repository, [VOOM](https://github.com/yutongwangBIT/VOOM).
### Need Install
- **Pangolin**
We use Pangolin for visualization and user interface. Download and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.
- **OpenCV**
We use OpenCV to manipulate images and features. Download and install instructions can be found at: http://opencv.org. Required at least 2.4.3. Tested with OpenCV 2.4.11 and OpenCV 3.2.
- **Eigen3**
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. Required at least 3.1.0.
- **Dlib**
We use Dlib in the comparison object data association method used by OA-SLAM. Download and install instructions can be found at: https://github.com/davisking/dlib
- **Protocol Buffers**
It is used for Osmap (see below). Download and install instructions can be found at: https://github.com/protocolbuffers/protobuf

### Included in the Thirdparty folder
- DBoW2 and g2o 
We use modified versions of the DBoW2 library to perform place recognition and g2o library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the Thirdparty folder.
- [Json](https://github.com/nlohmann/json) for I/O json files.
- [Osmap](https://github.com/AlejandroSilvestri/osmap) for map saving/loading. Modified version to handle objects.

## Compilation

1. Clone the repository recursively:

    ```git clone https://github.com/yutongwangBIT/GOReloc.git```
3. Build:
 
   ```sh build.sh```

## Data
1. [TUM RGBD](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
2. [LM Data](https://peringlab.org/lmdata/) Diamond sequences

We use the same detections in JSON files as in VOOM: https://github.com/yutongwangBIT/VOOM/tree/main/Data. If you would like to process your own dataset, please find the Python scripts in: https://github.com/yutongwangBIT/VOOM/tree/main/PythonScripts

   
## Run our system
All command lines can be found at https://github.com/yutongwangBIT/GOReloc/blob/main/script

An example usage on TUM Fr2_desk sequence:
```
cd bin/
TODO
```

## Code Explanation 
The key components related to this paper include the following:
1. **Graph Construction**: The `Graph` class is implemented in [Graph.cc](https://github.com/yutongwangBIT/GOReloc/blob/main/src/Graph.cc). Implementation of how the graph is constructed can be found in [Tracking::GrabImageRGBD](https://github.com/yutongwangBIT/GOReloc/blob/c87f7a0a6c3c2ca0a2db96b99bdd98f80becd31f/src/Tracking.cc#L370C5-L399C52), where it shows how the graph is built during the tracking phase of RGB-D SLAM. Please note that while we tested the function in RGB-D mode, we actually only use RGB images for relocalization. Consequently, the function can be also compatible with Mono and Stereo modes.
2. **Map Saving and Loading**: We utilized a third-party library, [OSMAP](https://github.com/AlejandroSilvestri/osmap), and extended it to include support for encoding and decoding objects and graphs.
3. **GOReloc Function**: The core implementation of the GOReloc function can be found in [Tracking::GOReloc](https://github.com/yutongwangBIT/GOReloc/blob/c87f7a0a6c3c2ca0a2db96b99bdd98f80becd31f/src/Tracking.cc#L2360C1-L2360C26), corresponding to the methods described in the paper.

## Publication
Please cite the corresponding RA-L paper:

	    @ARTICLE{wang2024ral,
                author={Wang, Yutong and Jiang, Chaoyang and Chen, Xieyuanli},
                journal={IEEE Robotics and Automation Letters}, 
                title={GOReloc: Graph-based Object-Level Relocalization for Visual SLAM}, 
                year={2024},
                volume={},
                number={},
                pages={1-8},
                doi={10.1109/LRA.2024.3442560}
	    }
