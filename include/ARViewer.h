/**
* This file is part of OA-SLAM.
*
* Copyright (C) 2022 Matthieu Zins <matthieu.zins@inria.fr>
* (Inria, LORIA, Université de Lorraine)
* OA-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* OA-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with OA-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/



#ifndef ARVIEWER_H
#define ARVIEWER_H

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Tracking.h"
#include "System.h"

#include <mutex>

namespace ORB_SLAM2
{

class Tracking;
class FrameDrawer;
class MapDrawer;
class System;

class ARViewer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    ARViewer(System* pSystem, FrameDrawer* pFrameDrawer, Map* pMap, Tracking *pTracking, const string &strSettingPath);

    // Main thread function. Draw points, keyframes, the current camera pose and the last processed
    // frame. Drawing is refreshed according to the camera fps. We use Pangolin.
    void Run();

    void RequestFinish();

    void RequestStop();

    bool isFinished();

    bool isStopped();

    void Release();

    bool isPaused();

    bool CheckFinish();

    void SetForcePause() {
        force_pause_ = true;
    }

    void SetCurrentCameraPose(const Matrix34d& Rt) {
        unique_lock<mutex> lock(mMutexCamera);
        Rt_.block<3, 4>(0, 0) = Rt;
    }

    void UpdateFrame(cv::Mat img);
    void DrawMapObjects();

    void AddModel(const std::string& model, const Eigen::Matrix4d& Twm) {
        // if (models_.size())
        //     return;
        unique_lock<mutex> lock(mMutexModels);
        models_.push_back(model);
        models_transf_.push_back(Twm);
        models_loaded_.push_back(false);
    }
    Matrix34d IdentifyPlane();
    void DrawLines();

    void SetProperties(bool disp_mesh, bool fix_size) {
        disp_mesh_ = disp_mesh;
        fix_mesh_size_ = fix_size;
    }

private:

    bool Stop();

    System* mpSystem;
    FrameDrawer* mpFrameDrawer;
    Map* mpMap;
    Tracking* mpTracker;

    // 1/fps in ms
    double mT;
    float mImageWidth, mImageHeight;

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    bool mbStopped;
    bool mbStopRequested;
    bool mbPaused;
    std::mutex mMutexStop;

    bool use_class_col_ = false;
    bool force_pause_ = false;
    double fx_, fy_;
    double cx_, cy_;
    double w_, h_;
    double znear_ = 0.01;
    double zfar_ = 10000.0;
    Eigen::Matrix4d K_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d Rt_ = Eigen::Matrix4d::Identity();
    std::mutex mMutexCamera;
    cv::Mat frame_;

    std::mutex mMutexModels;
    std::vector<std::string> models_;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> models_transf_;
    std::vector<bool> models_loaded_;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> lines_;
    std::unordered_map<MapObject*, double> size_by_object_;
    bool disp_mesh_ = true;
    bool fix_mesh_size_ = true;
};

}


#endif // ARVIEWER_H
	

