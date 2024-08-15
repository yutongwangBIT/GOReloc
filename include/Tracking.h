/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include"FrameDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include "Utils.h"
#include "Object.h"

#include <nlohmann/json.hpp>

#include <mutex>

#include <Eigen/Dense>

using json = nlohmann::json;

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;
class Detection;
class Ellipsoid;
class Graph;
class Object;
// enum  enumForceRelocalization: int;

class Tracking
{  

public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, const std::vector<Detection::Ptr>& detections, bool force_relocalize=false, bool bProcessDepth=false);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp, const std::vector<Detection::Ptr>& detections, bool force_relocalize);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

    const System* GetSystem() const {
        return mpSystem;
    }


public:
    //ADDED 
    json json_asso = json::array();

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3,
        FORCE_RELOC_POINTS=4,
        FORCE_RELOC_OBJECTS=5,
        FORCE_RELOC_OBJECTS_AND_POINTS=6,
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;
    cv::Mat im_rgb_;

    //ADDED FOR GRAPH RELOC
    //KeyFrame mRelocKeyFrame;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;


    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset();

    unsigned int GetCurrentFrameIdx() const {
        return current_frame_idx_;
    }
    Eigen::Matrix3d GetK() const {
        return K_;
    }

    std::vector<Detection::Ptr> GetCurrentFrameDetections() {
        return current_frame_detections_;
    }

    std::vector<Detection::Ptr> GetCurrentFrameGoodDetections() {
        return current_frame_good_detections_;
    }

    double GetCurrentMeanDepth() const {
        return current_mean_depth_;
    }

    /*Graph* GetCurrentFrameGraph(){
        return current_frame_graph_;
    }*/

protected:

    double dist_mat(cv::Mat a, cv::Mat b) {
        return cv::norm(a - b);
    }
    // Main tracking function. It is independent of the input sensor.
    void Track(bool use_object=false);

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    void ObjectsInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    bool Relocalization();
    //bool VOReloc();
    bool GOReloc();

    void UpdateLocalMap(bool use_object);
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames(bool use_object);

    bool TrackLocalMap(bool use_object);
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    bool check_ious(std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>>& ellipsoids,
                    std::vector<BBox2, Eigen::aligned_allocator<BBox2>>& bboxes, 
                    cv::Mat Rt, double thres, double &mean_iou);

    double get_mean_iou(std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> ellipsoids_tmp,
                        std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes_tmp, cv::Mat Rt);

    double get_mean_wasser_dist(std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> ellipsoids_tmp,
                            std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>> ellipses_tmp, cv::Mat Rt);


    bool OptPoseByKeyframes(std::vector<KeyFrame*> covKeyFrames, double cost);

    int UpdateMatches(std::map<int, std::vector<int>> match_candidates, cv::Mat Rt_mat, std::map<int, int> &tmp_matches);
    double ObjectMatchesPnP(std::map<int, int> matches, cv::Mat &Rt);
    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap;

    //Calibration matrix
    cv::Mat mK;
    Eigen::Matrix3d K_;
    cv::Mat mDistCoef;
    float mbf;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;

    size_t current_frame_idx_ = 0;
    bool createdNewKeyFrame_ = false;
    std::vector<Detection::Ptr> current_frame_detections_;
    std::vector<std::pair<float, float> > current_depth_data_per_det_;
    std::vector<Detection::Ptr> current_frame_good_detections_;
    Graph* current_frame_graph_;
    double current_mean_depth_ = 0.0;

    int count_useful_frames = 0;

    std::vector<int> count_da; //TOTAL, TP. FP. TN, FN
    float sum_iou = 0.0;
    float sum_wasser = 0.0;
    float sum_norm = 0.0;
    float sum_fa = 0.0;

    std::ofstream myfile;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
