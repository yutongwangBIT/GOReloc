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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/core/eigen.hpp>
//#include <opencv2/calib3d.hpp>
#include <opencv2/surface_matching/icp.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include "Optimizer.h"
#include "PnPsolver.h"
#include "Ellipsoid.h"
#include "Ellipse.h"
#include "MapObject.h"
#include "LocalObjectMapping.h"
#include "Localization.h"
#include "System.h"
#include "Utils.h"
#include "ARViewer.h"
#include "Graph.h"
#include "ObjectMatcher.h"
#include "p3p.h"
#include <iostream>

#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <unistd.h>
#include <Eigen/Dense>
#include <unistd.h>
#include <dlib/optimization/max_cost_assignment.h>
#include <chrono>


using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

namespace ORB_SLAM2
{


Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{

    //myfile.open ("/home/yutong/OA-SLAM/bin/reloc/reloc.txt");
    myfile.open ("/home/yutong/OA-SLAM/bin/reloc/detection_count_ana.txt");
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);
    K_ = cvToEigenMatrix<double, float, 3, 3>(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    //ADDED
    for(size_t i=0; i<5; i++){
        count_da.push_back(0);
    }
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLocalObjectMapper(LocalObjectMapping *obj_mapper)
{
    local_object_mapper_ = obj_mapper;
}


void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetARViewer(ARViewer *pARViewer)
{
    mpARViewer = pARViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, const std::vector<Detection::Ptr>& detections, bool force_relocalize, bool bProcessDepth)
{
    current_frame_idx_ = (current_frame_idx_ + 1) % (std::numeric_limits<size_t>::max()-1);
    mImGray = imRGB;
    cv::Mat imDepth = imD;
    
    imRGB.copyTo(im_rgb_);

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,bProcessDepth);

    if(imRGB.channels()==1){
        if(cv::countNonZero(imRGB)==0){
            cv::Mat a;
            return a;
        }
    }
    //ADDED TO PROCESS DETECTION TO AVOID SIMILAR DETS
    current_frame_detections_.clear();
    for(auto det1 : detections){
        bool has_similar_det = false;
         //0.2, 300, 0.3, 5, 0.4 for diamond
        //0.2, 300, 0.1, 10, 0.4 for TUM fr2
        /*if(det1->score < 0.2 || bbox_area(det1->bbox) < 300 ||
            bbox_area(det1->bbox) > 0.1*im_rgb_.rows*im_rgb_.cols ||
            is_near_boundary(det1->bbox, im_rgb_.cols, im_rgb_.rows, 10) ||
            bboxes_iou(det1->bbox, det1->ell.ComputeBbox())<0.4) 
            continue;*/
        double thres1, thres2, thres3, thres4, thres5, thres6;
        if(force_relocalize){
            thres1 = 0.1;
            thres2 = 300.0;
            thres3 = 0.3;
            thres4 = 0.4;
            thres5 = 0.3; //0.5
            thres6 = 0.6; //0.8
        }
        else{
            thres1 = 0.2;
            thres2 = 300.0;
            thres3 = 0.3;
            thres4 = 0.4;
            thres5 = 0.3;
            thres6 = 0.6;
        }
        if(det1->score < thres1 || bbox_area(det1->bbox) < thres2 ||
            bbox_area(det1->bbox) > thres3*im_rgb_.rows*im_rgb_.cols ||
            is_near_boundary(det1->bbox, im_rgb_.cols, im_rgb_.rows, 5) ||
            bboxes_iou(det1->bbox, det1->ell.ComputeBbox())<thres4) 
            continue;
        //if(det1->category_id==0 || det1->category_id==56) continue; //person & chair
        for (auto det2 : current_frame_detections_ ){
            double bbox_iou = bboxes_iou(det1->bbox, det2->bbox);
            if(det1->category_id == det2->category_id){
                if(bbox_iou>thres5) //0.3
                    has_similar_det = true;
            }
            else{
                if(bbox_iou>thres6) //0.6
                    has_similar_det = true;
            }
        }
        if(!has_similar_det)
            current_frame_detections_.push_back(det1);
    }

    current_depth_data_per_det_.clear();
    current_depth_data_per_det_.resize(current_frame_detections_.size(), std::make_pair(0.0f, 0.0f));

    for (size_t i = 0; i < current_frame_detections_.size(); ++i) {
        auto det = current_frame_detections_[i];
        auto bbox = det->bbox;
        //choose random pixels to estimate
        int number_pixels = 30;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis_u(bbox[0], bbox[2]);
        std::uniform_int_distribution<> dis_v(bbox[1], bbox[3]);
        float sum_d = 0.0f;
        float min_d = 100.0;
        float max_d = -1.0;
        float count_avg = 0.0f;
        for(int j = 0; j<number_pixels; j++){
            int u = dis_u(gen);
            int v = dis_v(gen);
            float d = imDepth.at<float>(v,u);
            //std::cout<<d<<",";
            if(d>0.0f){
                sum_d += d;
                count_avg += 1.0f;
                if(d<min_d) min_d = d;
                if(d>max_d) max_d = d;
            }
        }
        //std::cout<<std::endl;
        if(count_avg>0.0f){
            current_depth_data_per_det_[i].first = std::min(sum_d/count_avg,5.0f);
            current_depth_data_per_det_[i].second = std::min(std::max(max_d - min_d, 0.05f), 0.2f);
        }
    }

    //ADDED FOR GRAPH
    //std::cout<<"frame idx:"<<mCurrentFrame.mnId<<std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    mCurrentFrame.graph = new Graph();
    std::vector<Eigen::Vector2d> center_points; //TODO 3d points
    for (size_t i = 0; i < current_frame_detections_.size(); ++i) {
        BBox2 box = current_frame_detections_[i]->bbox;
        mCurrentFrame.graph->add_node(i, current_frame_detections_[i]->category_id,
            current_frame_detections_[i]->score, -1.0f, box, current_frame_detections_[i]->ell);
        Eigen::Vector2d center_point = (box.segment(0,2) + box.segment(2,2)) / 2.0;
        center_points.push_back(center_point);
    }
    int base_k = (force_relocalize) ? 5 : 4;
    int k = std::min(base_k, static_cast<int>(current_frame_detections_.size()-1)); //4
    for (size_t i = 0; i < current_frame_detections_.size(); ++i) {
        std::vector<pair<int,double>> distances;
        for (size_t j = 0; j < current_frame_detections_.size(); ++j) {
            if (i != j){
                double distance = (center_points[i] - center_points[j]).norm();
                distances.push_back(make_pair(j,distance));
            }
        }
        sort(distances.begin(),distances.end(),[](auto& left, auto& right) { 
            return left.second < right.second; 
        });
        for (int m=0; m<k; m++){
            mCurrentFrame.graph->add_edge(i, distances[m].first);
        }
    }
    mCurrentFrame.graph->compute_feature_vectors();
    //std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    //double ttrack0 = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    //std::cout<<"graph generation time:"<<ttrack0<<std::endl;
    //mCurrentFrame.graph->computeRandomWalkDescriptors();
    //current_frame_graph_ = mCurrentFrame.graph;
    
    //ENDED GENERATE GRAPH 

    if (force_relocalize)
    {
        auto t1 = high_resolution_clock::now();

        bool bOK = false;
        if (mpSystem->GetRelocalizationMode() == RELOC_POINTS) {
            std::cout << "Relocalize with points.\n";
            bOK = Relocalization();
        } else if (mpSystem->GetRelocalizationMode() == GOReloc_Mode){
            //bOK = VOReloc();
            bOK = GOReloc();
        }


        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;

        mpFrameDrawer->Update(this, true);    
        if (bOK){
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
        }
            
        mpSystem->relocalization_status = bOK;
        if (!bOK) 
        {
            cv::Mat a;
            return a;
        }
    }
    else
    {
        Track(true);
        mpFrameDrawer->Update(this);
    } 
    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp,
                                     const std::vector<Detection::Ptr>& detections, bool force_relocalize)
{
    std::cout<<"GrabImageMonocular:"<<std::endl;
    current_frame_idx_ = (current_frame_idx_ + 1) % (std::numeric_limits<size_t>::max()-1);
    mImGray = im;
    im.copyTo(im_rgb_);

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    current_frame_detections_ = detections;
    std::cout<<"current_frame_detections_:"<<current_frame_detections_.size()<<std::endl;
    current_frame_good_detections_.clear();
    for (auto det : current_frame_detections_) {
        if (det->score > 0.5) {
            // if (det->category_id != 73 && det->score > 0.5 ||  det->score > 0.7) { // for table scene to ignore book on the nappe
            current_frame_good_detections_.push_back(det);
        }
    }
    std::cout<<"current_frame_good_detections_:"<<current_frame_good_detections_.size()<<std::endl;

    if (force_relocalize)
    {

        bool bOK = false;
        if (mpSystem->GetRelocalizationMode() == RELOC_POINTS) {
            std::cout << "Relocalize with points.\n";
            bOK = Relocalization();
        } 

        mpFrameDrawer->Update(this);
        if (bOK)
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
        mpSystem->relocalization_status = bOK;
    }
    else
    {
        std::cout<<"before tracking"<<std::endl;

        Track();
        std::cout<<"after tracking"<<std::endl;
        // if (!mbOnlyTracking)    // if in localization-only mode, no neeed to track objects
        //     break;

        /////////////////////////////////// Objects Tracking ///////////////////////////////////
        // Update mean depth
        if (mState == Tracking::OK) {
            Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
            double z_mean = 0.0;
            unsigned int z_nb = 0;
            for(size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); i++) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP && !mCurrentFrame.mvbOutlier[i]) {
                    cv::Mat mp =  pMP->GetWorldPos();
                    Eigen::Vector4d p(mp.at<float>(0), mp.at<float>(1), mp.at<float>(2), 1.0);
                    Eigen::Vector3d p_cam = Rt * p;
                    z_mean += p_cam[2];
                    z_nb++;
                }
            }
            z_mean /= z_nb;
            // std::cout << "Mean depth = " << z_mean << "\n";
            current_mean_depth_ = z_mean;
        }

        std::cout << "Frame " << current_frame_idx_ << " ===========\n";
        // std::cout << "Created new KF: " << createdNewKeyFrame_ << "\n";
        std::cout << "Nb Object Tracks: " << objectTracks_.size() << "\n";
        std::cout << "Nb Map Objects  : " << mpMap->GetNumberMapObjects() << "\n";
        // for (auto tr : objectTracks_) {
        //     std::cout << "    - tr " << tr->GetId() << " : " << tr->GetNbObservations() << "\n";
        // }

        double MIN_2D_IOU_THRESH = 0.2;
        double MIN_3D_IOU_THRESH = 0.3;
        int TIME_DIFF_THRESH = 30;


        BBox2 img_bbox(0, 0, im.cols, im.rows);

        if (mState == Tracking::OK) {

            // Keep only detections with a certain score
            if (current_frame_good_detections_.size() != 0) {

                KeyFrame *kf = mpLastKeyFrame;
                if (!createdNewKeyFrame_)
                    kf = nullptr;


                // pre-compute all the projections of all ellipsoids which already reconstructed
                Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
                Matrix34d P;
                P = K_ * Rt;
                std::unordered_map<ObjectTrack::Ptr, BBox2> proj_bboxes;
                for (auto tr: objectTracks_) {
                    if (tr->GetStatus() == ObjectTrackStatus::INITIALIZED ||
                        tr->GetStatus() == ObjectTrackStatus::IN_MAP) {
                        MapObject* obj = tr->GetMapObject();
                        Eigen::Vector3d c = obj->GetEllipsoid().GetCenter();
                        double z = Rt.row(2).dot(c.homogeneous());
                        auto ell = obj->GetEllipsoid().project(P);
                        BBox2 bb = ell.ComputeBbox();
                        if (bboxes_intersection(bb, img_bbox) < 0.3 * bbox_area(bb)) {
                            continue;
                        }
                        proj_bboxes[tr] = ell.ComputeBbox();

                        // Check occlusions and keep only the nearest
                        std::unordered_set<ObjectTrack::Ptr> hidden;
                        for (auto it : proj_bboxes) {
                            if (it.first != tr && bboxes_iou(it.second, bb) > 0.9) {
                                Eigen::Vector3d c2 = it.first->GetMapObject()->GetEllipsoid().GetCenter();
                                double z2 = Rt.row(2).dot(c2.homogeneous());
                                if (z < z2) {
                                    // remove z2
                                    hidden.insert(it.first);
                                } else {
                                    // remove z
                                    hidden.insert(tr);
                                }
                                break;
                            }
                        }
                        for (auto hid : hidden) {
                            proj_bboxes.erase(hid);
                        }
                    }
                }

                // find possible tracks
                std::vector<ObjectTrack::Ptr> possible_tracks;
                for (auto tr : objectTracks_) {
                    auto bb = tr->GetLastBbox();
                    if (tr->GetLastObsFrameId() + 60 >= current_frame_idx_ &&
                        bboxes_intersection(bb, img_bbox) >= 0.3 * bbox_area(bb)) {
                        possible_tracks.push_back(tr);
                    } else if (proj_bboxes.find(tr) != proj_bboxes.end()) {
                        possible_tracks.push_back(tr);
                    }
                }

                // Associated map points to each detection
                std::vector<std::unordered_set<MapPoint*>> assoc_map_points(current_frame_good_detections_.size());
                for (size_t i = 0; i < current_frame_good_detections_.size(); ++i) {
                    for (size_t j = 0; j < mCurrentFrame.mvKeysUn.size(); ++j) {
                        if (mCurrentFrame.mvpMapPoints[j]) {
                            const auto& kp = mCurrentFrame.mvKeysUn[j];
                            MapPoint* corresp_map_point = mCurrentFrame.mvpMapPoints[j];
                            if (is_inside_bbox(kp.pt.x, kp.pt.y, current_frame_good_detections_[i]->bbox)) {
                                assoc_map_points[i].insert(corresp_map_point);
                            }
                        }
                    }
                }

                // Try to match detections to existing object track based on the associated map points
                int THRESHOLD_NB_MATCH = 10;
                std::vector<int> matched_by_points(current_frame_good_detections_.size(), -1);
                std::vector<std::vector<size_t>> nb_matched_points(current_frame_good_detections_.size(), std::vector<size_t>());
                for (size_t i = 0; i < current_frame_good_detections_.size(); ++i) {
                    int det_cat = current_frame_good_detections_[i]->category_id;
                    size_t max_nb_matches = 0;
                    size_t best_matched_track = 0;
                    for (size_t j = 0; j < possible_tracks.size(); ++j) {
                        auto tr_map_points = possible_tracks[j]->GetAssociatedMapPoints();
                        size_t n = count_set_map_intersection(assoc_map_points[i], tr_map_points);
                        if (n > max_nb_matches) {
                            max_nb_matches = n;
                            best_matched_track = j;
                        }

                        if (det_cat != possible_tracks[j]->GetCategoryId())
                            n = 0;
                        nb_matched_points[i].push_back(n);
                    }

                    if (max_nb_matches > THRESHOLD_NB_MATCH &&
                        current_frame_good_detections_[i]->category_id == possible_tracks[best_matched_track]->GetCategoryId()) {
                        matched_by_points[i] = best_matched_track;
                    }
                }


                int m = std::max(possible_tracks.size(), current_frame_good_detections_.size());
                dlib::matrix<long> cost = dlib::zeros_matrix<long>(m, m);
                std::vector<long> assignment(m, std::numeric_limits<long>::max()); // Important to have it in 'long', max_int is used to force assignment of tracks already matched using points
                if (current_frame_good_detections_.size() > 0)
                {
                    // std::cout << "Hungarian algorithm size " << m << "\n";
                    for (size_t di = 0; di < current_frame_good_detections_.size(); ++di) {
                        auto det = current_frame_good_detections_[di];

                        for (size_t ti = 0; ti < possible_tracks.size(); ++ti) {
                            auto tr = possible_tracks[ti];
                            if (tr->GetCategoryId() == det->category_id) {
                                double iou_2d = 0;
                                double iou_3d = 0;

                                if (tr->GetLastObsFrameId() + TIME_DIFF_THRESH >= current_frame_idx_)
                                    iou_2d = bboxes_iou(tr->GetLastBbox(), det->bbox);

                                if (proj_bboxes.find(tr) != proj_bboxes.end())
                                    iou_3d = bboxes_iou(proj_bboxes[tr], det->bbox);

                                if (iou_2d < MIN_2D_IOU_THRESH) iou_2d = 0;
                                if (iou_3d < MIN_3D_IOU_THRESH) iou_3d = 0;

                                // std::cout << "2D: " << iou_2d << "\n";
                                // std::cout << "3D: " << iou_3d << "\n";
                                cost(di, ti) = std::max(iou_2d, iou_3d) * 1000;
                            }
                        }

                        if (matched_by_points[di] != -1) {
                            cost(di, matched_by_points[di]) = std::numeric_limits<int>::max();
                        }
                    }

                    // for (size_t i = 0; i < current_frame_good_detections_.size(); ++i) {
                    //     for (size_t j = 0; j < possible_tracks.size(); ++j) {
                    //         // std::cout << i << " " << j << " " << nb_matched_points[i][j] << "\n";
                    //         cost(i, j) += nb_matched_points[i][j] * 1000;
                    //     }
                    // }

                    assignment = dlib::max_cost_assignment(cost); // solve
                }


                std::vector<ObjectTrack::Ptr> new_tracks;
                for (size_t di = 0; di < current_frame_good_detections_.size(); ++di) {
                    auto det = current_frame_good_detections_[di];
                    auto assigned_track_idx = assignment[di];
                    if (assigned_track_idx >= static_cast<long>(possible_tracks.size()) || cost(di, assigned_track_idx) == 0) {
                        // assigned to non-existing => means not assigned
                        auto tr = ObjectTrack::CreateNewObjectTrack(det->category_id, det->bbox, det->score, Rt,
                                                                    current_frame_idx_, this, kf);
                        // std::cout << "create new track " << tr->GetId() << "\n";
                        new_tracks.push_back(tr);
                    } else {
                        ObjectTrack::Ptr associated_track = possible_tracks[assigned_track_idx];
                        associated_track->AddDetection(det->bbox, det->score, Rt, current_frame_idx_, kf);
                        if (kf && associated_track->GetStatus() == ObjectTrackStatus::IN_MAP) {
                            // std::cout << "Add modified objects" << std::endl;
                            if (local_object_mapper_)
                                local_object_mapper_->InsertModifiedObject(associated_track->GetMapObject());
                        }
                    }
                }

                for (auto tr : new_tracks)
                    objectTracks_.push_back(tr);


                if (!mbOnlyTracking) {
                    for (auto& tr : objectTracks_) {
                        if (tr->GetLastObsFrameId() == current_frame_idx_) {
                            // Try reconstruct from points
                            if ((tr->GetNbObservations() > 10 && tr->GetStatus() == ObjectTrackStatus::ONLY_2D) ||
                                (tr->GetNbObservations() % 2 == 0 && tr->GetStatus() == ObjectTrackStatus::INITIALIZED)) {
                                // tr->ReconstructFromSamplesEllipsoid();
                                // tr->ReconstructFromSamplesCenter();

                            bool status_rec = tr->ReconstructFromCenter(); // try to reconstruct and change status to INITIALIZED if success
                            // tr->ReconstructFromLandmarks(mpMap);
                            // tr->ReconstructCrocco(false); // not working
                            if (status_rec)
                                tr->OptimizeReconstruction(mpMap);
                            }
                        }

                        // Try to optimize objects and insert in the map
                        if (tr->GetNbObservations() >= 40 && tr->GetStatus() == ObjectTrackStatus::INITIALIZED) {
                            tr->OptimizeReconstruction(mpMap);
                            // std::cout << "First opimitzation done.\n";
                            auto checked = tr->CheckReprojectionIoU(0.3);
                            // std::cout << "Check reprojection " << checked << ".\n";
                            if (checked) {
                                // Add object to map
                                tr->InsertInMap(mpMap);
                                // Add object in the local object mapping thread to run a fusion checking
                                if (local_object_mapper_)
                                    local_object_mapper_->InsertModifiedObject(tr->GetMapObject());
                            } else {
                                tr->SetIsBad(); // or only reset to ONLY_2D ?
                            }
                        }
                    }
                }
            }

            if (!mbOnlyTracking) {
                // Remove objects that are not tracked anymore and not initialized or in the map
                for (ObjectTrack::Ptr tr : objectTracks_) {
                    if (static_cast<int>(tr->GetLastObsFrameId()) < static_cast<int>(current_frame_idx_) - TIME_DIFF_THRESH
                        && tr->GetStatus() != ObjectTrackStatus::IN_MAP) {
                            tr->SetIsBad();
                        }
                }

                // Clean bad objects
                auto tr_it = objectTracks_.begin();
                while (tr_it != objectTracks_.end()) {
                    auto temp = *tr_it;
                    ++tr_it;
                    if (temp->IsBad())
                        RemoveTrack(temp);
                }
            }
        }

        std::cout << "Object Tracks: " << objectTracks_.size() << std::endl;
        mpFrameDrawer->Update(this);
        std::cout << "finish update " << std::endl;
    }

    if (mpARViewer) { // Update AR viewer camera
        mpARViewer->UpdateFrame(im_rgb_);
        if (mCurrentFrame.mTcw.rows == 4)
            mpARViewer->SetCurrentCameraPose(cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw));
    }

    return mCurrentFrame.mTcw.clone();
}

void Tracking::RemoveTrack(ObjectTrack::Ptr track)
{
    if (track->GetMapObject())
        mpMap->EraseMapObject(track->GetMapObject());
    objectTracks_.remove(track);
}

void Tracking::Track(bool use_object)
{
    createdNewKeyFrame_ = false;
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    // int debug = 0; // for starting a new map without reset

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD){
            StereoInitialization();
            if(use_object)
                ObjectsInitialization();
        }
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }

                // if (bOK)
                //     std::cout << "Tracking is OK\n";
                // else
                //     std::cout << "Tracking failed\n";
            }
            else
            {
                if (mpSystem->GetRelocalizationMode() == RELOC_POINTS) {
                    std::cout << "Relocalize with points.\n";
                    bOK = Relocalization();
                }

                if (bOK)
                    std::cout << "Relocalization is OK\n";
                else {
                    std::cout << "Relocalization failed\n";
                }
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated
            // std::cout << "Tracking: Mapping is disabled \n";
            if(mState==LOST)
            {
                if (mpSystem->GetRelocalizationMode() == RELOC_POINTS) {
                    std::cout << "Relocalize with points.\n";
                    bOK = Relocalization();
                }
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        ObjectMatcher obj_matcher = ObjectMatcher(K_);
        BBox2 img_bbox(0, 0, im_rgb_.cols, im_rgb_.rows);
        std::unordered_map<Object*, Ellipse> proj_bboxes;
        if(use_object && bOK){
            //std::cout<<"mCurrentFrame idx:"<<mCurrentFrame.mnId<<std::endl;
            //SAVE IM FOR DEBUGGING
            Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
            Matrix34d P;
            P = K_ * Rt;
            for(auto obj : mpMap->GetAllObjects()){
                //if(obj->GetLastObsFrameId() + 500 < mCurrentFrame.mnId  && !obj->GetFlagOptimized()){//already M frames not seen and it has been not usually seen
                    //TODO DELETE
                    //continue;
                //}
                auto proj = obj->GetEllipsoid().project(P);
                auto c3d = obj->GetEllipsoid().GetCenter();
                auto bb_proj = proj.ComputeBbox();
                double z = Rt.row(2).dot(c3d.homogeneous());
                if ( z < 0 || bboxes_intersection(bb_proj, img_bbox) < 0.1 * bbox_area(bb_proj) ){
                   // || is_near_boundary(bb_proj, im_rgb_.cols, im_rgb_.rows, -10) ) {
                    continue;
                }
                proj_bboxes[obj] = proj;
                // Check occlusions and keep only the nearest
                std::unordered_set<Object*> hidden;
                for (auto it : proj_bboxes) {
                    if (it.first != obj && bboxes_iou(it.second.ComputeBbox(), bb_proj) > 0.8) {
                        Eigen::Vector3d c2 = it.first->GetEllipsoid().GetCenter();
                        double z2 = Rt.row(2).dot(c2.homogeneous());
                        if (z < z2) {
                            // remove z2
                            hidden.insert(it.first);
                        } else {
                            // remove z
                            hidden.insert(obj);
                        }
                        break;
                    }
                }
                for (auto hid : hidden) {
                    proj_bboxes.erase(hid);
                }
            }

            //for(auto& [obj, proj] : proj_bboxes){
            //    auto axes = proj.GetAxes();
            //    double angle = proj.GetAngle();
            //    auto c = proj.GetCenter();
            //    cv::ellipse(im_rgb_, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, obj->GetColor(), 1);
            //}

            
            //int nmatches = obj_matcher.MatchObjectsByProjection(mCurrentFrame, proj_bboxes);
            //int nmatches = obj_matcher.MatchObjectsHungarian(mCurrentFrame, proj_bboxes);
            //int nmatches = obj_matcher.MatchObjectsIoU(mCurrentFrame, proj_bboxes);
            int nmatches = obj_matcher.MatchObjectsWasserDistance(mCurrentFrame, proj_bboxes);
            //std::cout<<"nmatches:"<<nmatches<<std::endl;

            //TODO: check features association according to object asscociation
            //And ggf. correct the frame poses.
            //if(nmatches>5){
            //    float average_intersect_ratio = obj_matcher.AssociateFeaturesWithObjects(mCurrentFrame);
                /*if(average_intersect_ratio > 0.0f){
                    ORBmatcher matcher(0.9,true);
                    // Project points seen in previous frame
                    int th=10;
                    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);
                }
            //}
                

            Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
            P = K_ * Rt;
            for(size_t i=0; i< mCurrentFrame.mvKeysUn.size(); i++ ){
                auto mp_frame = mCurrentFrame.mvpMapPoints[i];
                if(mp_frame){
                    cv::circle(im_rgb_,mCurrentFrame.mvKeysUn[i].pt,1,cv::Scalar(0,255,0),-1);
                    cv::Mat p3d = mp_frame->GetWorldPos();
                    cv::Mat p3Dc = mCurrentFrame.mTcw(cv::Rect(0, 0, 3, 3))*p3d + mCurrentFrame.mTcw(cv::Rect(3, 0, 1, 3));
                    // Project into Image
                    float invz = 1/p3Dc.at<float>(2);
                    float x = p3Dc.at<float>(0)*invz;;
                    float y = p3Dc.at<float>(1)*invz;;
                    float u = mCurrentFrame.fx*x+mCurrentFrame.cx;
                    float v = mCurrentFrame.fy*y+mCurrentFrame.cy;
                    cv::circle(im_rgb_,cv::Point2f(u, v),1,cv::Scalar(0,255,255),-1);
                    cv::line(im_rgb_, mCurrentFrame.mvKeysUn[i].pt, cv::Point2f(u, v), cv::Scalar(255,255,255), 1, cv::LINE_AA);
                }
            }

            for(auto& [node_id, attribute] : mCurrentFrame.graph->attributes){
                auto bb = attribute.bbox;
                cv::rectangle(im_rgb_, cv::Point2i(bb[0], bb[1]),
                                        cv::Point2i(bb[2], bb[3]),
                                        cv::Scalar(255, 255, 255),
                                        2);
                //auto ell_bb = Ellipse::FromBbox(bb);
                //auto axes_bb = ell_bb.GetAxes();
                //double angle_bb = ell_bb.GetAngle();
                //auto c_bb = ell_bb.GetCenter();
                //cv::ellipse(im_rgb_, cv::Point2f(c_bb[0], c_bb[1]), cv::Size2f(axes_bb[0], axes_bb[1]), TO_DEG(angle_bb), 0, 360, cv::Scalar(255, 255, 255), 1);
                auto ell = attribute.ell;
                auto axes = ell.GetAxes();
                double angle = ell.GetAngle();
                auto c = ell.GetCenter();
                //cv::ellipse(im_rgb_, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, cv::Scalar(255, 255, 255), 2);
                auto obj = attribute.obj;
                if(obj){
                    cv::putText(im_rgb_, std::to_string(attribute.label) + "|"+ std::to_string(node_id) + "|" + std::to_string(obj->GetId()),
                                            cv::Point2i(bb[0]-10, bb[1]-5), cv::FONT_HERSHEY_DUPLEX,
                                            0.2, cv::Scalar(255, 255, 255), 1, false);
                    auto proj = obj->GetEllipsoid().project(P);
                    auto c = proj.GetCenter();
                    auto axes = proj.GetAxes();
                    double angle = proj.GetAngle();
                    cv::ellipse(im_rgb_, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, cv::Scalar(255, 0, 0), 1);
                    auto vIndices_in_box = mCurrentFrame.GetFeaturesInBox(bb[0], bb[2], bb[1], bb[3]);
                    std::set<MapPoint*> tmp_set;
                    for(auto i : vIndices_in_box){
                        auto kp = mCurrentFrame.mvKeysUn[i].pt;
                        auto mp_frame = mCurrentFrame.mvpMapPoints[i];
                        if(mp_frame){
                            //cv::circle(im_rgb_,kp,1,cv::Scalar(0,255,0),-1);
                            tmp_set.insert(mp_frame);
                        }
                        //else
                        //    cv::circle(im_rgb_,kp,1,cv::Scalar(255,255,255),-1);
                    }
                    for(auto mp : obj->GetAssociatedMapPoints()){
                        cv::Mat p3d = mp->GetWorldPos();
                        cv::Mat p3Dc = mCurrentFrame.mTcw(cv::Rect(0, 0, 3, 3))*p3d + mCurrentFrame.mTcw(cv::Rect(3, 0, 1, 3));
                        // Project into Image
                        float invz = 1/p3Dc.at<float>(2);
                        float x = p3Dc.at<float>(0)*invz;;
                        float y = p3Dc.at<float>(1)*invz;;
                        float u = mCurrentFrame.fx*x+mCurrentFrame.cx;
                        float v = mCurrentFrame.fy*y+mCurrentFrame.cy;
                        if(tmp_set.count(mp) == 0)
                            cv::circle(im_rgb_,cv::Point2f(u, v),1,cv::Scalar(0,0,255),-1);
                        //else
                            //cv::circle(im_rgb_,cv::Point2f(u, v),1,cv::Scalar(0,255,255),-1);
                    }   
                            
                }
            }*/

            
        }

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap(use_object);
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap(use_object);
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // // Update drawer
        // mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame()) {
                CreateNewKeyFrame();
                createdNewKeyFrame_ = true;
            }

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        if(use_object && bOK){
            KeyFrame *kf = mpLastKeyFrame;
            if (!createdNewKeyFrame_)
                kf = nullptr;

            /* // PLOT FEATURE POINTS ONLY
            cv::Mat im_tmp;
            im_rgb_.copyTo(im_tmp); 

            const float r = 5;
            for(size_t i=0; i< mCurrentFrame.mvKeysUn.size(); i++ ){
                auto mp_frame = mCurrentFrame.mvpMapPoints[i];
                if(mp_frame){
                    if(mp_frame->isBad()) continue;
                    cv::Point2f pt1,pt2;
                    pt1.x=mCurrentFrame.mvKeysUn[i].pt.x-r;
                    pt1.y=mCurrentFrame.mvKeysUn[i].pt.y-r;
                    pt2.x=mCurrentFrame.mvKeysUn[i].pt.x+r;
                    pt2.y=mCurrentFrame.mvKeysUn[i].pt.y+r;
                    cv::circle(im_tmp,mCurrentFrame.mvKeysUn[i].pt,1,cv::Scalar(0,255,0),-1);
                    cv::rectangle(im_tmp,pt1,pt2,cv::Scalar(0,255,0));
                }
            }

            std::string image_path1 = "/home/yutong/VOOM/im_save/frames/" + to_string(mCurrentFrame.mnId) + ".png";
            cv::imwrite(image_path1.c_str(), im_tmp);

            // PLOT ELLIPSE ONLY
            cv::Mat im_tmp2;
            im_rgb_.copyTo(im_tmp2);
            for(auto [node_id, attribute] : mCurrentFrame.graph->attributes){
                auto bb_det = attribute.bbox;
                cv::rectangle(im_tmp2, cv::Point2i(bb_det[0], bb_det[1]),
                                        cv::Point2i(bb_det[2], bb_det[3]),
                                        cv::Scalar(255, 255, 255),
                                        2);
                auto ell_det = attribute.ell;
                auto c_det = ell_det.GetCenter();
                auto axes_det = ell_det.GetAxes();
                double angle_det = ell_det.GetAngle();
                if(axes_det[0] <= 0.001 || axes_det[1] <= 0.001)
                    continue;
                //cv::ellipse(im_tmp2, cv::Point2f(c_det[0], c_det[1]), cv::Size2f(axes_det[0], axes_det[1]), TO_DEG(angle_det), 0, 360, cv::Scalar(255, 255, 255), 2);
                for (int i = 0; i < 360; i += 16) {
                    cv::ellipse(im_tmp2, cv::Point2f(c_det[0], c_det[1]), cv::Size2f(axes_det[0], axes_det[1]), 
                            TO_DEG(angle_det), i, i+8, cv::Scalar(255, 255, 255), 2);
                }
            } 

            std::string image_path2 = "/home/yutong/VOOM/im_save/frames/" + to_string(mCurrentFrame.mnId) + "_ell.png";
            cv::imwrite(image_path2.c_str(), im_tmp2);

            // PLOT BOTH FEATURE and ELLIPSE
            cv::Mat im_tmp3;
            im_rgb_.copyTo(im_tmp3);
            vector<cv::Scalar> mvColor = std::vector<cv::Scalar>(mCurrentFrame.N, cv::Scalar(0,0,0,0));
            for(auto [node_id, attribute] : mCurrentFrame.graph->attributes){
                auto bb_det = attribute.bbox;
                cv::rectangle(im_tmp3, cv::Point2i(bb_det[0], bb_det[1]),
                                        cv::Point2i(bb_det[2], bb_det[3]),
                                        cv::Scalar(255, 255, 255),
                                        2);
                Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
                Matrix34d P;
                P = K_ * Rt;
                if(attribute.obj){
                    auto proj = attribute.obj->GetEllipsoid().project(P);
                    auto c = proj.GetCenter();
                    auto axes = proj.GetAxes();
                    double angle = proj.GetAngle();
                    cv::ellipse(im_tmp3, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, attribute.obj->GetColor(), 2);
                    vector<size_t> indecies_in_box = mCurrentFrame.GetFeaturesInBox(bb_det[0], bb_det[2], bb_det[1], bb_det[3]);
                    for(auto ind : indecies_in_box){
                        mvColor[ind] = attribute.obj->GetColor();
                    }
                }
            }

            for(size_t i=0; i< mCurrentFrame.mvKeysUn.size(); i++ ){
                auto mp_frame = mCurrentFrame.mvpMapPoints[i];
                if(mp_frame){
                    if(mp_frame->isBad()) continue;
                    cv::Point2f pt1,pt2;
                    pt1.x=mCurrentFrame.mvKeysUn[i].pt.x-r;
                    pt1.y=mCurrentFrame.mvKeysUn[i].pt.y-r;
                    pt2.x=mCurrentFrame.mvKeysUn[i].pt.x+r;
                    pt2.y=mCurrentFrame.mvKeysUn[i].pt.y+r;
                    if(!(mvColor[i][0]==0 && mvColor[i][1]==0 && mvColor[i][2]==0 && mvColor[i][3]==0)){
                        cv::circle(im_tmp3,mCurrentFrame.mvKeysUn[i].pt,1,mvColor[i],-1);
                        cv::rectangle(im_tmp3,pt1,pt2,mvColor[i]);
                    }
                    else{
                        cv::circle(im_tmp3,mCurrentFrame.mvKeysUn[i].pt,1,cv::Scalar(0,255,0),-1);
                        cv::rectangle(im_tmp3,pt1,pt2,cv::Scalar(0,255,0));
                    }
                }
            }
            
            std::string image_path3 = "/home/yutong/VOOM/im_save/frames/" + to_string(mCurrentFrame.mnId) + "_both.png";
            cv::imwrite(image_path3.c_str(), im_tmp3);*/

            if(kf){
                Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
                Matrix34d P;
                P = K_ * Rt;

                /*for(auto obj : mpMap->GetAllObjects()){
                    if(obj->isBad()) continue;
                    auto proj = obj->GetEllipsoid().project(P);
                    auto c3d = obj->GetEllipsoid().GetCenter();
                    auto bb_proj = proj.ComputeBbox();
                    double z = Rt.row(2).dot(c3d.homogeneous());
                    if ( z < 0 || bboxes_intersection(bb_proj, img_bbox) < 0.3 * bbox_area(bb_proj)
                        || is_near_boundary(bb_proj, im_rgb_.cols, im_rgb_.rows, -10) ) {
                        continue;
                    }
                    proj_bboxes[obj] = proj;
                    // Check occlusions and keep only the nearest
                    std::unordered_set<Object*> hidden;
                    for (auto it : proj_bboxes) {
                        if (it.first != obj && bboxes_iou(it.second.ComputeBbox(), bb_proj) > 0.8) {
                            Eigen::Vector3d c2 = it.first->GetEllipsoid().GetCenter();
                            double z2 = Rt.row(2).dot(c2.homogeneous());
                            if (z < z2) {
                                // remove z2
                                hidden.insert(it.first);
                            } else {
                                // remove z
                                hidden.insert(obj);
                            }
                            break;
                        }
                    }
                    for (auto hid : hidden) {
                        proj_bboxes.erase(hid);
                    }
                }

                //for(auto& [obj, proj] : proj_bboxes){
                //    auto axes = proj.GetAxes();
                //    double angle = proj.GetAngle();
                //    auto c = proj.GetCenter();
                    //cv::ellipse(im_rgb_, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, obj->GetColor(), 1);
                //}

                int nmatches = obj_matcher.MatchObjectsWasserDistance(mCurrentFrame, proj_bboxes);*/
                ////////////////////////////////////////////////////////////////////////////////////////////////

                /*for(size_t i=0; i< mCurrentFrame.mvKeysUn.size(); i++ ){
                    auto mp_frame = mCurrentFrame.mvpMapPoints[i];
                    if(mp_frame){
                        if(mp_frame->isBad()) continue;
                        cv::circle(im_rgb_,mCurrentFrame.mvKeysUn[i].pt,1,cv::Scalar(0,255,0),-1);
                        cv::Mat p3d = mp_frame->GetWorldPos();
                        cv::Mat p3Dc = mCurrentFrame.mTcw(cv::Rect(0, 0, 3, 3))*p3d + mCurrentFrame.mTcw(cv::Rect(3, 0, 1, 3));
                        // Project into Image
                        float invz = 1/p3Dc.at<float>(2);
                        float x = p3Dc.at<float>(0)*invz;;
                        float y = p3Dc.at<float>(1)*invz;;
                        float u = mCurrentFrame.fx*x+mCurrentFrame.cx;
                        float v = mCurrentFrame.fy*y+mCurrentFrame.cy;
                        //cv::circle(im_rgb_,cv::Point2f(u, v),1,cv::Scalar(0,255,255),-1);
                        //cv::line(im_rgb_, mCurrentFrame.mvKeysUn[i].pt, cv::Point2f(u, v), cv::Scalar(255,255,255), 1, cv::LINE_AA);
                    }
                }*/

                //for(auto& [obj, proj] : proj_bboxes){
                //    auto axes = proj.GetAxes();
                //    double angle = proj.GetAngle();
                //    auto c = proj.GetCenter();
                //    cv::ellipse(im_rgb_, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, obj->GetColor(), 1);
                //}

                for(auto [node_id, attribute] : mCurrentFrame.graph->attributes){
                    auto bb_det = attribute.bbox;
                    //cv::rectangle(im_rgb_, cv::Point2i(bb_det[0], bb_det[1]),
                    //                        cv::Point2i(bb_det[2], bb_det[3]),
                    //                        cv::Scalar(255, 255, 255),
                    //                        2);
                    if(attribute.obj){
                        //std::cout<<"node "<<node_id<<" is matched with object "<<attribute.obj->GetId()<<std::endl;
                        //check iou again???
                        auto proj = attribute.obj->GetEllipsoid().project(P);
                        auto bb_proj = proj.ComputeBbox();
                        double iou = bboxes_iou(bb_proj, bb_det);
                        auto c3d = attribute.obj->GetEllipsoid().GetCenter();
                        float z = Rt.row(2).dot(c3d.homogeneous());
                        if(iou>0.01 && abs(z-current_depth_data_per_det_[node_id].first)<3.0){
                            auto c = proj.GetCenter();
                            auto axes = proj.GetAxes();
                            double angle = proj.GetAngle();
                            //if(iou<0.3)
                            //    cv::ellipse(im_rgb_, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, cv::Scalar(0, 0, 255), 2);
                            //else
                            //    cv::ellipse(im_rgb_, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, cv::Scalar(0, 255, 0), 2);
                            //cv::ellipse(im_rgb_, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, attribute.obj->GetColor(), 2);
                            attribute.obj->AddDetection(attribute.label, bb_det, attribute.ell, attribute.confidence, Rt, mCurrentFrame.mnId, kf);
                            //attribute.obj->AddDetection(attribute.label, bb_det, Ellipse::FromBbox(bb_det), attribute.confidence, Rt, mCurrentFrame.mnId, kf);
                            //proj_bboxes.erase(attribute.obj);
                            //double dis_min = normalized_gaussian_wasserstein_2d(proj, Ellipse::FromBbox(bb_det), 10);
                            //std::cout<<"wasser:"<<dis_min<<std::endl;
                            //cv::putText(im_rgb_,  std::to_string(attribute.hue), cv::Point2i(bb_det[0]-10, bb_det[1]-5), cv::FONT_HERSHEY_DUPLEX,
                            //    0.55, cv::Scalar(255, 255, 0), 1, false);
                        }
                        else{//TODO??
                            //std::cout<<"BUT IOU IS NOT ENOUGH"<<std::endl;
                            //attribute.obj = nullptr;
                            continue;
                        }
                    }
                }

                //cv::putText(im_rgb_,  std::to_string(average_intersect_ratio), cv::Point(15, 15), cv::FONT_HERSHEY_DUPLEX,
                                //0.55, cv::Scalar(255, 255, 0), 1, false);

                for(auto [node_id, attribute] : kf->graph->attributes){
                    if(!attribute.obj){
                        //std::cout<<"not asscociated node id:"<<node_id<<std::endl;
                        //TODO check if match new
                        //create new
                        Object* obj = new Object(attribute.label, attribute.bbox, attribute.ell, attribute.confidence, current_depth_data_per_det_[node_id], K_,
                                    Rt, mCurrentFrame.mnId, kf);
                        if(obj->GetAssociatedMapPoints().size()<5){
                            delete obj;
                            continue;
                        }
                        mpMap->AddObject(obj);
                        kf->graph->attributes[node_id].obj = obj;
                        auto proj = obj->GetEllipsoid().project(P);
                        auto c = proj.GetCenter();
                        auto axes = proj.GetAxes();
                        double angle = proj.GetAngle();
                        //cv::ellipse(im_rgb_, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, cv::Scalar(0, 255, 255), 2);
                        //cv::ellipse(im_rgb_, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, obj->GetColor(), 2);
                    }
                }
                //std::string image_path = "/home/yutong/OA-SLAM/bin/tmp/paper_fr2/" + to_string(kf->mnId) + ".png";
                //cv::imwrite(image_path.c_str(), im_rgb_);
                //std::cout<<"saved im"<<std::endl;
            }

        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty() && mCurrentFrame.mpReferenceKF)
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}

void Tracking::ObjectsInitialization(){
    if(current_frame_detections_.empty()){
        std::cout<<"WARNING: NO DETECTION IN THE INITIALIZATION FRAME"<<std::endl;
    }
    //std::cout<<"current_frame_detections_ size:"<<current_frame_detections_.size()<<std::endl;
    Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
    int count = 0;
    for (size_t di = 0; di < current_frame_detections_.size(); ++di) {
        auto det = current_frame_detections_[di];
        //std::cout<<current_depth_data_per_det_[di].first<<",";
        if(current_depth_data_per_det_[di].first > 0.01f && current_depth_data_per_det_[di].first < 15.0f){
            Object* obj = new Object(det->category_id, det->bbox, det->ell, det->score, current_depth_data_per_det_[di], K_,
                          Rt, 0, mpReferenceKF);
            //std::cout<<"obj:"<<obj->GetId()<<" has mappoint size:"<<obj->GetAssociatedMapPoints().size()<<std::endl;
            mpMap->AddObject(obj);
            count += 1;
        }
    }
    std::vector<Object*> objects = mpMap->GetAllObjects();
    std::cout<<"count:"<<count<<"Map has "<<objects.size()<<" objects"<<std::endl;
}

void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        //added by YUTONG
        if(!im_rgb_.empty())
            im_rgb_.copyTo(pKFini->im); 

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }
    /*cv::Mat im_kf, img_h;
    mpReferenceKF->im.copyTo(im_kf);
    cv::hconcat(im_rgb_, im_kf, img_h);
    
    for(int i=0; i<mCurrentFrame.mvpMapPoints.size(); i++){
        cv::Point2f p_f0 = mCurrentFrame.mvKeysUn[i].pt;
        cv::circle(img_h, p_f0, 2, cv::Scalar(0,255,255), -1);
        auto mp = mCurrentFrame.mvpMapPoints[i];
        if(!mp)
            continue;
        int ind_mp_in_kf = mp->GetIndexInKeyFrame(mpReferenceKF);
        cv::circle(img_h,mpReferenceKF->mvKeysUn[ind_mp_in_kf].pt+cv::Point2f(im_kf.cols, 0),3,cv::Scalar(0,255,0),-1);
        
        cv::line(img_h,p_f0,mpReferenceKF->mvKeysUn[ind_mp_in_kf].pt+cv::Point2f(640,0),
                    cv::Scalar(0,255,0), 1, cv::LINE_AA); 
    }
    std::string image_path = "/home/yutong/OA-SLAM/bin/tmp/"+to_string(mCurrentFrame.mnId)
                                    +"_"+to_string(mpReferenceKF->mnId)+"_matches"+to_string(nmatchesMap)+".png";
    cv::imwrite(image_path.c_str(), img_h);*/
    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap(bool use_object)
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap(use_object);

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}


bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    //added by YUTONG
    if(!im_rgb_.empty())
        im_rgb_.copyTo(pKF->im); 

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::UpdateLocalMap(bool use_object)
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames(use_object);
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames(bool use_object) //YUTONG TODO
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;

    if(use_object){
        for(auto [node_id, attribute] : mCurrentFrame.graph->attributes){
            Object* obj = attribute.obj;
            if(!obj) continue;
            if(obj->isBad()) continue;
            auto mps = obj->GetAssociatedMapPoints();
            for(auto pMP : mps){
                if(pMP){
                    if(!pMP->isBad()){
                        const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                        for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                            keyframeCounter[it->first]++;
                    }
                }
            }
        }
    }
    
    if(keyframeCounter.size()<5){
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                    for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        /*vector<KeyFrame*> vNeighs = pKF->GetBestObjectCovisibilityKeyFrames(10);

        if(vNeighs.empty()){
            vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        }*/

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::GOReloc(){
    auto objects = mpMap->GetAllObjects();

    if(mCurrentFrame.graph->nodes.size()<4)
        return false;

    std::map<int, std::vector<int>> match_candidates 
                = mCurrentFrame.graph->GetMatchCandidatesOfEachNode(mpMap->graph_3d->feature_vectors, 3);

    int valid_candidates = 0;
    for(auto& [n, candis] : match_candidates){
        if(candis.size()>0)
            valid_candidates++;
    }
    
    int current_num_nodes = mCurrentFrame.graph->nodes.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, current_num_nodes - 1);
    int iterations = 30;
    int n_min_nodes = 3;
    int max_inliers = 0;
    std::map<int, int> bestMatch;
    cv::Mat Rt_mat;
    Graph* graph_2d = mCurrentFrame.graph;
    Graph* graph_3d = mpMap->graph_3d;

    if(valid_candidates<n_min_nodes)
        return false;

    for (int itr = 0; itr < iterations; ++itr) {
        std::vector<int> selected_nodes;
        // 确保选择三个不同的节点
        while (selected_nodes.size() < n_min_nodes) {
            int tmp_node = dis(gen);
            if(match_candidates[tmp_node].size()>0)
                selected_nodes.push_back(tmp_node);
        }

        std::vector<std::map<int, int>> randomMatches = std::vector<std::map<int, int>>();
        int iterations_3d = 0;

        while(randomMatches.size()<10 && iterations_3d<50){
            std::map<int, int> currentMatch; // 3d, 2d
            for(int i=0; i<n_min_nodes; i++){
                int selected_node = selected_nodes[i];
                std::vector<int> tmp_valid_candis = std::vector<int>();
                for(auto& [tmp_2d_node, tmp_3d_node] : currentMatch){
                    bool connected_2d = graph_2d->has_edge(selected_node, tmp_2d_node);
                    for(auto& candi : match_candidates[selected_node]){
                        bool connected_3d = graph_3d->has_edge(candi, tmp_3d_node);
                        if(graph_3d->attributes[candi].obj && connected_2d == connected_3d && candi != tmp_3d_node){
                            tmp_valid_candis.push_back(candi);
                        }
                    }
                }
                if(tmp_valid_candis.empty()){
                    if(i == 0)
                        tmp_valid_candis = match_candidates[selected_node];
                    else
                        break;
                }
                
                //random select a candi in the node
                std::uniform_int_distribution<> cand_dis(0, tmp_valid_candis.size() - 1);
                int candidate_index = cand_dis(gen);
                int candidate = tmp_valid_candis[candidate_index];
                currentMatch[selected_node] = candidate;
            }
            if(currentMatch.size()==n_min_nodes)
                randomMatches.push_back(currentMatch);
            iterations_3d++;
        }

        //std::cout<<"iteration:"<<itr<<std::endl;
        for(auto& currentMatch:randomMatches){
            //std::cout<<std::endl;
            //for(auto& [node_2d, node_3d] : currentMatch){
            //    std::cout<<"("<<node_2d<<", "<<node_3d<<")";
            //}
            //std::cout<<std::endl;
            cv::Mat Rt_tmp = cv::Mat::eye(4, 4, CV_32F);
            double cost = ObjectMatchesPnP(currentMatch, Rt_tmp);
            //std::cout<<"cost:"<<cost<<", T:"<<std::endl;
            if(cost>2) continue;
            //std::cout<<Rt_tmp<<std::endl;
            std::map<int, int> tmp_matches; //2d 3d
            int inliers = UpdateMatches(match_candidates, Rt_tmp, tmp_matches);
            //std::cout<<"inliers:"<<inliers<<std::endl;
            if(inliers > max_inliers){
                max_inliers = inliers;
                bestMatch = tmp_matches;
                Rt_mat = Rt_tmp;
            }
        }
    }

    

    std::cout<<"inliers:"<<max_inliers<<std::endl;
    if(Rt_mat.empty()) return false;
    mCurrentFrame.SetPose(Rt_mat);

    
    double cost = ObjectMatchesPnP(bestMatch, Rt_mat);
    if(cost > 0.0)
        mCurrentFrame.SetPose(Rt_mat);

    mnLastRelocFrameId = mCurrentFrame.mnId;
    return true;
}

double Tracking::ObjectMatchesPnP(std::map<int, int> matches, cv::Mat &Rt){
    std::vector<cv::Point3f> objectPoints_tmp;
    std::vector<cv::Point2f> imagePoints_tmp;
    std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>> ellipses_tmp;
    std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> ellipsoids_tmp;
    std::vector<double> depths_tmp;
    std::vector<pair<int, int>> vec_matches;
    std::vector<double> uncertainties;
    Eigen::Matrix3d K_inv = K_.inverse(); 
    Eigen::Matrix3d X, x;
    int i = 0;

    for(auto& [node_id_2d, candi_3d] : matches){
        auto obj = mpMap->graph_3d->attributes[candi_3d].obj;
        auto bbox = mCurrentFrame.graph->attributes[node_id_2d].bbox;
        Eigen::Vector2d center_2d = bbox_center(bbox);
        Eigen::Vector3d center_obj =  obj->GetEllipsoid().GetCenter();
        imagePoints_tmp.push_back(cv::Point2f(center_2d(0), center_2d(1)));
        objectPoints_tmp.push_back(cv::Point3f(center_obj(0), center_obj(1), center_obj(2)));
        ellipses_tmp.push_back(Ellipse::FromBbox(bbox));
        ellipsoids_tmp.push_back(obj->GetEllipsoid());
        X.col(i) = center_obj;
        x.col(i) = K_inv * center_2d.homogeneous();
        x.col(i).normalize();
        i++;
        int label = mpMap->graph_3d->attributes[candi_3d].label;
        double uncertainty = mpMap->graph_3d->node_cat_frequencies[candi_3d][label];
        uncertainties.push_back(uncertainty);
    }

    if(matches.size() == 3){
        Eigen::Matrix<Eigen::Matrix<double, 3, 4>, 4, 1> solutions;
        monocular_pose_estimator::P3P::computePoses(x, X, solutions);

        double best_cost = 3.0;

        for (size_t idx = 0; idx < 4; ++idx)
        {
            const Eigen::Matrix<double, 3, 4>& p = solutions(idx, 0);
            Eigen::Matrix3d o = p.block<3, 3>(0, 0);
            Eigen::Vector3d pos_est = p.col(3);

            // Check that the camera is in front of the scene
            bool behind_cam = false;
            for(auto ell : ellipsoids_tmp){
                if ((ell.GetCenter() - pos_est).dot(o.col(2)) < 0)
                    behind_cam = true;
            }
            if (behind_cam)
                continue;

            // ----- Evaluate coherence -----
            // project all ellipsoids
            Matrix34d pose, Rt_eigen;
            pose << o, pos_est;
            Rt_eigen << o.transpose(), -o.transpose() * pos_est;

            Matrix34d P_frame = K_ * Rt_eigen;

            double sum_wasser_dist = 0;
            for (int i = 0; i < ellipsoids_tmp.size(); i++){
                auto ell_3d = ellipsoids_tmp[i];
                auto proj_ell = ell_3d.project(P_frame);
                auto ell = ellipses_tmp[i];
                double wasser_dist = 1 - normalized_gaussian_wasserstein_2d(proj_ell, ell, 100);
                sum_wasser_dist += wasser_dist;
            }

            if(sum_wasser_dist<best_cost){
                best_cost = sum_wasser_dist;
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 4; ++j)
                        Rt.at<float>(i, j) = Rt_eigen(i, j);
                Rt.at<float>(3, 3) = 1.0;
            }

        }
        return best_cost;
    }
    else{
        int inliers = OptimizePosePnP(objectPoints_tmp, imagePoints_tmp, uncertainties, Rt, K_);
        if(inliers>0)
          return 1.0;
        else return 0.0;
    }

}

int Tracking::UpdateMatches(std::map<int, std::vector<int>> match_candidates, cv::Mat Rt_mat, std::map<int, int> &tmp_matches){
    Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(Rt_mat);
    Matrix34d P_frame;
    P_frame = K_ * Rt;
    for(auto& [node_id_2d, candidates] : match_candidates){
        if(candidates.empty()) continue;
        int best_candi = -1;
        double min_error = 1.0;
        while(best_candi == -1){
            for(size_t i=0; i<candidates.size(); i++){
                auto candi_3d = candidates[i];
                if(candi_3d < 0) continue;
                auto obj = mpMap->graph_3d->attributes[candi_3d].obj;
                auto bbox = mCurrentFrame.graph->attributes[node_id_2d].bbox;
                //PROJECTION ERROR:
                Eigen::Vector3d c = obj->GetEllipsoid().GetCenter();
                double z = Rt.row(2).dot(c.homogeneous());
                //double depth = current_depth_data_per_det_[node_id_2d].first;
                if(z<0) continue;
                //double depth_error = abs(z-depth);
                //PROJECTION ERROR:
                auto proj = obj->GetEllipsoid().project(P_frame);
                auto error_proj = 1 - normalized_gaussian_wasserstein_2d(Ellipse::FromBbox(bbox), proj, 100); 
                if(error_proj>0.5) continue; //0.5
                double weighted_error = error_proj;
                if(weighted_error<min_error){
                    min_error = weighted_error;
                    best_candi = i;
                }
            }
            if(best_candi>-1){
                auto obj = mpMap->graph_3d->attributes[candidates[best_candi]].obj;
                if(obj->last_obs_ids_and_max_iou.first.first == mCurrentFrame.mnId){
                    double error_last = obj->last_obs_ids_and_max_iou.second;
                    if(min_error > error_last){
                        candidates[best_candi] = -1;
                        min_error = 1.0;
                        best_candi = -1;
                        continue;
                    }//else{
                        //FIND THE BETTER FOR THE OTHER NODE:
                        //auto candidates_another = match_candidates[obj->last_obs_ids_and_max_iou.first.second];
                    //   best_matches.erase(obj->last_obs_ids_and_max_iou.first.second);
                    //}
                }
            }else{
                break;
            }
        }
        if(best_candi>-1){
            auto obj = mpMap->graph_3d->attributes[candidates[best_candi]].obj;
            tmp_matches[node_id_2d] = candidates[best_candi];
            obj->last_obs_ids_and_max_iou.first = std::make_pair(mCurrentFrame.mnId, node_id_2d);
            obj->last_obs_ids_and_max_iou.second = min_error;
        }
    }
    for(auto& [node_2d, candi_3d] : tmp_matches){
        auto obj = mpMap->graph_3d->attributes[candi_3d].obj;
        obj->last_obs_ids_and_max_iou.first = std::make_pair(-1, -1);
    }
    return tmp_matches.size();
}

bool Tracking::Relocalization()
{
    auto objects = mpMap->GetAllObjects();
    ORBmatcher matcher0(0.75,true);

    std::vector<tuple<pair<int, int>, double>> object_matches;
    if(mCurrentFrame.graph->nodes.size()<4){
        return false;
    }
    /*else if(mCurrentFrame.graph->nodes.size()<2){ //when detections not enough, graph vectors are not precise
        for(auto& [node_id, attribute] : mCurrentFrame.graph->attributes){
            int count_bow_matches = 0;
            Object* best_obj = nullptr;
            for(auto obj : objects){
                if(attribute.label == obj->GetCategoryId()){
                    auto set_mps_obj = obj->GetAssociatedMapPoints();
                    std::vector<MapPoint*> mps_obj(set_mps_obj.begin(), set_mps_obj.end());
                    auto bb_det = attribute.bbox;
                    std::vector<pair<size_t, MapPoint*>> map_ind_to_mp = 
                                matcher0.SearchByBoWInBoxGivenMapPoints(mCurrentFrame, mps_obj, bb_det, 100); 
                    if(map_ind_to_mp.size()>count_bow_matches){
                        count_bow_matches = map_ind_to_mp.size();
                        best_obj = obj;
                    }
                }
            }
            if(count_bow_matches>0){
                object_matches.push_back(make_tuple(make_pair(node_id, best_obj->GetId()), count_bow_matches));
            }
        }
    }
    else{
        double si = mCurrentFrame.graph->find_matches_score(mpMap->graph_3d->feature_vectors, object_matches, mCurrentFrame.graph->nodes.size(), 0.1);
    }

    if(object_matches.size()==0){
        return false;
    }*/

    count_useful_frames += 1;
    std::cout << "count_useful_frames:" << count_useful_frames << std::endl;


    std::vector<MapPoint*> mvpLocalMapPoints;
    mLastProcessedState = mState;

    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    std::cout<<"vpCandidateKFs:"<<vpCandidateKFs.size()<<std::endl;

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        //std::cout<<i<<std::endl;
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            //std::cout<<"nmatches:"<<nmatches<<std::endl;
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);
    int most_inliers = -1;
    int keyframe_idx_with_most_inliers = 0;

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            if(nInliers > most_inliers){ //ADDED BY YUTONG
                keyframe_idx_with_most_inliers = i;
                most_inliers = nInliers;
            }

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                        mvpLocalMapPoints.push_back(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    /*std::cout<<"mvpLocalMapPoints:"<<mvpLocalMapPoints.size()<<std::endl;
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // ADDED FOR DEBUG
    cv::Mat img_h;
    cv::Mat im_kf;
    cv::Mat im_f;
    im_rgb_.copyTo(im_f);
    vpCandidateKFs[keyframe_idx_with_most_inliers]->im.copyTo(im_kf);
    cv::hconcat(im_f, im_kf, img_h);

    float si = mpORBVocabulary->score(mCurrentFrame.mBowVec,vpCandidateKFs[keyframe_idx_with_most_inliers]->mBowVec);
    myfile << mCurrentFrame.mnId <<","<< si << ",T" << std::endl;*/
    
    /*for(int i=0; i<mCurrentFrame.mvKeys.size(); i++)
        cv::circle(im_f,mCurrentFrame.mvKeys[i].pt,2,cv::Scalar(255,0,0),-1);
    for(int i=0; i<vpCandidateKFs[keyframe_idx_with_most_inliers]->mvKeysUn.size(); i++)
        cv::circle(im_kf,vpCandidateKFs[keyframe_idx_with_most_inliers]->mvKeysUn[i].pt,2,cv::Scalar(255,0,0),-1);
    cv::hconcat(im_f, im_kf, img_h);
    //for(int i=0; i<mCurrentFrame.mvpMapPoints.size(); i++){
    for(int i=0; i<vvpMapPointMatches[keyframe_idx_with_most_inliers].size(); i++){
        auto mp = mCurrentFrame.mvpMapPoints[i];
        if(!mp)
            continue;
        cv::Mat p3d = mp->GetWorldPos();
        cv::Mat p3Dc = mCurrentFrame.mTcw(cv::Rect(0, 0, 3, 3))*p3d + mCurrentFrame.mTcw(cv::Rect(3, 0, 1, 3));
        // Project into Image
        float invz = 1/p3Dc.at<float>(2);
        float x = p3Dc.at<float>(0)*invz;;
        float y = p3Dc.at<float>(1)*invz;;
        float u = mCurrentFrame.fx*x+mCurrentFrame.cx;
        float v = mCurrentFrame.fy*y+mCurrentFrame.cy;
        cv::circle(img_h,cv::Point2f(u, v),4,cv::Scalar(0,255,0),-1);

        int ind_mp_in_kf = mp->GetIndexInKeyFrame(vpCandidateKFs[keyframe_idx_with_most_inliers]);
        if(ind_mp_in_kf == -1)
            continue;
        cv::circle(img_h,vpCandidateKFs[keyframe_idx_with_most_inliers]->mvKeysUn[ind_mp_in_kf].pt+cv::Point2f(640,0),
                    4,cv::Scalar(0,255,0),-1);
        cv::line(img_h, mCurrentFrame.mvKeys[i].pt, vpCandidateKFs[keyframe_idx_with_most_inliers]->mvKeysUn[ind_mp_in_kf].pt+cv::Point2f(640,0),
                    cv::Scalar(0,255,0), 1, cv::LINE_AA);
        
    }*/
    
    //std::string image_path = "/home/yutong/OA-SLAM/bin/reloc/points/"+
    //                to_string(mCurrentFrame.mnId)+"_"+to_string(bMatch)+".png";
    //cv::imwrite(image_path.c_str(), img_h);

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

bool Tracking::OptPoseByKeyframes(std::vector<KeyFrame*> covKeyFrames, double cost){
    ORBmatcher matcher(0.75, true);
    mvpLocalMapPoints.clear();
    int nKFs = covKeyFrames.size();
    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = covKeyFrames[i];

        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            
            if(nmatches<10) //10
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);
    int most_inliers = -1;
    int keyframe_idx_with_most_inliers = 0;
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            if(nInliers > most_inliers){
                keyframe_idx_with_most_inliers = i;
                most_inliers = nInliers;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10) 
                    continue;

                for(int io = 0; io<mCurrentFrame.N; io++){
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);
                }

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50) 
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,covKeyFrames[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        for(int io = 0; io<mCurrentFrame.N; io++)
                            if(mCurrentFrame.mvbOutlier[io])
                                mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50 )
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(mCurrentFrame,covKeyFrames[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io = 0; io<mCurrentFrame.N; io++){
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);
                                }
                            }
                        }
                    }
                }
                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)// && iou>0.05)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    return bMatch;
}


double Tracking::get_mean_iou(std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> ellipsoids_tmp,
                            std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes_tmp, cv::Mat Rt){
    Eigen::Matrix3d R_old;
    cv::cv2eigen(Rt(cv::Rect(0, 0, 3, 3)), R_old);
    Eigen::Vector3d T_old;
    cv::cv2eigen(Rt(cv::Rect(3, 0, 1, 3)), T_old);
    Matrix34d R_frame;
    R_frame << R_old, T_old;
    Matrix34d P_frame = K_ * R_frame;

    double sum_iou = 0.0;
    //double max_iou = 0.0;
    for (int i = 0; i < ellipsoids_tmp.size(); i++){
        auto ell = ellipsoids_tmp[i];
        auto bbox = bboxes_tmp[i];
        auto proj_bbox = ell.project(P_frame).ComputeBbox();
        double iou = bboxes_iou(bbox, proj_bbox);
        sum_iou += iou;
    }
    return sum_iou/ellipsoids_tmp.size();
}

double Tracking::get_mean_wasser_dist(std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> ellipsoids_tmp,
                            std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>> ellipses_tmp, cv::Mat Rt){
    Eigen::Matrix3d R_old;
    cv::cv2eigen(Rt(cv::Rect(0, 0, 3, 3)), R_old);
    Eigen::Vector3d T_old;
    cv::cv2eigen(Rt(cv::Rect(3, 0, 1, 3)), T_old);
    Matrix34d R_frame;
    R_frame << R_old, T_old;
    Matrix34d P_frame = K_ * R_frame;

    double sum_dist = 0.0;
    for (int i = 0; i < ellipsoids_tmp.size(); i++){
        auto ell_3d = ellipsoids_tmp[i];
        auto proj_ell = ell_3d.project(P_frame);
        auto ell = ellipses_tmp[i];
        double wasser_dis = normalized_gaussian_wasserstein_2d(proj_ell, ell, 100);
        sum_dist += wasser_dis;
    }
    return sum_dist/ellipsoids_tmp.size();
}

bool Tracking::check_ious(std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>>& ellipsoids,
                         std::vector<BBox2, Eigen::aligned_allocator<BBox2>>& bboxes, 
                         cv::Mat Rt, double thres, double &mean_iou){

    Eigen::Matrix3d R_old;
    cv::cv2eigen(Rt(cv::Rect(0, 0, 3, 3)), R_old);
    Eigen::Vector3d T_old;
    cv::cv2eigen(Rt(cv::Rect(3, 0, 1, 3)), T_old);
    Matrix34d R_frame;
    R_frame << R_old, T_old;
    Matrix34d P_frame = K_ * R_frame; 

    double sum_iou = 0.0;
    std::vector<int> inliers = std::vector<int>();

    for (int i = 0; i < ellipsoids.size(); i++){
        auto ell = ellipsoids[i];
        auto bbox = bboxes[i];
        auto proj_bbox = ell.project(P_frame).ComputeBbox();
        double iou = bboxes_iou(bbox, proj_bbox);
        double z = (ell.GetCenter() - T_old).dot(R_old.col(2)); // Check that the camera is in front of the scene too
        if(iou>0.1f && z>0){
            inliers.push_back(i);
            sum_iou += iou;
        }
    }

    //double mean_iou = 0.0;
    if (inliers.empty()){
        mean_iou = 0.0;
        ellipsoids.clear();
        bboxes.clear();
    }
    else{
        mean_iou = sum_iou / inliers.size();
        std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> ellipsoids_tmp;
        std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes_tmp;
        for(auto i: inliers){
            ellipsoids_tmp.push_back(ellipsoids[i]);
            bboxes_tmp.push_back(bboxes[i]);
        }
        ellipsoids.clear();
        bboxes.clear();
        ellipsoids = ellipsoids_tmp;
        bboxes = bboxes_tmp;
    }
    if (mean_iou < thres || bboxes.size()<4)
       return false;
    return true;
}



void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    if (local_object_mapper_) {
        cout << "Reseting Local Object Mapper...";
        local_object_mapper_->RequestReset();
        cout << " done" << endl;
    }

    // Reset Loop Closing
    /*if(mpLoopClosing){
        cout << "Reseting Loop Closing...";
        mpLoopClosing->RequestReset();
        cout << " done" << endl;
    }*/
        

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    objectTracks_.clear();
    mpMap->clear();


    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
