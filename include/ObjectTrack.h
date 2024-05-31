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


#ifndef OBJECT_TRACK_H
#define OBJECT_TRACK_H

#include "Utils.h"

#include <memory>
#include <list>
#include <iostream>

#include <Eigen/Dense>

#include<opencv2/core/core.hpp>

#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
// #include <ceres/ceres.h>

#include "Distance.h"
#include "Ellipse.h"
#include "Ellipsoid.h"
#include "Map.h"
#include "RingBuffer.h"
#include "MapObject.h"

static const size_t max_frames_history = 50;


namespace ORB_SLAM2
{

class Tracking;

enum class ObjectTrackStatus {
    ONLY_2D,
    INITIALIZED,
    IN_MAP,
    FULLY_OPTIMIZED,
    BAD
};

class ObjectTrack
{
public:
    typedef std::shared_ptr<ObjectTrack> Ptr;
    static unsigned int factory_id;

    static ObjectTrack::Ptr CreateNewObjectTrack(unsigned int cat, const BBox2& bbox, double score, const Matrix34d& Rt, unsigned int frame_idx, Tracking *tracker, KeyFrame *kf);

    bool ReconstructFromLandmarks(Map* map);
    bool ReconstructCrocco(bool use_two_passes=true);
    bool ReconstructFromCenter(bool use_keyframes=false);
    bool ReconstructFromSamplesEllipsoid();
    bool ReconstructFromSamplesCenter();
    bool ReconstructWithDepth(std::pair<float, float> depth_data);

    void AddDetection(const BBox2& bbox, double score, const Matrix34d& Rt, unsigned int frame_idx, KeyFrame* kf);

    unsigned int GetCategoryId() const {
        return category_id_;
    }
    
    unsigned int GetId() const {
        return id_;
    }

    // ------- For osmap ------
    void SetId(unsigned int id) {
        id_ = id;
    }

    void SetLastObsFrameId(int frame_id) {
        last_obs_frame_id_ = frame_id;
    }

    void SetColor(const cv::Scalar& color) {
        color_ = color;
    }

    void SetStatus(ObjectTrackStatus status) {
        status_ = status;
    }

    void SetMapObject(MapObject* map_obj) {
        map_object_ = std::unique_ptr<MapObject>(map_obj);
    }
    // ------------------------

    BBox2 GetLastBbox() const {
        return bboxes_.front();
    }

    size_t GetLastObsFrameId() const {
        return last_obs_frame_id_;
    }

    size_t GetNbObservations() const {
        return bboxes_.size();
    }
    size_t GetNbObservationsInKeyFrame() const {
        return keyframes_bboxes_.size();
    }

    cv::Scalar GetColor() const {
        return color_;
    }
    MapObject* GetMapObject() {
        return map_object_.get();
    }

    MapObject* GetMapObject2() {
        return map_object_2;
    }

    ~ObjectTrack();
    
    
    void OptimizeReconstruction(Map* map);
    void OptimizeReconstructionQuat(); //ADDED BY YUTONG

    bool CheckReprojectionIoU(double iou_threshold);
    double CheckReprojectionIoUInKeyFrames(double iou_threshold);

    // void OptimizeReconstructionCeres(Map* map);

    // copy detections in keyframes and remove bad keyframes
    std::tuple<std::vector<BBox2, Eigen::aligned_allocator<BBox2>>, std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>, std::vector<double>>
    CopyDetectionsInKeyFrames();

    void CleanBadKeyFrames();
    void RemoveKeyFrame(KeyFrame* kf) {
        unique_lock<mutex> lock(mutex_add_detection_);
        keyframes_bboxes_.erase(kf);
        keyframes_scores_.erase(kf);
    }

    bool IsBad() const {
        std::unique_lock<std::mutex> lock(mutex_status_);
        return status_ == ObjectTrackStatus::BAD;
    }

    void SetIsBad() {
        std::unique_lock<std::mutex> lock(mutex_status_);
        status_ = ObjectTrackStatus::BAD;
    }

    ObjectTrackStatus GetStatus() const {
        std::unique_lock<std::mutex> lock(mutex_status_); // REALLY NEEDED ?
        return status_;
    }

    double GetAngularDifference() const;

    void Merge(ObjectTrack *track);
    void UnMerge(ObjectTrack *track);

    void InsertInMap(Map *map) {
        std::cout<<"insert in map:"<<id_<<std::endl;
        //map->AddMapObject(map_object_.get());
        map->AddMapObject(map_object_2);
        status_ = ObjectTrackStatus::IN_MAP;
    }

    void ClearTrackingBuffers() {
        last_obs_frame_id_ = -1;
        bboxes_.clear();
        scores_.clear();
        Rts_.clear();
    }

    double GetLastObsScore() const {
        return last_obs_score_;
    }

    std::pair<std::unordered_map<KeyFrame*, BBox2,
                       std::hash<KeyFrame*>,
                       std::equal_to<KeyFrame*>,
                       Eigen::aligned_allocator<std::pair<KeyFrame const*, BBox2>>>, 
                       std::unordered_map<KeyFrame*, double>>
    CopyDetectionsMapInKeyFrames() const {
        unique_lock<mutex> lock(mutex_add_detection_);
        return {keyframes_bboxes_, keyframes_scores_};
    }

    void TryResetEllipsoidFromMaPoints();
    void AssociatePointsInsideEllipsoid(Map* map);


    std::unordered_map<MapPoint*, int> GetAssociatedMapPoints() const {
        std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
        return associated_map_points_;
    }

    std::unordered_map<MapPoint*, int> GetFilteredAssociatedMapPoints(int threshold) const {
        std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
        int limit = threshold; //  std::max(10.0, threshold * keyframes_bboxes_.size());

        std::unordered_map<MapPoint*, int> filtered;
        for (auto mp_cnt : associated_map_points_) {
            if (mp_cnt.second > limit) {
                filtered[mp_cnt.first] = mp_cnt.second;
            }
        }

        if (map_object_) {
            const auto& ellipsoid = map_object_->GetEllipsoid();
            for (auto mp_cnt : associated_map_points_) {
                cv::Mat p = mp_cnt.first->GetWorldPos();
                Eigen::Vector3d pos(p.at<float>(0), p.at<float>(1), p.at<float>(2));
                if (ellipsoid.IsInside(pos, 1.0))
                    filtered[mp_cnt.first] = mp_cnt.second;
            }
        }

        return filtered;
    }


// protected:
public:
    unsigned int category_id_;
    unsigned int id_;
    int last_obs_frame_id_ = -1; // no unsigned int: need to ba able to use -1
    double last_obs_score_ = 0.0;
    cv::Scalar color_;
    std::unique_ptr<MapObject> map_object_ = nullptr;
    MapObject* map_object_2 = nullptr;
    Tracking* tracker_ = nullptr;

    RingBuffer<BBox2, Eigen::aligned_allocator<BBox2>> bboxes_ = RingBuffer<BBox2, Eigen::aligned_allocator<BBox2>>(max_frames_history);
    RingBuffer<Matrix34d, Eigen::aligned_allocator<Matrix34d>> Rts_ = RingBuffer<Matrix34d, Eigen::aligned_allocator<Matrix34d>>(max_frames_history);
    RingBuffer<double> scores_ = RingBuffer<double>(max_frames_history);
    std::unordered_map<KeyFrame*, BBox2,
                       std::hash<KeyFrame*>,
                       std::equal_to<KeyFrame*>,
                       Eigen::aligned_allocator<std::pair<KeyFrame const*, BBox2>>> keyframes_bboxes_;
    std::unordered_map<KeyFrame*, double> keyframes_scores_;
    
    mutable std::mutex mutex_add_detection_;
    mutable std::mutex mutex_status_;
    mutable std::mutex mutex_associated_map_points_;
    ObjectTrackStatus status_ = ObjectTrackStatus::ONLY_2D;
    double unc_ = 0.0;
    std::unordered_map<MapPoint*, int> associated_map_points_;
    int degenerated_ellipsoid_ = 0;

};

}

#endif // OBJECT_TRACK_H
