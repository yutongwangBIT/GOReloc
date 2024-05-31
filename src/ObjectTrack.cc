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


#include "ObjectTrack.h"

#include "OptimizerObject.h"
#include "ColorManager.h"
#include "MapObject.h"
#include <mutex>
#include <fstream>
#include <memory>

#include "Reconstruction.h"

namespace ORB_SLAM2 
{

unsigned int ObjectTrack::factory_id = 0;
int chair_count = 0;
ObjectTrack::Ptr ObjectTrack::CreateNewObjectTrack(unsigned int cat, const BBox2& bbox, double score, const Matrix34d& Rt, unsigned int frame_idx, Tracking *tracker, KeyFrame *kf) {
    ObjectTrack::Ptr obj = std::make_shared<ObjectTrack>();
    obj->id_ = factory_id++;
    obj->category_id_ = cat;
    obj->bboxes_.reset(max_frames_history);
    obj->Rts_.reset(max_frames_history);
    obj->scores_.reset(max_frames_history);
    obj->bboxes_.push_front(bbox);
    obj->Rts_.push_front(Rt);
    obj->scores_.push_front(score);
    obj->last_obs_score_ = score;
    if (kf) {
        obj->keyframes_bboxes_[kf] = bbox;
        obj->keyframes_scores_[kf] = score;
    }
    obj->last_obs_frame_id_ = frame_idx;
    obj->color_ = RandomUniformColorGenerator::Generate();

    cv::Scalar col(0, 0, 0);
    if (cat == 56) { //chair
        if (chair_count == 0)
            col = cv::Scalar(255, 0, 140);
        else
            col = cv::Scalar(255, 245, 5);

        chair_count++;
    } else if (cat == 26 || cat == 24) { // bag
        col = cv::Scalar(184, 216, 176);
    } else if (cat == 58) { // plant
        col = cv::Scalar(6, 191, 0);
    } else if (cat == 75) { // vase
        col = cv::Scalar(173, 105, 42);
    } else if (cat == 49) { // orange
        col = cv::Scalar(144, 0, 255);
    } else if (cat == 73) { // book
        col = cv::Scalar(0, 110, 255);
    } else if (cat == 41) { // cup
        col = cv::Scalar(255, 0, 0);
    } else if (cat == 39) { // bottle
        col = cv::Scalar(63, 194, 177);
    } else if (cat == 67) { // phone
        col = cv::Scalar(255, 153, 0);
    }

    cv::Scalar bgr(col[2], col[1], col[0]);
    if (!(col[0] == 0 && col[1] == 0 && col[2] == 0))
        obj->color_ =  bgr;

    obj->tracker_ = tracker;
    obj->status_ = ObjectTrackStatus::ONLY_2D;
    obj->unc_ = 0.5; // ===> depends on the variance of the reconstruction gaussian curve
    return obj;
}

ObjectTrack::~ObjectTrack() {
    // if (map_object_) {
    //     delete map_object_;
    // }
}

bool ObjectTrack::ReconstructWithDepth(std::pair<float, float> depth_data){
    std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> Rts = Rts_.to_vector();
    std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes = bboxes_.to_vector();
    float avg_depth = depth_data.first;
    float diff_depth = depth_data.second;
    //CENTER
    auto bb = bboxes[0];
    auto Rt = Rts[0];
    Eigen::Vector2d bb_center = bbox_center(bb);
    auto K_ = tracker_->GetK();
    double u = (bb_center(0) - K_(0, 2)) / K_(0, 0);
    double v = (bb_center(1) - K_(1, 2)) / K_(1, 1);
    Eigen::Vector3d bb_center_cam(u*avg_depth, v*avg_depth, avg_depth);
    auto Rcw = Rt.block<3,3>(0,0);
    auto tcw = Rt.col(3);
    Eigen::Vector3d center_world = Rcw.transpose() * bb_center_cam + (-Rcw.transpose() * tcw);
    //ROTATION
    Eigen::Vector3d zc = bb_center_cam / bb_center_cam.norm();
    Eigen::Vector3d up_vec{0, -1, 0};
    Eigen::Vector3d xc = (-up_vec).cross(zc);
    xc = xc / xc.norm();
    Eigen::Vector3d yc = zc.cross(xc);
    Eigen::Matrix3d rot_cam;
    rot_cam.col(0) = xc;
    rot_cam.col(1) = yc;
    rot_cam.col(2) = zc;
    Eigen::Matrix3d rot_world = Rcw.transpose() * rot_cam;
    //AXES
    double width_in_img = bb[2] - bb[0];
    double height_in_img = bb[3] - bb[1];
    double width_in_world = (width_in_img/K_(0, 0))*avg_depth;
    double height_in_world = (height_in_img/K_(1, 1))*avg_depth;
    Eigen::Vector3d axes(0.5*width_in_world, 0.5*height_in_world, 0.5*diff_depth);
    //Ellipsoid ellipsoid = Ellipsoid(axes, Eigen::Matrix3d::Identity(), center_world);
    Ellipsoid ellipsoid = Ellipsoid(axes, rot_world, center_world);
    if (map_object_){ // if object was already reconstructed return
        map_object_->SetEllipsoid(ellipsoid);
        std::cout<<"object was already reconstructed"<<std::endl;
    }
    else
        map_object_ = std::make_unique<MapObject>(ellipsoid, this);
    
    map_object_2 = new MapObject(ellipsoid, this);
    //map_object_2->SetId(id_);
    
    if (status_ == ObjectTrackStatus::ONLY_2D)
        status_ = ObjectTrackStatus::INITIALIZED;

    return true;
}




bool ObjectTrack::ReconstructFromLandmarks(Map* map)
{

    // get current map points
    std::vector<MapPoint*> points = map->GetAllMapPoints();
    Eigen::Matrix<double, 3, Eigen::Dynamic> pts(3, points.size());
    for(size_t j = 0; j < points.size(); j++) {
            cv::Mat p = points[j]->GetWorldPos();
            pts(0, j) = p.at<float>(0);
            pts(1, j) = p.at<float>(1);
            pts(2, j) = p.at<float>(2);
    }

    std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> Rts = Rts_.to_vector();
    std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes = bboxes_.to_vector();

    auto [status, ellipsoid] = ReconstructEllipsoidFromLandmarks(bboxes, Rts, tracker_->GetK(), pts);

    if (!status)
        return false;
    
    if (map_object_) // if object was already reconstructed return
        map_object_->SetEllipsoid(ellipsoid);
    else
        map_object_ = std::make_unique<MapObject>(ellipsoid, this);

    if (status_ == ObjectTrackStatus::ONLY_2D)
        status_ = ObjectTrackStatus::INITIALIZED;

    return true;
}


bool ObjectTrack::ReconstructCrocco(bool use_two_passes)
{
    if (this->GetAngularDifference() < TO_RAD(10.0)) {
        // std::cerr << "Impossible to triangulate the center: not enough angular difference.\n";
        return false;
    }

    std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> Rts = Rts_.to_vector();
    std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes = bboxes_.to_vector();

    auto [status, ellipsoid] = ReconstructEllipsoidCrocco(bboxes, Rts, tracker_->GetK(), use_two_passes);

    if (!status)
        return false;

    if (map_object_) // if object was already reconstructed return
        map_object_->SetEllipsoid(ellipsoid);
    else
        map_object_ = std::make_unique<MapObject>(ellipsoid, this);

    if (status_ == ObjectTrackStatus::ONLY_2D)
        status_ = ObjectTrackStatus::INITIALIZED;

    return true;
}


bool ObjectTrack::ReconstructFromCenter(bool use_keyframes)
{
    if (this->GetAngularDifference() < TO_RAD(10.0)) {
        //std::cerr << "Impossible to triangulate the center: not enough angular difference.\n";
        return false;
    }

    std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> Rts;
    std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes;

    if (use_keyframes) {
        auto [bbs, poses, _] = this->CopyDetectionsInKeyFrames();
        bboxes = std::move(bbs);
        Rts = std::move(poses);
    } else {
        Rts = Rts_.to_vector();
        bboxes = bboxes_.to_vector();
    }




    auto [status, ellipsoid] = ReconstructEllipsoidFromCenters(bboxes, Rts, tracker_->GetK());

    if (!status)
        return false;

    if (map_object_) // if object was already reconstructed return
        map_object_->SetEllipsoid(ellipsoid);
    else
        map_object_ = std::make_unique<MapObject>(ellipsoid, this);

    if (status_ == ObjectTrackStatus::ONLY_2D)
        status_ = ObjectTrackStatus::INITIALIZED;

    return true;

}


void ObjectTrack::AddDetection(const BBox2& bbox, double score, const Matrix34d& Rt, unsigned int frame_idx, KeyFrame* kf)
{
    unique_lock<mutex> lock(mutex_add_detection_);
    bboxes_.push_front(bbox);
    Rts_.push_front(Rt);
    scores_.push_front(score);
    last_obs_frame_id_ = frame_idx;
    last_obs_score_ = score;
    if (kf) {
        keyframes_bboxes_[kf] = bbox;
        keyframes_scores_[kf] = score;

        std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
        for (size_t i = 0; i < kf->mvKeys.size(); ++i) {
            MapPoint* p = kf->mvpMapPoints[i];
            if (p) {
                auto kp = kf->mvKeys[i];
                if (is_inside_bbox(kp.pt.x , kp.pt.y, bbox)) {
                    if (associated_map_points_.find(p) == associated_map_points_.end())
                        associated_map_points_[p] = 1;
                      else
                        associated_map_points_[p]++;
                }
            }
        }
    }
    if (status_ == ObjectTrackStatus::IN_MAP) {
        // kalman uncertainty update
        double k = unc_ / (unc_ + std::exp(-score));
        unc_ = unc_ * (1.0 - k);
    }


}

std::tuple<std::vector<BBox2, Eigen::aligned_allocator<BBox2>>, std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>, std::vector<double>>
ObjectTrack::CopyDetectionsInKeyFrames()
{
    unique_lock<mutex> lock(mutex_add_detection_);
    std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes;
    std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> poses;
    std::vector<double> scores;
    if (map_object_) {
        bboxes.reserve(keyframes_bboxes_.size());
        poses.reserve(keyframes_bboxes_.size());
        scores.reserve(keyframes_bboxes_.size());
        int i = 0;
        std::vector<KeyFrame*> to_erase;
        for (auto& it : keyframes_bboxes_) {
            if (it.first->isToBeErased() || it.first->isBad()) {
                to_erase.push_back(it.first);
                continue;
            }
            bboxes.push_back(it.second);
            poses.push_back(cvToEigenMatrix<double, float, 3, 4>(it.first->GetPose()));
            scores.push_back(keyframes_scores_[it.first]);
            ++i;
        }
        for (auto* pt : to_erase) {
            keyframes_bboxes_.erase(pt);
            keyframes_scores_.erase(pt);
        }
    }
    return make_tuple(bboxes, poses, scores);
}


void ObjectTrack::OptimizeReconstruction(Map *map)
{
    if (!map_object_) {
        std::cerr << "Impossible to optimize ellipsoid. It first requires a initial reconstruction." << std::endl;
        return ;
    }
    const Ellipsoid& ellipsoid = map_object_->GetEllipsoid();
    const Eigen::Matrix3d& K = tracker_->GetK();

    // std::cout << "===============================> Start ellipsoid optimization " << id_ << std::endl;
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolver_6_1;
    BlockSolver_6_1::LinearSolverType *linear_solver = new g2o::LinearSolverDense<BlockSolver_6_1::PoseMatrixType>();


    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        new BlockSolver_6_1(linear_solver)
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);


    VertexEllipsoidNoRot* vertex = new VertexEllipsoidNoRot();
    // VertexEllipsoid* vertex = new VertexEllipsoid();
    vertex->setId(0);
    Eigen::Matrix<double, 6, 1> e;
    e << ellipsoid.GetAxes(), ellipsoid.GetCenter();
    // EllipsoidQuat ellipsoid_quat = EllipsoidQuat::FromEllipsoid(*ellipsoid_);
    // std::cout << "before optim: " << e.transpose() << "\n";;
    // vertex->setEstimate(ellipsoid_quat);
    vertex->setEstimate(e);
    optimizer.addVertex(vertex);

    auto it_bb = bboxes_.begin();
    auto it_Rt = Rts_.begin();
    auto it_s = scores_.begin();
    for (size_t i = 0; i < bboxes_.size() && it_bb != bboxes_.end() && it_Rt != Rts_.end() && it_s != scores_.end(); ++i, ++it_bb, ++it_Rt, ++it_s) {
        Eigen::Matrix<double, 3, 4> P = K * (*it_Rt);
        EdgeEllipsoidProjection *edge = new EdgeEllipsoidProjection(P, Ellipse::FromBbox(*it_bb), ellipsoid.GetOrientation());
        edge->setId(i);
        edge->setVertex(0, vertex);

        Eigen::Matrix<double, 1, 1> information_matrix = Eigen::Matrix<double, 1, 1>::Identity();
        information_matrix *= *it_s;
        edge->setInformation(information_matrix);
        optimizer.addEdge(edge);
    }
    optimizer.initializeOptimization();
    optimizer.optimize(8);
    Eigen::Matrix<double, 6, 1> ellipsoid_est = vertex->estimate();
    // EllipsoidQuat ellipsoid_quat_est = vertex->estimate();

    // std::cout << "after optim: " << vertex->estimate().transpose() << "\n\n";

    Ellipsoid new_ellipsoid(ellipsoid_est.head(3), Eigen::Matrix3d::Identity(), ellipsoid_est.tail(3));
    map_object_->SetEllipsoid(new_ellipsoid);
}

void ObjectTrack::OptimizeReconstructionQuat()
{
    if (!map_object_) {
        std::cerr << "Impossible to optimize ellipsoid. It first requires a initial reconstruction." << std::endl;
        return ;
    }
    const Ellipsoid& ellipsoid = map_object_->GetEllipsoid();
    const Eigen::Matrix3d& K = tracker_->GetK();

    std::cout << "===============================> Start ellipsoid optimization quat " << id_ << std::endl;
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 1>> BlockSolver;
    BlockSolver::LinearSolverType *linear_solver = new g2o::LinearSolverDense<BlockSolver::PoseMatrixType>();

    // std::cout << "Optim obj " << obj->GetTrack()->GetId()<< "\n";
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        new BlockSolver(linear_solver)
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);


    VertexEllipsoidQuat* vertex = new VertexEllipsoidQuat();
    vertex->setId(0);
    EllipsoidQuat ellipsoid_quat = EllipsoidQuat::FromEllipsoid(ellipsoid);
    vertex->setEstimate(ellipsoid_quat);
    optimizer.addVertex(vertex);

    unsigned int height = tracker_->mImGray.rows;
    unsigned int width = tracker_->mImGray.cols;
    auto it_bb = bboxes_.begin();
    auto it_Rt = Rts_.begin();
    for (size_t i = 0; i < bboxes_.size() && it_bb != bboxes_.end() && it_Rt != Rts_.end(); ++i, ++it_bb, ++it_Rt) {
        Eigen::Matrix<double, 3, 4> P = K * (*it_Rt);
        
        EdgeEllipsoidProjectionQuat *edge = new EdgeEllipsoidProjectionQuat(P, Ellipse::FromBbox(*it_bb), ellipsoid.GetOrientation());
        
        edge->setId(i);
        edge->setVertex(0, vertex);
        Eigen::Matrix<double, 1, 1> information_matrix = Eigen::Matrix<double, 1, 1>::Identity();
        edge->setInformation(information_matrix);

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        optimizer.addEdge(edge);
    }

    
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    EllipsoidQuat ellipsoid_quat_est = vertex->estimate();
    Ellipsoid new_ellipsoid = ellipsoid_quat_est.ToEllipsoid();
    map_object_->SetEllipsoid(new_ellipsoid);
}


bool ObjectTrack::CheckReprojectionIoU(double iou_threshold)
{
    if (status_ != ObjectTrackStatus::INITIALIZED)
        return false;

    const Ellipsoid& ellipsoid = map_object_->GetEllipsoid();
    const Eigen::Matrix3d& K = tracker_->GetK();

    bool valid = true;
    auto it_bb = bboxes_.begin();
    auto it_Rt = Rts_.begin();
    for (; it_bb != bboxes_.end() && it_Rt != Rts_.end(); ++it_bb, ++it_Rt) {
        Eigen::Matrix<double, 3, 4> P = K * (*it_Rt);
        Ellipse ell = ellipsoid.project(P);
        BBox2 proj_bb = ell.ComputeBbox();
        double iou = bboxes_iou(*it_bb, proj_bb);
        if (iou < iou_threshold)
        {
            valid  = false;
            break;
        }
    }
    return valid;
}


double ObjectTrack::CheckReprojectionIoUInKeyFrames(double iou_threshold)
{
    // called form the local mapping thread
    auto [bboxes, Rts, scores] = this->CopyDetectionsInKeyFrames();
    
    const Ellipsoid& ellipsoid = map_object_->GetEllipsoid();
    const Eigen::Matrix3d& K = tracker_->GetK();
    int valid = 0;
    for (size_t i = 0; i < bboxes.size(); ++i)
    {
        Eigen::Matrix<double, 3, 4> P = K * Rts[i];
        Ellipse ell = ellipsoid.project(P);
        BBox2 proj_bb = ell.ComputeBbox();
        double iou = bboxes_iou(bboxes[i], proj_bb);
        if (iou > iou_threshold)
        {
            ++valid;
        }
    }
    double valid_ratio = static_cast<double>(valid) / bboxes.size();
    return valid_ratio;
}

/// Compute the angle between bearing vectors going through the center
/// of the first and latest bboxes
double ObjectTrack::GetAngularDifference() const
{
    Eigen::Vector3d c0 = bbox_center(bboxes_.front()).homogeneous();
    Eigen::Matrix3d K = tracker_->GetK();
    Eigen::Matrix3d K_inv = K.inverse();
    Eigen::Vector3d v0 = K_inv * c0;
    v0.normalize();
    v0 = Rts_.front().block<3, 3>(0, 0).transpose() * v0;
    Eigen::Vector3d c1 = bbox_center(bboxes_.back()).homogeneous();
    Eigen::Vector3d v1 = K_inv * c1;
    v1.normalize();
    v1 = Rts_.back().block<3, 3>(0, 0).transpose() * v1;

    return std::atan2(v0.cross(v1).norm(), v0.dot(v1));
}


void ObjectTrack::Merge(ObjectTrack *track)
{
    std::unique_lock<std::mutex> lock(mutex_add_detection_);

    // std::cout << "second track size = " << track->keyframes_bboxes_.size() << "\n";
    // std::cout << "track size before = " << keyframes_bboxes_.size() << "\n";
    // for (auto kf : track->keyframes_bboxes_) {
    //     if (keyframes_bboxes_.find(kf.first) != keyframes_bboxes_.end()) {
    //         std::cout << "kf already exsits\n";
    //         std::cout << keyframes_bboxes_[kf.first].transpose() << "\n";
    //         std::cout << track->keyframes_bboxes_[kf.first].transpose() << "\n\n";

    //     }
    //     keyframes_bboxes_[kf.first] = kf.second;
    // }

    for (auto kf : track->keyframes_bboxes_) {
        keyframes_bboxes_[kf.first] = kf.second;
        keyframes_scores_[kf.first] = track->keyframes_scores_[kf.first];
    }

    // std::cout << "track size after = " << keyframes_bboxes_.size() << "\n";

    // track->SetIsBad();
}

void ObjectTrack::UnMerge(ObjectTrack *track) {
    for (auto kf : track->keyframes_bboxes_) {
        if (keyframes_bboxes_.find(kf.first) != keyframes_bboxes_.end()) {
            keyframes_bboxes_.erase(kf.first);
            keyframes_scores_.erase(kf.first);
        }
    }
}


void ObjectTrack::CleanBadKeyFrames()
{
    vector<KeyFrame*> to_remove;
    for (auto it : keyframes_bboxes_) {
        if (it.first->isBad() || it.first->isToBeErased()) {
            to_remove.push_back(it.first);
        }
    }
    for (auto* pt : to_remove) {
        keyframes_bboxes_.erase(pt);
        keyframes_scores_.erase(pt);
    }
    std::cout << "Cleaned " << to_remove.size() << " frames.\n";
}

bool ObjectTrack::ReconstructFromSamplesEllipsoid()
{
    // std::ofstream file(std::to_string(id_) + "_reconstruction.txt");
    double dmin = 0.1;
    double dmax = tracker_->GetCurrentMeanDepth() * 2;
    int n_samples = 100;

    const Eigen::Matrix3d& K = tracker_->GetK();
    Eigen::Matrix3d K_inv = K.inverse();

    std::stack<Matrix34d> Rts;
    for (const auto& Rt : Rts_) {
        Rts.push(Rt);
    }

    std::stack<BBox2> bboxes;
    double mean_size = 0.0;
    for (auto const& bb : bboxes_) {
        bboxes.push(bb);
        mean_size = 0.5 * ((bb[2] - bb[0]) / K(0, 0) + (bb[3] - bb[1]) / K(1, 1));
    }
    mean_size /= bboxes_.size();

    auto Rt0 = Rts.top();
    auto bb0 = bboxes.top();
    Rts.pop();
    bboxes.pop();

    Eigen::Matrix3d o0 = Rt0.block<3, 3>(0, 0).transpose();
    Eigen::Vector3d p0 = -o0 * Rt0.block<3, 1>(0, 3);

    std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> samples;
    Eigen::Vector3d dir_cam = K_inv * bbox_center(bb0).homogeneous();
    double step = (dmax - dmin) / n_samples;
    for (int i = 0; i < n_samples; ++i) {
        Eigen::Vector3d X_c = (dmin + i * step) * dir_cam;
        double dx = (bb0[2]-bb0[0]) / K(0, 0);
        double dy = (bb0[3]-bb0[1]) / K(1, 1);
        double dz = (dx + dy) * 0.5;
        Eigen::Vector3d X_w = o0 * X_c + p0;
        Eigen::Vector3d axes(dx, dy, dz);
        samples.push_back(Ellipsoid(axes, o0, X_w));
        // samples.push_back(Ellipsoid(axes, Eigen::Matrix3d::Identity(), X));
    }

    vector<double> accu(n_samples, 0.0);

    while (!Rts.empty()) {
        auto Rt = Rts.top();
        auto bb = bboxes.top();
        Rts.pop();
        bboxes.pop();
        Matrix34d P = K * Rt;
        for (int i = 0; i < n_samples; ++i) {
            Ellipse ell = samples[i].project(P);
            double iou = bboxes_iou(ell.ComputeBbox(), bb);
            accu[i] += iou;
        }

        // write in file
        // for (auto a : accu){
        //     file << a << " ";
            // std::cout << a << " ";
        // }
        // file << "\n";
    }

    // file.close();
    // std::cout << "\n";

    int best_idx = std::distance(accu.begin(), std::max_element(accu.begin(), accu.end()));
    Ellipsoid ellipsoid = samples[best_idx];

    if (map_object_)
        map_object_->SetEllipsoid(ellipsoid);
    else
        map_object_ = std::make_unique<MapObject>(ellipsoid, this);

    if (status_ == ObjectTrackStatus::ONLY_2D)
        status_ = ObjectTrackStatus::INITIALIZED;
    return true;
}


bool ObjectTrack::ReconstructFromSamplesCenter()
{
    std::ofstream file(std::to_string(id_) + "_reconstruction.txt");

    double dmin = 0.1;
    double dmax = 10.0;
    int n_samples = 1000;

    const Eigen::Matrix3d& K = tracker_->GetK();
    Eigen::Matrix3d K_inv = K.inverse();

    std::stack<Matrix34d> Rts;
    for (const auto& Rt : Rts_) {
        Rts.push(Rt);
    }

    std::stack<BBox2> bboxes;
    double mean_size = 0.0;
    for (auto const& bb : bboxes_) {
        bboxes.push(bb);
        mean_size = 0.5 * ((bb[2] - bb[0]) / K(0, 0) + (bb[3] - bb[1]) / K(1, 1));
    }
    mean_size /= bboxes_.size();

    auto Rt0 = Rts.top();
    auto bb0 = bboxes.top();
    Rts.pop();
    bboxes.pop();

    Eigen::Matrix3d o0 = Rt0.block<3, 3>(0, 0).transpose();
    Eigen::Vector3d p0 = -o0 * Rt0.block<3, 1>(0, 3);

    std::vector<Eigen::Vector3d> samples;
    // std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> samples;
    Eigen::Vector3d dir_cam = K_inv * bbox_center(bb0).homogeneous();
    double step = (dmax - dmin) / n_samples;
    for (int i = 0; i < n_samples; ++i) {
        Eigen::Vector3d X_c = (dmin + i * step) * dir_cam;
        Eigen::Vector3d X_w = o0 * X_c + p0;
        samples.push_back(X_w);
    }

    vector<double> accu(n_samples, 0.0);

    while (!Rts.empty()) {
        auto Rt = Rts.top();
        auto bb = bboxes.top();
        Eigen::Vector2d c = bbox_center(bb);
        Rts.pop();
        bboxes.pop();
        Matrix34d P = K * Rt;
        for (int i = 0; i < n_samples; ++i) {
            Eigen::Vector3d p = P * samples[i].homogeneous();
            p /= p[2];
            accu[i] += (p.head<2>() - c).norm();
        }
    }

    for (auto a : accu){
        file << a << "\n";
        // std::cout << a << " ";
    }
    file.close();
    // std::cout << "\n";

    int best_idx = std::distance(accu.begin(), std::min_element(accu.begin(), accu.end()));
    Eigen::Vector3d center = samples[best_idx];
    double d = (Rt0 * center.homogeneous()).z();
    mean_size *= d;
    Ellipsoid ellipsoid(Eigen::Vector3d(mean_size, mean_size, mean_size), Eigen::Matrix3d::Identity(), center);


    if (map_object_)
        map_object_->SetEllipsoid(ellipsoid);
    else
        map_object_ = std::make_unique<MapObject>(ellipsoid, this);

    if (status_ == ObjectTrackStatus::ONLY_2D)
        status_ = ObjectTrackStatus::INITIALIZED;
    return true;
}

void ObjectTrack::TryResetEllipsoidFromMaPoints()
{
    if (!map_object_)
        return;


    std::unordered_map<MapPoint*, int> tmp;
    Eigen::Vector3d axes = map_object_->GetEllipsoid().GetAxes();
    double size_ratio = std::min(std::min(axes[0], axes[1]), axes[2]) / std::max(std::max(axes[0], axes[1]), axes[2]);
    std::cout << "Object " << id_ << " size ratio = " << size_ratio << "\n";

    if (size_ratio < 0.01) {
        degenerated_ellipsoid_++;
    }

    if (degenerated_ellipsoid_ > 10) {
        std::cout << ":!:!!!*********************************************************::!:!:!:!:!:!:!:\n";
        std::cout << "Reconstruct from map points.\n";
        // Re-initialize ellipsoid using PCA on points
        if (associated_map_points_.size() < 5)
            return; // not enough points

        std::vector<Eigen::Vector3d> good_points;
        for (const auto& mp_cnt : associated_map_points_) {
            if (mp_cnt.second * 2 < keyframes_bboxes_.size()) {
                cv::Mat p = mp_cnt.first->GetWorldPos();
                Eigen::Vector3d pt(p.at<float>(0), p.at<float>(1), p.at<float>(2));
                good_points.push_back(pt);
            }
        }


        std::cout << "nb points " << good_points.size() << "\n";

        Eigen::MatrixX3d pts(good_points.size(), 3);
        Eigen::Vector3d center(0, 0, 0);
        for (size_t i = 0; i < good_points.size(); ++i) {
            pts.row(i) = good_points[i].transpose();
            center += good_points[i].transpose();
        }
        std::cout << "points\n" << pts << "\n";
        center /= pts.rows();
        std::cout << "center = " << center.transpose() << "\n";
        Eigen::MatrixX3d pts_centered = pts.rowwise() - center.transpose();


        Eigen::Matrix3d M = pts_centered.transpose() * pts_centered;
        M /= pts_centered.cols();
        std::cout << "M\n" << M << "\n";

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M);
        Eigen::Matrix3d R = solver.eigenvectors();
        Eigen::Vector3d s = 0.5 * solver.eigenvalues().cwiseAbs();//.cwiseSqrt();
        if (R.determinant() < 0) {
            R.col(2) *= -1;
        }
        std::cout << "reconstructed axes = " << s << "\n";

        map_object_->SetEllipsoid(Ellipsoid(s, R, center));
        degenerated_ellipsoid_ = 0;
    }
    /*else {
        auto points = tracker_->GetSystem()->mpMap->GetAllMapPoints();
        Ellipsoid ell = map_object_->GetEllipsoid();
        for (auto* p : points)
        {
            cv::Mat pos = p->GetWorldPos();
            Eigen::Vector3d pt(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
            if (ell.IsInside(pt)) {

            }

        }
    }*/
    // std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
}

void ObjectTrack::AssociatePointsInsideEllipsoid(Map* map)
{
    if (!map_object_)
        return;

    std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
    associated_map_points_.clear();
    size_t nb_kf = keyframes_bboxes_.size();
    auto points = map->GetAllMapPoints();
    Ellipsoid ell = map_object_->GetEllipsoid();
    for (auto* p : points)
    {
        cv::Mat pos = p->GetWorldPos();
        Eigen::Vector3d pt(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
        if (ell.IsInside(pt, 1.0)) {
            associated_map_points_[p] = nb_kf;
        }
    }
}


// void ObjectTrack::OptimizeReconstructionCeres(Map *map)
// {
//     std::cout << "Optim reconstruction " << id_ << "\n";
//     if (!map_object_) {
//         std::cerr << "Impossible to optimize ellipsoid. It first requires a initial reconstruction." << std::endl;
//         return ;
//     }
//     const Ellipsoid& ellipsoid = map_object_->GetEllipsoid();
//     const Eigen::Matrix3d& K = tracker_->GetK();


//     std::cout << "Start ellipsoid optimization" << std::endl;
    
//     const Eigen::Vector3d& axes = ellipsoid.GetAxes();
//     const Eigen::Matrix3d& orientation = ellipsoid.GetOrientation();
//     const Eigen::Vector3d& center = ellipsoid.GetCenter();
//     double params[9] = {
//         axes[0], axes[1], axes[2],
//         center[0], center[1], center[2], 0, 0, 0
//     };
//     // const auto& Q = ellipsoid_->AsDual();
//     // double params[9] = {
//     //     Q(0, 0), Q(0, 1), Q(0, 2), Q(0, 3),
//     //     Q(1, 1), Q(1, 2), Q(1, 3),
//     //     Q(2, 2), Q(2, 3)
//     // };
//     // double params[3] = {
//     //     center[0], center[1], center[2]
//     // };

//     ceres::Problem problem;
//     problem.AddParameterBlock(params, 6);

//     for (size_t i = 0; i < bboxes_.size(); ++i) {
//         Eigen::Matrix<double, 3, 4> P = K * Rts_[i];

//         BBox2 bb = bboxes_[i];
//         Eigen::Vector2d bb_center((bb[0] + bb[2])*0.5, (bb[1] + bb[3])*0.5);

//         problem.AddResidualBlock(
//             new ceres::NumericDiffCostFunction<EllipsoidProjectionError, ceres::CENTRAL, 1, 6>(
//                 new EllipsoidProjectionError(bboxes_[i], P, orientation, axes)
//                 // new EllipsoidProjectionError(bb_center, P, orientation, axes)
//             ), nullptr, params
//         );
//     }

//     problem.SetParameterLowerBound(params, 0, 0.01);
//     problem.SetParameterLowerBound(params, 1, 0.01);
//     problem.SetParameterLowerBound(params, 2, 0.01);


//     ceres::Solver::Options options;
//     options.minimizer_progress_to_stdout = false;
//     ceres::Solver::Summary summary;
//     std::cout << "initial: ";
//     for (int j = 0; j < 9; ++j) 
//         std::cout << params[j] << " "; 
//     std::cout << "\n";
//     Eigen::Matrix<double, 9, 1> v0;
//     v0 << params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8];
//     // v0 << params[0], params[1], params[2];

//     ceres::Solve(options, &problem, &summary);
//     // std::cout << summary.BriefReport() << "\n";
//     std::cout << "final: ";
//     for (int j = 0; j < 9; ++j)
//         std::cout << params[j] << " ";
//     std::cout << "\n\n";
//     // auto v1 = Eigen::Matrix<double, 3, 1>(params);
//     auto v1 = Eigen::Matrix<double, 9, 1>(params);
//     if ((v1-v0).norm() < 1e-3) 
//         std::cout << "############################################### NNNNNOOOOOOOOOOO\n";
//     else
//         std::cout << "############################################### YYYYYEEEEESSSSSSS **********************-----------------*************\n";

//     Ellipsoid new_ellipsoid(Eigen::Vector3d(params[0], params[1], params[2]), orientation, Eigen::Vector3d(params[3], params[4], params[5]));
//     map_object_->SetEllipsoid(new_ellipsoid);

//     optimized = true;
// }







}
;
