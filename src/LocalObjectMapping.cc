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


#include "LocalObjectMapping.h"

#include<mutex>
#include <unistd.h>

#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"

#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "OptimizerObject.h"
#include "MapObject.h"


namespace ORB_SLAM2
{

LocalObjectMapping::LocalObjectMapping(Map *pMap, Tracking *tracker):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false),
    mpMap(pMap), tracker_(tracker)
{
}

bool LocalObjectMapping::CheckModifiedObjects() 
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return !modified_objects_.empty();
}

void LocalObjectMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        // SetAcceptKeyFrames(false);
        // For now LocalObjectMapping always accept newly modified objects

        // while ... do reconstruction
        while (!modified_objects_.empty()) {

            MapObject *obj = modified_objects_.front();

            // obj->GetTrack()->TryResetEllipsoidFromMaPoints();
            // obj->GetTrack()->AssociatePointsInsideEllipsoid(mpMap);

            OptimizeReconstruction(obj);

            const Ellipsoid& ell = obj->GetEllipsoid();
            const Eigen::Vector3d& center = ell.GetCenter();
            BBox3 bb = ell.ComputeBbox();
            auto cat = obj->GetTrack()->GetCategoryId();

            // Check objects fusion
            const std::vector<MapObject*>& map_objects = mpMap->GetAllMapObjects();
            auto pts = obj->GetTrack()->GetFilteredAssociatedMapPoints(10);
            for (auto* obj2 : map_objects)
            {
                if (obj == obj2 || obj2->GetTrack()->GetCategoryId() != cat)
                    continue;

                const Ellipsoid& ell2 = obj2->GetEllipsoid();
                BBox3 bb2 = ell2.ComputeBbox();
                double iou = bboxes3d_iou(bb, bb2);
                auto inter = bboxes3d_intersection(bb, bb2);
                double rel_inter = inter / bbox3d_volume(bb);
                double rel_inter2 = inter / bbox3d_volume(bb2);

                auto pts_2 = obj2->GetTrack()->GetFilteredAssociatedMapPoints(10);
                int nb_common_points = count_map_intersection(pts, pts_2);

                //  if (ell.IsInside(ell2.GetCenter()) || ell2.IsInside(center) || iou > 0.4 || nb_common_points >= 10) { //0.4
                if (ell.IsInside(ell2.GetCenter()) || ell2.IsInside(center) || iou > 0.15 || rel_inter > 0.2 || rel_inter2 > 0.2 || nb_common_points >= 10) { //0.4
                // if (ell.IsInside(ell2.GetCenter()) || ell2.IsInside(center) || iou > 0.3 || rel_inter > 0.4 || rel_inter2 > 0.4 || nb_common_points >= 10) { //0.4
                    auto ellipsoid_save = obj2->GetEllipsoid();
                    auto merge_ok = obj2->Merge(obj);
                    if (merge_ok && obj2->GetTrack()->CheckReprojectionIoUInKeyFrames(0.2) > 0.5) {
                        obj->GetTrack()->SetIsBad(); // object will be removed from the map
                        break;
                    } else {
                        // undo fusion and reset ellipsoid
                        obj2->GetTrack()->UnMerge(obj->GetTrack());
                        obj2->SetEllipsoid(ellipsoid_save);
                    }
                }
            }

            {
                std::unique_lock<std::mutex> lock(mMutexNewKFs);
                modified_objects_.pop();
                modified_objects_set_.erase(obj);
            }
        }


        if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        // SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }
    SetFinish();
}


void LocalObjectMapping::OptimizeReconstruction(MapObject *obj)
{
    if (!obj || !obj->GetTrack())
        return;
    auto [bboxes, Rts, scores] = obj->GetTrack()->CopyDetectionsInKeyFrames();
    const Ellipsoid& ellipsoid = obj->GetEllipsoid();
    const Eigen::Matrix3d& K = tracker_->GetK();

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 1>> BlockSolver;
    BlockSolver::LinearSolverType *linear_solver = new g2o::LinearSolverDense<BlockSolver::PoseMatrixType>();

    // std::cout << "Optim obj " << obj->GetTrack()->GetId()<< "\n";
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        new BlockSolver(linear_solver)
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);


    // VertexEllipsoidNoRot* vertex = new VertexEllipsoidNoRot();
    VertexEllipsoidQuat* vertex = new VertexEllipsoidQuat();
    vertex->setId(0);
    EllipsoidQuat ellipsoid_quat = EllipsoidQuat::FromEllipsoid(ellipsoid);
    vertex->setEstimate(ellipsoid_quat);
    // Eigen::Matrix<double, 6, 1> e;
    // e << ellipsoid.GetAxes(), ellipsoid.GetCenter();
    // vertex->setEstimate(e);
    optimizer.addVertex(vertex);

    unsigned int height = tracker_->mImGray.rows;
    unsigned int width = tracker_->mImGray.cols;
    for (size_t i = 0; i < bboxes.size(); ++i) {
        Eigen::Matrix<double, 3, 4> P = K * Rts[i];
        // EdgeEllipsoidProjection *edge = new EdgeEllipsoidProjection(P, Ellipse::FromBbox(bboxes[i]), ellipsoid.GetOrientation());
        EdgeEllipsoidProjectionQuat *edge = new EdgeEllipsoidProjectionQuat(P, Ellipse::FromBbox(bboxes[i]), ellipsoid.GetOrientation());

        // EdgeEllipsoidProjectionQuatLevelSets *edge = new EdgeEllipsoidProjectionQuatLevelSets(P, Ellipse::FromBbox(bboxes[i]), ellipsoid.GetOrientation());
        // EdgeEllipsoidProjectionQuatAlg *edge = new EdgeEllipsoidProjectionQuatAlg(P, Ellipse::FromBbox(bboxes[i]), ellipsoid.GetOrientation());
        // EdgeEllipsoidProjectionQuatQBBox *edge = new EdgeEllipsoidProjectionQuatQBBox(P, Ellipse::FromBbox(bboxes[i]), ellipsoid.GetOrientation(), width, height);
        // EdgeEllipsoidProjectionQuatTangency *edge = new EdgeEllipsoidProjectionQuatTangency(P, Ellipse::FromBbox(bboxes[i]), ellipsoid.GetOrientation(), width, height);
        // EdgeEllipsoidProjectionBbox *edge = new EdgeEllipsoidProjectionBbox(P, Ellipse::FromBbox(bboxes[i]), ellipsoid.GetOrientation());
        edge->setId(i);
        edge->setVertex(0, vertex);
        Eigen::Matrix<double, 1, 1> information_matrix = Eigen::Matrix<double, 1, 1>::Identity();
        // information_matrix *= scores[i];
        // Eigen::MatrixXd information_matrix = Eigen::MatrixXd::Identity(24, 24);
        edge->setInformation(information_matrix);

        // edge->setMeasurement(*it_bb);
        // edge->setInformation(Eigen::Matrix4d::Identity());
        // edge->setInformation(W);
        // edge->setInformation(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(24, 24));
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        // edge->setRobustKernel(rk);
        // rk->setDelta(0.1);
        optimizer.addEdge(edge);
    }


    // auto mps = obj->GetTrack()->GetFilteredAssociatedMapPoints(0.5);
    // int idx = 0;
    // for (const auto& mp_cnt : mps) {
    //     cv::Mat p = mp_cnt.first->GetWorldPos();
    //     Eigen::Vector3d pos(p.at<float>(0), p.at<float>(1), p.at<float>(2));
    //     EdgeEllipsoidMapPoint *edge = new EdgeEllipsoidMapPoint(pos);
    //     edge->setId(bboxes.size() + idx);
    //     edge->setVertex(0, vertex);
    //     Eigen::Matrix<double, 1, 1> information_matrix = Eigen::Matrix<double, 1, 1>::Identity();
    //     // information_matrix *= scores[i];
    //     edge->setInformation(information_matrix);
    //     optimizer.addEdge(edge);
    //     ++idx;
    // }


    optimizer.initializeOptimization();
    optimizer.optimize(50);
    // Eigen::Matrix<double, 6, 1> ellipsoid_est = vertex->estimate();
    // Ellipsoid new_ellipsoid(ellipsoid_est.head(3), Eigen::Matrix3d::Identity(), ellipsoid_est.tail(3));

    EllipsoidQuat ellipsoid_quat_est = vertex->estimate();
    Ellipsoid new_ellipsoid = ellipsoid_quat_est.ToEllipsoid();

    obj->SetEllipsoid(new_ellipsoid);
}




void LocalObjectMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalObjectMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Object Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalObjectMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalObjectMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}


void LocalObjectMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalObjectMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalObjectMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalObjectMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalObjectMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalObjectMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalObjectMapping::InsertModifiedObject(MapObject* obj)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    if (!obj->GetTrack()->IsBad() && modified_objects_set_.find(obj) == modified_objects_set_.end()) {
        modified_objects_.push(obj);
        modified_objects_set_.insert(obj);
    }
}

void LocalObjectMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    // for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    //     delete *lit;
    // mlNewKeyFrames.clear();

    cout << "Local Object Mapping RELEASE" << endl;
}

} //namespace ORB_SLAM
