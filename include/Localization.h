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


#ifndef LOCALIZATION_H
#define LOCALIZATION_H

#include <vector>
#include <unordered_map>
#include "Ellipsoid.h"
#include "Utils.h"

namespace ORB_SLAM2
{

using Mapping = std::vector<std::vector<size_t>>;

std::pair<Mapping, Mapping> generate_possible_mappings(const std::vector<size_t> objects_category, const std::vector<size_t> detections_category);


std::tuple<int,
std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>,
std::vector<double>,
std::vector<std::vector<std::pair<size_t, size_t>>>,
std::vector<std::vector<std::pair<size_t, size_t>>>>
solveP3P_ransac(const std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>>& ellipsoids,
                const std::vector<size_t>& ellipsoids_categories,
                const std::vector<BBox2, Eigen::aligned_allocator<BBox2>>& detections,
                const std::vector<size_t>& detections_categories, const Eigen::Matrix3d& K, double THRESHOLD);

cv::Mat OptimizePoseFromObjects(const std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>>& ellipses,
                                const std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>>& ellipsoids,
                                cv::Mat Rt, const Eigen::Matrix3d& K);

int OptimizePoseFromObjects(const std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>>& ellipses,
                            const std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>>& ellipsoids,
                            cv::Mat &Rt, double &cost, const Eigen::Matrix3d& K, double th);

int OptimizePoseFromObjectsAndDepth(const std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>>& ellipses,
                                const std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>>& ellipsoids,
                                std::vector<double> &depths, cv::Mat &Rt, double &cost, const Eigen::Matrix3d& K, double th, bool bCheckInliers);

int OptimizePosePnP(const std::vector<cv::Point3f>& objectPoints, 
                    const std::vector<cv::Point2f>& imagePoints,
                    const std::vector<double>& uncertainties,
                    cv::Mat &Rt, 
                    const Eigen::Matrix3d& K);

}

#endif // LOCALIZATION_H