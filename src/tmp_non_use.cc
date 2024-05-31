/*
auto objects = mpMap->GetAllMapObjects();
    std::map<int, int> track_id_to_object_index;
    for (size_t i=0; i<objects.size(); i++){
        auto obj = objects[i];
        track_id_to_object_index[obj->GetTrack()->GetId()] = i;      
        //std::cout<<"track id:"<<obj->GetTrack()->GetId()<<", object index:"<<i<<"\n"; 
    }
    
    double max_iou = 0.0f;
    cv::Mat best_Rt = cv::Mat::eye(4, 4, CV_32F);
    for(auto& [pkf, matches]: kf_and_matches){
        //std::cout<<"matches:"<<matches.size()<<std::endl;
        std::vector<cv::Point3f> objectPoints;
        std::vector<cv::Point2f> imagePoints;
        std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>> ellipses;
        std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> ellipsoids;
        std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes;
        std::vector<int> object_ids;
        std::vector<pair<cv::Point2f, cv::Point2f>> center_points;
        for (auto& [match, score] : matches){
            auto det = current_frame_detections_[match.first];
            Eigen::Vector2d center_2d = bbox_center(det->bbox);
            Eigen::Vector2d center_2d_kf = bbox_center(pkf->graph->attributes[match.second].bbox);
            center_points.push_back(make_pair(cv::Point2f(center_2d(0), center_2d(1)),
                                               cv::Point2f(center_2d_kf(0), center_2d_kf(1))));
            auto track_id = pkf->graph->attributes[match.second].object_id;
            if (track_id_to_object_index.count(track_id) == 0)
                continue;
            auto object_id = track_id_to_object_index[track_id];
            auto obj = objects[object_id];            
            Eigen::Vector3d center_obj =  obj->GetEllipsoid().GetCenter();
            imagePoints.push_back(cv::Point2f(center_2d(0), center_2d(1)));
            objectPoints.push_back(cv::Point3f(center_obj(0), center_obj(1), center_obj(2)));
            bboxes.push_back(det->bbox);
            ellipses.push_back(Ellipse::FromBbox(det->bbox));
            ellipsoids.push_back(obj->GetEllipsoid());
            object_ids.push_back(object_id);
        } 
        if(ellipsoids.size()<4)
            continue; //todo, feature match
        //std::cout<<"object_ids:"<<object_ids.size()<<std::endl;
        cv::Mat distCoeffs;// = cv::Mat::zeros(4,1,CV_32F);
        cv::Mat r, rvec, tvec, tmp_r;
        cv::Mat inliers;
        cv::Mat cv_K_;
        cv::eigen2cv(K_, cv_K_);
        cv::Mat Rt_cv = cv::Mat::eye(4, 4, CV_32F);
        
        bool success = cv::solvePnPRansac(objectPoints, imagePoints, cv_K_, distCoeffs, rvec, tvec, 
                                        false, 50, 10.0f, 0.9, inliers);
        if(!success)
            continue;
        cv::Rodrigues(rvec, r);
        r.copyTo(Rt_cv(cv::Rect(0, 0, 3, 3)));
        tvec.copyTo(Rt_cv(cv::Rect(3, 0, 1, 3)));
        double mean_iou = get_mean_iou(ellipsoids, bboxes, Rt_cv);
        std::cout<<"mean_iou:"<<mean_iou<<std::endl;
        if (mean_iou > max_iou){
            max_iou = mean_iou;
            best_Rt = Rt_cv;
            best_candi = make_pair(pkf, object_ids);
            center_points_for_visual = center_points;
        }
        if(max_iou>0.1)
            break;
        
    }
    if(max_iou < 0.01)
        return false; //TODO PURE POINTS
    mCurrentFrame.SetPose(best_Rt);*/

    /*for (auto obj_ind : best_candi.second){ //tuple<pair<int, int>, int. double>
        auto obj = objects[obj_ind];
        auto ell =  obj->GetEllipsoid();
        // Check that the camera is in front of the scene //maybe not useful currently
        if ((ell.GetCenter() - T_old).dot(R_old.col(2)) < 0)
            continue;
        auto color =  obj->GetTrack()->GetColor();
        auto proj = ell.project(P_frame);
        object_projections_widgets.push_back(ObjectProjectionWidget(proj, 0,
                                                                    0, color,
                                                                    true,
                                                                    0.5));
        //mpFrameDrawer->DrawProjections(im_rgb_, object_projections_widgets);
    }*/