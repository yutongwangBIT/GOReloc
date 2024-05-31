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


#ifndef MAP_OBJECT_H
#define MAP_OBJECT_H

#include "Utils.h"

#include <random>
#include <memory>
#include <list>
#include <iostream>

#include <Eigen/Dense>


#include "Ellipse.h"
#include "Ellipsoid.h"
#include "Map.h"


namespace ORB_SLAM2
{

class ObjectTrack;

class MapObject
{
    public:
        MapObject(const Ellipsoid& ellipsoid, ObjectTrack* track) : ellipsoid_(ellipsoid), object_track_(track) {
            //std::cout<<"MapObject track id:"<<object_track_->GetId()<<std::endl;
            id_ = 0;//object_track_->GetId();
        }

        ObjectTrack* GetTrack() const {
            return object_track_;
        }

        const Ellipsoid& GetEllipsoid() const {
            std::unique_lock<std::mutex> lock(mutex_ellipsoid_);
            return ellipsoid_;
        }

        void SetEllipsoid(const Ellipsoid& ell) {
            std::unique_lock<std::mutex> lock(mutex_ellipsoid_);
            ellipsoid_ = ell;
        }

        bool Merge(MapObject* obj);

        void RemoveKeyFrameObservation(KeyFrame* kf);

        unsigned int GetId() const {
            return id_;
        }
        /*void SetId(unsigned int id) {
            std::cout<<"mapobj:"<<id_<<"set id to:"<<id<<std::endl;
            id_ = id;
        }*/

        bool operator<(const MapObject* rhs) const{
            return this->id_ < rhs->GetId();
        }

        


    protected:
        Ellipsoid ellipsoid_;
        ObjectTrack *object_track_ = nullptr;
        mutable std::mutex mutex_ellipsoid_;

        

        unsigned int id_ = 0;

        MapObject() = delete;
};


}

#endif // MAP_OBJECT_H
