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


#include "MapObject.h"
#include "ObjectTrack.h"


namespace ORB_SLAM2 
{
    bool MapObject::Merge(MapObject* obj) {
        // do something with the ellipsoid
        // for now just keep the initial one
        // std::unique_lock<std::mutex> lock(mutex_ellipsoid_);

        this->GetTrack()->Merge(obj->GetTrack());
        auto ret = this->GetTrack()->ReconstructFromCenter(true);
        return ret;
        //  auto checked = tr->CheckReprojectionIoU(0.3);

    }

    void MapObject::RemoveKeyFrameObservation(KeyFrame* kf)
    {
        this->object_track_->RemoveKeyFrame(kf);
    }

}
