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


#ifndef LOCALOBJECTMAPPING_H
#define LOCALOBJECTMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>
#include <unordered_set>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;
class MapPoint;
class MapObject;

class LocalObjectMapping
{
public:
    LocalObjectMapping(Map* pMap, Tracking* tracker);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

    bool CheckModifiedObjects();
    void InsertModifiedObject(MapObject* obj);
    void OptimizeReconstruction(MapObject *obj);

protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    LoopClosing* mpLoopCloser;

    std::list<KeyFrame*> mlNewKeyFrames;

    KeyFrame* mpCurrentKeyFrame;

    std::list<MapPoint*> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;
    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;

    std::mutex mMutexStop;

    Map* mpMap;
    Tracking* tracker_;
    // bool mbAcceptKeyFrames;
    // std::mutex mMutexAccept;

    std::queue<MapObject*> modified_objects_;
    std::unordered_set<MapObject*> modified_objects_set_;
};

} //namespace ORB_SLAM

#endif // LOCALOBJECTMAPPING_H
