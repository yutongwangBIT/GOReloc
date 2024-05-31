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

#include "Map.h"
#include "MapObject.h"
#include "ObjectTrack.h"
#include<mutex>

namespace ORB_SLAM2
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
    graph_3d = new Graph();//nullptr;
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint

    // Remove keyframes from objects observations
    for (auto* obj : map_objects_) {
        // std::cout << "Remove KF " << pKF << " from obj " << obj->GetTrack()->GetId() << std::endl;
        obj->RemoveKeyFrameObservation(pKF);
    }
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::SetTmpPointsPos(const cv::Mat Pos)
{
    unique_lock<mutex> lock(mMutexMap);
    tmpPointsPos = Pos;
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

cv::Mat Map::GetTmpPointsPos()
{
    unique_lock<mutex> lock(mMutexMap);
    return tmpPointsPos;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();

    for(set<MapObject*>::iterator sit=map_objects_.begin(), send=map_objects_.end(); sit!=send; sit++)
        delete *sit;
    map_objects_.clear();
}

void Map::AddObject(Object *obj){
    unique_lock<mutex> lock(mMutexMap);
    mspObjects.insert(obj);
}

std::vector<Object*> Map::GetAllObjects()
{
    unique_lock<mutex> lock(mMutexMap);
    return std::vector<Object*>(mspObjects.begin(),mspObjects.end());
}

void Map::AddMapObject(MapObject *obj)
{
    unique_lock<mutex> lock(mMutexMap);
    std::cout<<"brfore add mapobject"<<std::endl;
    for(auto o:map_objects_){
        std::cout<<o->GetId()<<",";
    }
    std::cout<<std::endl;
    map_objects_.insert(obj);
    std::cout<<"after add mapobject"<<std::endl;
    for(auto o:map_objects_){
        std::cout<<o->GetId()<<",";
    }
    std::cout<<std::endl;
    
    std::cout<<"map added a mo,"<<obj->GetId()<<", total size:"<<map_objects_.size()<<std::endl;
    //m_map_objects_by_tr_id_[tr_id] = obj;
}

/*MapObject* Map::GetObjWithTrId(int tr_id){
    unique_lock<mutex> lock(mMutexMap);
    if(m_map_objects_by_tr_id_.count(tr_id)>0)
        return m_map_objects_by_tr_id_[tr_id];
    else return nullptr;
}*/

vector<MapObject*> Map::GetAllMapObjects()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapObject*>(map_objects_.begin(),map_objects_.end());
}

void Map::EraseMapObject(MapObject *obj)
{
    unique_lock<mutex> lock(mMutexMap);
    std::cout<<"map erase an object,"<<obj->GetId()<<std::endl;
    map_objects_.erase(obj);

    // only erase the pointer
    // ObjectTrack is responsible to memory freeing
}

} //namespace ORB_SLAM
