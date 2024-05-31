/**
* This file is part of OSMAP.
*
* Copyright (C) 2018-2019 Alejandro Silvestri <alejandrosilvestri at gmail>
* For more information see <https://github.com/AlejandroSilvestri/osmap>
*
* OSMAP is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* OSMAP is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with OSMAP. If not, see <http://www.gnu.org/licenses/>.
*/

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


#include <fstream>
#include <iostream>
#include <assert.h>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <Eigen/Dense>

#include "Osmap.h"
#include "ObjectTrack.h"
#include "LocalObjectMapping.h"
#include "Utils.h"
#include "System.h"

// Option check macro
#define OPTION(OP) if(options[OP]) headerFile << #OP;

// Log variable
#define LOGV(VAR) if(verbose) cout << #VAR << ": " << VAR << endl;

using namespace std;
using namespace cv;

namespace ORB_SLAM2{

Osmap::Osmap(System &_system):
	map(static_cast<OsmapMap&>(*_system.mpMap)),
	tracker(static_cast<OsmapTracking&>(*_system.mpTracker)),
	keyFrameDatabase(*_system.mpKeyFrameDatabase),
	system(_system),
	currentFrame(_system.mpTracker->mCurrentFrame)
{
#ifndef OSMAP_DUMMY_MAP

	/* Every new MapPoint require a dummy pRefKF in its constructor, copying the following parameters:
	 *
	 * - mnFirstKFid(pRefKF->mnId)
	 * - mnFirstFrame(pRefKF->mnFrameId)
	 * - mpRefKF(pRefKF)
	 *
	 * A fake keyframe construction requires a Frame with
	 *
	 *  - mTcw (must be provided, error if not)
	 *  - Grid (already exists)
	 */
    Frame dummyFrame;
    dummyFrame.mTcw = Mat::eye(4, 4, CV_32F);
    pRefKF = new KeyFrame(dummyFrame, &map, &keyFrameDatabase);

#endif
};


void Osmap::mapSave(const string givenFilename, bool pauseThreads){
	// Stop threads
	if(pauseThreads){
		system.mpLocalMapper->RequestStop();
		while(!system.mpLocalMapper->isStopped()) usleep(1000);
	}

	// Strip out .yaml if present
	string baseFilename, filename, pathDirectory;
	parsePath(givenFilename, &filename, &pathDirectory);
	if(pathDirectory != "")
		chdir(pathDirectory.c_str());

	int length = filename.length();
	if(length>5 && filename.substr(length-5) == ".yaml")
	  baseFilename = filename.substr(length-5);
	else
	  baseFilename = filename;

	// Map depuration
	if(!options[NO_DEPURATION])
		depurate();


	// Actual saving
	filename = baseFilename + ".yaml";

	// Open YAML file for write, it will be the last file to close.
	// FileStorage https://docs.opencv.org/3.1.0/da/d56/classcv_1_1FileStorage.html
	FileStorage headerFile(filename, FileStorage::WRITE);
	if(!headerFile.isOpened()){
	// Is this necessary?
	 cerr << "Couldn't create file " << baseFilename << ".yaml, map not saved." << endl;
	 return;
	}

	// MapPoints
	if(!options[NO_MAPPOINTS_FILE]){
	  // Order mappoints by mnId
	  getMapPointsFromMap();

	  // New file
	  filename = baseFilename + ".mappoints";

	  // Serialize
	  cout << "Saving " << filename << endl;
	  headerFile << "mappointsFile" << filename;
	  headerFile << "nMappoints" << MapPointsSave(filename);
	}

	// MapObjects
	// if(!options[NO_MAPPOINTS_FILE]){
	  // Order mappoints track_id
	 // getMapObjectsFromMap();
	  getObjectsFromMap();

	  // New file
	  filename = baseFilename + ".objects";

	  // Serialize
	  cout << "Saving " << filename << endl;
	  headerFile << "objectsFile" << filename;
	  headerFile << "nObjects" << ObjectsSave(filename);
	// }

	// ObjectTracks
	// if(!options[NO_MAPPOINTS_FILE]){
	  // Order mappoints track_id
	//  getObjectTracksFromTracker();

	  // New file
	//  filename = baseFilename + ".objecttracks";

	  // Serialize
	//  cout << "Saving " << filename << endl;
	//  headerFile << "objecttracksFile" << filename;
	//  headerFile << "nObjectTracks" << ObjectTracksSave(filename);
	// }

	// K: grab camera calibration matrices.  Will be saved to yaml file later.
	if(!options[K_IN_KEYFRAME]) getVectorKFromKeyframes();

	// KeyFrames
	if(!options[NO_KEYFRAMES_FILE]){
	  getKeyFramesFromMap();

	  // New file
	  filename = baseFilename + ".keyframes";

	  // Serialize
	  cout << "Saving " << filename << endl;
	  headerFile << "keyframesFile" << filename;
	  headerFile << "nKeyframes" << KeyFramesSave(filename);

	  filename = baseFilename + ".graphs";
	  
	  cout << "Saving " << filename << endl;
	  headerFile << "graphsFile" << filename;
	  headerFile << "nGraphs" << GraphsSave(filename);
	}

	

	// Features
	if(!options[NO_FEATURES_FILE]){
	  filename = baseFilename + ".features";
	  cout << "Saving " << filename << endl;
	  headerFile << "featuresFile" << filename;
	  headerFile << "nFeatures" << featuresSave(filename);
	}


	// Save options, as an int
	headerFile << "Options" << (int) options.to_ulong();
	// Options
	if(options.any()){
	headerFile << "Options descriptions" << "[:";
	OPTION(NO_LOOPS)
	OPTION(NO_FEATURES_DESCRIPTORS)
	OPTION(K_IN_KEYFRAME)
	OPTION(ONLY_MAPPOINTS_FEATURES)
	OPTION(FEATURES_FILE_DELIMITED)
	OPTION(FEATURES_FILE_NOT_DELIMITED)
	OPTION(NO_MAPPOINTS_FILE)
	OPTION(NO_KEYFRAMES_FILE)
	OPTION(NO_FEATURES_FILE)
	headerFile << "]";
	}


	// K: camera calibration matrices, save to yaml at the end of file.
	if(!options[K_IN_KEYFRAME]){
	// Save K matrices in header file yaml
	headerFile << "cameraMatrices" << "[";
	for(auto pK:vectorK)
	   headerFile << "{:"  << "fx" << pK->at<float>(0,0) << "fy" << pK->at<float>(1,1) << "cx" << pK->at<float>(0,2) << "cy" << pK->at<float>(1,2) << "}";
	headerFile << "]";
	}

	// Save yaml file
	headerFile.release();

	// Clear temporary vectors
	clearVectors();

	if(pauseThreads && system.mpViewer)
	  system.mpViewer->Release();
}

void Osmap::mapLoad(string yamlFilename, bool noSetBad, bool pauseThreads, string mSensor){
	//std::cout << "osmap -3" << std::endl;
#ifndef OSMAP_DUMMY_MAP
	LOGV(system.mpTracker->mState)
	// Initialize currentFrame via calling GrabImageMonocular just in case, with a dummy image.
	if(system.mpTracker->mState == ORB_SLAM2::Tracking::NO_IMAGES_YET){
		Mat m = Mat::zeros(100, 100, CV_8U);
		if(mSensor == "mono"){
			system.mpTracker->GrabImageMonocular(m, 0.0, {}, false);
		}
		else if(mSensor == "rgbd"){
			system.mpTracker->GrabImageRGBD(m, m, 0.0, {});
		}
	}
#endif
	//std::cout << "osmap -2" << std::endl;
	if(pauseThreads){
		// Reset thr tracker to clean the map
		system.mpLocalMapper->Release();	// Release local mapper just in case it's stopped, because if it is stopped it can't be reset
		if (system.local_object_mapper_) system.local_object_mapper_->Release();	// Release local mapper just in case it's stopped, because if it is stopped it can't be reset
		system.mpTracker->Reset();
		// Here the system is reset, state is NO_IMAGE_YET

		// Stop LocalMapping and Viewer
		system.mpLocalMapper->RequestStop();
		if (system.local_object_mapper_) system.local_object_mapper_->RequestStop();
		if (system.mpViewer) system.mpViewer	    ->RequestStop();
		while(!system.mpLocalMapper->isStopped()) usleep(1000);
		//if (system.local_object_mapper_) while(!system.local_object_mapper_->isStopped()) usleep(1000);
		if (system.mpViewer) while(!system.mpViewer     ->isStopped()) usleep(1000);
	}
	//std::cout << "osmap -1" << std::endl;
#if !defined OSMAP_DUMMY_MAP && !defined OS1
	if(system.mpTracker->mlRelativeFramePoses.empty()){
		// Add dummy point to trajectory recorder to avoid errors.  The point is in the origin of the map's reference system.
		system.mpTracker->mlRelativeFramePoses.push_back(cv::Mat::eye(4,4,CV_32F));
		system.mpTracker->mlpReferences.push_back(NULL);
		system.mpTracker->mlFrameTimes.push_back(0.0);
		system.mpTracker->mlbLost.push_back(true);
	}
#endif
//std::cout << "osmap -0.5" << std::endl;
	LOGV(system.mpLocalMapper->isStopped())
	if (system.local_object_mapper_) LOGV(system.local_object_mapper_->isStopped())
	if (system.mpViewer) LOGV(system.mpViewer     ->isStopped())

	string filename;
	int intOptions;

	// Open YAML
	//std::cout << "osmap 0" << std::endl;
	cv::FileStorage headerFile(yamlFilename, cv::FileStorage::READ);
	//std::cout << "osmap 1" << std::endl;
	// Options
	headerFile["Options"] >> intOptions;
	options = intOptions;

	// K
	if(!options[K_IN_KEYFRAME]){
		vectorK.clear();
		FileNode cameraMatrices = headerFile["cameraMatrices"];
		FileNodeIterator it = cameraMatrices.begin(), it_end = cameraMatrices.end();
		for( ; it != it_end; ++it){
			Mat *k = new Mat();
			*k = Mat::eye(3,3,CV_32F);
			k->at<float>(0,0) = (*it)["fx"];
			k->at<float>(1,1) = (*it)["fy"];
			k->at<float>(0,2) = (*it)["cx"];
			k->at<float>(1,2) = (*it)["cy"];
			vectorK.push_back(k);
		}
	}
	//std::cout << "osmap 2" << std::endl;
	// Save current directory
	char buf[4096];
	getcwd(buf, 4096);
	std::string current_dir(buf);

	// Change directory
	string pathDirectory;
	parsePath(yamlFilename, NULL, &pathDirectory);
	if(pathDirectory != "")
		chdir(pathDirectory.c_str());


	// MapPoints
	vectorMapPoints.clear();
	if(!options[NO_MAPPOINTS_FILE]){
		headerFile["mappointsFile"] >> filename;
		MapPointsLoad(filename);
	}

	vectorObjects.clear();
	headerFile["objectsFile"] >> filename;
	ObjectsLoad(filename);

	/*vectorMapObjects.clear();
	headerFile["mapobjectsFile"] >> filename;
	MapObjectsLoad(filename);

	vectorObjectTracks.clear();
	headerFile["objecttracksFile"] >> filename;
	ObjectTracksLoad(filename);*/


	// KeyFrames
	vectorKeyFrames.clear();
	if(!options[NO_KEYFRAMES_FILE]){
		headerFile["keyframesFile"] >> filename;
		KeyFramesLoad(filename);
		headerFile["graphsFile"] >> filename;
		GraphsLoad(filename);
	}

	// Features
	if(!options[NO_FEATURES_FILE]){
		headerFile["featuresFile"] >> filename;
		cout << "Loading features from " << filename << " ..." << endl;
		featuresLoad(filename);
	}

	// Close yaml file
	headerFile.release();

	// Rebuild
	rebuild(noSetBad);

	// Copy to map
	setMapPointsToMap();
	setKeyFramesToMap();
	setObjectToMap();
	//setMapObjectToMap();
	//setObjectTracksToTracker();

	// Release temporary vectors
	clearVectors();

#ifndef OSMAP_DUMMY_MAP
// Lost state, the system must relocalize itself in the just loaded map.
	system.mpTracker->mState = ORB_SLAM2::Tracking::LOST;
#endif

	if(pauseThreads){
		// Resume threads

		// Reactivate viewer.  Do not reactivate localMapper because the system resumes in "only tracking" mode immediatly after loading.
		if (system.mpViewer) system.mpViewer->Release();

		// If a map is loaded, local mapping and local object mapping are disabled
		// system.mpLocalMapper->Release();
		// system.local_object_mapper_->Release();

		// Tracking do this when going to LOST state.
		// Invoked after viewer.Release() because of mutex.
		//std::cout<<"mpFrameDrawer Update"<<std::endl;
		//system.mpFrameDrawer->Update(system.mpTracker);
	}

	// Go back to the initial working directory
	if(current_dir != "")
		chdir(current_dir.c_str());
}

int Osmap::MapPointsSave(string filename){
	ofstream file;
	file.open(filename, std::ofstream::binary);

	// Serialize
	SerializedMappointArray serializedMappointArray;
	int nMP = serialize(vectorMapPoints, serializedMappointArray);

	// Closing
	if (!serializedMappointArray.SerializeToOstream(&file))
		// Signals the error
		nMP = -1;
	file.close();

	return nMP;
}

int Osmap::MapPointsLoad(string filename){
	ifstream file;
	file.open(filename, ifstream::binary);

	SerializedMappointArray serializedMappointArray;
	serializedMappointArray.ParseFromIstream(&file);
	int nMP = deserialize(serializedMappointArray, vectorMapPoints);
	cout << "Mappoints loaded: " << nMP << endl;

	file.close();
	return nMP;
}

int Osmap::ObjectsSave(string filename){
	ofstream file;
	file.open(filename, std::ofstream::binary);

	// Serialize
	SerializedObjectArray serializedObjectArray;
	int nMO = serialize(vectorObjects, serializedObjectArray);

	// Closing
	if (!serializedObjectArray.SerializeToOstream(&file))
		// Signals the error
		nMO = -1;
	file.close();

	return nMO;
}

int Osmap::ObjectsLoad(string filename) {
	ifstream file;
	file.open(filename, ifstream::binary);

	SerializedObjectArray serializedObjectArray;
	serializedObjectArray.ParseFromIstream(&file);
	int nMO = deserialize(serializedObjectArray, vectorObjects);
	cout << "objects loaded: " << nMO << endl;

	file.close();
	return nMO;
}

//TOBE DELETE
int Osmap::MapObjectsSave(string filename){
	ofstream file;
	file.open(filename, std::ofstream::binary);

	// Serialize
	SerializedMapobjectArray serializedMapobjectArray;
	int nMO = serialize(vectorMapObjects, serializedMapobjectArray);

	// Closing
	if (!serializedMapobjectArray.SerializeToOstream(&file))
		// Signals the error
		nMO = -1;
	file.close();

	return nMO;
}


int Osmap::MapObjectsLoad(string filename) {
	ifstream file;
	file.open(filename, ifstream::binary);

	SerializedMapobjectArray serializedMapobjectArray;
	serializedMapobjectArray.ParseFromIstream(&file);
	int nMO = deserialize(serializedMapobjectArray, vectorMapObjects);
	cout << "Mapobjects loaded: " << nMO << endl;

	file.close();
	return nMO;
}

int Osmap::ObjectTracksSave(string filename){
	ofstream file;
	file.open(filename, std::ofstream::binary);

	// Serialize
	SerializedObjectTrackArray serializedObjecttrackArray;
	int nOT = serialize(vectorObjectTracks, serializedObjecttrackArray);

	// Closing
	if (!serializedObjecttrackArray.SerializeToOstream(&file))
		// Signals the error
		nOT = -1;
	file.close();
	return nOT;
}


int Osmap::ObjectTracksLoad(string filename) {
	ifstream file;
	file.open(filename, ifstream::binary);

	SerializedObjectTrackArray serializedObjecttrackArray;
	serializedObjecttrackArray.ParseFromIstream(&file);
	int nOT = deserialize(serializedObjecttrackArray, vectorObjectTracks);
	cout << "Objecttracks loaded: " << nOT << endl;

	file.close();
	return nOT;
}

int Osmap::KeyFramesSave(string filename){
	ofstream file;
	file.open(filename, std::ofstream::binary);

	// Serialize
	SerializedKeyframeArray serializedKeyFrameArray;
	int nKF = serialize(vectorKeyFrames, serializedKeyFrameArray);

	// Closing
	if (!serializedKeyFrameArray.SerializeToOstream(&file))
		// Signals the error
		nKF = -1;
	file.close();

	return nKF;
}

int Osmap::GraphsSave(string filename){
	ofstream file;
	file.open(filename, std::ofstream::binary);

	// Serialize
	SerializedGraphArray serializedGraphArray;
	int nKF = serialize(vectorKeyFrames, serializedGraphArray);

	// Closing
	if (!serializedGraphArray.SerializeToOstream(&file))
		// Signals the error
		nKF = -1;
	file.close();

	return nKF;
}

int Osmap::KeyFramesLoad(string filename){
	ifstream file;
	file.open(filename, ifstream::binary);
#ifndef OSMAP_DUMMY_MAP
	if(!currentFrame.mTcw.dims)	// if map is no initialized, currentFrame has no pose, a pose is needed to create keyframes.
		currentFrame.mTcw = Mat::eye(4, 4, CV_32F);
#endif
	SerializedKeyframeArray serializedKeyFrameArray;
	serializedKeyFrameArray.ParseFromIstream(&file);
	int nKF = deserialize(serializedKeyFrameArray, vectorKeyFrames);
	cout << "Keyframes loaded: "
		<< nKF << endl;
	file.close();
	return nKF;
}

int Osmap::GraphsLoad(string filename){
	ifstream file;
	file.open(filename, ifstream::binary);
	SerializedGraphArray serializedGraphArray;
	serializedGraphArray.ParseFromIstream(&file);
	int nGraphs = deserialize(serializedGraphArray, vectorKeyFrames);
	cout << "Graphs loaded: "
		<< nGraphs << endl;
	file.close();
	return nGraphs;
}

int Osmap::featuresSave(string filename){
	int nFeatures = 0;
	ofstream file;

	file.open(filename, ofstream::binary);
	if(
		options[FEATURES_FILE_DELIMITED] ||
		(!options[FEATURES_FILE_NOT_DELIMITED] && countFeatures() > FEATURES_MESSAGE_LIMIT)
	){
		// Saving with delimited ad hoc file format
		// Loop serializing blocks of no more than FEATURES_MESSAGE_LIMIT features, using Kendon Varda's function

		options.set(FEATURES_FILE_DELIMITED);

		// This Protocol Buffers stream must be deleted before closing file.  It happens automatically at }.
		::google::protobuf::io::OstreamOutputStream protocolbuffersStream(&file);
		vector<OsmapKeyFrame*> vectorBlock;
		vectorBlock.reserve(FEATURES_MESSAGE_LIMIT/30);

		auto it = vectorKeyFrames.begin();
		while(it != vectorKeyFrames.end()){
			unsigned int n = (*it)->N;
			vectorBlock.clear();
			do{
				vectorBlock.push_back(*it);
				++it;
				if(it == vectorKeyFrames.end()) break;
				KeyFrame *KF = *it;
				n += KF->N;
			} while(n <= FEATURES_MESSAGE_LIMIT);

			SerializedKeyframeFeaturesArray serializedKeyframeFeaturesArray;
			nFeatures += serialize(vectorBlock, serializedKeyframeFeaturesArray);
			writeDelimitedTo(serializedKeyframeFeaturesArray, &protocolbuffersStream);
		}
	}else{
		options.set(FEATURES_FILE_NOT_DELIMITED);
		SerializedKeyframeFeaturesArray serializedKeyframeFeaturesArray;
		nFeatures = serialize(vectorKeyFrames, serializedKeyframeFeaturesArray);
		if (!serializedKeyframeFeaturesArray.SerializeToOstream(&file)){
			cerr << "Error while serializing features file without delimitation." << endl;
			nFeatures = -1;
		}
	}
	file.close();

	return nFeatures;
}

int Osmap::featuresLoad(string filename){
	int nFeatures = 0;
	ifstream file;
	file.open(filename, ifstream::binary);
	auto *googleStream = new ::google::protobuf::io::IstreamInputStream(&file);
	SerializedKeyframeFeaturesArray serializedKeyframeFeaturesArray;
	if(options[FEATURES_FILE_DELIMITED]){
		while(true)
			if(readDelimitedFrom(googleStream, &serializedKeyframeFeaturesArray)){
				nFeatures += deserialize(serializedKeyframeFeaturesArray);
				cout << "Features deserialized in loop: "
					 << nFeatures << endl;
			}
			else
				break;
	} else {
		// Not delimited, pure Protocol Buffers
		serializedKeyframeFeaturesArray.ParseFromIstream(&file);
		nFeatures = deserialize(serializedKeyframeFeaturesArray);
	  }
	cout << "Features loaded: " << nFeatures << endl;
	file.close();
	return nFeatures;
}

void Osmap::getMapPointsFromMap(){
	  vectorMapPoints.clear();
	  vectorMapPoints.reserve(map.mspMapPoints.size());
	  std::transform(map.mspMapPoints.begin(), map.mspMapPoints.end(), std::back_inserter(vectorMapPoints), [](MapPoint *pMP)->OsmapMapPoint*{return static_cast<OsmapMapPoint*>(pMP);});
	  sort(vectorMapPoints.begin(), vectorMapPoints.end(), [](const MapPoint* a, const MapPoint* b){return a->mnId < b->mnId;});
}

void Osmap::setMapPointsToMap(){
	map.mspMapPoints.clear();
	copy(vectorMapPoints.begin(), vectorMapPoints.end(), inserter(map.mspMapPoints, map.mspMapPoints.end()));
}

//OBJECTS
void Osmap::getObjectsFromMap(){
	  vectorObjects.clear();
	  vectorObjects.reserve(map.mspObjects.size());
	  std::transform(map.mspObjects.begin(), map.mspObjects.end(), std::back_inserter(vectorObjects), [](Object *pMO)->OsmapObject*{return static_cast<OsmapObject*>(pMO);});
	  sort(vectorObjects.begin(), vectorObjects.end(), [](const Object* a, const Object* b){return a->GetId() < b->GetId();});
}
void Osmap::setObjectToMap() {
	map.mspObjects.clear();
	std::copy(vectorObjects_out.begin(), vectorObjects_out.end(), std::inserter(map.mspObjects, map.mspObjects.end()));
	std::cout<<"map has objects:"<<map.mspObjects.size()<<std::endl;
}

//TOBE DELETED
void Osmap::getMapObjectsFromMap(){
	  vectorMapObjects.clear();
	  vectorMapObjects.reserve(map.map_objects_.size());
	  std::transform(map.map_objects_.begin(), map.map_objects_.end(), std::back_inserter(vectorMapObjects), [](MapObject *pMO)->OsmapMapObject*{return static_cast<OsmapMapObject*>(pMO);});
	  sort(vectorMapObjects.begin(), vectorMapObjects.end(), [](const MapObject* a, const MapObject* b){return a->GetTrack()->GetId() < b->GetTrack()->GetId();});
}
void Osmap::setMapObjectToMap() {
	map.map_objects_.clear();
	std::copy(vectorMapObjects_out.begin(), vectorMapObjects_out.end(), std::inserter(map.map_objects_, map.map_objects_.end()));
}

void Osmap::getObjectTracksFromTracker(){
	vectorObjectTracks.clear();
	const auto& ots = tracker.GetObjectTracks();
	vectorObjectTracks.reserve(ots.size());
	for (auto ot : ots){
		// save only objects that are in the map
		if (ot->GetStatus() == ObjectTrackStatus::IN_MAP) {
			vectorObjectTracks.push_back(static_cast<OsmapObjectTrack*>(ot.get()));
		}
	}
	// vectorObjectTracks.reserve(ot.size());
	// std::transform(ot.begin(), ot.end(), std::back_inserter(vectorObjectTracks), [](ObjectTrack::Ptr pOT)->OsmapObjectTrack*{return static_cast<OsmapObjectTrack*>(pOT.get());});
	// sort(vectorObjectTracks.begin(), vectorObjectTracks.end(), [](const ObjectTrack* a, const ObjectTrack* b){return a->GetId() < b->GetId();});
}

void Osmap::setObjectTracksToTracker() {
	tracker.objectTracks_.clear();
	for (auto pt : vectorObjectTracks_out) {
		tracker.objectTracks_.push_back(pt);
	}
}

void Osmap::getKeyFramesFromMap(){
	// Order keyframes by mnId
	vectorKeyFrames.clear();
	vectorKeyFrames.reserve(map.mspKeyFrames.size());
	std::transform(map.mspKeyFrames.begin(), map.mspKeyFrames.end(), std::back_inserter(vectorKeyFrames), [](KeyFrame *pKF)->OsmapKeyFrame*{return static_cast<OsmapKeyFrame*>(pKF);});
	sort(vectorKeyFrames.begin(), vectorKeyFrames.end(), [](const KeyFrame *a, const KeyFrame *b){return a->mnId < b->mnId;});
}

void Osmap::setKeyFramesToMap(){
	map.mspKeyFrames.clear();
	copy(vectorKeyFrames.begin(), vectorKeyFrames.end(), inserter(map.mspKeyFrames, map.mspKeyFrames.end()));
}



void Osmap::clearVectors(){
	keyframeid2vectorkIdx.clear();
	vectorKeyFrames.clear();
	vectorMapPoints.clear();
	vectorObjects.clear();
	vectorMapObjects.clear();
	vectorMapObjects_out.clear();
	vectorObjectTracks.clear();
	vectorObjectTracks_out.clear();
	vectorK.clear();
}

void Osmap::parsePath(const string &path, string *filename, string *pathDirectory){
	size_t pos = path.find_last_of("\\/");
	if(std::string::npos == pos)
		// No directory separator, file is assumed.
		pos = 0;
	else
		// Last directory separator (/) will be in pathDirectory, not in filename.
		pos++;

	if(pathDirectory)
		*pathDirectory = path.substr(0, pos);
	if(filename)
		*filename = path.substr(pos);
	return;
}


void Osmap::depurate(){
	// First erase MapPoint from KeyFrames, and then erase KeyFrames from MapPoints.

	// NULL out bad MapPoints in KeyFrame::mvpMapPoints
	for(auto pKF: map.mspKeyFrames){
		// NULL out bad MapPoints and warns if not in map.  Usually doesn't find anything.
		auto pOKF = static_cast<OsmapKeyFrame*>(pKF);
		auto &pMPs = pOKF->mvpMapPoints;
		for(int i=pMPs.size(); --i>=0;){
			auto pOMP = static_cast<OsmapMapPoint *>(pMPs[i]);

			if(!pOMP) continue;	// Ignore if NULL

			if(pOMP->mbBad){
				// If MapPoint is bad, NULL it in keyframe's observations.
				//cerr << "depurate(): Nullifying bad MapPoint " << pOMP->mnId << " in KeyFrame " << pOKF->mnId << endl;
				pMPs[i] = NULL;
			} else if(!map.mspMapPoints.count(pOMP) && !options[NO_APPEND_FOUND_MAPPOINTS]){
				// If MapPoint is not in map, append it to the map
				map.mspMapPoints.insert(pOMP);
				//cout << "depurate(): APPEND_FOUND_MAPPOINTS: MapPoint " << pOMP->mnId << " added to map. ";
			}
		}
	}
}

void Osmap::rebuild(bool noSetBad){
	/*
	 * On every KeyFrame:
	 * - Builds the map database
	 * - UpdateConnections to rebuild covisibility graph
	 * - MapPoint::AddObservation on each point to rebuild MapPoint:mObservations y MapPoint:mObs
	 */
	cout << "Rebuilding map:" << endl;
	keyFrameDatabase.clear();

	if(noSetBad)
		options.set(NO_SET_BAD);

	log("Processing", vectorKeyFrames.size(), "keyframes");
	for(auto *pKF : vectorKeyFrames){
		LOGV(pKF);
		LOGV(pKF->mnId);

		pKF->mbNotErase = !pKF->mspLoopEdges.empty();
		LOGV(pKF->mbNotErase);

		// Build BoW vectors
		pKF->ComputeBoW();
		log("BoW computed");

		// Build many pose matrices
		pKF->SetPose(pKF->Tcw);
		log("Pose set");

		/*
		 * Rebuilding grid.
		 * Code from Frame::AssignFeaturesToGrid()
		 */
		std::vector<std::size_t> grid[pKF->mnGridCols][pKF->mnGridRows];
		int nReserve = 0.5f*pKF->N/(pKF->mnGridCols*pKF->mnGridRows);
		for(int i=0; i<pKF->mnGridCols;i++)
			for (int j=0; j<pKF->mnGridRows;j++)
				grid[i][j].reserve(nReserve);
		log("Grid built");

		for(int i=0;i<pKF->N;i++){
			const cv::KeyPoint &kp = pKF->mvKeysUn[i];
			int posX = round((kp.pt.x-pKF->mnMinX)*pKF->mfGridElementWidthInv);
			int posY = round((kp.pt.y-pKF->mnMinY)*pKF->mfGridElementHeightInv);

			//Keypoint's coordinates are undistorted, which could cause to go out of the image
			if(!(posX<0 || posX>=pKF->mnGridCols || posY<0 || posY>=pKF->mnGridRows))
				grid[posX][posY].push_back(i);
		}
		log("Grid full");

		pKF->mGrid.resize(pKF->mnGridCols);
		for(int i=0; i < pKF->mnGridCols;i++){
			pKF->mGrid[i].resize(pKF->mnGridRows);
			for(int j=0; j < pKF->mnGridRows; j++)
				pKF->mGrid[i][j] = grid[i][j];
		}
		log("Grid fitted");

		// Append keyframe to the database
		keyFrameDatabase.add(pKF);

		// Rebuild MapPoints obvervations
		size_t n = pKF->mvpMapPoints.size();
		for(size_t i=0; i<n; i++){
			MapPoint *pMP = pKF->mvpMapPoints[i];
			if(pMP)
				pMP->AddObservation(pKF, i);
		}
		log("Observations rebuilt");

		// Calling UpdateConnections in mnId order rebuilds the covisibility graph and the spanning tree.
		pKF->UpdateConnections();
	}

	// Last KeyFrame's id
	map.mnMaxKFid = vectorKeyFrames.back()->mnId;

	// Next KeyFrame id
	KeyFrame::nNextId = map.mnMaxKFid + 1;

	// Retry on isolated keyframes
	for(auto *pKF : vectorKeyFrames)
		if(pKF->mConnectedKeyFrameWeights.empty()){
			log("Isolated keyframe pKF:", pKF);
			pKF->UpdateConnections();
			if(!options[NO_SET_BAD] && pKF->mConnectedKeyFrameWeights.empty() && pKF->mnId){
				// If this keyframe is isolated (and it isn't keyframe zero), erase it.
				cerr << "Isolated keyframe " << pKF->mnId << " set bad." << endl;
				pKF->SetBadFlag();
			}
		}



	/*
	 * Check and fix the spanning tree created with UpdateConnections.
	 * Rebuilds the spanning tree asigning a mpParent to every orphan KeyFrame without, except that with id 0.
 	 * It ends when every KeyFrame has a parent.
	 */

	// mvpKeyFrameOrigins should be empty at this point, and must contain only one element, the first keyframe.
	map.mvpKeyFrameOrigins.clear();
	map.mvpKeyFrameOrigins.push_back(*vectorKeyFrames.begin());

	// Number of parents assigned in each iteration and in total.  Usually 0.
	int nParents = -1, nParentsTotal = 0;
	log("Rebuilding spanning tree.");
	while(nParents){
		nParents = 0;
		for(auto pKF: vectorKeyFrames)
			if(!pKF->mpParent && pKF->mnId)	// Process all keyframes without parent, exccept id 0
				for(auto *pConnectedKF : pKF->mvpOrderedConnectedKeyFrames){
					auto poConnectedKF = static_cast<OsmapKeyFrame*>(pConnectedKF);
					if(poConnectedKF->mpParent || poConnectedKF->mnId == 0){	// Parent found: not orphan or id 0
						nParents++;
						pKF->ChangeParent(pConnectedKF);
						break;
					}
				}
		nParentsTotal += nParents;
		log("Parents assigned in this loop:", nParents);
	}
	log("Parents assigned in total:", nParentsTotal);

	/*
	 * On every MapPoint:
	 * - Rebuilds mpRefKF as the first observation, which should be the KeyFrame with the lowest id
	 * - Rebuilds many properties with UpdateNormalAndDepth()
	 */
	log("Processing", vectorMapPoints.size(), "mappoints.");
	for(OsmapMapPoint *pMP : vectorMapPoints){
		LOGV(pMP)
		LOGV(pMP->mnId)
		// Rebuilds mpRefKF.  Requires mObservations.
		if(!options[NO_SET_BAD] && pMP->mnId && pMP->mObservations.empty()){
			cerr << "MP " << pMP->mnId << " without observations." << "  Set bad." << endl;
			pMP->SetBadFlag();
			continue;
		}

		// Asumes the first observation in mappoint has the lowest mnId.  Processed keyframes in mnId order ensures this.
		auto pair = (*pMP->mObservations.begin());
		pMP->mpRefKF = pair.first;

		/* UpdateNormalAndDepth() requires prior rebuilding of mpRefKF, and rebuilds:
		 * - mNormalVector
		 * - mfMinDistance
		 * - mfMaxDistance
		 */
		pMP->UpdateNormalAndDepth();
	}
	MapPoint::nNextId = vectorMapPoints.back()->mnId + 1;

	//-------------------------------------------------------------
	//--------- Rebuild Objects --------------
	//-------------------------------------------------------------
	vectorObjects_out.clear();
  std::map<unsigned int, Object*>	objects_tmp;
	std::vector<pair<unsigned int, Eigen::Vector3d>> center_points_3d;
	for (auto* obj : vectorObjects) {
		Eigen::Vector3d center_point_3d = obj->ellipsoid.GetCenter(); 
    center_points_3d.push_back(make_pair(obj->object_id, center_point_3d));
		//std::cout<<"load object:"<<obj->object_id<<" with ell Axes:"<<obj->ellipsoid.GetAxes()<<std::endl;
	  Object* new_obj = new Object(obj->ellipsoid, obj->object_id, obj->cat_ids);
		int count = 0;
		for(auto id : obj->corres_mp_ids){
			auto mp = getMapPoint(id);
			if(mp){
				new_obj->InsertNewAsscoatedMapPoint(mp);
				count+=1;
			}
		}
		// add more mappoints
		auto ellipsoid = new_obj->GetEllipsoid();
		for(OsmapMapPoint *pMP : vectorMapPoints){
			cv::Mat p = pMP->GetWorldPos();
			Eigen::Vector3d pos(p.at<float>(0), p.at<float>(1), p.at<float>(2));
			if (ellipsoid.IsInside(pos, 5.0))
				new_obj->InsertNewAsscoatedMapPoint(pMP);
		}
		//std::cout<<"load object:"<<new_obj->GetId()<<" with mps:"<<obj->corres_mp_ids.size()<<", actual:"<<count<<std::endl;
		vectorObjects_out.push_back(new_obj);
		//new_obj->SetColor(cv::Scalar(220, 220, 220));
		objects_tmp[obj->object_id] = new_obj;
	}
	std::cout << "Rebuilt " << vectorObjects_out.size() << " map objects.\n";

	//--------- Generate 3D Graph --------------
	const int k = 3;
	std::map<int, std::vector<pair<int,double>> > obj_index_to_neighbours;
	for (size_t i=0; i<center_points_3d.size(); i++){
			std::vector<pair<int,double>> distances;
			for (size_t j=0; j<center_points_3d.size(); j++){
					if (i != j){
							double distance = (center_points_3d[i].second - center_points_3d[j].second).norm();
							distances.push_back(make_pair(center_points_3d[j].first,distance));
					}
			}
			sort(distances.begin(),distances.end(),[](auto& left, auto& right) { 
					return left.second < right.second; 
			});
			for (int m=0; m<k; m++){
					obj_index_to_neighbours[center_points_3d[i].first].push_back(distances[m]);
			}
	} //YUTONG: using distance to generate graph can make some edges be ignored

	vector<pair<int,int>> edge_list, node_labels;

	for(auto pkf : vectorKeyFrames){
		//std::cout<<"graph nodes:"<<pkf->graph->nodes.size()<<std::endl;
		for (auto& [node_id, neighbours] : pkf->graph->nodes){
			auto obj_id = pkf->graph->attributes[node_id].object_id;
			//std::cout<<"graph obj_id:"<<obj_id<<std::endl;
			if(objects_tmp.count(obj_id)>0){
				//ADD OBSERVATION FOR OBJECT COVISIBILITY GRAPH
				objects_tmp[obj_id]->AddObservation(pkf);
				pkf->graph->attributes[node_id].obj = objects_tmp[obj_id];
				
			  auto tmp_pair = make_pair(obj_id, pkf->graph->attributes[node_id].label);
				//auto tmp_pair = make_pair(obj_id, (int)objects_tmp[obj_id]->GetCategoryId());
				if(std::find(node_labels.begin(), node_labels.end(), tmp_pair) == node_labels.end()){
					node_labels.push_back(tmp_pair);
					//std::cout<<"node_labels:"<<tmp_pair.first<<std::endl;
				}
			}
			for (auto n_id : pkf->graph->nodes[node_id]){
        auto neighbour_id = pkf->graph->attributes[n_id].object_id;
				if(objects_tmp.count(neighbour_id)>0){
					auto tmp_pair = make_pair(obj_id, neighbour_id);
					if(std::find(edge_list.begin(), edge_list.end(), tmp_pair) == edge_list.end()){
						edge_list.push_back(tmp_pair);
					}
				}
			}
			//ADD NEIGHBOURS TOO
			for(auto [neighbour_id, s] : obj_index_to_neighbours[obj_id]){
				if(objects_tmp.count(neighbour_id)>0){
					auto tmp_pair = make_pair(obj_id, neighbour_id);
					if(std::find(edge_list.begin(), edge_list.end(), tmp_pair) == edge_list.end()){
						edge_list.push_back(tmp_pair);
					}
				}
			}
		}
	}
	map.graph_3d = new ORB_SLAM2::Graph(edge_list, node_labels);

	for(auto [node_id, neighbours] : map.graph_3d->nodes){
		if(objects_tmp.count(node_id)==0) continue;
		map.graph_3d->attributes[node_id].obj = objects_tmp[node_id];
		if(map.graph_3d->attributes[node_id].obj->GetAllCategoryIds().size()>1){
			int total_counts = 0;
			for(auto [cat_id, count] : map.graph_3d->attributes[node_id].obj->GetAllCategoryIds()){
				total_counts += count;
			}
			for(auto [cat_id, count] : map.graph_3d->attributes[node_id].obj->GetAllCategoryIds()){
				double freq = static_cast<double>(count)/total_counts;
				std::cout<<"node_id:"<<node_id<<", cat_id:"<<cat_id<<", freq:"<<freq<<std::endl;
				map.graph_3d->node_cat_frequencies[node_id][cat_id] = freq;
			}
		}
	}

	for(auto [node_id, neighbours] : map.graph_3d->nodes){
		if(objects_tmp.count(node_id)==0) continue;
		Eigen::VectorXd feature_vector(80);
		feature_vector.setZero(); 
		for (int neighbour : neighbours) {
			for(auto& [cat_id, freq] : map.graph_3d->node_cat_frequencies[neighbour]){
				feature_vector[cat_id] += freq;
			}
		}
		feature_vector.normalize();
		//for(auto& [cat_id, freq] : map.graph_3d->node_cat_frequencies[node_id]){
		//	map.graph_3d->feature_vectors[cat_id].push_back(make_pair(node_id, feature_vector));
		//}
		std::cout<<"node_id:"<<node_id<<std::endl;
		auto label = map.graph_3d->attributes[node_id].obj->GetCategoryId();
		std::cout<<"label:"<<label<<std::endl;
		map.graph_3d->feature_vectors[label].push_back(make_pair(node_id, feature_vector));

		/*for (int neighbour : neighbours) {
				feature_vector[map.graph_3d->attributes[neighbour].label] += 1.0f; //todo multiple labels
		}
		//auto label = map.graph_3d->attributes[node_id].obj->GetCategoryId();
		//map.graph_3d->feature_vectors[label].push_back(make_pair(node_id, feature_vector));
		//MULTI LABEL
		for(auto [cat_id, count] : map.graph_3d->attributes[node_id].obj->GetAllCategoryIds()){
			std::cout<<"cat_id:"<<cat_id<<std::endl;
			map.graph_3d->feature_vectors[cat_id].push_back(make_pair(node_id, feature_vector));
		}
		if(map.graph_3d->attributes[node_id].obj->GetAllCategoryIds().size()>1){
			std::cout<<"size:"<<map.graph_3d->attributes[node_id].obj->GetAllCategoryIds().size()<<std::endl;
			for(auto [cat_id, count] : map.graph_3d->attributes[node_id].obj->GetAllCategoryIds()){
				std::cout<<"cat_id:"<<cat_id<<", count:"<<count<<std::endl;
			}
		}*/
	}

	std::cout<<"REBUILD"<<std::endl;
	// First rebuild object tracks
	/*vectorObjectTracks_out.clear();
	unsigned int max_id = 0;
	for (auto* ot: vectorObjectTracks) {
		max_id = std::max(max_id, ot->id);

		// create new object track with dummy detection
		auto new_ot = ObjectTrack::CreateNewObjectTrack(ot->cat, BBox2(), 0.0, Matrix34d(), 0, dynamic_cast<Tracking*>(&tracker), nullptr);
		new_ot->SetId(ot->id);
		new_ot->SetColor(ot->color);
		new_ot->SetStatus(ot->status);
		new_ot->SetLastObsFrameId(-1);

		for (size_t i = 0; i < ot->kf_bboxes.size(); ++i) {
			OsmapKeyFrame *kf = nullptr;
			for(auto *pKF : vectorKeyFrames) {
				if (pKF->mnId == ot->kf_indices[i]) {
					kf = pKF;
					break;
				}
			}
			if (kf) {
				new_ot->AddDetection(ot->kf_bboxes[i],
									 ot->kf_scores[i],
									 ORB_SLAM2::cvToEigenMatrix<double, float, 3, 4>(kf->GetPose()),
									 kf->mnId,
									 kf);
			}
		}

		// Clear trakcing buffers with consecutive frames. Only the keyframes can be loaded form the map
		new_ot->ClearTrackingBuffers();
		// we re-insert only the keyframes so it does not make sense to set the last frame id
		new_ot->SetLastObsFrameId(-1);
		vectorObjectTracks_out.push_back(new_ot);
	}
	std::cout << "Rebuilt " << vectorObjectTracks_out.size() << " object tracks.\n";
	ObjectTrack::factory_id = max_id + 1;

	// Secondly rebuild map objects
	vectorMapObjects_out.clear();
	for (auto* mo : vectorMapObjects) {
		ObjectTrack* corresponding_ot = nullptr;
		for (auto ot : vectorObjectTracks_out) {
			if (ot->GetId() == mo->object_track_id) {
				corresponding_ot = ot.get();
			}
		}
		MapObject* new_mo = new MapObject(mo->ellipsoid, corresponding_ot);
		if (corresponding_ot) {
			corresponding_ot->SetMapObject(new_mo);
		}
		vectorMapObjects_out.push_back(new_mo);
	}
	std::cout << "Rebuilt " << vectorMapObjects_out.size() << " map objects.\n";*/

}

void Osmap::getVectorKFromKeyframes(){
  vectorK.clear();
  keyframeid2vectorkIdx.resize(KeyFrame::nNextId);	// Assume map is not ill formed so nNextId is ok, thus no keyframe's id is bigger than this.
  fill(keyframeid2vectorkIdx.begin(), keyframeid2vectorkIdx.end(), 0);	// Fill with index 0 to prevent segfault from unknown bugs.

  if(vectorKeyFrames.empty())
	  getKeyFramesFromMap();

  //for(auto &pKF:map.mspKeyFrames){
  for(auto pKF: vectorKeyFrames){
    // Test if K can be found in vectorK.  If new, add it to the end of vectorK.
    //Mat &K = const_cast<cv::Mat &> (pKF->mK);
    const Mat &K = pKF->mK;

    // Will be the index of K in vectorK
    unsigned int i;
    for(i=0; i<vectorK.size(); i++){
      const Mat &vK = *vectorK[i];

      // Tests: break if found

      // Quick test
      if(K.data == vK.data) break;

      // Slow test, compare each element
/*
      if(
        K.at<float>(0,0) == vK.at<float>(0,0) &&
        K.at<float>(1,1) == vK.at<float>(1,1) &&
        K.at<float>(0,2) == vK.at<float>(0,2) &&
        K.at<float>(1,2) == vK.at<float>(1,2)
      ) break;
*/
#define DELTA 0.1
      if(
        abs(K.at<float>(0,0) - vK.at<float>(0,0)) < DELTA &&
        abs(K.at<float>(1,1) - vK.at<float>(1,1)) < DELTA &&
        abs(K.at<float>(0,2) - vK.at<float>(0,2)) < DELTA &&
        abs(K.at<float>(1,2) - vK.at<float>(1,2)) < DELTA
      ) break;
    }

    // if not found, push
    if(i>=vectorK.size()){
      // add new K
      vectorK.push_back(&K);
    }

    // i is the vectorK index for this keyframe
    keyframeid2vectorkIdx[ pKF->mnId ] = i;
  }
}

int Osmap::countFeatures(){
	int n=0;
	for(auto pKP : vectorKeyFrames)
		n += pKP->N;

	return n;
}


// Utilities
MapPoint *Osmap::getMapPoint(unsigned int id){
  //for(auto pMP : map.mspMapPoints)
  for(auto pMP : vectorMapPoints)
    if(pMP->mnId == id)
    	return pMP;

  // Not found
  return NULL;
}

OsmapKeyFrame *Osmap::getKeyFrame(unsigned int id){
  //for(auto it = map.mspKeyFrames.begin(); it != map.mspKeyFrames.end(); ++it)
  for(auto pKF : vectorKeyFrames)
	if(pKF->mnId == id)
	  return pKF;

  // If not found
  return NULL;
}




// K matrix ================================================================================================
void Osmap::serialize(const Mat &k, SerializedK *serializedK){
  serializedK->set_fx(k.at<float>(0,0));
  serializedK->set_fy(k.at<float>(1,1));
  serializedK->set_cx(k.at<float>(0,2));
  serializedK->set_cy(k.at<float>(1,2));
}

void Osmap::deserialize(const SerializedK &serializedK, Mat &m){
  m = Mat::eye(3,3,CV_32F);
  m.at<float>(0,0) = serializedK.fx();
  m.at<float>(1,1) = serializedK.fy();
  m.at<float>(0,2) = serializedK.cx();
  m.at<float>(1,2) = serializedK.cy();
}

void Osmap::serialize(const vector<Mat*> &vK, SerializedKArray &serializedKArray){

}

void Osmap::deserialize(const SerializedKArray &serializedKArray, vector<Mat*> &vK){

}



// Descriptor ================================================================================================
void Osmap::serialize(const Mat &m, SerializedDescriptor *serializedDescriptor){
  assert(m.rows == 1 && m.cols == 32);
  for(unsigned int i = 0; i<8; i++)
	serializedDescriptor->add_block(((unsigned int*)m.data)[i]);
}

void Osmap::deserialize(const SerializedDescriptor &serializedDescriptor, Mat &m){
  assert(serializedDescriptor.block_size() == 8);
  m = Mat(1, 32, CV_8UC1);
  for(unsigned int i = 0; i<8; i++)
	((unsigned int*)m.data)[i] = serializedDescriptor.block(i);
}

// Pose ================================================================================================
void Osmap::serialize(const Mat &m, SerializedPose *serializedPose){
  float *pElement = (float*) m.data;
  for(unsigned int i = 0; i<12; i++)
    serializedPose->add_element(pElement[i]);
}

void Osmap::deserialize(const SerializedPose &serializedPose, Mat &m){
  assert(serializedPose.element_size() == 12);
  m = Mat::eye(4,4,CV_32F);
  float *pElement = (float*) m.data;
  for(unsigned int i = 0; i<12; i++)
	pElement[i] = serializedPose.element(i);
}


// Ellipsoid ================================================================================================
void Osmap::serialize(const Ellipsoid &m, SerializedEllipsoid *serializedEllipsoid){
  Eigen::Matrix4d Q = m.AsDual();
  for (int i = 0; i < 4; ++i) {
	for (int j = 0; j < 4; ++j) {
		serializedEllipsoid->add_element(Q(i, j));
	}
  }
}

void Osmap::deserialize(const SerializedEllipsoid &serializedEllipsoid, Ellipsoid &m){
  assert(serializedEllipsoid.element_size() == 16);
  Eigen::Matrix4d Q;
  for(unsigned int i = 0; i < 16; i++) {
	Q(i/4, i%4) = serializedEllipsoid.element(i);
  }
  m = Ellipsoid(Q);
}


// BBox ================================================================================================
void Osmap::serialize(const BBox2 &bb, SerializedBBox2 *serializedBBox){
  serializedBBox->set_xmin(bb[0]);
  serializedBBox->set_ymin(bb[1]);
  serializedBBox->set_xmax(bb[2]);
  serializedBBox->set_ymax(bb[3]);
}

void Osmap::deserialize(const SerializedBBox2 &serializedBBox, BBox2 &bb){
  bb[0] = serializedBBox.xmin();
  bb[1] = serializedBBox.ymin();
  bb[2] = serializedBBox.xmax();
  bb[3] = serializedBBox.ymax();
}

// Position ================================================================================================
void Osmap::serialize(const Mat &m, SerializedPosition *serializedPosition){
  serializedPosition->set_x(m.at<float>(0,0));
  serializedPosition->set_y(m.at<float>(1,0));
  serializedPosition->set_z(m.at<float>(2,0));
}

void Osmap::deserialize(const SerializedPosition &serializedPosition, Mat &m){
  m = Mat(3,1,CV_32F);
  m.at<float>(0,0) = serializedPosition.x();
  m.at<float>(1,0) = serializedPosition.y();
  m.at<float>(2,0) = serializedPosition.z();
}

// KeyPoint ================================================================================================
void Osmap::serialize(const KeyPoint &kp, SerializedKeypoint *serializedKeypoint){
  serializedKeypoint->set_ptx(kp.pt.x);
  serializedKeypoint->set_pty(kp.pt.y);
  serializedKeypoint->set_octave(kp.octave);
  serializedKeypoint->set_angle(kp.angle);
}

void Osmap::deserialize(const SerializedKeypoint &serializedKeypoint, KeyPoint &kp){
  kp.pt.x   = serializedKeypoint.ptx();
  kp.pt.y   = serializedKeypoint.pty();
  kp.octave = serializedKeypoint.octave();
  kp.angle  = serializedKeypoint.angle();
}



// MapPoint ================================================================================================
void Osmap::serialize(const OsmapMapPoint &mappoint, SerializedMappoint *serializedMappoint){
  serializedMappoint->set_id(mappoint.mnId);
  serialize(mappoint.mWorldPos, serializedMappoint->mutable_position());
  serializedMappoint->set_visible(mappoint.mnVisible);
  serializedMappoint->set_found(mappoint.mnFound);
  //if(options[NO_FEATURES_DESCRIPTORS])	// This is the only descriptor to serialize	** This line is disable to force mappoint descriptor serialization, while it's not being reconstructed in rebuild. **
    serialize(mappoint.mDescriptor, serializedMappoint->mutable_briefdescriptor());
}

OsmapMapPoint *Osmap::deserialize(const SerializedMappoint &serializedMappoint){
  OsmapMapPoint *pMappoint = new OsmapMapPoint(this);

  pMappoint->mnId        = serializedMappoint.id();
  pMappoint->mnVisible   = serializedMappoint.visible();
  pMappoint->mnFound     = serializedMappoint.found();
  if(serializedMappoint.has_briefdescriptor()) deserialize(serializedMappoint.briefdescriptor(), pMappoint->mDescriptor);
  if(serializedMappoint.has_position())        deserialize(serializedMappoint.position(),        pMappoint->mWorldPos  );

  return pMappoint;
}

int Osmap::serialize(const vector<OsmapMapPoint*>& vectorMP, SerializedMappointArray &serializedMappointArray){
  for(auto pMP : vectorMP)
    serialize(*pMP, serializedMappointArray.add_mappoint());

  return vectorMP.size();
}


int Osmap::deserialize(const SerializedMappointArray &serializedMappointArray, vector<OsmapMapPoint*>& vectorMapPoints){
  int i, n = serializedMappointArray.mappoint_size();
  for(i=0; i<n; i++)
	vectorMapPoints.push_back(deserialize(serializedMappointArray.mappoint(i)));

  return i;
}

// Object ================================================================================================
void Osmap::serialize(const OsmapObject &object, SerializedObject  *serializedObject){
	serialize(object.ellipsoid_, serializedObject->mutable_ellipsoid());
	serializedObject->set_object_id(object.GetId());
	//serializedObject->set_cat_id(object.GetCategoryId());
	int count = 0;
	for(auto [cat_id, cat_count] : object.GetAllCategoryIds()){
		serializedObject->add_cat_ids(cat_id);
		serializedObject->add_cat_counts(cat_count);
		count+=1;
	}
	for(auto mp : object.GetAssociatedMapPoints()){
		if(mp){
			if(!mp->isBad()){
				serializedObject->add_mappoint_id(mp->mnId);
				//count+=1;
			}
		}
	}
	std::cout<<"object:"<<object.GetId()<<"has cat_ids:"<<count<<std::endl;
}

OsmapObject *Osmap::deserialize(const SerializedObject &serializedObject){
	Ellipsoid ell;
  deserialize(serializedObject.ellipsoid(), ell);
	std::unordered_map<int, int> cat_ids;
	int j, m = serializedObject.cat_ids_size();
	for(j=0; j<m; j++){
		cat_ids[serializedObject.cat_ids(j)] = serializedObject.cat_counts(j);
	}
	//std::cout<<"load object:"<<serializedObject.object_id()<<" with ell center:"<<ell.GetCenter()<<std::endl;
	//OsmapObject *pObject = new OsmapObject(ell, serializedObject.object_id(), serializedObject.cat_id()); // just create an object
	OsmapObject *pObject = new OsmapObject(ell, serializedObject.object_id(), cat_ids);
	pObject->ellipsoid = ell;
	pObject->object_id = serializedObject.object_id(); 
  //pObject->cat_id = serializedObject.cat_id(); 
	pObject->cat_ids = cat_ids;
	//std::cout<<"object:"<<serializedObject.object_id()<<"has cat_ids:"<<cat_ids.size()<<std::endl;
	int i, n = serializedObject.mappoint_id_size();
  for(i=0; i<n; i++){
		pObject->corres_mp_ids.push_back(serializedObject.mappoint_id(i));
	}
	//std::cout<<"OsmapObject:"<<pObject->object_id<<" with ell center:"<<pObject->ellipsoid.GetCenter()<<std::endl; 
	return pObject;
}

int Osmap::serialize(const vector<OsmapObject*>& vectorObjects, SerializedObjectArray &serializedObjectArray){
	int count = 0;
	for(auto pObj : vectorObjects){
		if(pObj->isBad()) continue;
		serialize(*pObj, serializedObjectArray.add_object());
		count += 1;
	}
	return count;
}

int Osmap::deserialize(const SerializedObjectArray &serializedObjectArray, vector<OsmapObject*>& vectorObjects){
	int i, n = serializedObjectArray.object_size();
  for(i=0; i<n; i++)
		vectorObjects.push_back(deserialize(serializedObjectArray.object(i)));
  return i;
}

// TODO DELETE MapObject ================================================================================================

void Osmap::serialize(const OsmapMapObject &mapobject, SerializedMapobject *serializedMapobject){
  serialize(mapobject.ellipsoid_, serializedMapobject->mutable_ellipsoid());
  serializedMapobject->set_object_track_id(mapobject.object_track_->GetId());
}

OsmapMapObject *Osmap::deserialize(const SerializedMapobject &serializedMapobject){
  Ellipsoid ell;
  deserialize(serializedMapobject.ellipsoid(), ell);
  OsmapMapObject *pMapobject = new OsmapMapObject(ell); // just create an object
  pMapobject->ellipsoid = ell; // set ellipsoid
  pMapobject->object_track_id = serializedMapobject.object_track_id(); // set object_track id
  return pMapobject;
}

int Osmap::serialize(const vector<OsmapMapObject*>& vectorMO, SerializedMapobjectArray &serializedMapobjectArray){
  for(auto pMO : vectorMO)
    serialize(*pMO, serializedMapobjectArray.add_mapobject());
  return vectorMO.size();
}

int Osmap::deserialize(const SerializedMapobjectArray &serializedMapobjectArray, vector<OsmapMapObject*>& vectorMapObjects){
  int i, n = serializedMapobjectArray.mapobject_size();
  for(i=0; i<n; i++)
	vectorMapObjects.push_back(deserialize(serializedMapobjectArray.mapobject(i)));
  return i;
}

// ObjectTrack ================================================================================================

void Osmap::serialize(const OsmapObjectTrack &objecttrack, SerializedObjectTrack *serializedObjectTrack){
  serializedObjectTrack->set_id(objecttrack.id_);
  serializedObjectTrack->set_cat(objecttrack.category_id_);
  serializedObjectTrack->set_last_frame_id(objecttrack.last_obs_frame_id_);
  cv::Scalar color = objecttrack.color_;
  serializedObjectTrack->set_color_r(color[0]);
  serializedObjectTrack->set_color_g(color[1]);
  serializedObjectTrack->set_color_b(color[2]);
  auto status = objecttrack.status_;
  int status_int = 0;
  if (status == ObjectTrackStatus::BAD)
	status_int = 0;
  else if (status == ObjectTrackStatus::ONLY_2D)
	status_int = 1;
  else if (status == ObjectTrackStatus::INITIALIZED)
	status_int = 2;
  else if (status == ObjectTrackStatus::IN_MAP)
	status_int = 3;
  serializedObjectTrack->set_status(status_int);
  const auto& kf_bboxes = objecttrack.keyframes_bboxes_;
  for (auto it : kf_bboxes) {
    serialize(it.second, serializedObjectTrack->add_kf_bboxes());
    serializedObjectTrack->add_kf_indices(it.first->mnId);
  }
  const auto& kf_scores = objecttrack.keyframes_scores_;
  for (auto it : kf_scores) {
    serializedObjectTrack->add_kf_scores(it.second);
  }
}

OsmapObjectTrack *Osmap::deserialize(const SerializedObjectTrack &serializedObjectTrack){
  OsmapObjectTrack *pObjecttrack = new OsmapObjectTrack(this);
  pObjecttrack->id = serializedObjectTrack.id();
  pObjecttrack->cat = serializedObjectTrack.cat();
  pObjecttrack->last_frame_id = serializedObjectTrack.last_frame_id();
  pObjecttrack->color = cv::Scalar(serializedObjectTrack.color_r(),
								   serializedObjectTrack.color_g(),
								   serializedObjectTrack.color_b());
  std::vector<ObjectTrackStatus> status = {ObjectTrackStatus::BAD, ObjectTrackStatus::ONLY_2D, 
  										   ObjectTrackStatus::INITIALIZED, ObjectTrackStatus::IN_MAP};
  pObjecttrack->status = status[serializedObjectTrack.status()];
  assert(serializedObjectTrack.kf_bboxes_size() == serializedObjectTrack.kf_indices_size());
  size_t n = serializedObjectTrack.kf_bboxes_size();
  pObjecttrack->kf_bboxes.resize(n);
  pObjecttrack->kf_indices.resize(n);
  pObjecttrack->kf_scores.resize(n);
  for (size_t i = 0; i < n; ++i) {
    deserialize(serializedObjectTrack.kf_bboxes(i), pObjecttrack->kf_bboxes[i]);
    pObjecttrack->kf_indices[i] = serializedObjectTrack.kf_indices(i);
    pObjecttrack->kf_scores[i] = serializedObjectTrack.kf_scores(i);
  }
  // retrieve keyframes_bboxes
  return pObjecttrack;
}

int Osmap::serialize(const vector<OsmapObjectTrack*>& vectorOT, SerializedObjectTrackArray &serializedObjectTrackArray){
  for(auto pOT : vectorOT)
    serialize(*pOT, serializedObjectTrackArray.add_objecttrack());
  return vectorOT.size();
}

int Osmap::deserialize(const SerializedObjectTrackArray &serializedObjectTrackArray, vector<OsmapObjectTrack*>& vectorObjectTracks){
  int i, n = serializedObjectTrackArray.objecttrack_size();
  for(i=0; i<n; i++)
	vectorObjectTracks.push_back(deserialize(serializedObjectTrackArray.objecttrack(i)));
  return i;
}

// KeyFrame ================================================================================================
void Osmap::serialize(const OsmapKeyFrame &keyframe, SerializedKeyframe *serializedKeyframe){
  serializedKeyframe->set_id(keyframe.mnId);
  serialize(keyframe.Tcw, serializedKeyframe->mutable_pose());
  serializedKeyframe->set_timestamp(keyframe.mTimeStamp);
  if(options[K_IN_KEYFRAME])
	serialize(keyframe.mK, serializedKeyframe->mutable_kmatrix());
  else
	serializedKeyframe->set_kindex(keyframeid2vectorkIdx[keyframe.mnId]);
  if(!keyframe.mspLoopEdges.empty())
	for(auto loopKF : keyframe.mspLoopEdges)
		// Only serialize id of keyframes already serialized, to easy deserialization.
		if(keyframe.mnId > loopKF->mnId)
			serializedKeyframe->add_loopedgesids(loopKF->mnId);
}

OsmapKeyFrame *Osmap::deserialize(const SerializedKeyframe &serializedKeyframe){
	OsmapKeyFrame *pKeyframe = new OsmapKeyFrame(this);

  pKeyframe->mnId = serializedKeyframe.id();
  const_cast<double&>(pKeyframe->mTimeStamp) = serializedKeyframe.timestamp();

  if(serializedKeyframe.has_pose())
	  deserialize(serializedKeyframe.pose(), pKeyframe->Tcw);

  if(serializedKeyframe.has_kmatrix())
	  // serialized with K_IN_KEYFRAME option, doesn't use K list in yaml
	  deserialize(serializedKeyframe.kmatrix(), const_cast<cv::Mat&>(pKeyframe->mK));
  else
	  // serialized with default no K_IN_KEYFRAME option, K list in yaml
	  const_cast<cv::Mat&>(pKeyframe->mK) = *vectorK[serializedKeyframe.kindex()];

  if(serializedKeyframe.loopedgesids_size()){
	// Only ids of keyframes already deserialized and present on vectorKeyFrames
	for(int i=0; i<serializedKeyframe.loopedgesids_size(); i++){
	  unsigned int loopEdgeId = serializedKeyframe.loopedgesids(i);
	  OsmapKeyFrame *loopEdgeKF = getKeyFrame(loopEdgeId);
	  loopEdgeKF->mspLoopEdges.insert(pKeyframe);
	  pKeyframe->mspLoopEdges.insert(loopEdgeKF);
	}
  }

  return pKeyframe;
}


int Osmap::serialize(const vector<OsmapKeyFrame*>& vectorKF, SerializedKeyframeArray &serializedKeyframeArray){
  for(auto pKF: vectorKF)
    serialize(*pKF, serializedKeyframeArray.add_keyframe());

  return vectorKF.size();
}



int Osmap::deserialize(const SerializedKeyframeArray &serializedKeyframeArray, vector<OsmapKeyFrame*>& vectorKeyFrames){
  int i, n = serializedKeyframeArray.keyframe_size();
  for(i=0; i<n; i++)
		vectorKeyFrames.push_back(deserialize(serializedKeyframeArray.keyframe(i)));

  return i;
}


// Graph ================================================================================================
int Osmap::serialize(const vector<OsmapKeyFrame*>& vectorKF, SerializedGraphArray &serializedGraphArray){
  for(auto pKF: vectorKF)
    serialize(*pKF->graph, serializedGraphArray.add_graph());

  return serializedGraphArray.graph_size();
}

int Osmap::deserialize(const SerializedGraphArray &serializedGraphArray, vector<OsmapKeyFrame*>& vectorKeyFrames){
  int m = vectorKeyFrames.size();
  int n = serializedGraphArray.graph_size();
  //std::cout<<"!!!! GRAPH SIZE !!!!"<<n<<std::endl;
  if (m != n)
		std::cout<<"!!!! GRAPH SIZE NOT EQUAL TO KEYFRAME SIZE !!!!"<<std::endl;
  int i = 0;
  for (auto pKF: vectorKeyFrames){
		pKF->graph = new Graph();
		deserialize(serializedGraphArray.graph(i), pKF->graph);
		//std::cout<<"Graph pKF id:"<<pKF->mnId<<","<<i<<",";
		i++;
  }
	std::cout<<std::endl;
  return i;
}

void Osmap::deserialize(const SerializedGraph &serializedGraph, Graph *graph){
	std::vector<pair<int,int>> edge_list, node_labels;
	int i, n = serializedGraph.node_size();
  //std::cout<<"!!!! NODE SIZE !!!!"<<i<<","<<n<<std::endl;
	std::vector<Node> nodes;
	for(i=0; i<n; i++){
		Node node = {-1, -1, -1, std::vector<int>()};
		deserialize(serializedGraph.node(i), node);
		nodes.push_back(node);
		graph->add_node(node.id, node.category_id, 0.0, 0.0, node.bbox);
		graph->attributes[node.id].object_id = node.object_id;
		//std::cout<<"graph->attributes[node.id].object_id:"<<graph->attributes[node.id].object_id<<std::endl;
	}
	for(auto node:nodes){
		for (auto n_id : node.neighbours){
			graph->add_edge(node.id, n_id);
		}
	}
	//std::cout<<"Graph nodes:"<<nodes.size()<<std::endl;
	graph->compute_feature_vectors();
}

void Osmap::deserialize(const SerializedNode &serializedNode, Node& node){
	node.id = serializedNode.node_id();
	node.object_id = serializedNode.object_id();
	node.category_id = serializedNode.category_id();
	deserialize(serializedNode.bbox(), node.bbox);
	for (auto n_id : serializedNode.neighbours_ids())
		node.neighbours.push_back(n_id);
}

void Osmap::serialize(Graph& graph, SerializedGraph *serializedGraph){
  auto nodes = graph.nodes;
  auto attributes = graph.attributes;
  for (auto n : nodes){
	  /*auto node_id = node.first;
	  auto neighbours = node.second;
	  auto label = attributes[node.first].label;*/
		int object_id = -1;
		if(attributes[n.first].obj){
			object_id = attributes[n.first].obj->GetId();
			//std::cout<<"object_id:"<<object_id<<std::endl;
		}
		Node node = {n.first, object_id, attributes[n.first].label, n.second, attributes[n.first].bbox};
		serialize(node, serializedGraph->add_node());
  }
}

void Osmap::serialize(Node node, SerializedNode *serializedNode){
	serializedNode->set_node_id(node.id);
	serializedNode->set_object_id(node.object_id);
	//std::cout<<"node.object_id:"<<node.object_id<<std::endl;
	serializedNode->set_category_id(node.category_id);
	serialize(node.bbox, serializedNode->mutable_bbox());
	for (auto n_id : node.neighbours){
		serializedNode->add_neighbours_ids(n_id);
	}	
}


// Feature ================================================================================================
void Osmap::serialize(const OsmapKeyFrame &keyframe, SerializedKeyframeFeatures *serializedKeyframeFeatures){
  serializedKeyframeFeatures->set_keyframe_id(keyframe.mnId);
  for(int i=0; i<keyframe.N; i++){
	if(!options[ONLY_MAPPOINTS_FEATURES] || keyframe.mvpMapPoints[i]){	// If chosen to only save mappoints features, check if there is a mappoint.
		SerializedFeature &serializedFeature = *serializedKeyframeFeatures->add_feature();

		// KeyPoint
		serialize(keyframe.mvKeysUn[i], serializedFeature.mutable_keypoint());

		// If there is a MapPoint, serialize it
		if(keyframe.mvpMapPoints[i])
		  serializedFeature.set_mappoint_id(keyframe.mvpMapPoints[i]->mnId);

		// Serialize descriptor but skip if chosen to not do so.
		if(!options[NO_FEATURES_DESCRIPTORS])	//
		  serialize(keyframe.mDescriptors.row(i), serializedFeature.mutable_briefdescriptor());
	}
  }
}

OsmapKeyFrame *Osmap::deserialize(const SerializedKeyframeFeatures &serializedKeyframeFeatures){
  unsigned int KFid = serializedKeyframeFeatures.keyframe_id();
  OsmapKeyFrame *pKF = getKeyFrame(KFid);
  if(pKF){
	  int n = serializedKeyframeFeatures.feature_size();
	  const_cast<int&>(pKF->N) = n;
	  const_cast<std::vector<cv::KeyPoint>&>(pKF->mvKeysUn).resize(n);
	  pKF->mvpMapPoints.resize(n);
	  const_cast<cv::Mat&>(pKF->mDescriptors) = Mat(n, 32, CV_8UC1);	// n descriptors

// ORB-SLAM2 needs to have set mvuRight and mvDepth even though they are not used in monocular.  DUMMY_MAP and OS1 don't have these properties.
#if !defined OSMAP_DUMMY_MAP && !defined OS1
	  const_cast<std::vector<float>&>(pKF->mvuRight) = vector<float>(n,-1.0f);
	  const_cast<std::vector<float>&>(pKF->mvDepth) = vector<float>(n,-1.0f);
#endif
	  for(int i=0; i<n; i++){
		const SerializedFeature &feature = serializedKeyframeFeatures.feature(i);
		if(feature.mappoint_id())		  pKF->mvpMapPoints[i] = getMapPoint(feature.mappoint_id());
		if(feature.has_keypoint())    	  deserialize(feature.keypoint(), const_cast<cv::KeyPoint&>(pKF->mvKeysUn[i]));
		if(feature.has_briefdescriptor()){
			Mat descriptor;
			deserialize(feature.briefdescriptor(), descriptor);
			descriptor.copyTo(pKF->mDescriptors.row(i));
		}
	  }
  } else {
	  cerr << "KeyFrame id "<< KFid << "not found while deserializing features: skipped.  Inconsistence between keyframes and features serialization files." << endl;
  }
  return pKF;
}


int Osmap::serialize(const vector<OsmapKeyFrame*> &vectorKF, SerializedKeyframeFeaturesArray &serializedKeyframeFeaturesArray){
  unsigned int nFeatures = 0;
  for(auto pKF:vectorKF){
    serialize(*pKF, serializedKeyframeFeaturesArray.add_feature());
    nFeatures += pKF->N;
  }

  return nFeatures;
}


int Osmap::deserialize(const SerializedKeyframeFeaturesArray &serializedKeyframeFeaturesArray){
  int nFeatures = 0, i, n = serializedKeyframeFeaturesArray.feature_size();
  for(i=0; i<n; i++){
    KeyFrame *pKF=deserialize(serializedKeyframeFeaturesArray.feature(i));
	if(pKF)
		nFeatures += pKF->N;
  }

  return nFeatures;
}



// Kendon Varda's code to serialize many messages in one file, from https://stackoverflow.com/questions/2340730/are-there-c-equivalents-for-the-protocol-buffers-delimited-i-o-functions-in-ja
// Returns false if error, true if ok.
bool Osmap::writeDelimitedTo(
    const google::protobuf::MessageLite& message,
    google::protobuf::io::ZeroCopyOutputStream* rawOutput
){
  // We create a new coded stream for each message.  Don't worry, this is fast.
  google::protobuf::io::CodedOutputStream output(rawOutput);

  // Write the size.
  const int size = message.ByteSize();
  output.WriteVarint32(size);
  uint8_t* buffer = output.GetDirectBufferForNBytesAndAdvance(size);
  if (buffer != NULL) {
    // Optimization:  The message fits in one buffer, so use the faster
    // direct-to-array serialization path.
    message.SerializeWithCachedSizesToArray(buffer);
  } else {
    // Slightly-slower path when the message is multiple buffers.
    message.SerializeWithCachedSizes(&output);
    if (output.HadError()){
      cerr << "Error in writeDelimitedTo." << endl;
      return false;
    }
  }
  return true;
}

bool Osmap::readDelimitedFrom(
    google::protobuf::io::ZeroCopyInputStream* rawInput,
    google::protobuf::MessageLite* message
){
  // We create a new coded stream for each message.  Don't worry, this is fast,
  // and it makes sure the 64MB total size limit is imposed per-message rather
  // than on the whole stream.  (See the CodedInputStream interface for more
  // info on this limit.)
  google::protobuf::io::CodedInputStream input(rawInput);

  // Read the size.
  uint32_t size;
  if (!input.ReadVarint32(&size)) return false;

  // Tell the stream not to read beyond that size.
  google::protobuf::io::CodedInputStream::Limit limit =
      input.PushLimit(size);

  // Parse the message.
  if (!message->MergeFromCodedStream(&input)) return false;
  if (!input.ConsumedEntireMessage()) return false;

  // Release the limit.
  input.PopLimit(limit);

  return true;
};


/*
 * Orbslam adapter.  Class wrappers.
 */
#ifndef OSMAP_DUMMY_MAP

OsmapMapPoint::OsmapMapPoint(Osmap *osmap):
	MapPoint(Mat(), osmap->pRefKF, &osmap->map)
{};

OsmapObject::OsmapObject(const Ellipsoid& ell, unsigned int id, std::unordered_map<int, int> cats):
	Object(ell, id, cats)
{};

OsmapMapObject::OsmapMapObject(const Ellipsoid& ell):
	MapObject(ell, nullptr)
{};

OsmapObjectTrack::OsmapObjectTrack(Osmap *osmap):
	ObjectTrack()
{};

OsmapKeyFrame::OsmapKeyFrame(Osmap *osmap):
	KeyFrame(osmap->currentFrame, &osmap->map, &osmap->keyFrameDatabase)
{};

#else

OsmapMapPoint::OsmapMapPoint(Osmap *osmap):
	MapPoint(osmap)
{};

OsmapKeyFrame::OsmapKeyFrame(Osmap *osmap):
	KeyFrame(osmap)
{};

#endif


}	// namespace ORB_SLAM2
