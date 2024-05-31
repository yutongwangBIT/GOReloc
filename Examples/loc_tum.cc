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


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>
#include <experimental/filesystem>

#include <System.h>
#include "Converter.h"
#include "Osmap.h"
#include <nlohmann/json.hpp>
#include "Utils.h"

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<long double> &vTimestamps);

int main(int argc, char **argv)
{
    srand(time(nullptr));
    std::cout << "C++ version: " << __cplusplus << std::endl;

    if(argc != 11)
    {
        cerr << endl << "Usage:\n"
                        " ./oa-slam_localization\n"
                        "      vocabulary_file\n"
                        "      camera_file\n"
                        "      path_to_image_sequence (.txt file listing the images or a folder with rgb.txt)\n"
                        "      detections_file (.json file with detections or .onnx yolov5 weights)\n"
                        "      categories_to_ignore_file (file containing the categories to ignore (one category_id per line))\n"
                        "      map_file (.yaml)\n"
                        "      relocalization_mode ('points' or 'goreloc')\n"
                        "      output_name \n"
                        "      force_relocalization_on_each_frame (0 or 1)\n";
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::string vocabulary_file = string(argv[1]);
    std::string parameters_file = string(argv[2]);
    string path_to_images = string(argv[3]);
    string strAssociationFilename = string(argv[4]);
    std::string detections_file(argv[5]);
    std::string categories_to_ignore_file(argv[6]);
    string map_file = string(argv[7]);
    string reloc_mode = string(argv[8]);
    string output_name = string(argv[9]);
    bool force_reloc = std::stoi(argv[10]);


    // possible to pass 'webcam_X' where 'X' is the webcam id
    bool use_webcam = false;
    int webcam_id = 0;
    if (path_to_images.size() >= 6 && path_to_images.substr(0, 6) == "webcam") {
        use_webcam = true;
        if (path_to_images.size() > 7) {
            webcam_id = std::stoi(path_to_images.substr(7));
        }
    }

    string output_folder = output_name;
    if (output_folder.back() != '/')
        output_folder += "/";
    fs::create_directories(output_folder);

    // Get map folder absolute path
    int l = map_file.find_last_of('/') + 1;
    std::string map_folder = map_file.substr(0, l);
    if (map_folder[0] != '/') {
        fs::path map_folder_abs = fs::current_path() / map_folder;
        map_folder = map_folder_abs.string();
    }

    // Load object detections
    auto extension = get_file_extension(detections_file);
    std::shared_ptr<ORB_SLAM2::ImageDetectionsManager> detector = nullptr;
    bool detect_from_file = false;
    if (extension == "json") { // load from external detections file
        detector = std::make_shared<ORB_SLAM2::DetectionsFromFile>(detections_file);
        detect_from_file = true;
    } else {
        std::cout << "Invalid detection file. It should be .json "
                      "No detections will be obtained.\n";
    }


    ORB_SLAM2::enumRelocalizationMode relocalization_mode = ORB_SLAM2::RELOC_POINTS;
    if (reloc_mode == string("points"))
        relocalization_mode = ORB_SLAM2::RELOC_POINTS;
    else if (reloc_mode == std::string("goreloc"))
        relocalization_mode = ORB_SLAM2::GOReloc_Mode;
    else {
        std::cerr << "Error: Invalid parameter for relocalization mode. "
                     "It should be 'points' or 'goreloc'.\n";
        return 1;
    }

    // Load images
    cv::VideoCapture cap;
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<long double> vTimestamps;
    if (!use_webcam) {
        LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);
    }
    else{
        if (cap.open(webcam_id)) {
            std::cout << "Opened webcam: " << webcam_id << "\n";
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        } else {
            std::cerr << "Failed to open webcam: " << webcam_id << "\n";
            return -1;
        }
    }

    int start = 0; // 700;//500;//700;
    int nImages = vstrImageFilenamesRGB.size() - start; //576

    ORB_SLAM2::System SLAM(vocabulary_file, parameters_file, ORB_SLAM2::System::RGBD, true, false, 0);
    SLAM.SetRelocalizationMode(relocalization_mode);
    SLAM.ActivateLocalizationMode();
    SLAM.map_folder = map_folder;

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.reserve(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    ORB_SLAM2::Osmap osmap = ORB_SLAM2::Osmap(SLAM);
    std::cout << "Start loading map from:" << map_file << std::endl;
    osmap.mapLoad(map_file, false, true, "rgbd");
    std::cout << "End of loading map" << std::endl;
    //===============================================================================
    std::cout << "read keyframe json file and save im path to each keyframe." <<std::endl;
    std::vector<pair<int, std::string>> keyframe_filenames;
    std::string filename_json = "/home/yutong/OA-SLAM/Data/diamond_all/keyframes_poses_diamond_all.json";
    std::ifstream fin_keyframe(filename_json);
    if (!fin_keyframe.is_open())
    {
        std::cerr << "Warning failed to open file: " << filename_json << std::endl;
    }
    json data;
    fin_keyframe >> data;
    ORB_SLAM2::Map* map = SLAM.mpMap;
    for (auto& frame : data)
    {
        int id = frame["mnId"].get<int>();
        std::string name = frame["file_name"].get<std::string>();
        //std::cout << name <<std::endl;
        for(auto pkf : map->getKeyFrames()){
            if(id == pkf->mnId){
                cv::Mat im = cv::imread(name);//, cv::IMREAD_UNCHANGED);
                pkf->im = im;
                break;
            }
        }
    }

    //===============================================================================
    //SAVE MAP OBJECTS TO JSON
    json json_objects = json::array();
    auto map_objects = map->GetAllObjects();
    for(auto obj:map_objects){
        json object_j;
        object_j["id"] = obj->GetId();
        object_j["cat"] = obj->GetCategoryId();
        auto ellipsoid = obj->GetEllipsoid();
        auto c = ellipsoid.GetCenter();
        auto a = ellipsoid.GetAxes();
        auto r = ellipsoid.GetOrientation();
        json json_center = json::array({static_cast<double>(c.x()), static_cast<double>(c.y()), static_cast<double>(c.z())});
        json json_axes = json::array({static_cast<double>(a.x()), static_cast<double>(a.y()), static_cast<double>(a.z())});
        json json_rotation = json::array();
        for(int i=0; i<3; i++){
            json json_row = json::array();
            for(int j=0; j<3; j++){
                json_row.push_back(r(i,j));
            }
            json_rotation.push_back(json_row);
        }
        object_j["center"] = json_center;
        object_j["axes"] = json_axes;
        object_j["matrix"] = json_rotation;
        json_objects.push_back(object_j);
    }
    std::ofstream file_objects("/home/yutong/OA-SLAM/Data/map_objects_fr2_all.json");
    file_objects << json_objects.dump(4);
    file_objects.close();
    //===============================================================================

    // Main loop
    cv::Mat imRGB, imD;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
    poses.reserve(nImages);
    std::vector<std::string> filenames;
    filenames.reserve(nImages);
    std::vector<bool> reloc_status;
    reloc_status.reserve(nImages);
    int ni = 0;
    std::vector<long double> times;
    int count_success = 0;
    
    while (1)
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        std::string filename;
        if (use_webcam) {
            cap >> imRGB;  // get image from webcam
            filename = "frame_" + std::to_string(ni) + ".png";
        }
        else
        {
            filename = path_to_images + '/' + vstrImageFilenamesRGB[ni+start];//path_to_images + vstrImageFilenames[ni];
            std::cout<<"filename:"<<filename<<std::endl;
            imRGB = cv::imread(filename);//, cv::IMREAD_UNCHANGED);  // read image from disk
            imD = cv::imread(path_to_images + '/' + vstrImageFilenamesD[ni+start],CV_LOAD_IMAGE_UNCHANGED);
            
        }
        long double tframe = (ni+start) < vTimestamps.size() ? vTimestamps[ni+start] : std::time(nullptr);
        
        
        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image: "
                 << filename << endl;
            return 1;
        }
        filenames.push_back(filename);

        // Get object detections
        std::vector<ORB_SLAM2::Detection::Ptr> detections;
        if (detector) {
            if (detect_from_file)
                detections = detector->detect(filename); // from detections file
            else
                detections = detector->detect(imRGB);  // from neural network
        }

        //std::cout<<"detections:"<<detections.size()<<std::endl;

        // Pass the image and detections to the SLAM system
        cv::Mat m = SLAM.TrackRGBD(imRGB,imD,tframe, detections, true, false);
        reloc_status.push_back(SLAM.relocalization_status);

        if (!m.empty()){
            poses.push_back(ORB_SLAM2::cvToEigenMatrix<double, float, 4, 4>(m));
            times.push_back(tframe);
            count_success += 1;
        }
        //else
            //poses.push_back(Eigen::Matrix4d::Identity());

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack.push_back(ttrack);

        if (SLAM.ShouldQuit())
            break;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];
        //std::cout<<"vTimestamps[ni+1]:"<<vTimestamps[ni+1]<<"tframe:"<<tframe<<"ttrack:"<<ttrack<<", T:"<<T<<std::endl;
        if(ttrack<T)
            usleep((T-ttrack)*1e5);

        ++ni;
        //ni += 5;
        if (ni >= nImages)
            break;

        
    }
    double success_rate = count_success/vTimesTrack.size();
    // Stop all threads
    SLAM.Shutdown();
    SLAM.SaveAssociationJson(output_folder+"association_results.json");
  
    // Save camera trajectory
    json json_data;
    ofstream f;
    string filename_tum = output_folder+"CameraTrajectory.txt";
    f.open(filename_tum.c_str());
    for (size_t i = 0; i < poses.size(); ++i)
    {
        Eigen::Matrix4d m = poses[i];
        json R({{m(0, 0), m(0, 1), m(0, 2)},
                {m(1, 0), m(1, 1), m(1, 2)},
                {m(2, 0), m(2, 1), m(2, 2)}});
        json t({m(0, 3), m(1, 3), m(2, 3)});
        json image_data;
        image_data["time"] = times[i];
        image_data["R"] = R;
        image_data["t"] = t;
        json_data.push_back(image_data);
        //cv::Mat m_ = ORB_SLAM2::eigenToCvMatrix<float, double, 4, 4>(m);
        cv::Mat m_ = cv::Mat::zeros(3, 3, CV_32F);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m_.at<float>(i, j) = m(i, j);
        cv::Mat Rwc = m_.rowRange(0,3).colRange(0,3).t();//m_.rowRange(0,3).colRange(0,3).t();
        //cv::Mat twc = m_.colRange(0,3).row(3);//-Rwc*m_.rowRange(0,3).col(3);
        cv::Mat t_ = cv::Mat::zeros(3, 1, CV_32F);
        t_.at<float>(0) = m(0, 3);
        t_.at<float>(1) = m(1, 3);
        t_.at<float>(2) = m(2, 3);
        cv::Mat twc = -Rwc*t_;
        vector<float> q = ORB_SLAM2::Converter::toQuaternion(Rwc);
        long double time = times[i];
        f << setprecision(15) << time << " " <<  setprecision(9) << twc.at<float>(0) << " " 
        << twc.at<float>(1) << " " << twc.at<float>(2)
        << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    std::cout << "Saved all frames trajectory to " << filename_tum << endl;
    std::cout<<"total tracked images: "<<vTimesTrack.size()<<std::endl;
    std::cout<<"reloc success rate is: "<<count_success<<std::endl;

    std::ofstream json_file(output_folder + "camera_poses_" + output_name + ".json");
    json_file << json_data;
    json_file.close();
    std::cout << "Saved " << poses.size() << " poses.\n";

    // Save camera trajectory
    //SLAM.SaveTrajectoryTUM(output_folder+"CameraTrajectory.txt");
    //SLAM.SaveKeyFrameTrajectoryTUM(output_folder+"KeyFrameTrajectory.txt");


    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int n=0; n<vTimesTrack.size(); n++)
    {
        totaltime+=vTimesTrack[n];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[vTimesTrack.size()/2] << endl;
    cout << "mean tracking time: " << totaltime/vTimesTrack.size() << endl;
    return 0;
}


void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<long double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}