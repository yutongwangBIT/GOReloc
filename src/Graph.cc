

#include "Graph.h"
#include "Object.h"

namespace ORB_SLAM2 
{
    Graph::Graph(){
        category_ids_statistics = Eigen::VectorXd::Zero(80);
    }

    Graph::Graph(vector<pair<int,int>> edge_list, vector<pair<int,int>> node_labels) {
        for (auto& [node_id,label] : node_labels) {
            add_node(node_id,label);
        }
        category_ids_statistics = Eigen::VectorXd::Zero(80);
        for (auto& [node1,node2] : edge_list) {
            add_edge(node1,node2);
        }
        //compute_feature_vectors(); 
    }

    void Graph::add_node(int node_id, int label, float confidence, float hue, Eigen::Vector4d bbox, Ellipse ell, float depth) {
        Attribute attribute = {-1, label, confidence, hue, bbox, ell, nullptr, depth};
        nodes[node_id] = vector<int>();
        attributes[node_id] = attribute;
    }

    void Graph::add_edge(int node1, int node2, float weight) {
        if (node1 > node2) {
            swap(node1,node2);
        }
        if (!has_edge(node1,node2)){
            edges[make_pair(node1,node2)] = weight;
            nodes[node1].push_back(node2);
            nodes[node2].push_back(node1);
        }
    }

    void Graph::compute_catagory_statistics(){
        category_ids_statistics.setZero();
        for(auto& node : nodes)
            category_ids_statistics[attributes[node.first].label] += 1;
        for(size_t i=0; i<category_ids_statistics.size(); i++){
            category_ids_statistics[i] /= nodes.size();
        } 
    } 

    void Graph::compute_feature_vectors() {
        compute_catagory_statistics();
        feature_vectors.clear();
        for (auto& [node_id, neighbours] : nodes) {
            Eigen::VectorXd feature_vector(80);
            feature_vector.setZero(); 
            for (int neighbour : neighbours) {
                feature_vector[attributes[neighbour].label] += 
                            1-category_ids_statistics[attributes[neighbour].label];//1.0f;
            }
            feature_vector.normalize();
            feature_vectors[attributes[node_id].label].push_back(make_pair(node_id, feature_vector));
        }
    }


    Eigen::VectorXd Graph::compute_feature_vector_node(int node_id){
        Eigen::VectorXd feature_vector(80);
        feature_vector.setZero(); 
        for (int neighbour : nodes[node_id]) {
            feature_vector[attributes[neighbour].label] += 1.0f;
        }
        return feature_vector;
    }

    void Graph::compute_feature_vectors_map() { // currently not used, it is calculated in Osmap.
        feature_vectors.clear();
        for (auto& [node_id, neighbours] : nodes) {
            Eigen::VectorXd feature_vector(80);
            feature_vector.setZero(); 
            for (int neighbour : neighbours) {
                auto obj = attributes[neighbour].obj;
                if(obj){
                    for(auto [label, count] : obj->GetAllCategoryIds()){ //TODO FREQUENCY FOR WEIGHT
                        feature_vector[label] += 1.0f;
                    }
                }
            }
            if(attributes[node_id].obj){
                for(auto [label, count] : attributes[node_id].obj->GetAllCategoryIds()){
                    feature_vectors[label].push_back(make_pair(node_id, feature_vector));
                }
            }
        }
    }

    Eigen::VectorXd Graph::compute_feature_vector_node_3d(int node_id){// currently not used, it is calculated in Osmap.
        Eigen::VectorXd feature_vector(80);
        feature_vector.setZero(); 
        for (int neighbour : nodes[node_id]) {
            auto obj = attributes[neighbour].obj;
            if(obj){
                for(auto [label, count] : obj->GetAllCategoryIds()){ //TODO FREQUENCY FOR WEIGHT
                    feature_vector[label] += 1.0f;
                }
                //feature_vector[obj->GetCategoryId()] += 1.0f;
            }
        }
        return feature_vector;
    }

    std::map<int, std::vector<int>> Graph::GetMatchCandidatesOfEachNode(map<int, std::vector<pair<int, Eigen::VectorXd>>> feature_vectors_another, int k){
        std::map<int, std::vector<int>> match_candidates;
        for (auto& [label, feature_vectors_per_label1] : feature_vectors){
            auto feature_vectors_per_label2 = feature_vectors_another[label];
            //std::cout<<"1:"<<feature_vectors_per_label1.size()<<"2:"<<feature_vectors_per_label2.size()<<std::endl;
            for(size_t j=0; j<feature_vectors_per_label1.size(); j++){
                int node_id1 = feature_vectors_per_label1[j].first;
                match_candidates[node_id1] = std::vector<int>();
                Eigen::VectorXd feature1 = feature_vectors_per_label1[j].second;
                vector<pair<double, int> > vPairs;
                for(size_t l=0; l<feature_vectors_per_label2.size(); l++){
                    Eigen::VectorXd feature2 = feature_vectors_per_label2[l].second;
                    double sim = feature1.dot(feature2)/(feature1.norm() * feature2.norm());
                    if(sim>0.01){
                        vPairs.push_back(make_pair(sim, feature_vectors_per_label2[l].first));
                    }
                }
                if(!vPairs.empty()){
                    sort(vPairs.begin(),vPairs.end(), [](pair<double, int> a, pair<double, int> b) {
                        return a.first > b.first;
                    });
                    int i = 0;
                    while(i<vPairs.size() && i<k){
                        match_candidates[node_id1].push_back(vPairs[i].second);
                        i++;
                    }
                }
            }
        }
        return match_candidates;
    }

}
