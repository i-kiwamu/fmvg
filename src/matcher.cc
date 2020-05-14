#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <set>
#include <numeric>

#define CERES_FOUND true  // needs to include cv::sfm::reconstruct

#include <opencv2/opencv.hpp>
#include <opencv2/sfm.hpp>

#include <exiv2/exiv2.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graphviz.hpp>

#include "photo.h"
#include "matcher.h"

using std::cout;
using std::cerr;
using std::endl;

namespace fmvg {

void SfM::initializeSfM(const PhotoList& input_photo_list, const bool saveDebug) {
    photo_list_ = input_photo_list;
    saveDebug_ = saveDebug;

    // detect features
    std::vector<cv::KeyPoint> key_points;
    cv::Mat descriptor;
    for (auto p : photo_list_) {
        detectFeatures(p, key_points, descriptor);
        key_points_vec_.push_back(key_points);
        descriptor_vec_.push_back(descriptor);
    }
}  // SfM::initializeSfM


// detect features by BRISK
void SfM::detectFeatures(
    const Photo& photo,
    std::vector<cv::KeyPoint>& key_points,
    cv::OutputArray descriptor
) {
    auto brisk = cv::BRISK::create();
    brisk->detectAndCompute(
        photo.getMatCorrected(),
        cv::noArray(),
        key_points,
        descriptor
    );
    if (brisk->descriptorType() != CV_32F)
        descriptor.getMat().convertTo(descriptor, CV_32F);
}  // SfM::detectFeatures


// match two photos with Lowe's ratio test
std::vector<cv::DMatch> SfM::matchWithRatioTest(
    const cv::DescriptorMatcher* matcher,
    const cv::Mat& descriptor1,
    const cv::Mat& descriptor2
) {
    // Raw match
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);

    // filter matches (Lowe's ratio test)
    const float ratio_threshold = 0.7f;
    std::vector<cv::DMatch> matched_vec;
    matched_vec.reserve(knn_matches.size());
    for (size_t k = 0; k < knn_matches.size(); ++k) {
        const cv::DMatch x = knn_matches[k][0];
        if (x.distance < ratio_threshold * knn_matches[k][1].distance) {
            matched_vec.push_back(x);
        }
    }
    return matched_vec;
}  // SfM::matchWithRatioTest


// match two photos with reciprocity filter & RANSAC
std::vector<cv::DMatch> SfM::matchTwoPhotos(
    const cv::DescriptorMatcher* matcher,
    int i,
    int j
) {
    // target KeyPoint and descriptor for photo i & j
    std::vector<cv::KeyPoint> keypoints1 = key_points_vec_[i];
    std::vector<cv::KeyPoint> keypoints2 = key_points_vec_[j];
    cv::Mat descriptor1 = descriptor_vec_[i];
    cv::Mat descriptor2 = descriptor_vec_[j];

    std::vector<cv::DMatch> matched_vec = \
        matchWithRatioTest(matcher, descriptor1, descriptor2);
    std::vector<cv::DMatch> matched_vec_rev = \
        matchWithRatioTest(matcher, descriptor2, descriptor1);
    int n_matched_vec = matched_vec.size();

    // reciprocity test filter
    // Only accept match if 1 mathces 2 AND 2 matches 1.
    std::vector<cv::DMatch> matched_vec_merged;
    if (n_matched_vec > 0) {
        for (const cv::DMatch& dmr : matched_vec_rev) {
            bool found = false;
            for (const cv::DMatch& dm : matched_vec) {
                if (dmr.queryIdx == dm.trainIdx and \
                    dmr.trainIdx == dm.queryIdx)
                {
                    matched_vec_merged.push_back(dm);
                    found = true;
                    break;
                }
            }
            if (found) continue;
        }
    }
    int n_matched_vec_merged = matched_vec_merged.size();

    // epipolar constraint to remove outliers by RANSAC
    std::vector<uint8_t> inlinersMask(n_matched_vec_merged);
    std::vector<cv::Point2f> points1, points2;
    std::vector<cv::DMatch> matched_vec_epi;
    matched_vec_epi.reserve(n_matched_vec_merged);
    if (n_matched_vec_merged > 0) {
        for (const cv::DMatch& dm : matched_vec_merged) {
            points1.push_back(keypoints1[dm.queryIdx].pt);
            points2.push_back(keypoints2[dm.trainIdx].pt);
        }
        cv::findFundamentalMat(points1, points2, inlinersMask);

        for (size_t m = 0; m < n_matched_vec_merged; ++m) {
            if (inlinersMask[m])
                matched_vec_epi.push_back(matched_vec_merged[m]);
        }
    }

    cout << "Matching " + std::to_string(i) + " and "
         << std::to_string(j) + ": "
         << std::to_string(matched_vec_epi.size()) + " / "
         << std::to_string(matched_vec_merged.size()) + " / "
         << std::to_string(matched_vec.size()) << endl;

    return matched_vec_epi;
}  // SfM::matchTwoPhotos


// match all photos
void SfM::matchAll() {
    int n_photos = photo_list_.getNumPhotos();

    // search matches by FLANN (k nearest neighbors method)
    auto matcher = cv::DescriptorMatcher::create(
        cv::DescriptorMatcher::FLANNBASED
    );

    for (int i = 0; i < n_photos-1; ++i) {
        if (key_points_vec_[i].empty()) {
            detectFeatures(
                photo_list_.getPhotoVector()[i],
                key_points_vec_[i],
                descriptor_vec_[i]
            );
        }
        for (int j = i+1; j < n_photos; ++j) {
            if (key_points_vec_[j].empty()) {
                detectFeatures(
                    photo_list_.getPhotoVector()[j],
                    key_points_vec_[j],
                    descriptor_vec_[j]
                );
            }
            std::vector<cv::DMatch> match_ij = \
                matchTwoPhotos(matcher, i, j);
            matched_map_[std::make_pair(i,j)] = match_ij;
        }
    }
}  // SfM::matchAll


void SfM::buildTracks() {
    struct PhotoFeature {
        int photo_id;
        size_t feature_id;
    };
    typedef boost::adjacency_list <
        boost::listS,
        boost::vecS,
        boost::undirectedS,
        PhotoFeature
    > Graph;
    typedef boost::graph_traits <Graph>::vertex_descriptor Vertex;

    std::map<std::pair<int, int>, Vertex> vertex_by_photo_feature;

    Graph g;

    // add vertices: photo features
    for (int i = 0; i < key_points_vec_.size(); ++i) {
        std::vector<cv::KeyPoint> key_points = key_points_vec_[i];
        for (size_t j = 0; j < key_points.size(); ++j) {
            Vertex v = boost::add_vertex(g);
            g[v].photo_id = i;
            g[v].feature_id = j;
            vertex_by_photo_feature[std::make_pair(i,j)] = v;
        }
    }

    // add edges: feature matches
    for (const auto& match : matched_map_) {
        for (const cv::DMatch& dm : match.second) {
            Vertex& vI = vertex_by_photo_feature[
                std::make_pair(match.first.first,dm.queryIdx)];
            Vertex& vJ = vertex_by_photo_feature[
                std::make_pair(match.first.second, dm.trainIdx)];
            boost::add_edge(vI, vJ, g);
        }
    }

    using Filtered = boost::filtered_graph<
        Graph,
        boost::keep_all,
        std::function<bool(Vertex)>
    >;
    Filtered gFiltered(g, boost::keep_all{}, [&g](Vertex vd) {
        return degree(vd, g) > 0;
    });

    // get connected components
    std::vector<int> component(boost::num_vertices(gFiltered), -1);
    int num = boost::connected_components(gFiltered, &component[0]);
    std::map<int, std::vector<Vertex>> components_map;
    for (size_t i = 0; i < component.size(); ++i) {
        if (component[i] >= 0)
            components_map[component[i]].push_back(i);
    }

    // filter bad components (with more than 1 feature from a single photo)
    std::vector<int> vertex_in_good_component(
        boost::num_vertices(gFiltered),
        -1
    );
    std::map<int, std::vector<Vertex>> good_components_map;
    for (const auto& c : components_map) {
        std::set<int> photos_in_component;
        bool is_component_good = true;
        for (int j = 0; j < c.second.size(); ++j) {
            const int photo_id = g[c.second[j]].photo_id;
            if (photos_in_component.count(photo_id) > 0) {
                // photo already represented in this component
                is_component_good = false;
                break;
            } else {
                photos_in_component.insert(photo_id);
            }
        }
        if (is_component_good) {
            for (int j = 0; j < c.second.size(); ++j)
                vertex_in_good_component[c.second[j]] = 1;
            good_components_map[c.first] = c.second;
        }
    }

    Filtered g_good_components(
        g,
        boost::keep_all{},
        [&vertex_in_good_component](Vertex vd) {
            return vertex_in_good_component[vd] > 0;
        }
    );

    cout << "Total number of components found: " + \
        std::to_string(components_map.size()) << endl;
    cout << "Number of good components: " + \
        std::to_string(good_components_map.size()) << endl;
    
    const int accum_components = std::accumulate(
        good_components_map.begin(),
        good_components_map.end(),
        0,
        [](int a, std::pair<const int, std::vector<Vertex> >& v) {
            return a+v.second.size();
        });
    cout << "Average component size: " + \
        std::to_string((float)accum_components /
                       (float)(good_components_map.size())) << endl;
    
    // write tracks graph
    if (saveDebug_) {
        struct my_node_writer {
            my_node_writer(Graph& g_) : g(g_) {};
            void operator()(std::ostream& out, Vertex v) {
                const int imgId = g[v].photo_id;
                out << " [label=\"" << imgId << "\" style=filled]";
            };
            Graph g;
        };
        std::ofstream ofs("match_graph_good_components.gv");
        write_graphviz(ofs, g_good_components, my_node_writer(g));
        cout << "Please export Graphviz file to SVG as below:" << endl;
        cout << "  dot -Kdot -Tsvg match_graph_good_components.gv > match_graph_good_components.svg" << endl;
    }
    
    // each component is a track
    const size_t n_views = photo_list_.getNumPhotos();
    tracks_.resize(n_views);
    for (int i = 0; i < n_views; ++i) {
        tracks_[i].create(2, good_components_map.size(), CV_64FC1);
        tracks_[i].setTo(-1.0);
    }
    int i = 0;
    for (auto c = good_components_map.begin();
         c != good_components_map.end();
         ++c, ++i) {
        for (const int v : c->second) {
            const int image_id = g[v].photo_id;
            const size_t feature_id = g[v].feature_id;
            const cv::Point2f p = key_points_vec_[image_id][feature_id].pt;
            tracks_[image_id].at<double>(0, i) = p.x;
            tracks_[image_id].at<double>(1, i) = p.y;
        }
    }
}  // SfM::buildTracks

bool SfM::reconstructFromTracks() {
    const int n_photos = photo_list_.getNumPhotos();
    // const cv::Size photo_size = photo_list_.getPixelDims();
    cv::Mat K = cv::Mat(photo_list_.getIntrinsicMatrix());
    std::vector<cv::Mat> Rs;
    std::vector<cv::Mat> Ts;
    // std::vector<cv::Matx33d> Rs_orig = photo_list_.getRotationMatrixVec();
    // std::vector<cv::Vec3d> Ts_orig = photo_list_.getTranslationVec();
    // for (int i = 0; i < n_photos; ++i) {
    //     Rs.emplace_back(cv::Mat(Rs_orig[i]));
    //     Ts.emplace_back(cv::Mat(Ts_orig[i]));
    // }
    cv::sfm::reconstruct(tracks_, Rs, Ts, K, points3d_, true);

    cout << "Reconstruction" << endl;
    cout << "  Estimated 3D points: " + std::to_string(points3d_.size()) << endl;
    cout << "  Estimated cameras: " + std::to_string(Rs.size()) << endl;
    cout << "  Refined intrinsics: " << endl;
    cout << K << endl;

    if (Rs.size() != n_photos) {
        cerr << "Unable to reconstruct all camera views (" + \
            std::to_string(n_photos) + ")" << endl;
        return false;
    }
    if (tracks_[0].cols != points3d_.size()) {
        cerr << "Unable to reconstruct all tracks (" + \
            std::to_string(tracks_[0].cols) + ")" << endl;
    }

    // create the point cloud
    point_cloud_.clear();
    for (const auto &p : points3d_) point_cloud_.emplace_back(cv::Vec3f(p));

    return true;
}

// run SfM
void SfM::runSfM() {
    matchAll();
    buildTracks();
    reconstructFromTracks();
}  // SfM::runSfM
}  // namespace fmvg
