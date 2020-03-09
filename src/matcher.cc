#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <opencv2/opencv.hpp>
#include <exiv2/exiv2.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include "photo.h"
#include "matcher.h"

using std::cout;
using std::cerr;
using std::endl;

namespace fmvg {

void SfM::initializeSfM(const PhotoList& input_photo_list) {
    photo_list_ = input_photo_list;

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


// void buildTracks(
//     std::map<std::pair<int,int>, std::vector<cv::DMatch>>& matched_map
// ) {
//     struct ImageFeature {
//         std::string image;
//         size_t featureID;
//     };
//     typedef boost::adjacency_list <
//         boost::listS,
//         boost::vecS,
//         boost::undirectedS,
//         ImageFeature
//     > Graph;
//     typedef boost::graph_traits <Graph>::vertex_descriptor Vertex;

//     std::map<std::pair<std::string, int>, Vertex> vertexByImageFeature;

//     Graph g;

//     // add vertices - image features
//     for (const auto& match : matched_map) {
//         for (const cv::DMatch& dm : match.second)
//     }

// }

// run SfM
void SfM::runSfM() {
    matchAll();
}  // SfM::runSfM
}  // namespace fmvg
