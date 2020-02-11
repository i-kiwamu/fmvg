#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <exiv2/exiv2.hpp>

#include "photo.h"
#include "matcher.h"

namespace fmvg {

std::vector<cv::DMatch> matcher_ker(cv::Mat img1, cv::Mat img2) {
    // detect key points by BRISK
    std::vector<cv::KeyPoint> key_points1, key_points2;
    cv::Mat descriptor1, descriptor2;
    auto brisk = cv::BRISK::create();
    brisk->detectAndCompute(img1, cv::noArray(), key_points1, descriptor1);
    brisk->detectAndCompute(img2, cv::noArray(), key_points2, descriptor2);
    if (descriptor1.type() != CV_32F)
        descriptor1.convertTo(descriptor1, CV_32F);
    if (descriptor2.type() != CV_32F)
        descriptor2.convertTo(descriptor2, CV_32F);

    // search matches by FLANN (k nearest neighbors method)
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);

    // filter matches (Lowe's ratio test)
    const float ratio_threshold = 0.7f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_threshold * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    return good_matches;
}  // matcher_ker

std::unordered_map<int, std::vector<cv::DMatch>> matcher(const std::vector<Photo>& photos) {
    std::unordered_map<int, std::vector<cv::DMatch>> all_matches;
    size_t n_photos = photos.size();
    for (size_t i = 0; i < n_photos-1; ++i) {
        for (size_t j = i+1; j < n_photos; ++j) {
            int key = i * n_photos + j;
            cv::Mat img_i = photos[i].getMat();
            cv::Mat img_j = photos[j].getMat();
            all_matches[key] = matcher_ker(img_i, img_j);
        }
    }
    return all_matches;
}  // matcher

}  // namespace fmvg
