#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <exiv2/exiv2.hpp>

#include "photo.h"
#include "matcher.h"

namespace fmvg {

void matcher_ker(
    cv::Mat img1,
    cv::Mat img2, 
    cv::OutputArray matched_points1,
    cv::OutputArray matched_points2
) {
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
    matched_points1.create(knn_matches.size(), 2, CV_16F);
    matched_points2.create(knn_matches.size(), 2, CV_16F);
    for (size_t i = 0; i < knn_matches.size(); i++) {
        cv::DMatch x = knn_matches[i][0];
        if (x.distance < ratio_threshold * knn_matches[i][1].distance) {
            cv::Mat(key_points1[x.queryIdx].pt).copyTo(matched_points1.getMatRef(i));
            cv::Mat(key_points2[x.trainIdx].pt).copyTo(matched_points2.getMatRef(i));
        }
    }
}  // matcher_ker

void match_all(const std::vector<cv::Mat>& img_vec) {
    size_t n_img = img_vec.size();
    for (size_t i = 0; i < n_img-1; ++i) {
        for (size_t j = i+1; j < n_img; ++j) {
            cv::Mat matched_points_i, matched_points_j;
            matcher_ker(img_vec[i], img_vec[j], matched_points_i, matched_points_j);
        }
    }
}  // matcher

}  // namespace fmvg
