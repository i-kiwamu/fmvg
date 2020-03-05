#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <exiv2/exiv2.hpp>

#include "photo.h"
#include "matcher.h"

namespace fmvg {

void detectFeatures(
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
}  //detectFeatures

void matchTwoPhotos(
    const int i,
    const int j,
    const std::vector<std::vector<cv::KeyPoint>>& key_points_vec,
    const std::vector<cv::Mat>& descriptor_vec,
    cv::OutputArray matched_points
) {
    const std::vector<cv::KeyPoint>& key_points2 = key_points_vec[i+j];
    cv::Mat descriptor1 = descriptor_vec[i];
    cv::Mat descriptor2 = descriptor_vec[i+j];

    // search matches by FLANN (k nearest neighbors method)
    auto matcher = cv::DescriptorMatcher::create(
        cv::DescriptorMatcher::FLANNBASED
    );
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);

    // filter matches (Lowe's ratio test)
    const float ratio_threshold = 0.7f;
    for (size_t k = 0; k < knn_matches.size(); ++k) {
        cv::DMatch x = knn_matches[k][0];
        if (x.distance < ratio_threshold * knn_matches[k][1].distance) {
            float* matched_points_k = \
                matched_points.getMat().ptr<float>(x.queryIdx);
            matched_points_k[j*2+0] = key_points2[x.trainIdx].pt.x;
            matched_points_k[j*2+1] = key_points2[x.trainIdx].pt.y;
        }
    }
}  // matchTwoPhotos

void matchAll(
    const PhotoList& photos,
    std::vector<std::vector<cv::KeyPoint>>& key_points_vec,
    std::vector<cv::Mat>& matched_points_vec
) {
    int n_photos = photos.getNumPhotos();
    std::vector<cv::Mat> descriptor_vec(n_photos);

    for (int i = 0; i < n_photos-1; ++i) {
        if (key_points_vec[i].empty()) {
            detectFeatures(
                photos.getPhotoVector()[i],
                key_points_vec[i],
                descriptor_vec[i]
            );
        }
        int n_key_points_i = key_points_vec[i].size();
        cv::Mat& matched_points = matched_points_vec[i];
        matched_points.create(n_key_points_i, n_photos-i, CV_32FC2);
        matched_points.setTo(cv::Scalar(-1.0f, -1.0f));
        for (int j = i+1; j < n_photos; ++j) {
            if (key_points_vec[j].empty()) {
                detectFeatures(
                    photos.getPhotoVector()[j],
                    key_points_vec[j],
                    descriptor_vec[j]
                );
            }
            matchTwoPhotos(
                i, j-i,
                key_points_vec,
                descriptor_vec,
                matched_points
            );
        }
    }
}  // matchAll

}  // namespace fmvg
