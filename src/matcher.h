#ifndef FMVG_MATCHER_H_
#define FMVG_MATCHER_H_

namespace fmvg {

void detectFeatures(
    const Photo& photo,
    std::vector<cv::KeyPoint>& key_points,
    cv::OutputArray descriptor
);

void matchTwoPhotos(
    const int i,
    const int j,
    const std::vector<std::vector<cv::KeyPoint>>& key_points_vec,
    const std::vector<cv::Mat>& descriptor_vec,
    cv::OutputArray matched_points
);

void matchAll(
    const PhotoList& photos,
    std::vector<std::vector<cv::KeyPoint>>& key_points_vec,
    std::vector<cv::Mat>& matched_points_vec
);

}  // namespace fmvg

#endif  // FMVG_MATCHER_H_
