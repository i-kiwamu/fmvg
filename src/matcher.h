#ifndef FMVG_MATCHER_H_
#define FMVG_MATCHER_H_

namespace fmvg {

// class SfM {
//     // data members
//     PhotoList photo_list_;
//     std::vector<std::vector<cv::KeyPoint>> key_points_vec_;
//     std::vector<cv::Mat> descriptor_vec_;
//     std::map<std::pair<int,int>, std::vector<cv::DMatch>> matched_map_;

// };  // class SfM

void detectFeatures(
    const Photo& photo,
    std::vector<cv::KeyPoint>& key_points,
    cv::OutputArray descriptor
);

std::vector<cv::DMatch> matchTwoPhotos(
    const cv::DescriptorMatcher* matcher,
    int i,
    int j,
    const std::vector<std::vector<cv::KeyPoint>>& keypoints_vec,
    const std::vector<cv::Mat>& descriptor_vec
);

void matchAll(
    const PhotoList& photos,
    std::map<std::pair<int,int>, std::vector<cv::DMatch>>& matched_map
);

}  // namespace fmvg

#endif  // FMVG_MATCHER_H_
