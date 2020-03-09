#ifndef FMVG_MATCHER_H_
#define FMVG_MATCHER_H_

namespace fmvg {

class SfM {
    // data members
    PhotoList photo_list_;
    std::vector<std::vector<cv::KeyPoint>> key_points_vec_;
    std::vector<cv::Mat> descriptor_vec_;
    std::map<std::pair<int,int>, std::vector<cv::DMatch>> matched_map_;
    std::vector<cv::Vec2d> matched_points_photo_vec_;
    std::vector<cv::Vec3d> matched_points_world_vec_;
    std::vector<cv::Vec3d> point_cloud_;
    std::vector<cv::Vec3b> point_cloud_color_;

    // private member functions
    void initializeSfM(const PhotoList& input_photo_list);
    void detectFeatures(
        const Photo& photo,
        std::vector<cv::KeyPoint>& key_points,
        cv::OutputArray descriptor
    );
    std::vector<cv::DMatch> matchWithRatioTest(
        const cv::DescriptorMatcher* matcher,
        const cv::Mat& descriptor1,
        const cv::Mat& descriptor2
    );
    std::vector<cv::DMatch> matchTwoPhotos(
        const cv::DescriptorMatcher* matcher,
        int i, int j
    );
    void matchAll();
    void buildTracks();


public:
    // constructor
    SfM() {}
    SfM(const PhotoList& input_photo_list) {
        initializeSfM(input_photo_list);
    };

    // public member functions
    void runSfM();

};  // class SfM

}  // namespace fmvg

#endif  // FMVG_MATCHER_H_
