#ifndef FMVG_MATCHER_H_
#define FMVG_MATCHER_H_

namespace fmvg {

void matcher_ker(
    cv::Mat img1,
    cv::Mat img2,
    cv::OutputArray matched_points1,
    cv::OutputArray matched_points2
);

void match_all(const std::vector<cv::Mat>& img_vec);

}  // namespace fmvg

#endif  // FMVG_MATCHER_H_
