#ifndef FMVG_MATCHER_H_
#define FMVG_MATCHER_H_

namespace fmvg {

std::vector<cv::DMatch> matcher_ker(cv::Mat img1, cv::Mat img2);
std::unordered_map<int, std::vector<cv::DMatch>> matcher(const std::vector<Photo>& photos);

}  // namespace fmvg

#endif  // FMVG_MATCHER_H_
