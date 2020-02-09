#ifndef SEQUOIAORTHO_MATCHER_H_
#define SEQUOIAORTHO_MATCHER_H_

namespace sequoiaortho {

std::vector<cv::DMatch> matcher_ker(cv::Mat img1, cv::Mat img2);
std::unordered_map<int, std::vector<cv::DMatch>> matcher(const std::vector<cv::Mat> &imgs);

}  // namespace sequoiaortho

#endif  // SEQUOIAORTHO_MATCHER_H_
