#ifndef FMVG_PHOTO_CORRECTION_H_
#define FMVG_PHOTO_CORRECTION_H_

namespace fmvg {

std::vector<cv::Mat> undistortPhotos(PhotoList& inputPhotos);

}  // namespace fmvg

#endif  // FMVG_PHOTO_CORRECTION_H_
