#include <vector>
#include <opencv2/opencv.hpp>
#include <exiv2/exiv2.hpp>
#include "photo.h"
#include "matcher.h"
#include "photo_correction.h"

using std::cout;
using std::cerr;
using std::endl;

namespace fmvg {

std::vector<cv::Mat> undistortPhotos(PhotoList& inputPhotos) {
    // initialize undistortion function
    cv::Matx33d intrinsic_matrix = inputPhotos.getIntrinsicMatrix();
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(
        intrinsic_matrix,
        inputPhotos.getDistortCoeff(),
        cv::Mat(),
        intrinsic_matrix,
        inputPhotos.getPhotoVector()[0].getMatOriginal().size(),
        CV_16SC2,
        map1,
        map2
    );

    std::vector<cv::Mat> photo_undistort_vec;
    for (auto p = inputPhotos.begin(); p != inputPhotos.end(); ++p) {
        cv::Mat photo_undistort;
        cv::remap(
            p->getMatOriginal(),
            photo_undistort,
            map1,
            map2,
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar()
        );
        photo_undistort_vec.emplace_back(photo_undistort);
    }

    return photo_undistort_vec;
}

}  // namespace fmvg
