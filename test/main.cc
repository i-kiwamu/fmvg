#include <iostream>
#include <dirent.h>  // find files in a directory
#include <string>
#include <time.h>
#include <vector>
#include <map>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  // for imshow
#include <exiv2/exiv2.hpp>

#include "photo.h"
#include "matcher.h"
// #include "bundle_adjuster.h"

using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char** argv) {
    if (argc == 1) {
        cerr << "ERROR: The program requires more than one argument of directory to contain images!" << endl;
        return 1;
    }

    // read image files
    fmvg::PhotoList photos;
    std::vector<std::string> args(argv, argv + argc);
    args.erase(args.begin());  // remove program file name

    photos.readFromFiles(args);

    int n_photos = photos.getNumPhotos();
    std::vector<fmvg::Photo> photo_vec = photos.getPhotoVector();
    cout << "Photo list" << endl;
    cout << "  Number of photos: " << n_photos << endl;
    cout << "  Model type: " << photos.getModelType() << endl;
    cout << "  Pixel focal length: " << photos.getPixelFocalLengths() << endl;
    cout << "  Principal point: " << photos.getPrincipalPoint() << endl;
    cout << "  Distortion coefficients: " << photos.getDistortCoeff() << endl;
    cout << endl;
    for (auto p = photos.begin(); p != photos.end(); ++p) {
        cout << "Date&time: " << p->getDateTime() << endl;
        cout << "Pixel dimensions: " << p->getPixelDims() << endl;
        cout << "GPS: " << p->getCameraPositionInit() << endl;
        cout << "Camera matrix: " << p->getCameraMatrixInit() << endl;
        cout << endl;
    }

    // cv::Vec2d pp = photos.getPhotoVector()[n_photos-1].getPrincipalPoint();
    // cv::Point2f ppf;
    // ppf.x = (float)pp[0];
    // ppf.y = (float)pp[1];
    // cout << "Recalculate GPS last: "
    //      << 
    //      << endl;

    // // show
    // cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Corrected", cv::WINDOW_AUTOSIZE);
    // for (auto p : photos) {
    //     cv::imshow("Original", p.getMatOriginal());
    //     cv::imshow("Corrected", p.getMatCorrected());
    //     cv::waitKey(0);
    // }
    // cv::destroyAllWindows();

    // matches
    fmvg::SfM sfm(photos);
    sfm.runSfM();
    // cout << matched_points_vec[0] << endl;

    // bundle adjustment
    // fmvg::bundleAdjuster(photos, matched_points_vec);

    return 0;
}
