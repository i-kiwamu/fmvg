#include <iostream>
#include <dirent.h>  // find files in a directory
#include <string>
#include <time.h>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <exiv2/exiv2.hpp>
#include "photo.h"
#include "matcher.h"

// namespace fs = std::filesystem;
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
    cout << "Photo list" << endl;
    cout << "  Number of photos: " << photos.getPhotoVector().size() << endl;
    cout << "  Model type: " << photos.getModelType() << endl;
    cout << "  Pixel focal length: " << photos.getPixelFocalLengths() << endl;
    cout << "  Principal point: " << photos.getPrincipalPoint() << endl;
    cout << "  Distortion coefficients: " << photos.getDistortionCoefficients() << endl;
    cout << endl;
    for (auto p = photos.begin(); p != photos.end(); ++p) {
        cout << "Date&time: " << p->getDateTime() << endl;
        cout << "Pixel dimensions: " << p->getPixelDims() << endl;
        cout << "GPS: (" << p->getGPSLatitude() << ", "
             << p->getGPSLongitude() << ", "
             << p->getGPSAltitude() << ")" << endl;
        cout << endl;
    }

    // // matches
    // std::unordered_map<int, std::vector<cv::DMatch>> all_matches = fmvg::matcher(photos);

    // // show all matches
    // for (const auto& d : all_matches) {
    //     int j = d.first % n_photos;
    //     int i = (d.first - j) / n_photos;
    //     cout << i << "-" << j << endl;
    //     for (const auto& x : d.second) {
    //         cout << "  " << x.queryIdx << " --> " << x.trainIdx << endl;
    //     }
    // }

    return 0;
}
