#include <iostream>
// #include <filesystem>
#include <dirent.h>  // find files in a directory
#include <string>
#include <time.h>
// #include <vector>
// #include <unordered_map>
#include <opencv2/opencv.hpp>
#include <exiv2/exiv2.hpp>
#include "photo.h"
#include "matcher.h"

// namespace fs = std::filesystem;
using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "ERROR: The program requires only one argument of directory to contain images!" << endl;
        return 1;
    }

    // read image files
    fmvg::PhotoList input_photos;

    // C++17
    // for(auto& p : fs::directory_iterator(argv[1])) {
    //     fs::path path = p.path();
    //     if (fs::is_regular_file(path)) {
    //         std::string path_str = path.string();
    //         std::string suffix = path_str.substr(path_str.size() - 4);
    //         if (suffix == ".JPG" | suffix == ".TIF") {
    //             input_imgs.push_back(cv::imread(path_str, -1));
    //         }
    //     }
    // }
    // C++11
    input_photos.readFromDir(argv[1]);
    cout << "Photo list" << endl;
    std::size_t n_photos = input_photos.getPhotoVector().size();
    cout << "  Number of photos: " << n_photos << endl;
    cout << "  Model type: " 
         << input_photos.getModelType() << endl;
    cout << "  Pixel focal length: "
         << input_photos.getPixelFocalLengths() << endl;
    cout << "  Principal point: "
         << input_photos.getPrincipalPoint() << endl;
    cout << "  Distortion coefficients: "
         << input_photos.getDistortionCoefficients() << endl;
    for (std::size_t i = 0; i < n_photos; ++i) {
        cout << "Photo #" << i << endl;
        cout << "  Date time: " << endl;
    }

    // matches
    std::vector<fmvg::Photo> photos = input_photos.getPhotoVector();
    std::unordered_map<int, std::vector<cv::DMatch>> all_matches = fmvg::matcher(photos);

    // show all matches
    for (const auto& d : all_matches) {
        int n_photos = photos.size();
        int j = d.first % n_photos;
        int i = (d.first - j) / n_photos;
        cout << i << "-" << j << endl;
        for (const auto& x : d.second) {
            cout << "  " << x.queryIdx << " --> " << x.trainIdx << endl;
        }
    }

    return 0;
}
