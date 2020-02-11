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
    std::vector<fmvg::Photo> input_photos;

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
    std::string dir_path = argv[1];
    const char* path = argv[1];
    DIR *dp;
    dirent* entry;
    dp = opendir(path);
    if (dp == NULL) exit(1);
    do {
        entry = readdir(dp);
        if (entry != NULL) {
            if (entry->d_type == DT_REG) {
                std::string path_str = entry->d_name;
                std::string suffix = path_str.substr(path_str.size() - 4);
                if (suffix == ".JPG" | suffix == ".TIF") {
                    fmvg::Photo photo;
                    photo.readFromFile(dir_path + "/" + path_str);
                    cout << path_str << endl;
                    cout << "  Datetime: " << photo.getDateTime() << endl;
                    cout << "  Pixel dims: " << photo.getPixelDims() << endl;
                    cout << "  GPS: (" << photo.getGPSLatitude() << ", "
                        << photo.getGPSLongitude() << ", "
                        << photo.getGPSAltitude() << ")" << endl;
                    cout << "  Model type: " << photo.getModelType() << endl;
                    cout << "  Pixel focal length: " << photo.getPixelFocalLengths() << endl;
                    cout << "  Principal point: " << photo.getPrincipalPoint() << endl;
                    cout << "  Distortion coefficients: " << photo.getDistortionCoefficients() << endl;
                    input_photos.push_back(photo);
                }
            }
        }
    } while (entry != NULL);

    // matches
    std::unordered_map<int, std::vector<cv::DMatch>> all_matches = fmvg::matcher(input_photos);

    // show all matches
    for (const auto& d : all_matches) {
        int n_photos = input_photos.size();
        int j = d.first % n_photos;
        int i = (d.first - j) / n_photos;
        cout << i << "-" << j << endl;
        for (const auto& x : d.second) {
            cout << "  " << x.queryIdx << " --> " << x.trainIdx << endl;
        }
    }

    return 0;
}
