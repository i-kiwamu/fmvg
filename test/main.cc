#include <iostream>
// #include <filesystem>
#include <string>
#include <time.h>
// #include <vector>
// #include <unordered_map>
#include <opencv2/opencv.hpp>
#include <preparation.h>
#include <matcher.h>

namespace sqo = sequoiaortho;
// namespace fs = std::filesystem;
using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "ERROR: The program requires only one argument of directory to contain images!" << endl;
        return 1;
    }

    // read an image file
    sqo::Photo img;
    img.readFromFile(argv[1]);

    cout << "Datetime: " << img.getDateTime() << endl;
    cout << "Pixel dimension X: " << img.getPixelDimX() << endl;
    cout << "Pixel dimension Y: " << img.getPixelDimY() << endl;
    cout << "Focal length init: " << img.getFocalLengthInit() << endl;
    cout << "GPS Longitude: " << img.getGPSLongitude() << endl;
    cout << "GPS Latitude: " << img.getGPSLatitude() << endl;
    cout << "GPS Altitude: " << img.getGPSAltitude() << endl;

    // // read image files
    // std::vector<cv::Mat> input_imgs;
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

    // // matches
    // std::unordered_map<int, std::vector<cv::DMatch>> all_matches = sqo::matcher(input_imgs);

    // // show all matches
    // for (const auto& d : all_matches) {
    //     int n_imgs = input_imgs.size();
    //     int j = d.first % n_imgs;
    //     int i = (d.first - j) / n_imgs;
    //     cout << i << "-" << j << endl;
    //     for (const auto& x : d.second) {
    //         cout << "  " << x.queryIdx << " --> " << x.trainIdx << endl;
    //     }
    // }

    return 0;
}
