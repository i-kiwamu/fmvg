#include <iostream>
#include <utility>
#include <time.h>
#include <cmath>  // for pow
#include <vector>
#include <opencv2/opencv.hpp>
#include <exiv2/exiv2.hpp>
#include "preparation.h"

using std::cout;
using std::cerr;
using std::endl;

namespace sequoiaortho {

time_t Photo::getDateTime() {
    return datetime_;
}  // Photo::getDateTime()

int Photo::getPixelDimX() {
    return pixel_dimension_x_;
}  // Photo::getPixelDimX()

int Photo::getPixelDimY() {
    return pixel_dimension_y_;
}  // Photo::getPixelDimY()

cv::Size Photo::getPixelDims() {
    return cv::Size(pixel_dimension_x_, pixel_dimension_y_);
}  // Photo::getPixelDims()

double Photo::getFocalLengthInit() {
    return focal_length_init_;
}  // Photo::getFocalLengthInit()

double Photo::getGPSLatitude() {
    return gps_latitude_;
}  // Photo::getGPSLatitude()

double Photo::getGPSLongitude() {
    return gps_longitude_;
}  // Photo::getGPSLongitude()

double Photo::getGPSAltitude() {
    return gps_altitude_;
}  // Photo::getGPSAltitude()

cv::Mat Photo::getMat() {
    return mat_;
}  // Photo::getMat()

bool Photo::readFromFile(std::string file_path) {
    // aliases
    using ekey = Exiv2::ExifKey;

    // read EXIF data
    Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(file_path);
    assert(image.get() != 0);
    image->readMetadata();
    Exiv2::ExifData &exifData = image->exifData();
    if (exifData.empty()) {
        std::string error(file_path);
        error += ": No EXIF data found in the file";
        cerr << error << endl;
        return false;
    }

    // datetime
    Exiv2::ExifData::iterator pos = \
        exifData.findKey(ekey("Exif.Photo.DateTimeOriginal"));
    if (pos == exifData.end()) {
        cerr << "ERROR: Exif.Photo.DateTimeOriginal was not found!" << endl;
        return false;
    }
    std::string dts = pos->toString();
    struct tm datetime_struct = {
        std::stoi(dts.substr(17, 2)),
        std::stoi(dts.substr(14, 2)),
        std::stoi(dts.substr(11, 2)),
        std::stoi(dts.substr(8, 2)),
        std::stoi(dts.substr(5, 2)),
        std::stoi(dts.substr(0, 4)),
    };
    datetime_ = std::mktime(&datetime_struct);

    // pixel dimensions
    pos = exifData.findKey(ekey("Exif.Photo.PixelXDimension"));
    if (pos == exifData.end()) {
        cerr << "ERROR: Exif.Photo.PixelXDimension was not found!" << endl;
        return false;
    }
    pixel_dimension_x_ = pos->toLong();

    pos = exifData.findKey(ekey("Exif.Photo.PixelYDimension"));
    if (pos == exifData.end()) {
        cerr << "ERROR: Exif.Photo.PixelYDimension was not found!" << endl;
        return false;
    }
    pixel_dimension_y_ = pos->toLong();

    // focal length
    pos = exifData.findKey(ekey("Exif.Photo.FocalLength"));
    if (pos == exifData.end()) {
        cerr << "ERROR: Exif.Photo.FocalLength was not found!" << endl;
        return false;
    }
    Exiv2::Rational flr = pos->toRational();
    focal_length_init_ = (double)flr.first / flr.second;

    // GPS xyz
    gps_longitude_ = 0.0;
    gps_latitude_ = 0.0;
    pos = exifData.findKey(ekey("Exif.GPSInfo.GPSLongitude"));
    if (pos == exifData.end()) {
        cerr << "ERROR: Exif.GPSInfo.GPSLongitude was not found!" << endl;
        return false;
    }
    for (int i = 0; i < pos->count(); ++i) {
        Exiv2::Rational xr = pos->toRational(i);
        gps_longitude_ += (double)xr.first / xr.second / std::pow(60, i);
    }

    pos = exifData.findKey(ekey("Exif.GPSInfo.GPSLatitude"));
    if (pos == exifData.end()) {
        cerr << "ERROR: Exif.GPSInfo.GPSLatitude was not found!" << endl;
        return false;
    }
    for (int i = 0; i < pos->count(); ++i) {
        Exiv2::Rational yr = pos->toRational(i);
        gps_latitude_ += (double)yr.first / yr.second / std::pow(60, i);
    }

    pos = exifData.findKey(ekey("Exif.GPSInfo.GPSAltitude"));
    if (pos == exifData.end()) {
        cerr << "ERROR: Exif.GPSInfo.GPSAltitude was not found!" << endl;
        return false;
    }
    Exiv2::Rational zr = pos->toRational();
    gps_altitude_ = (double)zr.first / zr.second;

    // OpenCV mat
    this->mat_ = cv::imread(file_path);
    if (mat_.data == NULL) return false;

    // return
    return true;
}  // Photo::readFromFile

}  // namespace sequoiaortho
