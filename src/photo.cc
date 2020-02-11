#include <iostream>
#include <utility>
#include <time.h>
#include <cmath>  // for pow
#include <string>
#include <sstream>  // for string split
#include <vector>
#include <opencv2/opencv.hpp>
#include <exiv2/exiv2.hpp>
#include "photo.h"

using std::cout;
using std::cerr;
using std::endl;

namespace fmvg {

time_t Photo::getDateTime() const {
    return datetime_;
}  // Photo::getDateTime()

cv::Size Photo::getPixelDims() const {
    return pixel_dimensions_;
}  // Photo::getPixelDims()

double Photo::getGPSLatitude() const {
    return gps_latitude_;
}  // Photo::getGPSLatitude()

double Photo::getGPSLongitude() const {
    return gps_longitude_;
}  // Photo::getGPSLongitude()

double Photo::getGPSAltitude() const {
    return gps_altitude_;
}  // Photo::getGPSAltitude()

std::string Photo::getModelType() const {
    return model_type_;
}  // Photo::getModelType()

cv::Vec2d Photo::getPixelFocalLengths() const {
    return pixel_focal_lengths_;
}  // Photo::getPixelFocalLengths()

cv::Vec2d Photo::getPrincipalPoint() const {
    return principal_point_;
}  // Photo::getPrincipalPoint()

cv::Vec6d Photo::getDistortionCoefficients() const {
    return distortion_coefficients_;
}  // Photo::getDistortionCoefficients()

cv::Mat Photo::getMat() const {
    return mat_;
}  // Photo::getMat()

bool Photo::readExifData(Exiv2::ExifData exifData) {
    Exiv2::ExifData::iterator pos_e;
    using ekey = Exiv2::ExifKey;

    // datetime
    pos_e = exifData.findKey(ekey("Exif.Photo.DateTimeOriginal"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.Photo.DateTimeOriginal was not found!" << endl;
        return false;
    }
    std::string dts = pos_e->toString();
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
    pos_e = exifData.findKey(ekey("Exif.Photo.PixelXDimension"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.Photo.PixelXDimension was not found!" << endl;
        return false;
    }
    pixel_dimensions_.width = pos_e->toLong();

    pos_e = exifData.findKey(ekey("Exif.Photo.PixelYDimension"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.Photo.PixelYDimension was not found!" << endl;
        return false;
    }
    pixel_dimensions_.height = pos_e->toLong();

    // GPS xyz
    gps_longitude_ = 0.0;
    gps_latitude_ = 0.0;
    pos_e = exifData.findKey(ekey("Exif.GPSInfo.GPSLongitude"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.GPSInfo.GPSLongitude was not found!" << endl;
        return false;
    }
    for (int i = 0; i < pos_e->count(); ++i) {
        Exiv2::Rational xr = pos_e->toRational(i);
        gps_longitude_ += (double)xr.first / xr.second / std::pow(60, i);
    }

    pos_e = exifData.findKey(ekey("Exif.GPSInfo.GPSLatitude"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.GPSInfo.GPSLatitude was not found!" << endl;
        return false;
    }
    for (int i = 0; i < pos_e->count(); ++i) {
        Exiv2::Rational yr = pos_e->toRational(i);
        gps_latitude_ += (double)yr.first / yr.second / std::pow(60, i);
    }

    pos_e = exifData.findKey(ekey("Exif.GPSInfo.GPSAltitude"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.GPSInfo.GPSAltitude was not found!" << endl;
        return false;
    }
    Exiv2::Rational zr = pos_e->toRational();
    gps_altitude_ = (double)zr.first / zr.second;

    // result
    return true;
}  // Photo::readExifData

bool Photo::readXmpData(Exiv2::ExifData exifData, Exiv2::XmpData xmpData) {
    Exiv2::ExifData::iterator pos_e;
    Exiv2::XmpData::iterator pos_x;
    using ekey = Exiv2::ExifKey;
    using xkey = Exiv2::XmpKey;

    // model type
    pos_x = xmpData.findKey(xkey("Xmp.Camera.ModelType"));
    if (pos_x == xmpData.end()) {
        cerr << "ERROR: Xmp.Camera.ModelType was not found!" << endl;
        return false;
    }
    model_type_ = pos_x->toString();

    // pixel resolutions
    pos_e = exifData.findKey(ekey("Exif.Photo.FocalPlaneXResolution"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.Photo.FocalPlaneXResolution was not found!" << endl;
        return false;
    }
    Exiv2::Rational resxr = pos_e->toRational();
    double pixel_per_mm_x = (double)resxr.first / resxr.second;

    pos_e = exifData.findKey(ekey("Exif.Photo.FocalPlaneYResolution"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.Photo.FocalPlaneYResolution was not found!" << endl;
        return false;
    }
    Exiv2::Rational resyr = pos_e->toRational();
    double pixel_per_mm_y = (double)resyr.first / resyr.second;

    // focal length in pixel
    if (model_type_ == "perspective") {
        pos_x = xmpData.findKey(xkey("Xmp.Camera.PerspectiveFocalLength"));
        if (pos_x == xmpData.end()) {
            cerr << "ERROR: Xmp.Camera.PerspectiveFocalLength was not found!" << endl;
            return false;
        }
        pixel_focal_lengths_ = cv::Vec2d(
            (double)pos_x->toFloat() * pixel_per_mm_x,
            (double)pos_x->toFloat() * pixel_per_mm_y
        );
    } else if (model_type_ == "fisheye") {
        pos_e = exifData.findKey(ekey("Exif.Photo.FocalLength"));
        if (pos_e == exifData.end()) {
            cerr << "ERROR: Exif.Photo.FocalLength was not found!" << endl;
            return false;
        }
        Exiv2::Rational flr = pos_e->toRational();
        pixel_focal_lengths_ = cv::Vec2d(
            (double)flr.first / flr.second * pixel_per_mm_x,
            (double)flr.first / flr.second * pixel_per_mm_y
        );
    } else {
        cerr << "ERROR: Invalid model type (neither perspective nor fisheye)" << endl;
        return false;
    }

    // principal point
    pos_x = xmpData.findKey(xkey("Xmp.Camera.PrincipalPoint"));
    if (pos_x == xmpData.end()) {
        cerr << "ERROR: Xmp.Camera.PrincipalPoint was not found!" << endl;
        return false;
    }
    std::string xmp_text_str = pos_x->toString();
    std::stringstream ss(xmp_text_str);
    std::string buffer;
    for (std::size_t i = 0; i < 2; ++i) {
        std::getline(ss, buffer, ',');
        principal_point_[i] = std::stod(buffer);
    }

    // distortion coefficients (dc0-dc5)
    //   for perspective model,
    //     x_d = -fx*((1 + dc0*r^2 + dc1*r^4 + dc2*r^6)*x_h + dc4*(r^2 + 2*x_h^2)) + cx
    //     y_d = -fy*((1 + dc0*r^2 + dc1*r^4 + dc2*r^6)*y_h + dc3*(r^2 + 2*y_h^2)) + cy
    //     where fx & fy are focal lengths, r2 = x_h^2 + y_h^2, and cx & cy are principal points
    //     Here, dc5 is not used
    //   for fisheye lens model,
    //     (x_d, y_d) = ((dc0 dc1) (dc2 dc3)) (x_h, y_h) + (cx, cy)
    //     Here, dc4 & dc5 are not used
    distortion_coefficients_ = cv::Vec6d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if (model_type_ == "perspective") {
        pos_x = xmpData.findKey(xkey("Xmp.Camera.PerspectiveDistortion"));
        if (pos_x == xmpData.end()) {
            cerr << "ERROR: Xmp.Camera.PerspectiveDistortion was not found!" << endl;
            return false;
        }
        xmp_text_str = pos_x->toString();
        std::stringstream ss(xmp_text_str);
        for (std::size_t i = 0; i < 5; ++i) {
            std::getline(ss, buffer, ',');
            distortion_coefficients_[i] = std::stod(buffer);
        }
    } else if (model_type_ == "fisheye") {
        pos_x = xmpData.findKey(xkey("Xmp.Camera.FisheyeAffineMatrix"));
        if (pos_x == xmpData.end()) {
            cerr << "ERROR: Xmp.Camera.FisheyeAffineMatrix was not found!" << endl;
            return false;
        }
        xmp_text_str = pos_x->toString();
        std::stringstream ss(xmp_text_str);
        for (std::size_t i = 0; i < 4; ++i) {
            std::getline(ss, buffer, ',');
            distortion_coefficients_[i] = std::stod(buffer);
        }
    } else {
        cerr << "ERROR: Invalid model type (neither perspective nor fisheye)" << endl;
        return false;
    }

    // result
    return true;
}  // Photo::readXmpData

bool Photo::readFromFile(std::string file_path) {
    // check
    bool check;

    // read image & its metadata
    Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(file_path);
    assert(image.get() != 0);
    image->readMetadata();

    // read EXIF data
    Exiv2::ExifData &exifData = image->exifData();
    if (exifData.empty()) {
        std::string error(file_path);
        error += ": No EXIF data found in the file";
        cerr << error << endl;
        return false;
    }
    check = readExifData(exifData);

    // read XMP data
    Exiv2::XmpData &xmpData = image->xmpData();
    if (xmpData.empty()) {
        std::string error(file_path);
        error += ": No XMP data found in the file";
        cerr << error << endl;
        return false;
    }
    check = readXmpData(exifData, xmpData);

    // OpenCV mat
    this->mat_ = cv::imread(file_path);
    if (mat_.data == NULL) return false;

    // return check
    return check;
}  // Photo::readFromFile

}  // namespace fmvg
