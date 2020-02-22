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

// aliases
using ekey = Exiv2::ExifKey;
using xkey = Exiv2::XmpKey;

// Photo class

// datetime_
time_t Photo::getDateTime() const {
    return datetime_;
}  // Photo::getDateTime

bool Photo::setDateTime(const Exiv2::ExifData& exifData) {
    Exiv2::ExifData::const_iterator pos_e = \
        exifData.findKey(ekey("Exif.Photo.DateTimeOriginal"));
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

    return true;
}  // Photo::setDateTime


// pixel_dimensions_
cv::Size Photo::getPixelDims() const {
    return pixel_dimensions_;
}  // Photo::getPixelDims

bool Photo::setPixelDims(const Exiv2::ExifData& exifData) {
    Exiv2::ExifData::const_iterator pos_e = \
        exifData.findKey(ekey("Exif.Photo.PixelXDimension"));
    
    // width
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.Photo.PixelXDimension was not found!" << endl;
        return false;
    }
    pixel_dimensions_.width = pos_e->toLong();

    // height
    pos_e = exifData.findKey(ekey("Exif.Photo.PixelYDimension"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.Photo.PixelYDimension was not found!" << endl;
        return false;
    }
    pixel_dimensions_.height = pos_e->toLong();

    return true;
}  // Photo::setPixelDims


// gps_latitude_
double Photo::getGPSLatitude() const {
    return gps_latitude_;
}  // Photo::getGPSLatitude

bool Photo::setGPSLatitude(const Exiv2::ExifData& exifData) {
    gps_latitude_ = 0.0;
    Exiv2::ExifData::const_iterator pos_e = \
        exifData.findKey(ekey("Exif.GPSInfo.GPSLatitude"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.GPSInfo.GPSLatitude was not found!" << endl;
        return false;
    }
    for (int i = 0; i < pos_e->count(); ++i) {
        Exiv2::Rational yr = pos_e->toRational(i);
        gps_latitude_ += (double)yr.first / yr.second / std::pow(60, i);
    }

    return true;
}  // Photo::setGPSLatitude


// gps_longitude_
double Photo::getGPSLongitude() const {
    return gps_longitude_;
}  // Photo::getGPSLongitude

bool Photo::setGPSLongitude(const Exiv2::ExifData& exifData) {
    gps_longitude_ = 0.0;
    Exiv2::ExifData::const_iterator pos_e = \
        exifData.findKey(ekey("Exif.GPSInfo.GPSLongitude"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.GPSInfo.GPSLongitude was not found!" << endl;
        return false;
    }
    for (int i = 0; i < pos_e->count(); ++i) {
        Exiv2::Rational xr = pos_e->toRational(i);
        gps_longitude_ += (double)xr.first / xr.second / std::pow(60, i);
    }

    return true;
}  // Photo::setGPSLongitude


// gps_altitude_
double Photo::getGPSAltitude() const {
    return gps_altitude_;
}  // Photo::getGPSAltitude

bool Photo::setGPSAltitude(const Exiv2::ExifData& exifData) {
    gps_altitude_ = 0.0;
    Exiv2::ExifData::const_iterator pos_e = \
        exifData.findKey(ekey("Exif.GPSInfo.GPSAltitude"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.GPSInfo.GPSAltitude was not found!" << endl;
        return false;
    }
    Exiv2::Rational zr = pos_e->toRational();
    gps_altitude_ = (double)zr.first / zr.second;

    return true;
}  // Photo::setGPSAltitude


// mat_
cv::Mat Photo::getMatOriginal() const {
    return mat_original_;
}  // Photo::getMatOriginal


// read EXIF data
bool Photo::readExifData(const Exiv2::ExifData& exifData) {
    // check
    bool check;

    // datetime
    check = setDateTime(exifData);

    // pixel dimensions
    check &= setPixelDims(exifData);

    // GPS xyz
    check &= setGPSLatitude(exifData);
    check &= setGPSLongitude(exifData);
    check &= setGPSAltitude(exifData);

    return check;
}


// read from file
bool Photo::readFromFile(const std::string& file_path) {
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

    // OpenCV mat
    this->mat_original_ = cv::imread(file_path);
    if (mat_original_.data == NULL) return false;
    // mat_corrected_ = cv::Mat(mat_original_);

    // return check
    return check;
}  // Photo::readFromFile


// PhotoList class
// iterator
PhotoList::iterator PhotoList::begin() {
    return photo_vector_.begin();
}
PhotoList::iterator PhotoList::end() {
    return photo_vector_.end();
}


// photo_vector_
std::vector<Photo> PhotoList::getPhotoVector() const {
    return photo_vector_;
}  // PhotoList::getPhotoVector


// model_type_
std::string PhotoList::getModelType() const {
    return model_type_;
}  // PhotoList::getModelType

bool PhotoList::setModelType(const Exiv2::XmpData& xmpData) {
    Exiv2::XmpData::const_iterator pos_x = \
        xmpData.findKey(xkey("Xmp.Camera.ModelType"));
    if (pos_x == xmpData.end()) {
        cerr << "ERROR: Xmp.Camera.ModelType was not found!" << endl;
        return false;
    }
    model_type_ = pos_x->toString();

    return true;
}  // PhotoList::setModelType


// pixel_per_mm_
cv::Vec2d PhotoList::getPixelPerMM() const {
    return pixel_per_mm_;
}  // PhotoList::getPixelPerMM

bool PhotoList::setPixelPerMM(const Exiv2::ExifData& exifData) {
    // x-axis
    Exiv2::ExifData::const_iterator pos_e = \
        exifData.findKey(ekey("Exif.Photo.FocalPlaneXResolution"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.Photo.FocalPlaneXResolution was not found!" << endl;
        return false;
    }
    Exiv2::Rational resxr = pos_e->toRational();
    pixel_per_mm_[0] = (double)resxr.first / resxr.second;

    // y-axis
    pos_e = exifData.findKey(ekey("Exif.Photo.FocalPlaneYResolution"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.Photo.FocalPlaneYResolution was not found!" << endl;
        return false;
    }
    Exiv2::Rational resyr = pos_e->toRational();
    pixel_per_mm_[1] = (double)resyr.first / resyr.second;

    return true;
}  // PhotoList::setPixelPerMM


// pixel_focal_lengths_
cv::Vec2d PhotoList::getPixelFocalLengths() const {
    return pixel_focal_lengths_;
}  // PhotoList::getPixelFocalLengths

bool PhotoList::setPixelFocalLengths(const Exiv2::ExifData& exifData,
                                     const Exiv2::XmpData& xmpData) {
    Exiv2::ExifData::const_iterator pos_e;
    Exiv2::XmpData::const_iterator pos_x;

    if (model_type_ == "perspective") {
        pos_x = xmpData.findKey(xkey("Xmp.Camera.PerspectiveFocalLength"));
        if (pos_x == xmpData.end()) {
            cerr << "ERROR: Xmp.Camera.PerspectiveFocalLength was not found!" << endl;
            return false;
        }
        pixel_focal_lengths_ = cv::Vec2d(
            (double)pos_x->toFloat() * pixel_per_mm_[0],
            (double)pos_x->toFloat() * pixel_per_mm_[1]
        );
    } else if (model_type_ == "fisheye") {
        pos_e = exifData.findKey(ekey("Exif.Photo.FocalLength"));
        if (pos_e == exifData.end()) {
            cerr << "ERROR: Exif.Photo.FocalLength was not found!" << endl;
            return false;
        }
        Exiv2::Rational flr = pos_e->toRational();
        pixel_focal_lengths_ = cv::Vec2d(
            (double)flr.first / flr.second * pixel_per_mm_[0],
            (double)flr.first / flr.second * pixel_per_mm_[1]
        );
    } else {
        cerr << "ERROR: Invalid model type (neither perspective nor fisheye)" << endl;
        return false;
    }

    return true;
}  // PhotoList::setPixelFocalLengths


// principal_point_
cv::Vec2d PhotoList::getPrincipalPoint() const {
    return principal_point_;
}  // PhotoList::getPrincipalPoint

bool PhotoList::setPrincipalPoint(const Exiv2::XmpData& xmpData) {
    Exiv2::XmpData::const_iterator pos_x = \
        xmpData.findKey(xkey("Xmp.Camera.PrincipalPoint"));
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

    return true;
}  // PhotoList::setPrincipalPoint


// distortion_coefficients_
cv::Mat PhotoList::getDistortCoeff() const {
    return distortion_coefficients_;
}  // PhotoList::getDistortionCoefficients

bool PhotoList::setDistortCoeffP(const Exiv2::XmpData& xmpData) {
    Exiv2::XmpData::const_iterator pos_x;
    std::string xmp_text_str;
    std::string buffer;
    distortion_coefficients_ = cv::Mat(1, 5, CV_64FC1);
    pos_x = xmpData.findKey(xkey("Xmp.Camera.PerspectiveDistortion"));
    if (pos_x == xmpData.end()) {
        cerr << "ERROR: Xmp.Camera.PerspectiveDistortion was not found!" << endl;
        return false;
    }
    xmp_text_str = pos_x->toString();
    std::stringstream ss(xmp_text_str);
    for (std::size_t i = 0; i < 5; ++i) {
        std::getline(ss, buffer, ',');
        distortion_coefficients_.at<float>(i) = std::stod(buffer);
    }
    return true;
}  // PhotoList::setDistortCoeffP

bool PhotoList::setDistortCoeffF(const Exiv2::XmpData& xmpData) {
    Exiv2::XmpData::const_iterator pos_x;
    std::string xmp_text_str;
    std::string buffer;
    distortion_coefficients_ = cv::Mat(1, 4, CV_64FC1);
    pos_x = xmpData.findKey(xkey("Xmp.Camera.FisheyeAffineMatrix"));
    if (pos_x == xmpData.end()) {
        cerr << "ERROR: Xmp.Camera.FisheyeAffineMatrix was not found!" << endl;
        return false;
    }
    xmp_text_str = pos_x->toString();
    std::stringstream ss(xmp_text_str);
    for (std::size_t i = 0; i < 4; ++i) {
        std::getline(ss, buffer, ',');
        distortion_coefficients_.at<float>(i) = std::stod(buffer);
    }
    return true;
}  // PhotoList::setDistortCoeffF

bool PhotoList::setDistortCoeff(const Exiv2::XmpData& xmpData) {
    // distortion coefficients: dc0-dc5
    // (1) perspective model,
    //   x_d = -fx*((1 + dc0*r^2 + dc1*r^4 + dc2*r^6)*x_h + 2*dc3*x_h*y_h + dc4*(r^2 + 2*x_h^2)) + cx
    //   y_d = -fy*((1 + dc0*r^2 + dc1*r^4 + dc2*r^6)*y_h + dc3*(r^2 + 2*y_h^2)) + 2*dc4*x_h*y_h + cy
    //   where fx & fy are focal lengths, r2 = x_h^2 + y_h^2, and cx & cy are principal points
    // (2) fisheye lens model,
    //   (x_d, y_d) = ((dc0 dc1) (dc2 dc3)) (x_h, y_h) + (cx, cy)
    if (model_type_ == "perspective")
        setDistortCoeffP(xmpData);
    else if (model_type_ == "fisheye")
        setDistortCoeffF(xmpData);
    else {
        cerr << "ERROR: Invalid model type (neither perspective nor fisheye)" << endl;
        return false;
    }
}  // PhotoList::setDistortCoeff


cv::Matx33d PhotoList::getIntrinsicMatrix() {
    cv::Matx33d intrinsic_matrix = cv::Matx33f::zeros();
    intrinsic_matrix(0, 0) = pixel_focal_lengths_[0];
    intrinsic_matrix(1, 1) = pixel_focal_lengths_[1];
    intrinsic_matrix(2, 2) = 1.0;
    intrinsic_matrix(0, 2) = principal_point_[0];
    intrinsic_matrix(1, 2) = principal_point_[1];
    return intrinsic_matrix;
}  // PhotoList::getIntrinsicMatrix


bool PhotoList::readXmpData(const Exiv2::ExifData& exifData,
                            const Exiv2::XmpData& xmpData) {
    // check
    bool check;

    // model type
    check = setModelType(xmpData);

    // pixel resolutions
    check &= setPixelPerMM(exifData);

    // focal length in pixel
    check &= setPixelFocalLengths(exifData, xmpData);

    // principal point
    check &= setPrincipalPoint(xmpData);

    // distortion coefficients
    check &= setDistortCoeff(xmpData);

    return check;
    
}  // PhotoList::readXmpData()

bool PhotoList::readFromFiles(const std::vector<std::string> files) {
    // check
    bool check = true;
    bool is_first = true;

    // seek photo file
    Photo photo;
    std::string suffix;
    for (auto file_path : files) {
        suffix = file_path.substr(file_path.size() - 4);
        if (suffix == ".JPG" | suffix == ".TIF") {
            photo.readFromFile(file_path);
            photo_vector_.emplace_back(photo);  // emplace_back?

            if (is_first) {  // execute only at first
                // read image & its metadata
                Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(file_path);
                assert(image.get() != 0);
                image->readMetadata();

                Exiv2::ExifData& exifData = image->exifData();
                if (exifData.empty()) {
                    std::string error(file_path);
                    error += ": No EXIF data found in the file";
                    cerr << error << endl;
                    return false;
                }

                Exiv2::XmpData& xmpData = image->xmpData();
                if (xmpData.empty()) {
                    std::string error(file_path);
                    error += ": No XMP data found in the file";
                    cerr << error << endl;
                    return false;
                }

                check &= readXmpData(exifData, xmpData);
                is_first = false;
            }
        }
    }
    return check;
}  // PhotoList::readFromFiles

}  // namespace fmvg
