#include <iostream>
#include <utility>
#include <time.h>
#include <cmath>  // for pow
#include <string>
#include <sstream>  // for string split
#include <vector>
#include <dirent.h>  // to find files in a directory
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
        pos_e = exifData.findKey(ekey("Exif.GPSInfo.GPSLatitude"));
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
        pos_e = exifData.findKey(ekey("Exif.GPSInfo.GPSLongitude"));
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
        pos_e = exifData.findKey(ekey("Exif.GPSInfo.GPSAltitude"));
    if (pos_e == exifData.end()) {
        cerr << "ERROR: Exif.GPSInfo.GPSAltitude was not found!" << endl;
        return false;
    }
    Exiv2::Rational zr = pos_e->toRational();
    gps_altitude_ = (double)zr.first / zr.second;

    return true;
}  // Photo::setGPSAltitude


// mat_
cv::Mat Photo::getMat() const {
    return mat_;
}  // Photo::getMat


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
    this->mat_ = cv::imread(file_path);
    if (mat_.data == NULL) return false;

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
        pos_x = xmpData.findKey(xkey("Xmp.Camera.ModelType"));
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
        pos_e = exifData.findKey(ekey("Exif.Photo.FocalPlaneXResolution"));
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

    return true;
}  // PhotoList::setPrincipalPoint


// distortion_coefficients_
cv::Vec6d PhotoList::getDistortionCoefficients() const {
    return distortion_coefficients_;
}  // PhotoList::getDistortionCoefficients

bool PhotoList::setDistortionCoefficients(const Exiv2::ExifData& exifData,
                                          const Exiv2::XmpData& xmpData) {
    // distortion coefficients: dc0-dc5
    // (1) perspective model,
    //   x_d = -fx*((1 + dc0*r^2 + dc1*r^4 + dc2*r^6)*x_h + 2*dc3*x_h*y_h + dc4*(r^2 + 2*x_h^2)) + cx
    //   y_d = -fy*((1 + dc0*r^2 + dc1*r^4 + dc2*r^6)*y_h + dc3*(r^2 + 2*y_h^2)) + 2*dc4*x_h*y_h + cy
    //   where fx & fy are focal lengths, r2 = x_h^2 + y_h^2, and cx & cy are principal points
    //   Here, dc5 is not used
    // (2) fisheye lens model,
    //   (x_d, y_d) = ((dc0 dc1) (dc2 dc3)) (x_h, y_h) + (cx, cy)
    //   Here, dc4 & dc5 are not used
    Exiv2::ExifData::const_iterator pos_e;
    Exiv2::XmpData::const_iterator pos_x;
    std::string xmp_text_str;
    std::string buffer;
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
    }

    return true;
}  // PhotoList::setDistortionCoefficients

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
    check &= setDistortionCoefficients(exifData, xmpData);

    return check;
    
}  // PhotoList::readXmpData()

bool PhotoList::readFromDir(const std::string& dir_path) {
    // check
    bool check;
    bool is_first = true;

    // seek photo file
    DIR *dp;
    dirent* entry;
    dp = opendir(dir_path.c_str());
    if (dp == NULL) exit(1);
    do {
        entry = readdir(dp);
        if (entry->d_type == DT_REG) {
            std::string file_path = entry->d_name;
            std::string suffix = file_path.substr(file_path.size() - 4);
            if (suffix == ".JPG" | suffix == ".TIF") {
                // read photo file
                photo_vector_.push_back(fmvg::Photo(dir_path + "/" + file_path));

                if (is_first) {  // execute only at first
                    // read image & its metadata
                    Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(file_path);
                    assert(image.get() != 0);
                    image->readMetadata();

                    // read EXIF data
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

                    check = readXmpData(exifData, xmpData);
                    is_first = false;
                }
            }
        }
    } while (entry != NULL);
}

}  // namespace fmvg
