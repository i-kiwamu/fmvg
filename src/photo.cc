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


// compare all elements of cv array
bool isEqualAllElems(cv::InputArray a1, cv::InputArray a2) {
    cv::Mat m1 = a1.getMat(),
            m2 = a2.getMat();
    if (m1.size() != m2.size() || m1.channels() != m2.channels() || m1.type() != m1.type())
        return false;
    else {
        if (m1.depth() == CV_64F) {
            double epsl = 1.0e-06;
            return cv::countNonZero(cv::abs(m1 - m2) >= epsl) == 0;
        } else {
            return cv::countNonZero(m1 != m2) == 0;
        }
    }
}

/////////////////////////////////////////////////////////////////////
// Photo class

// datetime_
time_t Photo::getDateTime() const {
    return datetime_;
}  // Photo::getDateTime

void Photo::setDateTime(const Exiv2::ExifData& exifData) {
    Exiv2::ExifData::const_iterator pos_e = \
        exifData.findKey(ekey("Exif.Photo.DateTimeOriginal"));
    if (pos_e == exifData.end())
        throw std::invalid_argument("Exif.Photo.DateTimeOriginal was not found!");

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
}  // Photo::setDateTime


// model_type_
std::string Photo::getModelType() const {
    return model_type_;
}  // Photo::getModelType

void Photo::setModelType(const Exiv2::XmpData& xmpData) {
    Exiv2::XmpData::const_iterator pos_x = \
        xmpData.findKey(xkey("Xmp.Camera.ModelType"));
    if (pos_x == xmpData.end())
        throw std::invalid_argument("Xmp.Camera.ModelType was not found!");

    model_type_ = pos_x->toString();
}  // Photo::setModelType


// pixel_dimensions_
cv::Size Photo::getPixelDims() const {
    return pixel_dimensions_;
}  // Photo::getPixelDims

void Photo::setPixelDims(const Exiv2::ExifData& exifData) {
    Exiv2::ExifData::const_iterator pos_e = \
        exifData.findKey(ekey("Exif.Photo.PixelXDimension"));
    
    // width
    if (pos_e == exifData.end())
        throw std::invalid_argument("Exif.Photo.PixelXDimension was not found!");
    pixel_dimensions_.width = pos_e->toLong();

    // height
    pos_e = exifData.findKey(ekey("Exif.Photo.PixelYDimension"));
    if (pos_e == exifData.end())
        throw std::invalid_argument("Exif.Photo.PixelYDimension was not found!");
    pixel_dimensions_.height = pos_e->toLong();
}  // Photo::setPixelDims

// homogeneous scale factor f0
double Photo::getF0() const {
    cv::Size d = getPixelDims();
    return (d.width + d.height) / 2.0;
}


// gps_latitude_
double Photo::getGPSLatitude() const {
    return gps_latitude_;
}  // Photo::getGPSLatitude

void Photo::setGPSLatitude(const Exiv2::ExifData& exifData) {
    gps_latitude_ = 0.0;
    Exiv2::ExifData::const_iterator pos_e = \
        exifData.findKey(ekey("Exif.GPSInfo.GPSLatitude"));
    if (pos_e == exifData.end())
        throw std::invalid_argument("Exif.GPSInfo.GPSLatitude was not found!");

    for (int i = 0; i < pos_e->count(); ++i) {
        Exiv2::Rational yr = pos_e->toRational(i);
        gps_latitude_ += (double)yr.first / yr.second / std::pow(60, i);
    }
}  // Photo::setGPSLatitude


// gps_longitude_
double Photo::getGPSLongitude() const {
    return gps_longitude_;
}  // Photo::getGPSLongitude

void Photo::setGPSLongitude(const Exiv2::ExifData& exifData) {
    gps_longitude_ = 0.0;
    Exiv2::ExifData::const_iterator pos_e = \
        exifData.findKey(ekey("Exif.GPSInfo.GPSLongitude"));
    if (pos_e == exifData.end())
        throw std::invalid_argument("Exif.GPSInfo.GPSLongitude was not found!");

    for (int i = 0; i < pos_e->count(); ++i) {
        Exiv2::Rational xr = pos_e->toRational(i);
        gps_longitude_ += (double)xr.first / xr.second / std::pow(60, i);
    }
}  // Photo::setGPSLongitude


// gps_altitude_
double Photo::getGPSAltitude() const {
    return gps_altitude_;
}  // Photo::getGPSAltitude

void Photo::setGPSAltitude(const Exiv2::ExifData& exifData) {
    gps_altitude_ = 0.0;
    Exiv2::ExifData::const_iterator pos_e = \
        exifData.findKey(ekey("Exif.GPSInfo.GPSAltitude"));
    if (pos_e == exifData.end())
        throw std::invalid_argument("Exif.GPSInfo.GPSAltitude was not found!");

    Exiv2::Rational zr = pos_e->toRational();
    gps_altitude_ = (double)zr.first / zr.second;
}  // Photo::setGPSAltitude


// get initial camera position
cv::Vec3d Photo::getCameraPositionInit() const {
    cv::Vec3d pos(getGPSLatitude(), getGPSLongitude(), getGPSAltitude());
    return pos;
}


// pixel_per_mm_
cv::Vec2d Photo::getPixelPerMM() const {
    return pixel_per_mm_;
}  // Photo::getPixelPerMM

void Photo::setPixelPerMM(const Exiv2::ExifData& exifData) {
    // x-axis
    Exiv2::ExifData::const_iterator pos_e = \
        exifData.findKey(ekey("Exif.Photo.FocalPlaneXResolution"));
    if (pos_e == exifData.end())
        throw std::invalid_argument("Exif.Photo.FocalPlaneXResolution was not found!");

    Exiv2::Rational resxr = pos_e->toRational();
    pixel_per_mm_[0] = (double)resxr.first / resxr.second;

    // y-axis
    pos_e = exifData.findKey(ekey("Exif.Photo.FocalPlaneYResolution"));
    if (pos_e == exifData.end())
        throw std::invalid_argument("Exif.Photo.FocalPlaneYResolution was not found!");

    Exiv2::Rational resyr = pos_e->toRational();
    pixel_per_mm_[1] = (double)resyr.first / resyr.second;
}  // Photo::setPixelPerMM


// pixel_focal_lengths_
cv::Vec2d Photo::getPixelFocalLengths() const {
    return pixel_focal_lengths_;
}  // Photo::getPixelFocalLengths

void Photo::setPixelFocalLengths(
    const Exiv2::ExifData& exifData,
    const Exiv2::XmpData& xmpData
) {
    Exiv2::ExifData::const_iterator pos_e;
    Exiv2::XmpData::const_iterator pos_x;

    if (model_type_ == "perspective") {
        pos_x = xmpData.findKey(xkey("Xmp.Camera.PerspectiveFocalLength"));
        if (pos_x == xmpData.end())
            throw std::invalid_argument("Xmp.Camera.PerspectiveFocalLength was not found!");

        pixel_focal_lengths_ = cv::Vec2d(
            (double)pos_x->toFloat() * pixel_per_mm_[0],
            (double)pos_x->toFloat() * pixel_per_mm_[1]
        );
    } else if (model_type_ == "fisheye") {
        pos_e = exifData.findKey(ekey("Exif.Photo.FocalLength"));
        if (pos_e == exifData.end())
            throw std::invalid_argument("Exif.Photo.FocalLength was not found!");

        Exiv2::Rational flr = pos_e->toRational();
        pixel_focal_lengths_ = cv::Vec2d(
            (double)flr.first / flr.second * pixel_per_mm_[0],
            (double)flr.first / flr.second * pixel_per_mm_[1]
        );
    } else
        throw std::invalid_argument("Invalid model type (neither perspective nor fisheye)");
}  // Photo::setPixelFocalLengths


// principal_point_
cv::Vec2d Photo::getPrincipalPoint() const {
    return principal_point_;
}  // Photo::getPrincipalPoint

void Photo::setPrincipalPoint(const Exiv2::XmpData& xmpData) {
    Exiv2::XmpData::const_iterator pos_x = \
        xmpData.findKey(xkey("Xmp.Camera.PrincipalPoint"));
    if (pos_x == xmpData.end())
        throw std::invalid_argument("Xmp.Camera.PrincipalPoint was not found!");

    std::string xmp_text_str = pos_x->toString();
    std::stringstream ss(xmp_text_str);
    std::string buffer;
    for (std::size_t i = 0; i < 2; ++i) {
        std::getline(ss, buffer, ',');
        principal_point_[i] = std::stod(buffer) * pixel_per_mm_[i];
    }
}  // Photo::setPrincipalPoint


// distortion_coefficients_
cv::Mat Photo::getDistortCoeff() const {
    return distortion_coefficients_;
}  // PhotoList::getDistortionCoefficients

void Photo::setDistortCoeffP(const Exiv2::XmpData& xmpData) {
    Exiv2::XmpData::const_iterator pos_x;
    std::string xmp_text_str;
    std::string buffer;
    distortion_coefficients_ = cv::Mat(1, 5, CV_64F);
    pos_x = xmpData.findKey(xkey("Xmp.Camera.PerspectiveDistortion"));
    if (pos_x == xmpData.end())
        throw std::invalid_argument("Xmp.Camera.PerspectiveDistortion was not found!");

    xmp_text_str = pos_x->toString();
    std::stringstream ss(xmp_text_str);
    std::vector<int> dcp_index = {0, 1, 4, 2, 3};
    for (std::size_t i = 0; i < 5; ++i) {
        std::getline(ss, buffer, ',');
        distortion_coefficients_.at<double>(dcp_index[i]) = std::stod(buffer);
    }
}  // Photo::setDistortCoeffP

void Photo::setDistortCoeffF(const Exiv2::XmpData& xmpData) {
    Exiv2::XmpData::const_iterator pos_x;
    std::string xmp_text_str;
    std::string buffer;
    distortion_coefficients_ = cv::Mat(1, 8, CV_64F);
    pos_x = xmpData.findKey(xkey("Xmp.Camera.FisheyePolynomial"));
    if (pos_x == xmpData.end())
        throw std::invalid_argument("Xmp.Camera.FisheyePolynomial was not found!");

    xmp_text_str = pos_x->toString();
    std::stringstream ss(xmp_text_str);
    for (std::size_t i = 0; i < 4; ++i) {
        std::getline(ss, buffer, ',');
        distortion_coefficients_.at<double>(i) = std::stod(buffer);
    }

    pos_x = xmpData.findKey(xkey("Xmp.Camera.FisheyeAffineMatrix"));
    if (pos_x == xmpData.end())
        throw std::invalid_argument("Xmp.Camera.FisheyeAffineMatrix was not found!");

    xmp_text_str = pos_x->toString();
    std::stringstream ss1(xmp_text_str);
    for (std::size_t i = 4; i < 8; ++i) {
        std::getline(ss1, buffer, ',');
        distortion_coefficients_.at<double>(i) = std::stod(buffer);
    }
}  // Photo::setDistortCoeffF

void Photo::setDistortCoeff(const Exiv2::XmpData& xmpData) {
    // distortion coefficients: dc0-dc5
    // (1) perspective model,
    //   x_d = -fx*((1 + dc0*r^2 + dc1*r^4 + dc2*r^6)*x_h + 2*dc3*x_h*y_h + dc4*(r^2 + 2*x_h^2)) + cx
    //   y_d = -fy*((1 + dc0*r^2 + dc1*r^4 + dc2*r^6)*y_h + dc3*(r^2 + 2*y_h^2)) + 2*dc4*x_h*y_h + cy
    //   where fx & fy are focal lengths, r = sqrt(x_h^2 + y_h^2), and cx & cy are principal points
    // (2) fisheye lens model,
    //   (x_d, y_d) = ((dc4 dc5) (dc6 dc7)) (x_h, y_h) + (cx, cy)
    //   (x_h, y_h) = rho * (X', Y') / sqrt(X'^2 + Y'^2)
    //   where rho = theta + dc1*theta^2 + dc2*theta^3 + dc3*theta^4, theta = 2*arctan(r)/pi
    if (model_type_ == "perspective")
        setDistortCoeffP(xmpData);
    else if (model_type_ == "fisheye")
        setDistortCoeffF(xmpData);
    else
        throw std::invalid_argument("Invalid model type (neither perspective nor fisheye)");
}  // Photo::setDistortCoeff


// intrinsic matrix
cv::Matx33d Photo::getIntrinsicMatrix() const {
    cv::Matx33d intrinsic_matrix = cv::Matx33d::zeros();
    intrinsic_matrix(0, 0) = pixel_focal_lengths_[0];
    intrinsic_matrix(1, 1) = pixel_focal_lengths_[1];
    intrinsic_matrix(2, 2) = getF0();
    intrinsic_matrix(0, 2) = principal_point_[0];
    intrinsic_matrix(1, 2) = principal_point_[1];
    return intrinsic_matrix;
}  // Photo::getIntrinsicMatrix


// rotation_matrix_init_
void Photo::setRotationMatrixInit(const Exiv2::XmpData& xmpData) {
    double deg2rad = CV_PI / 180.0;
    Exiv2::XmpData::const_iterator pos_x;

    // yaw (around Z axis)
    pos_x = xmpData.findKey(xkey("Xmp.Camera.Yaw"));
    if (pos_x == xmpData.end())
        throw std::invalid_argument("Xmp.Camera.Yaw was not found!");
    double yaw = pos_x->toFloat() * deg2rad;
    cv::Matx33d Rz_ypr = cv::Matx33d::eye();
    Rz_ypr(0,0) = std::cos(yaw);
    Rz_ypr(1,1) = std::cos(yaw);
    Rz_ypr(0,1) = -std::sin(yaw);
    Rz_ypr(1,0) = std::sin(yaw);

    // pitch (around Y axis)
    pos_x = xmpData.findKey(xkey("Xmp.Camera.Pitch"));
    if (pos_x == xmpData.end())
        throw std::invalid_argument("Xmp.Camera.Pitch was not found!");
    double pitch = pos_x->toFloat() * deg2rad;
    cv::Matx33d Ry_ypr = cv::Matx33d::eye();
    Ry_ypr(0,0) = std::cos(pitch);
    Ry_ypr(2,2) = std::cos(pitch);
    Ry_ypr(0,2) = std::sin(pitch);
    Ry_ypr(2,0) = -std::sin(pitch);

    // roll (around X axis)
    pos_x = xmpData.findKey(xkey("Xmp.Camera.Roll"));
    if (pos_x == xmpData.end())
        throw std::invalid_argument("Xmp.Camera.Roll was not found!");
    double roll = pos_x->toFloat() * deg2rad;
    cv::Matx33d Rx_ypr = cv::Matx33d::eye();
    Rx_ypr(1,1) = std::cos(roll);
    Rx_ypr(2,2) = std::cos(roll);
    Rx_ypr(1,2) = -std::sin(roll);
    Rx_ypr(2,1) = std::sin(roll);

    cv::Matx33d R_ypr = Rz_ypr * Ry_ypr * Rx_ypr;

    double delta = 1e-06;
    cv::Vec3d p1(getGPSLatitude()+delta, getGPSLongitude(), getGPSAltitude());
    cv::Vec3d p2(getGPSLatitude()-delta, getGPSLongitude(), getGPSAltitude());
    cv::Vec3d xn = p1 - p2;
    xn = xn / cv::norm(xn);
    cv::Vec3d zn(0.0, 0.0, -1.0);
    cv::Vec3d yn = zn.cross(xn);
    cv::Matx33d Ce;
    for (int i = 0; i < 3; ++i) {
        Ce(i,0) = xn[i];
        Ce(i,1) = yn[i];
        Ce(i,2) = zn[i];
    }

    cv::Matx33d Cb = cv::Matx33d::zeros();
    Cb(0,1) = 1.0;
    Cb(1,0) = 1.0;
    Cb(2,2) = -1.0;

    cv::Mat R_opk = (cv::Mat)(Ce * R_ypr * Cb);
    R_opk.copyTo(rotation_matrix_init_);
}  // Photo::setRotationMatrixInit

cv::Matx33d Photo::getRotationMatrixInit() const {
    return rotation_matrix_init_;
}  // Photo::getRotationMatrixInit


// camera_matrix_init_
void Photo::setCameraMatrixInit() {
    cv::Matx33d K = getIntrinsicMatrix();
    cv::Matx33d RT = rotation_matrix_init_.t();
    cv::Matx34d It;
    cv::hconcat(cv::Matx33d::eye(), -getCameraPositionInit(), It);
    camera_matrix_init_ = K * RT * It;
}  // Photo::setCameraMatrixInit

cv::Matx34d Photo::getCameraMatrixInit() const {
    return camera_matrix_init_;
}


// set new camera matrix
void Photo::setCameraMatrix(const cv::Mat new_camera_matrix) {
    new_camera_matrix.copyTo(camera_matrix_);
}  // Photo::setCameraMatrix


// mat_original_
cv::Mat Photo::getMatOriginal() const {
    return mat_original_;
}  // Photo::getMatOriginal


// mat_corrected_
cv::Mat Photo::getMatCorrected() const {
    return mat_corrected_;
}  // Photo::getMatCorrected


// set meta data
void Photo::setMetaData(
    const Exiv2::ExifData& exifData,
    const Exiv2::XmpData& xmpData
) {
    try {
        // datetime
        setDateTime(exifData);

        // model type
        setModelType(xmpData);

        // pixel dimensions
        setPixelDims(exifData);

        // GPS xyz
        setGPSLatitude(exifData);
        setGPSLongitude(exifData);
        setGPSAltitude(exifData);

        // pixel per mm
        setPixelPerMM(exifData);

        // pixel focal lenngths
        setPixelFocalLengths(exifData, xmpData);

        // principal point
        setPrincipalPoint(xmpData);

        // distortion coefficients
        setDistortCoeff(xmpData);

        // rotation matrix initial
        setRotationMatrixInit(xmpData);

        // camera matrix initial
        setCameraMatrixInit();
    }
    catch (std::invalid_argument& e) {
        cerr << e.what() << endl;
    }
}  // Photo::setMetaData


// read from file
bool Photo::readFromFile(const std::string& file_path) {
    // read image file
    Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(file_path);
    if(image.get() == 0) {
        cerr << "Meta data cannot be read!" << endl;
        return false;
    }
    image->readMetadata();

    // read EXIF data
    Exiv2::ExifData &exifData = image->exifData();
    if (exifData.empty()) {
        cerr << "No EXIF data found in the file!" << endl;
        return false;
    }
    
    // read XMP data
    Exiv2::XmpData &xmpData = image->xmpData();
    if (xmpData.empty()) {
        cerr << "No XMP data found in the file!" << endl;
        return false;
    }
    
    // set meta data
    try {
        setMetaData(exifData, xmpData);
    }
    catch (std::invalid_argument& e) {
        cerr << e.what() << endl;
        return false;
    }

    // OpenCV mat
    this->mat_original_ = cv::imread(file_path);
    if (mat_original_.data == NULL) {
        cerr << "Image not found!" << endl;
        return false;
    }

    return true;
}  // Photo::readFromFile

void Photo::correctPhoto(const cv::Mat x_map, const cv::Mat y_map) {
    cv::remap(
        mat_original_,
        mat_corrected_,
        x_map,
        y_map,
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar()
    );
}


/////////////////////////////////////////////////////////////////////
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


// get number of photos
size_t PhotoList::getNumPhotos() const {
    return photo_vector_.size();
}


// model_type_
std::string PhotoList::getModelType() const {
    return model_type_;
}  // PhotoList::getModelType

void PhotoList::setModelType(const Photo& photo) {
    model_type_ = photo.getModelType();
}  // PhotoList::setModelType


// pixel_dimensions_
cv::Size2i PhotoList::getPixelDims() const {
    return pixel_dimensions_;
}  // PhotoList::getPixelDims

void PhotoList::setPixelDims(const Photo& photo) {
    pixel_dimensions_ = photo.getPixelDims();
}  // PhotoList::setPixelDims

double PhotoList::getF0() const {
    cv::Size d = getPixelDims();
    return (d.width + d.height) / 2.0;
}


// pixel_per_mm_
cv::Vec2d PhotoList::getPixelPerMM() const {
    return pixel_per_mm_;
}  // PhotoList::getPixelPerMM

void PhotoList::setPixelPerMM(const Photo& photo) {
    pixel_per_mm_ = photo.getPixelPerMM();
}  // PhotoList::setPixelPerMM


// pixel_focal_lengths_
cv::Vec2d PhotoList::getPixelFocalLengths() const {
    return pixel_focal_lengths_;
}  // PhotoList::getPixelFocalLengths

void PhotoList::setPixelFocalLengths(const Photo& photo) {
    pixel_focal_lengths_ = photo.getPixelFocalLengths();
}  // PhotoList::setPixelFocalLengths


// principal_point_
cv::Vec2d PhotoList::getPrincipalPoint() const {
    return principal_point_;
}  // PhotoList::getPrincipalPoint

void PhotoList::setPrincipalPoint(const Photo& photo) {
    principal_point_ = photo.getPrincipalPoint();
}  // PhotoList::setPrincipalPoint


// distortion_coefficients_
cv::Mat PhotoList::getDistortCoeff() const {
    return distortion_coefficients_;
}  // PhotoList::getDistortionCoefficients

void PhotoList::setDistortCoeff(const Photo& photo) {
    distortion_coefficients_ = photo.getDistortCoeff();
}  // PhotoList::setDistortCoeff


cv::Matx33d PhotoList::getIntrinsicMatrix() {
    cv::Matx33d intrinsic_matrix = cv::Matx33d::zeros();
    intrinsic_matrix(0, 0) = pixel_focal_lengths_[0];
    intrinsic_matrix(1, 1) = pixel_focal_lengths_[1];
    intrinsic_matrix(2, 2) = getF0();
    intrinsic_matrix(0, 2) = principal_point_[0];
    intrinsic_matrix(1, 2) = principal_point_[1];
    return intrinsic_matrix;
}  // PhotoList::getIntrinsicMatrix


void PhotoList::setMetaData() {
    Photo photo = photo_vector_[0];
    setModelType(photo);
    setPixelDims(photo);
    setPixelPerMM(photo);
    setPixelFocalLengths(photo);
    setPrincipalPoint(photo);
    setDistortCoeff(photo);

    // check all the meta data are the same
    for (size_t i = 1; i < n_photos_; ++i) {
        photo = photo_vector_[i];
        if (model_type_ != photo.getModelType())
            cerr << "Different model types are included!" << endl;
        if (pixel_dimensions_ != photo.getPixelDims())
            cerr << "Different pixel dimensions are included!" << endl;
        if (pixel_per_mm_ != photo.getPixelPerMM())
            cerr << "Different pixel per mm are included!" << endl;
        if (pixel_focal_lengths_ != photo.getPixelFocalLengths())
            cerr << "Different pixel focal lengths are included!" << endl;
        if (principal_point_ != photo.getPrincipalPoint())
            cerr << "Different principal points are included!" << endl;
        if (!isEqualAllElems(distortion_coefficients_, photo.getDistortCoeff()))
            cerr << "Different distortion coefficients are included!" << endl;
    }
}  // PhotoList::setMetaData


void PhotoList::correctPhotoList() {
    cv::Matx33d intrinsic_matrix = getIntrinsicMatrix();
    cv::Mat x_map, y_map;
    if (model_type_ == "perspective") {
        cv::Matx33d K (intrinsic_matrix);
        K(0, 0) = -K(0, 0);
        K(1, 1) = -K(1, 1);
        K(2, 2) = 1.0;
        cv::initUndistortRectifyMap(
            K,
            getDistortCoeff(),
            cv::Mat(),
            K,
            getPixelDims(),
            CV_16SC2,
            x_map,
            y_map
        );
    } else if (model_type_ == "fisheye") {
        double f0 = getF0();
        cv::Vec2d c = getPrincipalPoint();
        cv::Mat k = getDistortCoeff();
        cv::Matx22d Q(k.at<double>(4), k.at<double>(5), k.at<double>(6), k.at<double>(7));
        cv::Matx22d iQ = Q.inv(cv::DECOMP_SVD);
        cv::Size mat_size = getPixelDims();
        x_map.create(mat_size, CV_16SC2);
        y_map.create(mat_size, CV_16UC1);

        for (size_t i = 0; i < mat_size.height; ++i) {
            short* mx = (short*)x_map.ptr<float>(i);
            unsigned short* my = (unsigned short*)y_map.ptr<float>(i);
            for (size_t j = 0; j < mat_size.width; ++j) {
                cv::Vec2d pi = cv::Vec2d((double)j, (double)i);
                cv::Vec2d pw = iQ * (pi - c);
            
                double theta_d = std::sqrt(pw[0]*pw[0] + pw[1]*pw[1]);
                double scale = 1.0;
                if (theta_d > 1e-8) {
                    double theta = theta_d;
                    const double EPS = 1e-8;
                    for (int q = 0; q < 50; ++q) {
                        double theta2 = theta*theta,
                               theta3 = theta2*theta;
                        double k1_theta1 = k.at<double>(1) * theta,
                               k2_theta2 = k.at<double>(2) * theta2,
                               k3_theta3 = k.at<double>(3) * theta3;
                        double theta_fix = (theta * (1 + k1_theta1 + k2_theta2 + k3_theta3) - theta_d) /
                            (1 + 2*k1_theta1 + 3*k2_theta2 + 4*k3_theta3);
                        theta = theta - theta_fix;
                        if (std::fabs(theta_fix) < EPS)
                            break;
                    }
                    scale = std::tan(theta * CV_PI / 2.0) / theta_d;
                }
                cv::Vec2d pu = pw * scale;
                cv::Vec3d pr = intrinsic_matrix * cv::Vec3d(pu[0], pu[1], 1.0);
                cv::Vec3d fi(f0*pr[0]/pr[2], f0*pr[1]/pr[2]);
                int iu = cv::saturate_cast<int>(fi[0]*cv::INTER_TAB_SIZE);
                int iv = cv::saturate_cast<int>(fi[1]*cv::INTER_TAB_SIZE);
                mx[j*2+0] = (short)(iu >> cv::INTER_BITS);
                mx[j*2+1] = (short)(iv >> cv::INTER_BITS);
                my[j] = (unsigned short)((iv & (cv::INTER_TAB_SIZE-1))*cv::INTER_TAB_SIZE +
                    (iu & (cv::INTER_TAB_SIZE-1)));
            }
        }
    } else
        throw std::invalid_argument("Invalid model type (neither perspective nor fisheye)");

    for (size_t i = 0; i < getNumPhotos(); ++i) {
        Photo p = photo_vector_[i];
        p.correctPhoto(x_map, y_map);
        photo_vector_[i] = p;
    }
}


bool PhotoList::readFromFiles(const std::vector<std::string> files) {
    // seek photo files
    Photo photo;
    std::string suffix;
    for (auto file_path : files) {
        suffix = file_path.substr(file_path.size() - 4);
        if (suffix == ".JPG" | suffix == ".TIF") {
            photo.readFromFile(file_path);
            photo_vector_.emplace_back(photo);
        }
    }
    n_photos_ = photo_vector_.size();
    if (n_photos_ == 0) {
        cerr << "No photo found!" << endl;
        return false;
    }

    // set meta data
    setMetaData();

    // correct photo
    try {
        correctPhotoList();
    } catch (std::invalid_argument& e) {
        cerr << e.what() << endl;
    }

    return true;
}  // PhotoList::readFromFiles

}  // namespace fmvg
