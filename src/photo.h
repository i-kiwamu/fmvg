#ifndef FMVG_PHOTO_H_
#define FMVG_PHOTO_H_

namespace fmvg {

class Photo {
    // data members
    time_t datetime_;
    cv::Size2i pixel_dimensions_;
    double gps_longitude_;
    double gps_latitude_;
    double gps_altitude_;
    std::string model_type_;
    cv::Vec2d pixel_focal_lengths_;
    cv::Vec2d principal_point_;
    cv::Vec6d distortion_coefficients_;
    cv::Mat mat_;

    // private member functions
    bool readExifData(Exiv2::ExifData exifData);
    bool readXmpData(Exiv2::ExifData exifData, Exiv2::XmpData xmpData);

public:
    // constructor
    Photo() {}
    
    // definition equality & relational operators
    bool operator==(const Photo& p) const {
        bool test = datetime_ == p.datetime_ &&
            pixel_dimensions_ == p.pixel_dimensions_ &&
            gps_longitude_ == p.gps_longitude_ &&
            gps_latitude_ == p.gps_latitude_ &&
            gps_altitude_ == p.gps_altitude_ &&
            model_type_ == p.model_type_ &&
            pixel_focal_lengths_ == p.pixel_focal_lengths_ &&
            principal_point_ == p.principal_point_ &&
            distortion_coefficients_ == p.distortion_coefficients_ &&
            mat_.dims == p.mat_.dims &&
            mat_.type() == p.mat_.type() &&
            mat_.channels() == p.mat_.channels();
        if (test) {
            cv::Mat test_elem = mat_ == p.mat_;
            return cv::countNonZero(test_elem) == 0;
        }
        else return false;
    }  // operator==

    // public member functions
    time_t getDateTime() const;
    cv::Size2i getPixelDims() const;
    double getGPSLongitude() const;
    double getGPSLatitude() const;
    double getGPSAltitude() const;
    std::string getModelType() const;
    cv::Vec2d getPixelFocalLengths() const;
    cv::Vec2d getPrincipalPoint() const;
    cv::Vec6d getDistortionCoefficients() const;
    cv::Mat getMat() const;
    bool readFromFile(std::string file_path);
};  // class Photo

}  // namespace fmvg

#endif  // FMVG_PHOTO_H_
