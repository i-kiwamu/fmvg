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
    cv::Mat mat_;

    // private member functions
    bool setDateTime(const Exiv2::ExifData& exifData);
    bool setPixelDims(const Exiv2::ExifData& exifData);
    bool setGPSLongitude(const Exiv2::ExifData& exifData);
    bool setGPSLatitude(const Exiv2::ExifData& exifDatar);
    bool setGPSAltitude(const Exiv2::ExifData& exifData);

public:
    // definition equality & relational operators
    bool operator==(const Photo& p) const {
        bool test = datetime_ == p.datetime_ &&
            pixel_dimensions_ == p.pixel_dimensions_ &&
            gps_longitude_ == p.gps_longitude_ &&
            gps_latitude_ == p.gps_latitude_ &&
            gps_altitude_ == p.gps_altitude_ &&
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
    cv::Mat getMat() const;
    bool readExifData(const Exiv2::ExifData& exifData);
    bool readFromFile(const std::string& file_path);

    // constructor
    Photo() {}
    Photo(const std::string& file_path) {
        readFromFile(file_path);
    }
};  // class Photo

class PhotoList {
    // data members
    std::vector<Photo> photo_vector_;
    std::string model_type_;
    cv::Vec2d pixel_per_mm_;
    cv::Vec2d pixel_focal_lengths_;
    cv::Vec2d principal_point_;
    cv::Vec6d distortion_coefficients_;

    // private member functions
    bool setModelType(const Exiv2::XmpData& xmpData);
    bool setPixelPerMM(const Exiv2::ExifData& exifData);
    bool setPixelFocalLengths(const Exiv2::ExifData& exifData,
                              const Exiv2::XmpData& xmpData);
    bool setPrincipalPoint(const Exiv2::XmpData& xmpData);
    bool setDistortionCoefficients(const Exiv2::ExifData& exifData,
                                   const Exiv2::XmpData& xmpData);

public:
    // constructor
    PhotoList() {}

    // deifinition equality operator
    bool operator==(const PhotoList& pl) const {
        return photo_vector_ == pl.photo_vector_ &&
            model_type_ == pl.model_type_ &&
            pixel_per_mm_ == pl.pixel_per_mm_ &&
            pixel_focal_lengths_ == pl.pixel_focal_lengths_ &&
            principal_point_ == pl.principal_point_ &&
            distortion_coefficients_ == pl.distortion_coefficients_;
    }  // operator==

    // iterator
    typedef std::vector<Photo>::iterator iterator;
    iterator begin();
    iterator end();

    // const iterator
    std::vector<Photo>::const_iterator begin() const;
    std::vector<Photo>::const_iterator end() const;

    // public member function
    std::vector<Photo> getPhotoVector() const;
    std::string getModelType() const;
    cv::Vec2d getPixelPerMM() const;
    cv::Vec2d getPixelFocalLengths() const;
    cv::Vec2d getPrincipalPoint() const;
    cv::Vec6d getDistortionCoefficients() const;
    bool readXmpData(const Exiv2::ExifData& exifData, const Exiv2::XmpData& xmpData);
    bool readFromFiles(const std::vector<std::string>);
};  // class PhotoList

}  // namespace fmvg

#endif  // FMVG_PHOTO_H_
