#ifndef FMVG_PHOTO_H_
#define FMVG_PHOTO_H_

namespace fmvg {

bool isEqualAllElems(cv::InputArray a1, cv::InputArray a2);

class Photo {
    // data members
    time_t datetime_;
    std::string model_type_;
    cv::Size pixel_dimensions_;
    double gps_longitude_;
    double gps_latitude_;
    double gps_altitude_;
    cv::Vec2d pixel_per_mm_;
    cv::Vec2d pixel_focal_lengths_;
    cv::Vec2d principal_point_;
    cv::Mat distortion_coefficients_;
    cv::Matx33d rotation_matrix_init_;
    cv::Matx34d camera_matrix_;
    cv::Mat mat_original_;
    cv::Mat mat_corrected_;

    // private member functions
    void setDateTime(const Exiv2::ExifData& exifData);
    void setModelType(const Exiv2::XmpData& xmpData);
    void setPixelDims(const Exiv2::ExifData& exifData);
    void setGPSLongitude(const Exiv2::ExifData& exifData);
    void setGPSLatitude(const Exiv2::ExifData& exifDatar);
    void setGPSAltitude(const Exiv2::ExifData& exifData);
    void setPixelPerMM(const Exiv2::ExifData& exifData);
    void setPixelFocalLengths(const Exiv2::ExifData& exifData,
                              const Exiv2::XmpData& xmpData);
    void setPrincipalPoint(const Exiv2::XmpData& xmpData);
    void setDistortCoeffP(const Exiv2::XmpData& xmpData);
    void setDistortCoeffF(const Exiv2::XmpData& xmpData);
    void setDistortCoeff(const Exiv2::XmpData& xmpData);
    void setRotationMatrixInit(const Exiv2::XmpData& xmpData);
    void setMetaData(const Exiv2::ExifData& exifData, const Exiv2::XmpData& xmpData);

public:
    // definition equality & relational operators
    bool operator==(const Photo& p) const {
        bool test = datetime_ == p.datetime_ &&
            pixel_dimensions_ == p.pixel_dimensions_ &&
            gps_longitude_ == p.gps_longitude_ &&
            gps_latitude_ == p.gps_latitude_ &&
            gps_altitude_ == p.gps_altitude_ &&
            mat_original_.dims == p.mat_original_.dims &&
            mat_original_.type() == p.mat_original_.type() &&
            mat_original_.channels() == p.mat_original_.channels();
        if (test)
            return isEqualAllElems(mat_original_, p.mat_original_);
        else return false;
    }  // operator==

    // public member functions
    time_t getDateTime() const;
    cv::Size getPixelDims() const;
    double getF0() const;
    double getGPSLongitude() const;
    double getGPSLatitude() const;
    double getGPSAltitude() const;
    cv::Vec3d getCameraPositionInit() const;
    std::string getModelType() const;
    cv::Vec2d getPixelPerMM() const;
    cv::Vec2d getPixelFocalLengths() const;
    cv::Vec2d getPrincipalPoint() const;
    cv::Mat getDistortCoeff() const;
    cv::Matx33d getRotationMatrixInit() const;
    cv::Matx34d getCameraMatrixInit() const;
    cv::Mat getMatOriginal() const;
    cv::Mat getMatCorrected() const;
    bool readFromFile(const std::string& file_path);
    void correctPhoto(const cv::Mat map1, const cv::Mat map2);

    // constructor
    Photo() {}
    Photo(const std::string& file_path) {
        readFromFile(file_path);
    }
};  // class Photo

class PhotoList {
    // data members
    std::vector<Photo> photo_vector_;
    size_t n_photos_;
    std::string model_type_;
    cv::Size2i pixel_dimensions_;
    cv::Vec2d pixel_per_mm_;
    cv::Vec2d pixel_focal_lengths_;
    cv::Vec2d principal_point_;
    cv::Mat distortion_coefficients_;

    // private member functions
    void setModelType(const Photo& photo);
    void setPixelDims(const Photo& photo);
    void setPixelPerMM(const Photo& photo);
    void setPixelFocalLengths(const Photo& photo);
    void setPrincipalPoint(const Photo& photo);
    void setDistortCoeff(const Photo& photo);
    void setMetaData();
    void correctPhotoList();

public:
    // constructor
    PhotoList() {}
    PhotoList(const std::vector<std::string> file_paths) {
        readFromFiles(file_paths);
    }

    // deifinition equality operator
    bool operator==(const PhotoList& pl) const {
        return photo_vector_ == pl.photo_vector_ &&
            model_type_ == pl.model_type_ &&
            pixel_per_mm_ == pl.pixel_per_mm_ &&
            pixel_focal_lengths_ == pl.pixel_focal_lengths_ &&
            principal_point_ == pl.principal_point_ &&
            isEqualAllElems(distortion_coefficients_, pl.distortion_coefficients_);
    }  // operator==

    // iterator
    typedef std::vector<Photo>::iterator iterator;
    iterator begin();
    iterator end();

    // const iterator
    typedef std::vector<Photo>::const_iterator const_iterator;
    const_iterator begin() const;
    const_iterator end() const;

    // public member function
    std::vector<Photo> getPhotoVector() const;
    size_t getNumPhotos() const;
    std::string getModelType() const;
    cv::Size2i getPixelDims() const;
    double getF0() const;
    cv::Vec2d getPixelPerMM() const;
    cv::Vec2d getPixelFocalLengths() const;
    cv::Vec2d getPrincipalPoint() const;
    cv::Mat getDistortCoeff() const;
    cv::Matx33d getIntrinsicMatrix();
    bool readFromFiles(const std::vector<std::string>);
};  // class PhotoList

}  // namespace fmvg

#endif  // FMVG_PHOTO_H_
