#ifndef PREPARATION_H
#define PREPARATION_H

namespace sequoiaortho {

class Photo {
    // data members
    time_t datetime_;
    int pixel_dimension_x_;
    int pixel_dimension_y_;
    double focal_length_init_;
    double gps_longitude_;
    double gps_latitude_;
    double gps_altitude_;
    cv::Mat mat_;

public:
    // constructor
    Photo() {}
    
    // definition equality & relational operators
    bool operator==(const Photo& p) const {
        bool test = datetime_ == p.datetime_ &&
            pixel_dimension_x_ == p.pixel_dimension_x_ &&
            pixel_dimension_y_ == p.pixel_dimension_y_ &&
            focal_length_init_ == p.focal_length_init_ &&
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

    time_t getDateTime();
    int getPixelDimX();
    int getPixelDimY();
    cv::Size getPixelDims();
    double getFocalLengthInit();
    double getGPSLongitude();
    double getGPSLatitude();
    double getGPSAltitude();
    cv::Mat getMat();
    bool readFromFile(std::string file_path);
};  // class Photo

}  // namespace sequoiaortho

#endif
