//
// Created by 86176 on 2023/6/17.
//

#ifndef FISHEYE2CYLINDRICAL_PROJECTION_H
#define FISHEYE2CYLINDRICAL_PROJECTION_H

#include <opencv2/core.hpp>
#include <Dense>
#include <Core>
class Projection
{
public:
    virtual std::array<float,2> Project3dTo2darray(const cv::Matx14f& cam_points, bool do_clip = false, double invalid_value = std::numeric_limits<double>::quiet_NaN()) const = 0;
    virtual cv::Matx14f Project2dTo3darray(const std::array<int,2> array_lens_points) const = 0;
    virtual cv::Size GetSize() const = 0;
};


class CylindricalProjection : public Projection {
public:
    CylindricalProjection(const cv::Mat &translation, const cv::Mat& rotation, const cv::Size& size,
                          const cv::Point2f& principle_point, const cv::Point2f& focal_length);

    std::array<float,2> Project3dTo2darray(const cv::Matx14f& screen_points, bool do_clip = false, double invalid_value = std::numeric_limits<double>::quiet_NaN()) const override;
    cv::Matx14f Project2dTo3darray(const std::array<int,2> array_lens_points) const override;
    cv::Size GetSize() const override;

private:
    cv::Mat ApplyClip(const cv::Mat& points, const cv::Mat& clip_source);

private:
    cv::Matx44f pose_t;
    cv::Matx44f inv_pose_;
    cv::Size size_;
    cv::Point2f principle_point_;
    cv::Point2f focal_length_;
};

class OpencvCamProjection : public Projection {
public:
    OpencvCamProjection(const cv::Mat& translation, const cv::Mat& rotation, const cv::Size& size,
                        const cv::Point2f& principle_point, const cv::Point2f& focal_length, const std::vector<double>& distortion_params);

    std::array<float,2> Project3dTo2darray(const cv::Matx14f& world_points, bool do_clip = false, double invalid_value = std::numeric_limits<double>::quiet_NaN()) const override;
    cv::Matx14f Project2dTo3darray(const std::array<int,2> array_lens_points) const override;
    cv::Size GetSize() const override;

private:
    double ThetaToRho(double theta) const;
    cv::Mat ApplyClip(const cv::Mat& points, const cv::Mat& clip_source) const;

private:
    cv::Matx44f pose_t;
    cv::Matx44f inv_pose_;
    cv::Size size_;
    cv::Point2f principle_point_;
    cv::Point2f focal_length_;
    cv::Mat coefficients_;
    cv::Mat power_;
    cv::Mat focal_matrix_;
};

cv::Mat CreateImgProjectionMaps(const Projection* source_cam, const Projection* destination_cam);

#endif //FISHEYE2CYLINDRICAL_PROJECTION_H
