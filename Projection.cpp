//
// Created by 86176 on 2023/6/17.
//

#include <opencv2/opencv.hpp>
#include "Projection.h"
#include "math.h"
#include <chrono>
#include <Dense>
#include <Core>
#include <array>

cv::Mat ensure_point_list(const cv::Mat& points, int dim, bool concatenate = true, bool crop = true) {
    assert(points.type() == CV_32F);
    assert(points.rows > 0 && points.cols == dim);

    cv::Mat cropped_points = points.clone();
    if (crop) {
        for (int test_dim = 4; test_dim > dim; --test_dim) {
            if (cropped_points.cols == test_dim) {
                int new_cols = test_dim - 1;
                assert(cropped_points.col(new_cols).isContinuous() && cv::norm(cropped_points.col(new_cols) - cv::Scalar(1)) < 1e-6);
                cropped_points = cropped_points.colRange(0, new_cols);
            }
        }
    }

    cv::Mat result = cropped_points;
    if (concatenate && cropped_points.cols == dim - 1) {
        cv::hconcat(cropped_points, cv::Mat::ones(cropped_points.rows, 1, CV_32F), result);
    }

    assert(result.cols == dim);
    return result;
}

OpencvCamProjection::OpencvCamProjection(const cv::Mat &translation, const cv::Mat &rotation, const cv::Size &size,
                                         const cv::Point2f &principle_point, const cv::Point2f &focal_length,
                                         const std::vector<double> &distortion_params)
{
    cv::Matx44f pose = {rotation.at<double>(0,0),rotation.at<double>(0,1),rotation.at<double>(0,2),translation.at<double>(0),
                        rotation.at<double>(1,0),rotation.at<double>(1,1),rotation.at<double>(1,2),translation.at<double>(1),
                        rotation.at<double>(2,0),rotation.at<double>(2,1),rotation.at<double>(2,2),translation.at<double>(2),
                        0,0,0,1};
    // translation.copyTo(pose.col(3).rowRange(0, 3));
    // rotation.copyTo(pose.colRange(0, 3).rowRange(0, 3));
    pose_t = pose.t();
    inv_pose_ = pose.inv();
    size_ = size;
    principle_point_ = principle_point;
    coefficients_ = cv::Mat(distortion_params).clone();
    focal_length_ = focal_length;
}


std::array<float,2> OpencvCamProjection::Project3dTo2darray(const cv::Matx14f &world_points, bool do_clip, double invalid_value) const
{
    cv::Matx14f camera_points = world_points * inv_pose_.t();
    float chi = sqrt(camera_points(0,0)*(camera_points(0,0)) + camera_points(0,1)*(camera_points(0,1)));
    
    float theta = CV_PI / 2 - atan2(camera_points(0,2), chi);
    float rho = ThetaToRho(theta);
    float lenspoints01 = rho * (1/chi) * camera_points(0,0);
    float lenspoints02 = rho *(1/chi)* camera_points(0,1);
    float screenpoints01 = lenspoints01 * focal_length_.x + principle_point_.x;
    float screenpoints02 = lenspoints02 * focal_length_.y + principle_point_.y;

    std::array<float,2> screen_points_array = {screenpoints01, screenpoints02};
    return screen_points_array;
}



cv::Matx14f OpencvCamProjection::Project2dTo3darray(const std::array<int,2> lens_points) const
{
    return cv::Matx14f();
}


double OpencvCamProjection::ThetaToRho(double theta) const {
    double result = 0;
    for(int i = 0; i < coefficients_.rows; ++i)
    {
        result += coefficients_.at<double>(i) * pow(theta, (i+1)*2);
    }
    return (result+1)*theta;
}

cv::Mat OpencvCamProjection::ApplyClip(const cv::Mat &points, const cv::Mat &clip_source) const {
    cv::Mat result = points.clone();
    cv::Mat mask = (clip_source.col(0) < 0) | (clip_source.col(0) >= size_.width) |
                   (clip_source.col(1) < 0) | (clip_source.col(1) >= size_.height);
    result.setTo(std::numeric_limits<double>::quiet_NaN(), mask);
    return result;
}

cv::Size OpencvCamProjection::GetSize() const {
    return size_;
}

CylindricalProjection::CylindricalProjection(const cv::Mat &translation, const cv::Mat &rotation,
                                             const cv::Size &size, const cv::Point2f &principle_point,
                                             const cv::Point2f &focal_length) 
{
    // cv::Mat pose = cv::Mat::eye(4, 4, CV_32F);
    cv::Matx44f pose = {rotation.at<double>(0,0),rotation.at<double>(0,1),rotation.at<double>(0,2),translation.at<double>(0),
                        rotation.at<double>(1,0),rotation.at<double>(1,1),rotation.at<double>(1,2),translation.at<double>(1),
                        rotation.at<double>(2,0),rotation.at<double>(2,1),rotation.at<double>(2,2),translation.at<double>(2),
                        0,0,0,1};
    // translation.copyTo(pose.col(3).rowRange(0, 3));
    // rotation.copyTo(pose.colRange(0, 3).rowRange(0, 3));

    pose_t = pose.t();
    inv_pose_  = pose_t.inv();
    size_ = size;
    principle_point_ = principle_point;
    focal_length_ = focal_length;
}


std::array<float,2> CylindricalProjection::Project3dTo2darray(const cv::Matx14f &screen_points, bool do_clip, double invalid_value) const
{
    std::array<float,2> points;
    return points;
}


cv::Matx14f CylindricalProjection::Project2dTo3darray(const std::array<int,2> lens_points) const
{
    std::array<float,2> image_points = {lens_points[0]*1.0f,lens_points[1]*1.0f};
    image_points[0] =(image_points[0] - principle_point_.x)/focal_length_.x;
    image_points[1] =(image_points[1] - principle_point_.y)/focal_length_.y;
    
    
    float theta = image_points[0];
    float scale = 1/sqrt(image_points[1] * image_points[1] + 1);
    cv::Matx14f camera_points(sin(theta)*scale,image_points[1]*scale,cos(theta)*scale,1);
    cv::Matx14f world_points = camera_points * pose_t;
    return world_points;
}



cv::Mat CylindricalProjection::ApplyClip(const cv::Mat &points, const cv::Mat &clip_source) {
    return cv::Mat();
}

cv::Size CylindricalProjection::GetSize() const {
    return size_;
}

cv::Mat CreateImgProjectionMaps(const Projection* source_cam, const Projection* destination_cam)
{
    cv::Mat source_points = cv::Mat::zeros(1300,1920, CV_32FC2);
    cv::Mat destination_points = cv::Mat::zeros(1300,1920, CV_32FC2);

    // 遍历柱面图像坐标
    
    for (int x = 0; x < source_cam->GetSize().height; ++x)
    {
        for (int y = 0; y < source_cam->GetSize().width; ++y)
        {
            std::array<int,2>array_lens_points = {x, y};
            //  柱面图转归一化平面的世界坐标
            cv::Matx14f source_world_points = destination_cam->Project2dTo3darray(array_lens_points);
            // 归一化平面的世界坐标转鱼眼原图图像坐标

            std::array<float,2> destination_screen_points_array = source_cam->Project3dTo2darray(source_world_points);
            // if (!cv::checkRange(destination_screen_points)) {
            //     continue;
            // }

            // float dest_x = destination_screen_points.at<float>(0);
            // float dest_y = destination_screen_points.at<float>(1);
            float dest_x = destination_screen_points_array[0];
            float dest_y = destination_screen_points_array[1];


            if (dest_x < 0 || dest_x >= destination_cam->GetSize().height ||
                dest_y < 0 || dest_y >= destination_cam->GetSize().width) {
                continue;
            }

            // source_points保存鱼眼原图到柱面映射的坐标
            source_points.at<cv::Vec2f>(dest_y, dest_x) = cv::Vec2f(x, y);
            // destination_points保存柱面图到鱼眼原图的映射坐标
            destination_points.at<cv::Vec2f>(y, x) = cv::Vec2f(dest_x, dest_y);
            // destination_points.at<cv::Vec2f>(y, x) = cv::Vec2f(x, y);
            // std::cout << destination_points.at<cv::Vec2f>(y, x)<<std::endl;
        }
    }
    // cv::Mat source2;
    // std::cout<<destination_points.at<cv::Vec2f>(300,300)<<std::endl;
    // convertMaps(destination_points,source2, , sourcey ,CV_32FC2, true);
    // return destination_points;
    return source_points;

}
