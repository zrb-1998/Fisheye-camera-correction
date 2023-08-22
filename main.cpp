#include <iostream>
#include <json/json.h>
#include <fstream>
#include <Dense>
#include <Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "Projection.h"
#include <chrono>

using namespace std;


cv::Mat distort_matrix = cv::Mat_<double>(4,1);
cv::Mat rotation_angle = cv::Mat_<double>(3, 1);
cv::Mat rotation_matrix = cv::Mat_<double>(3, 3);
cv::Mat trans_matrix = cv::Mat_<double>(3,1);
cv::Mat external_matrix = cv::Mat::eye(4, 4, CV_32F);
// cv::Mat internal_matrix = cv::Mat::eye(3, 3, CV_64F);
cv::Point2f principle_matrix;
std::vector<double> distortion_params;
cv::Point2f focal_length;
cv::Size size;

//从文件中读取Json
void readFileJson()
{
    Json::Reader reader;
    Json::Value root;

    ifstream in("D:/work/fisheye2cylindrical/samples/params/camera_1.json",ios::binary);

    if(!in.is_open()){
        cout<<"Error opening"<<endl;
        return;
    }

    if(reader.parse(in,root)){
        principle_matrix = cv::Point2f(root["center_u"].asDouble(),root["center_v"].asDouble());
        focal_length = cv::Point2f(root["focal_u"].asDouble(),root["focal_v"].asDouble());
        size  = cv::Size(root["image_height"].asInt(),root["image_width"].asInt());

        const Json::Value distort = root["distort"];
        for(unsigned int i = 0;i < distort.size(); i++)
        {
            distort_matrix.at<double>(i, 0) = distort[i].asDouble();
            distortion_params.push_back(distort[i].asDouble());
        }

        const Json::Value exter_rot = root["vcs"]["rotation"];
        for(unsigned int i=0;i<exter_rot.size();i++)
        {
            rotation_angle.at<double>(i, 0)  = exter_rot[i].asDouble();
        }

        const Json::Value exter_trans = root["vcs"]["translation"];
        for(unsigned int i=0;i<exter_trans.size();i++)
        {
            trans_matrix.at<double>(i, 0) = exter_trans[i].asDouble();
        }
    }
    in.close();
}


cv::Mat makecylindricalrot(const cv::Mat &rotation_matrix)
{
    cv::Mat matrotation = rotation_matrix;
    Eigen::Matrix3d matrix_33d;
    cv2eigen(matrotation,matrix_33d);  
    Eigen::Vector3d eulerAngle =  matrix_33d.eulerAngles(2,0,2);
    // eulerAngle=eulerAngle/(CV_PI / 2);
    for(int i=0;i<3;i++)
    {
        eulerAngle(i)=round(eulerAngle(i)/(CV_PI / 2))*(CV_PI / 2);
    }
    matrix_33d = Eigen::AngleAxisd(eulerAngle[0], Eigen::Vector3d::UnitZ()) * 
                       Eigen::AngleAxisd(eulerAngle[1], Eigen::Vector3d::UnitX()) * 
                       Eigen::AngleAxisd(eulerAngle[2], Eigen::Vector3d::UnitZ());
    cv::Mat cylrotationmatrix;
    eigen2cv(matrix_33d,cylrotationmatrix);
    return cylrotationmatrix;
}

int main() {
    readFileJson();
    
    cv::Rodrigues(rotation_angle, rotation_matrix);
   
    rotation_matrix.copyTo(external_matrix(cv::Range(0, 3), cv::Range(0, 3)));
    trans_matrix.copyTo(external_matrix(cv::Range(0, 3), cv::Range(3, 4)));
    cv::Mat rotation;
    
    Projection* opencv_cam = new OpencvCamProjection(trans_matrix, rotation_matrix, size, principle_matrix,focal_length, distortion_params);
    Projection* cylindrical_cam = new CylindricalProjection(trans_matrix, rotation=makecylindricalrot(rotation_matrix), size, principle_matrix, focal_length);
    
    // cv::Mat fisheye_image = cv::imread("D:/work/fisheye2cylindrical/samples/images/input_1.jpg");
    cv::Mat fisheye_image = cv::imread("D:/work/fisheye2cylindrical/samples/cylin_res/test_byd_6.png");

    cv::waitKey(0);
    auto starttime = std::chrono::system_clock::now();
    cv::Mat map1 = CreateImgProjectionMaps(opencv_cam, cylindrical_cam);
    
    cv::Mat map;
    cv::Mat cylindrical_image ;
    if (!fisheye_image.empty())
    {
        remap(fisheye_image,cylindrical_image, map1, map, cv::InterpolationFlags::INTER_NEAREST);
        std::chrono::duration<double> diff = std::chrono::system_clock::now()- starttime;
        std::cout << "CreateImgProjectionMaps need time:" << diff.count() << "s" << std::endl;
        // cv::line(cylindrical_image, cv::Point(0, principle_matrix.y), cv::Point(cylindrical_image.size().width, principle_matrix.y), cv::Scalar(0, 0, 255));
        cv::imwrite("samples/cylin_res/test_byd_10.png",cylindrical_image);

        cv::imshow("cylindrical_image",cylindrical_image);
        cv::waitKey(0); 
    }
    
}