
#include <iostream>

#include "armor_predictor.hpp"
#include "behavior.hpp"
#include "compensator.hpp"
#include "detector.hpp"
#include "ekf.hpp"
#include "hik_camera.hpp"
#include "robot.hpp"

using namespace rm_auto_aim;
std::string jsonFilePath = "/home/cy/rv_detector/detector/src/json/data.json";
std::string jsonTrackerFilePath="/home/cy/rv_detector/detector/src/json/scrollbar.json";
cv::Mat matrix;
cv::Mat coeffs;
int binary_thres;
int ballet_speed;
int detector_color;
double max_angle;
std::string serial_name;

void SaveConfigToJson(const std::string& filepath) {//将数据保存到json文件中
    cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "binary_thres" << binary_thres;
        fs << "ballet_speed" << ballet_speed;
        fs.release();
    } else {
        std::cerr << "Error: Unable to open file for writing: " << filepath << std::endl;
    }
}

void onTrackbarChange(int new_value, void* userdata) {//滑动条的回调函数
    std::string* variableName = reinterpret_cast<std::string*>(userdata);
    if (*variableName == "binary_thres") {
        binary_thres = new_value;
    } else if (*variableName == "ballet_speed") {
        ballet_speed = new_value;
    }
    SaveConfigToJson(jsonTrackerFilePath);
}

void ReadTrackerConfigFromJson(const std::string &jsonFilePath) {//从json文件中读取数据
    cv::FileStorage fs(jsonFilePath, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

    if (!fs.isOpened()) {
        std::cout << "Error: Could not open the JSON file: " << jsonTrackerFilePath << std::endl;
        return;
    }

    fs["binary_thres"] >> binary_thres;
    fs["ballet_speed"] >> ballet_speed;
    std::cout<<"binary_thres:"<<binary_thres<<std::endl;
    std::cout<<"ballet_speed:"<<ballet_speed<<std::endl;
    fs.release();
}
void ReadConfigFromJson(const std::string &jsonFilePath) {//从json文件中读取数据
    cv::FileStorage fs(jsonFilePath, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

    if (!fs.isOpened()) {
        std::cout << "Error: Could not open the JSON file: " << jsonFilePath << std::endl;
        return;
    }
    fs["detector_color"] >> detector_color;
    fs["max_angle"] >> max_angle;
    fs["serial_name"]>>serial_name;
    std::cout<<"detector_color:"<<detector_color<<std::endl;
    std::cout<<"max_angle:"<<max_angle<<std::endl;
    std::cout<<"serial_name:"<<serial_name<<std::endl;
    fs.release();
}


const int kIMAGE_HEIGHT = 480;
const int kIMAGE_WIDTH = 640;

float SMALL_ARMOR_WIDTH = 135;
float SMALL_ARMOR_HEIGHT = 55;
float LARGE_ARMOR_WIDTH = 225;
float LARGE_ARMOR_HEIGHT = 55;

double small_half_y = SMALL_ARMOR_WIDTH / 2.0 / 1000.0;
double small_half_z = SMALL_ARMOR_HEIGHT/ 2.0 / 1000.0;
double large_half_y = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
double large_half_z =  LARGE_ARMOR_HEIGHT/ 2.0 / 1000.0;

// Init camera_matrix and dist_coeffs
// Init camera_matrix and dist_coeffs
cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1.0218063113893762e+03,
              0,
              3.1950000000000000e+02,
              0,
              1.0218063113893762e+03,
              2.3950000000000000e+02,
              0,
              0,
              1.0);
cv::Mat dist_coeffs =
    (cv::Mat_<double>(5, 1) << -6.6213667569873166e-02,
            1.2059610363567468e-01,
            0,
            0,-7.3367662006313394e-01);

void usbrun()
{
  cv::Mat frame;
  cv::VideoCapture cap(0);

  while (true)
  {
    cap >> frame;
    cv::imshow("RESULT", frame);
    if ('q' == cv::waitKey(10))
    {
      break;
    }
  }
}

// -----------------------添加海康相机----------------------------------
HikCamera cam_;

void hik_prepare()
{
  cam_.Open(0);
  cam_.Setup(640, 480); // width height
}

// -----------------------SolvePnp-------------------------------------
void SolvePnp(std::vector<Armor> &armors, const cv::Mat camera_matrix_,
              const cv::Mat dist_coeffs_)
{
  // image center distance sort
  std::sort(armors.begin(), armors.end(), [](Armor &a, Armor &b)
            { return abs(a.center.x - kIMAGE_WIDTH / 2) <=
                     abs(b.center.x - kIMAGE_WIDTH / 2); });

  if (!armors.empty())
  {
    auto armor = armors.front();
    armors.clear();
    // Start from bottom left in clockwise order
    // Model coordinate: x forward, y left, z up
    std::vector<cv::Point3f> small_armor_points_;
    std::vector<cv::Point3f> large_armor_points_;
    small_armor_points_.emplace_back(
        cv::Point3f(0, small_half_y, -small_half_z));
    small_armor_points_.emplace_back(
        cv::Point3f(0, small_half_y, small_half_z));
    small_armor_points_.emplace_back(
        cv::Point3f(0, -small_half_y, small_half_z));
    small_armor_points_.emplace_back(
        cv::Point3f(0, -small_half_y, -small_half_z));

    large_armor_points_.emplace_back(
        cv::Point3f(0, large_half_y, -large_half_z));
    large_armor_points_.emplace_back(
        cv::Point3f(0, large_half_y, large_half_z));
    large_armor_points_.emplace_back(
        cv::Point3f(0, -large_half_y, large_half_z));
    large_armor_points_.emplace_back(
        cv::Point3f(0, -large_half_y, -large_half_z));

    armor.image_armor_points.emplace_back(armor.left_light.bottom);
    armor.image_armor_points.emplace_back(armor.left_light.top);
    armor.image_armor_points.emplace_back(armor.right_light.top);
    armor.image_armor_points.emplace_back(armor.right_light.bottom);

    auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_
                                                        : large_armor_points_;
    cv::solvePnP(object_points, armor.image_armor_points, camera_matrix_,
                 dist_coeffs_, armor.rvec, armor.tvec, false,
                 cv::SOLVEPNP_IPPE);
    armors.emplace_back(armor);
  }
  return;
}

// ------------------------Compensator-------------------------------------
Compensator compensator_(game::Arm::kINFANTRY, camera_matrix, dist_coeffs);


// -------------------------Robot------------------------------------------

// -------------------------Behaviour--------------------------------------
Behavior behavior_;

const cv::Mat LinerModel(const cv::Mat x0, double delta_t)
{
  Eigen::Matrix<double, 5, 1> x;
  cv::Mat x1;
  cv::eigen2cv(x, x1);
  x1.at<double>(0) = x0.at<double>(0) + delta_t * x0.at<double>(1); // 0.1
  x1.at<double>(1) = x0.at<double>(1);                              // 100
  x1.at<double>(2) = x0.at<double>(2) + delta_t * x0.at<double>(3); // 0.1
  x1.at<double>(3) = x0.at<double>(3);                              // 100
  x1.at<double>(4) = x0.at<double>(4);                              // 0.01
  SPDLOG_DEBUG("LinerModel");
  return x1;
}

int main()
{
  ReadConfigFromJson(jsonFilePath);
  ReadTrackerConfigFromJson(jsonTrackerFilePath);
  Detector rv;
  // rv.binary_thres = 160;
  rv.binary_thres = binary_thres;

  rv.detect_color = detector_color;
  rv.l.max_angle = max_angle;
  rv.l.max_ratio = 0.7;
  rv.l.min_ratio = 0.1;

  rv.a.min_light_ratio = 0.7;
  rv.a.min_small_center_distance = 0.7;
  rv.a.max_small_center_distance = 3.2;
  rv.a.min_large_center_distance = 3.2;
  rv.a.max_large_center_distance = 5.8;
  rv.a.max_angle = 70.0;
  compensator_.ReadJson(jsonFilePath);
  cv::Mat frame;
  cv::VideoCapture cap(0);

  auto model_path =
      "/home/cy/rv_detector/detector/model/mlp.onnx";
  auto label_path =
      "/home/cy/rv_detector/detector/model/label.txt";

  double threshold = 0.7;

  std::vector<std::string> ignore_classes = {"negative"};
  rv.classifier = std::make_unique<NumberClassifier>(model_path, label_path,
                                                     threshold, ignore_classes);

  // double alpha = 0.8;
  // cv::Mat darkframe;

  // 注释为原USB相机代码
  //  while (true)
  //  {
  //      cap >> frame;
  //      frame.convertTo(darkframe, -1, alpha, 0);
  //      auto armors = rv.detect(darkframe);
  //      rv.drawResults(darkframe);
  //      cv::imshow("RESULT", darkframe);
  //      cv::waitKey(1);
  //  }
  Robot robot_;
  EKF ekf_;
  bool is_last = 0;
  cv::Mat measurement, Xp, last_Xp;
  Eigen::Matrix<double, 5, 1> Xr;
  Eigen::Matrix<double, 3, 1> Yr;

  hik_prepare();
//-------------------滑动条---------------------
std::string window_handle_="params";
cv::namedWindow(window_handle_, 1);

int max_value = 200;  // 确保滑动条的最大值适合你的应用需求

cv::createTrackbar("binary_thres", window_handle_, &rv.binary_thres, max_value, onTrackbarChange, new std::string("binary_thres"));
cv::setTrackbarPos("binary_thres", window_handle_, rv.binary_thres);

cv::createTrackbar("ballet_speed", window_handle_, &ballet_speed, max_value, onTrackbarChange, new std::string("ballet_speed"));
cv::setTrackbarPos("ballet_speed", window_handle_, ballet_speed);

//--------------------------------------------
  //打开串口  命令行 ls /dev/ttyUSB*查看
 robot_.Init(serial_name);
  while (true)
  {

    if (!cam_.GetFrame(frame))
      continue;
    // frame.convertTo(darkframe, -1, alpha, 0);
    //  auto armors = rv.detect(darkframe);
    //  rv.drawResults(darkframe);
    //  cv::imshow("RESULT", darkframe);
    auto armors = rv.detect(frame);
    rv.drawResults(frame);

    if (!armors.empty())
    {
      // 选取距离中心最近的armor，并进行solvepnp解算
      SolvePnp(armors, camera_matrix, dist_coeffs);

      // Armor Predict
      auto armor_ = armors.front();
      /* 获取世界坐标并进行ekf预测，由于相机和装甲板都在运动，所以需要以一个静止的坐标系作参考（交给你们写了）
         这里应该是以陀螺仪（c板）的坐标系（上电后不再变换）为参考系，投影和反投影都需要多一层转换
         xyz坐标都需要严格查看单位，这里的单位为m
       */
      auto coord = armor_.tvec;
      double x_pos = coord.at<double>(0, 0);
      double y_pos = coord.at<double>(1, 0);
      double z_pos = coord.at<double>(2, 0);
      Xr << x_pos, 0, y_pos, 0, z_pos;
      Yr << x_pos, y_pos, z_pos;

      cv::eigen2cv(Xr, measurement);
      if (ekf_.Config(Xr, 10, 5) == true)
      {
        // auto Yr = ConvertFromCoordToEuler(Xr);
        ekf_.Predict(Xr);
        auto Xe = ekf_.Update(Yr);

        double predict_time =
            std::sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos) / 1.;
        SPDLOG_WARN("predict_time:{}", predict_time);
        Xp = LinerModel(Xe, predict_time);
        std::cout << "coord:" << coord << "\n";
        std::cout << "Xp:" << Xp << "\n";

        if (!is_last)
        {
          last_Xp = Xp;
          is_last = 1;
        }
        double norm = cv::norm(last_Xp - Xp);
        last_Xp = Xp;
        SPDLOG_WARN("----------Xp norm------------:{}", norm);
        if (norm > 3)
        {
          // ekf_.Init
        }

        // 设置反投影坐标与预测坐标
        armor_.SetPredictCoord(Xp);
        // 反投影到图像中，黄圈为Predict，红点为Follow
        bool is_predict = compensator_.ArmorReProject(armor_, frame);

        // 解算euler与重力补偿（调raw和pitch）
        compensator_.Apply(armor_, ballet_speed, robot_.GetEuler(),
                           game::AimMethod::kARMOR, is_predict);

        // // aim the armor
        behavior_.Aim(armor_.GetAimEuler());

        // // // 发送数据包到电控
       robot_.Pack(behavior_.GetData(), 9999);

        if (!is_predict)
        {
          ekf_.error_frame_++;
        }
      }
    }

    cv::imshow("RESULT", frame);
    cv::waitKey(1);
  }
}