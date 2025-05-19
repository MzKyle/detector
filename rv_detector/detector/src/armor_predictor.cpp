#include "armor_predictor.Zhpp"

#include <spdlog/spdlog.h>

#include <execution>
#include <string>

#include "Eigen/Dense"
#include "detector.hpp"
#include "ekf.hpp"

using namespace rm_auto_aim;

Eigen::Matrix<double, 3, 1> ConvertFromCoordToEuler(
    Eigen::Matrix<double, 5, 1> coord) {
  Eigen::Matrix<double, 3, 1> result;
  // pitch
  result[0] = ceres::atan2(
      coord[4], ceres::sqrt(coord[0] * coord[0] + coord[2] * coord[2]));
  // yaw
  result[1] = ceres::atan2(coord[2], coord[0]);
  // distance
  result[2] = ceres::sqrt(coord[0] * coord[0] + coord[2] * coord[2] +
                          coord[4] * coord[4]);
  return result;
}

const cv::Mat LinerModel(const cv::Mat x0, double delta_t) {
  Eigen::Matrix<double, 5, 1> x;
  cv::Mat x1;
  cv::eigen2cv(x, x1);
  x1.at<double>(0) = x0.at<double>(0) + delta_t * x0.at<double>(1);  // 0.1
  x1.at<double>(1) = x0.at<double>(1);                               // 100
  x1.at<double>(2) = x0.at<double>(2) + delta_t * x0.at<double>(3);  // 0.1
  x1.at<double>(3) = x0.at<double>(3);                               // 100
  x1.at<double>(4) = x0.at<double>(4);                               // 0.01
  SPDLOG_DEBUG("LinerModel");
  return x1;
}

cv::Mat ArmorPredictor::ArmorCoord() {
  duration_predict_.Start();
  if (!armors_.empty()) {
    armor_ = armors_.front();

    /* 获取世界坐标并进行ekf预测 */
    auto coord = armor_.tvec;
    double x_pos = coord.at<double>(0, 0);
    double y_pos = coord.at<double>(1, 0);
    double z_pos = coord.at<double>(2, 0);
    std::cout << coord << "\n";
    Eigen::Matrix<double, 5, 1> Xr;
    Eigen::Matrix<double, 3, 1> Yr;
    Xr << x_pos, 0, y_pos, 0, z_pos;
    Yr << x_pos, y_pos, z_pos;
    cv::Mat measurement, Xp;
    cv::eigen2cv(Xr, measurement);
    ekf_.Config(Xr, 10);
    // auto Yr = ConvertFromCoordToEuler(Xr);
    ekf_.Predict(Xr);
    auto Xe = ekf_.Update(Yr);

    double predict_time =
        std::sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos) / speed_ /
        1000;
    Xp = LinerModel(Xe, predict_time);
    std::cout << Xp << "\n";

    // TODO(MC): 从世界坐标系转换到图像坐标系,在图像中显示
    return Xp;
  }
  duration_predict_.Calc("Predict Armor");
}

ArmorPredictor::ArmorPredictor() { SPDLOG_TRACE("Constructed."); }

ArmorPredictor::ArmorPredictor(std::vector<Armor> armors) {
  armors_ = armors;
  SPDLOG_TRACE("Constructed.");
}

ArmorPredictor::~ArmorPredictor() { SPDLOG_TRACE("Destructed."); }

const std::vector<Armor> &ArmorPredictor::Predict() {
  predicts_.clear();
  cv::Mat Xp = ArmorCoord();   // armor coord predict
  armor_.SetPredictCoord(Xp);  // set the predict point world
  predicts_.emplace_back(armor_);
  return predicts_;
}
