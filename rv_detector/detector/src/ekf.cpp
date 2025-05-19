#include "ekf.hpp"

#include <cmath>
#include <vector>

#include "spdlog/spdlog.h"

std::vector<ceres::Jet<double, 5>> LinerModel(
    const std::vector<ceres::Jet<double, 5>> x0, double delta_t)
{
  std::vector<ceres::Jet<double, 5>> x1(5);
  x1[0] = x0[0] + delta_t * x0[1]; // 0.1
  x1[1] = x0[1];                   // 100
  x1[2] = x0[2] + delta_t * x0[3]; // 0.1
  x1[3] = x0[3];                   // 100
  x1[4] = x0[4];                   // 0.01
  SPDLOG_DEBUG("LinerModel");
  return x1;
}

std::vector<ceres::Jet<double, 5>> ConvertFromCoordToEuler(
    const std::vector<ceres::Jet<double, 5>> coord)
{
  std::vector<ceres::Jet<double, 5>> result(3);
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

void EKF::InnerInit(const Matx51d &Xe)
{
  start_frame_ = 0;
  error_frame_ = 0;
  cv::cv2eigen(Xe, EXe);

  // 调整卡尔曼滤波需要调整下面的矩阵，但我不太懂
  EP = EMatx55d::Identity();
  EQ = EMatx55d::Identity();
  ER = EMatx33d::Identity();
}

EKF::EKF(const Matx51d &Xe) { InnerInit(Xe); }

EKF::~EKF() { SPDLOG_TRACE("Destruted."); }

void EKF::Init(const std::vector<double> &vec)
{
  if (method_ == component::FilterMethod::kUNKNOWN)
    method_ = component::FilterMethod::kEKF;
  InnerInit(Matx51d(vec[0], vec[1], vec[2], vec[3], vec[4]));
}

const cv::Mat &EKF::Predict(const EMatx51d &measurements)
{
  std::vector<ceres::Jet<double, 5>> EXe_auto_jet(5), EXp_auto_jet(5);

  for (int i = 0; i < 5; i++)
  {
    EXe[i] = measurements(i, 0);
    EXe_auto_jet[i].a = EXe[i];
    EXe_auto_jet[i].v[i] = 1;
  }
  EXp_auto_jet = LinerModel(EXe_auto_jet, delta_t_);
  for (int i = 0; i < 5; i++)
  {
    EXp[i] = EXp_auto_jet[i].a;
    EF.block(i, 0, 1, 5) = EXp_auto_jet[i].v.transpose();
  }

  EP = EF * EP * EF.transpose() + EQ;
  cv::eigen2cv(EXp, Xp);
  SPDLOG_DEBUG("Predicted");

  return Xp;
}

const cv::Mat &EKF::Update(const EMatx31d &measurements)
{
  std::vector<ceres::Jet<double, 5>> EXp_auto_jet(5), EYp_auto_jet(3);

  EMatx31d Y = measurements;

  for (int i = 0; i < 5; i++)
  {
    EXp_auto_jet[i].a = EXp[i];
    EXp_auto_jet[i].v[i] = 1;
  }
  // EYp_auto_jet = ConvertFromCoordToEuler(EXp_auto_jet);
  for (int i = 0; i < 3; i++)
  {
    EYp_auto_jet[i] = EXp_auto_jet[i * 2]; // 5 to 3 已经转换成Euler的情况下不用加这行
    EYp[i] = EYp_auto_jet[i].a;
    EH.block(i, 0, 1, 5) = EYp_auto_jet[i].v.transpose();
  }
  EK = EP * EH.transpose() * (EH * EP * EH.transpose() + ER).inverse();
  EXe = EXp + EK * (Y - EYp);
  EP = (EMatx55d::Identity() - EK * EH) * EP;
  cv::eigen2cv(EXe, Xe);

  SPDLOG_DEBUG("Updated");
  return Xe;
}

/*
@brief 配置ekf滤波器
@param measurements 传入x vx y vy z
@param frame_count 多少帧更新一次速度，可调整跟踪速度
@param kT 时间常数，用于计算速度
*/
bool EKF::Config(EMatx51d &measurements, unsigned int frame_count, double kT)
{
  // 配置速度以实际反投影的效果决定，这里一开始xy轴是反的
  double y = measurements(0, 0);
  double x = measurements(2, 0);

  // if (error_frame_ > 20) {
  //   start_frame_ = 0;
  //   error_frame_ = 0;
  //   coords_.clear();
  // }

  cv::Point2d pt(x, y);
  if (start_frame_ < frame_count)
  {
    coords_.emplace_back(pt);
    start_frame_++;
    return false;
  }
  else
  {
    coords_.emplace_back(pt);
    measurements(1, 0) =
        -(coords_[start_frame_ - 1].x - coords_[start_frame_ - frame_count].x) /
        kT;
    measurements(3, 0) =
        -(coords_[start_frame_ - 1].y - coords_[start_frame_ - frame_count].y) /
        kT;
    start_frame_++;
    SPDLOG_WARN("start_frame_:{}", start_frame_);
    if (start_frame_ >= 10000)
    {
      start_frame_ = 0;
      coords_.clear();
    }
    return true;
  }
}

void EKF::SetDeltaT(double delta_t) { delta_t_ = delta_t; }
