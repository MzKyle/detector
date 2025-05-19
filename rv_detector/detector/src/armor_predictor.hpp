#pragma once

#include <chrono>
#include <vector>

#include "common.hpp"
#include "detector.hpp"
#include "ekf.hpp"
#include "timer.hpp"

using namespace rm_auto_aim;

// TODO(MC) 1st: move compensator to predictor & world  to image
// TODO(MC) 2nd: complete antitop, auto-predictor and buff predictor
class ArmorPredictor {
 private:
  EKF ekf_;
  Armor armor_, last_armor_;
  std::vector<Armor> armors_;
  component::Timer duration_direction_, duration_predict_;
  bool same_armor_ = 0;
  double speed_;
  std::vector<Armor> predicts_;

  cv::Mat ArmorCoord();
  void ArmorProjection(cv::Mat Xp);

 public:
  ArmorPredictor();
  explicit ArmorPredictor(std::vector<Armor> armors);
  ~ArmorPredictor();

  void VisualizePrediction(const cv::Mat &output, int add_lable);
  const std::vector<Armor> &Predict();
};
