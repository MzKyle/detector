#pragma once

#include <vector>

#include "common.hpp"
#include "opencv2/opencv.hpp"

class Filter {
 private:
  bool predict_condition_;

 public:
  component::FilterMethod method_ = component::FilterMethod::kUNKNOWN;
  unsigned int measurements_, states_;
  std::vector<cv::Point2d> coords_;
  unsigned int start_frame_, error_frame_;
};
