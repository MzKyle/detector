#pragma once

#include "armor.hpp"
#include "common.hpp"
#include "opencv2/opencv.hpp"
#include "tbb/concurrent_vector.h"

using namespace rm_auto_aim;

class Compensator {
 private:
  double distance_;
  cv::Mat cam_mat_, distor_coff_;
  double gun_cam_distance_; /* 枪口到镜头的距离 */
  game::Arm arm_;
  game::AimMethod method_;
  double real_img_ratio_;
  bool is_predict_; //TODO:
  void SolveAngles(Armor& armor, const component::Euler& euler);
  void CompensateGravity(Armor& armor, const double ballet_speed,
                         game::AimMethod method);

 public:
  Compensator();
  Compensator(const game::Arm& arm, cv::Mat cam_mat, cv::Mat distor_coff);
  ~Compensator();

  void SetArm(const game::Arm& arm);

  void Apply(std::vector<Armor>& armors, const double ballet_speed,
             const component::Euler& euler, game::AimMethod method);

  void Apply(Armor& armor, const double ballet_speed,
             const component::Euler& euler, game::AimMethod method,
             bool is_predict);

  void VisualizeResult(tbb::concurrent_vector<Armor>& armors,
                       const cv::Mat& output, int verbose = 1);
  double pitchTrajectoryCompensation(double s, double z, double v);
  double monoDirectionalAirResistanceModel(double s, double v, double angle);
  bool ArmorReProject(Armor& armor, cv::Mat& output);
  void ReadJson(const std::string &jsonFilePath);
};

namespace draw {

const auto kCV_FONT = cv::FONT_HERSHEY_SIMPLEX;

const cv::Scalar kBLUE(255., 0., 0.);
const cv::Scalar kGREEN(0., 255., 0.);
const cv::Scalar kRED(0., 0., 255.);
const cv::Scalar kYELLOW(0., 255., 255.);
const cv::Scalar kBLACK(0., 0., 0.);

void VisualizeLabel(const cv::Mat& output, const std::string& label,
                    int level = 1, const cv::Scalar& color = kGREEN);

}  // namespace draw