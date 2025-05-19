#include "compensator.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace
{

  double yaw;


  // 重力加速度
  const double kG = 9.7945;

  // 更改枪口到镜头的y值，影响pitch轴的计算，关闭重力补偿下近距离打中，远距离偏低，说明解算没问题
  // 实际测量单位为m， 不准的话以上述从近到远调出的参数为准

  const double kINFANTRY = 0.046;//老步
  //const double kINFANTRY = -0.  //舵步
  const double kHERO = 0.071;
  const double kSENTRY = -0.05;//镜头在枪管上方
  const double kBIG_ARMOR = 230. / 127 * cos(15. / 180 * M_PI);
  const double kSMALL_ARMOR = 135. / 125 * cos(15. / 180 * M_PI);
  const int kIMAGE_HEIGHT = 480;
  const int kIMAGE_WIDTH = 640;

} // namespace

Compensator::Compensator() { SPDLOG_TRACE("Constructed."); }

Compensator::Compensator(const game::Arm &arm, cv::Mat cam_mat,
                         cv::Mat distor_coff)
{
  SPDLOG_TRACE("Constructed.");
  SetArm(arm);
  cam_mat_ = cam_mat;
  distor_coff_ = distor_coff;
}

Compensator::~Compensator() { SPDLOG_TRACE("Destructed."); }

/*
@brief 设置相机与枪口的距离，兵种不同参数不同
@param game::Arm 传入机器人类型
*/
void Compensator::SetArm(const game::Arm &arm)
{
  arm_ = arm;
  if (arm == game::Arm::kINFANTRY)
    gun_cam_distance_ = kINFANTRY;
  else if (arm == game::Arm::kHERO)
    gun_cam_distance_ = kHERO;
  else if (arm == game::Arm::kSENTRY)
    gun_cam_distance_ = kSENTRY;
}

void Compensator::SolveAngles(Armor &armor, const component::Euler &euler)
{
  component::Euler aiming_eulr;
  double x_pos, y_pos, z_pos;

  // 预测不准这里写死is_predict用Follow
  if (is_predict_)
  { // Predict
    SPDLOG_CRITICAL("coord_pre:{}, {}, {}", armor.coord_pre.at<double>(0, 0),
                    armor.coord_pre.at<double>(1, 0),
                    armor.coord_pre.at<double>(2, 0));
    x_pos = armor.coord_pre.at<double>(0, 0);
    y_pos = armor.coord_pre.at<double>(1, 0) - gun_cam_distance_;
    // y_pos = armor.coord_pre.at<double>(1, 0) + gun_cam_distance_;  //相机在上面的兵种用这个
    z_pos = armor.coord_pre.at<double>(2, 0);
  }
  else
  { // Follow
    x_pos = armor.tvec.at<double>(0, 0);
    y_pos = armor.tvec.at<double>(1, 0) - gun_cam_distance_;
    // y_pos = armor.tvec.at<double>(1, 0) + gun_cam_distance_;  //相机在上面的兵种用这个
    z_pos = armor.tvec.at<double>(2, 0);
  }
  SPDLOG_WARN("x : {}, y : {}, z : {} ", x_pos, y_pos, z_pos);
  SPDLOG_INFO("initial pitch : {}, initial yaw : {}", euler.pitch, euler.yaw);

  // 单位为m
  distance_ = sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos) / 1.;
  SPDLOG_INFO("distance:{}", distance_);

  if (distance_ > 100) // 这个参数可以改动，用于消除相机畸变
  {
    // PinHoleSolver
    double ax = cam_mat_.at<double>(0, 0);
    double ay = cam_mat_.at<double>(1, 1);
    double u0 = cam_mat_.at<double>(0, 2);
    double v0 = cam_mat_.at<double>(1, 2);

    std::vector<cv::Point2f> out;
    cv::undistortPoints(std::vector<cv::Point2f>{armor.center}, out, cam_mat_,
                        distor_coff_, cv::noArray(), cam_mat_);
    aiming_eulr.pitch = atan((out.front().y - v0) / ay);
    aiming_eulr.yaw = -atan((out.front().x - u0) / ax);
  }
  else
  {
    // P4PSolver
    aiming_eulr.pitch = -atan(y_pos / sqrt(x_pos * x_pos + z_pos * z_pos));
    aiming_eulr.yaw = -atan(x_pos / z_pos);
  }

  // 没有重力补偿的情况下的pitch yaw解算补偿
  SPDLOG_INFO("compensator pitch : {}", aiming_eulr.pitch);
  SPDLOG_INFO("compensator yaw : {}", aiming_eulr.yaw);

  aiming_eulr.pitch = aiming_eulr.pitch + euler.pitch;
  aiming_eulr.yaw = aiming_eulr.yaw + euler.yaw;

  // 经过解算补偿计算后的无重力补偿角度
  SPDLOG_INFO("final pitch : {}", aiming_eulr.pitch);
  SPDLOG_INFO("final yaw : {}", aiming_eulr.yaw);
  armor.SetAimEuler(aiming_eulr);
}

void Compensator::Apply(std::vector<Armor> &armors, const double ballet_speed,
                        const component::Euler &euler, game::AimMethod method)
{
  auto &armor = armors.front();
  // if (armor.GetModel() == game::Model::kUNKNOWN) {
  //   armor.SetModel(game::Model::kINFANTRY);
  //   SPDLOG_ERROR("Hasn't set model.");
  // }
  SolveAngles(armor, euler);
  CompensateGravity(armor, ballet_speed, method);
}

void Compensator::Apply(Armor &armor, const double ballet_speed,
                        const component::Euler &euler, game::AimMethod method,
                        bool is_predict)
{
  // 测试解算和重力补偿时下面代码没有用处，上场时开启，没有定义函数的话就从qdu-rm-ai复制粘贴

  // if (armor.GetModel() == game::Model::kUNKNOWN) {
  //   armor.SetModel(game::Model::kINFANTRY);
  //   SPDLOG_ERROR("Hasn't set model.");
  // }
  // is_predict_ = is_predict;
  is_predict_ = 0;
  if (is_predict_)
    SPDLOG_WARN("Predict Model!!!!!!!");
  else
    SPDLOG_WARN("Follow Model!!!!!!!!!");
  SolveAngles(armor, euler);
  CompensateGravity(armor, ballet_speed, method);
}



void Compensator::CompensateGravity(Armor& armor, const double ballet_speed,
                                    game::AimMethod method) {
  if (method == game::AimMethod::kARMOR || method == game::AimMethod::kBUFF) {

    component::Euler aiming_eulr = armor.GetAimEuler();
    double angle = aiming_eulr.pitch;
    SPDLOG_INFO("The speed of ballet is {}",ballet_speed);

      double target_y = distance_ * sin(angle);
      double temp_y = target_y;
      double x = distance_ * cos(angle);   // 敌方和己方的水平距离
      double k1;   // 用于计算空气阻力模型中的一个系数，在不同的距离下这个空气阻力系数也会不同，弹丸不同也会不同
          // 动态调整空气阻力系数k1基于子弹速度
      if (ballet_speed < 15) {
          k1 = 0.13;  // 速度小于15m/s
      } else if (ballet_speed < 18) {
          k1 = 0.05; // 速度在15m/s到18m/s之间
      } else if (ballet_speed < 22) {
          k1 = 0.01; // 速度在18m/s到22m/s之间
      } else {
          k1 = 0.005; // 速度超过22m/s
      }
      for (int i = 0; i < 10; i++) {
          double vy = ballet_speed * sin(angle);  //  竖直方向的初速度
          double vx = ballet_speed * cos(angle);
          double t = (exp(x*k1)-1)/(k1*vx);
          double real_y = vy * t - kG * t * t / 2;
          double deltaH = target_y - real_y;
          temp_y += deltaH;
          double x_pos = armor.tvec.at<double>(0, 0);
          double z_pos = armor.tvec.at<double>(2, 0);
          angle = atan(temp_y / sqrt(x_pos * x_pos + z_pos * z_pos));
      }
      double x_pos = armor.tvec.at<double>(0, 0);
      double z_pos = armor.tvec.at<double>(2, 0);
      angle =  atan( (temp_y - kHERO ) / sqrt(x_pos * x_pos + z_pos * z_pos));//注意修改KHERO
      SPDLOG_INFO("aim angle is {}" , aiming_eulr.pitch);
      SPDLOG_INFO("angle is{}" , angle);
      SPDLOG_INFO("diff is {}", angle-aiming_eulr.pitch);
      aiming_eulr.pitch = angle;
      armor.SetAimEuler(aiming_eulr);
      SPDLOG_DEBUG("Armor Euler is setted");
    }
  }

bool Compensator::ArmorReProject(Armor &armor, cv::Mat &output)
{
  std::vector<cv::Point2f> out_points_predict;
  std::vector<cv::Point2f> out_points_follow;

  // 装甲板中心是世界坐标系原点，即（0, 0, 0），用于排查相机参数误差问题
  cv::Point3f center(0, 0, 0);
  std::vector<cv::Point3f> center_vec;
  center_vec.emplace_back(center);

  // Yellow Predict
  cv::projectPoints(armor.coord_reprojection, armor.rvec, armor.tvec, cam_mat_,
                    distor_coff_, out_points_predict);
  cv::circle(output, out_points_predict[0], 6, draw::kYELLOW, 5);

  // Red Follow
  cv::projectPoints(center_vec, armor.rvec, armor.tvec, cam_mat_, distor_coff_,
                    out_points_follow);
  cv::circle(output, out_points_follow[0], 3, draw::kRED, -1);

  double difference = cv::norm(out_points_predict[0] - out_points_follow[0]);
  SPDLOG_CRITICAL("Predict and Follow diff:{}", difference);
  if (difference > 28)
    return false;
  // std::string label = cv::format(
  //     "point world(x,y,z) is %lf, %lf, %lf", armor.coord_pre.at<double>(0,
  //     0), armor.coord_pre.at<double>(1, 0), armor.coord_pre.at<double>(2,
  //     0));
  // draw::VisualizeLabel(output, label, 3, draw::kYELLOW);
  return true;
}
void Compensator::ReadJson(const std::string &jsonFilePath) {
    cv::FileStorage fs(jsonFilePath, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

    if (!fs.isOpened()) {
        std::cerr << "Error: Could not open the JSON file: " << jsonFilePath << std::endl;
        return;
    }

    fs["yaw"] >> yaw;

    std::cout << "Yaw: " << yaw << std::endl;
    fs.release();
}


#ifdef RMU2021
/**
 * @brief Angle θ required to hit coordinate (x, y)
 *
 * {\displaystyle \tan \theta ={\left({\frac {v^{2}\pm {\sqrt
 * {v^{4}-g(gx^{2}+2yv^{2})}}}{gx}}\right)}}
 *
 * @param target 目标坐标
 * @return double 出射角度
 */
double Compensator::SolveSurfaceLanchAngle(cv::Point2f target,
                                           double ballet_speed)
{
  const double v_2 = pow(ballet_speed, 2);
  const double up_base =
      std::sqrt(std::pow(ballet_speed, 4) -
                kG * (kG * std::pow(target.x, 2) + 2 * target.y * v_2));
  const double low = kG * target.x;
  const double ans1 = std::atan2(v_2 + up_base, low);
  const double ans2 = std::atan2(v_2 - up_base, low);

  if (std::isnan(ans1))
    return std::isnan(ans2) ? 0. : ans2;
  if (std::isnan(ans2))
    return std::isnan(ans1) ? 0. : ans1;
  return std::min(ans1, ans2);
}

cv::Vec3f Compensator::EstimateWorldCoord(Armor &armor)
{
  cv::Mat rot_vec, trans_vec;
  cv::solvePnP(armor.PhysicVertices(), armor.ImageVertices(), cam_mat_,
               distor_coff_, rot_vec, trans_vec, false, cv::SOLVEPNP_ITERATIVE);
  armor.SetRotVec(rot_vec), armor.SetTransVec(trans_vec);
  cv::Mat world_coord =
      ((cv::Vec2f(armor.ImageCenter()) * cam_mat_.inv() - trans_vec) *
       armor.GetRotMat().inv());
  return cv::Vec3f(world_coord);
}
#endif
