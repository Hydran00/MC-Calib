// calibration.cpp
#include <Eigen/Dense>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <open3d/Open3D.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <thread>
#include <vector>

using namespace sl;
using namespace open3d;
using namespace std;

// Load relative transformation from YAML calibration file
Eigen::Matrix4d LoadCalibrationYAML(const std::string &filename) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    cerr << "Failed to open YAML file: " << filename << endl;
    return Eigen::Matrix4d::Identity();
  }

  cv::Mat pose0, pose1;
  fs["camera_0"]["camera_pose_matrix"] >> pose0;
  fs["camera_1"]["camera_pose_matrix"] >> pose1;
  fs.release();

  Eigen::Matrix4d eigen_pose0, eigen_pose1;
  cv::cv2eigen(pose0, eigen_pose0);
  cv::cv2eigen(pose1, eigen_pose1);

  return eigen_pose0.inverse() * eigen_pose1;
}

// Convert ZED XYZRGBA to Open3D PointCloud
std::shared_ptr<geometry::PointCloud> ZEDToOpen3D(Mat &mat) {
  int width = mat.getWidth();
  int height = mat.getHeight();

  auto cloud = make_shared<geometry::PointCloud>();
  cloud->points_.reserve(width * height);
  cloud->colors_.reserve(width * height);

  float *ptr = mat.getPtr<float>(MEM::CPU);

#pragma omp parallel
  {
    vector<Eigen::Vector3d> points_private;
    vector<Eigen::Vector3d> colors_private;
    points_private.reserve(width * height / omp_get_max_threads());
    colors_private.reserve(width * height / omp_get_max_threads());

#pragma omp for nowait
    for (int i = 0; i < width * height; i++) {
      float x = ptr[i * 4 + 0];
      float y = ptr[i * 4 + 1];
      float z = ptr[i * 4 + 2];
      float rgba_f = ptr[i * 4 + 3];

      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        continue;

      points_private.emplace_back(x, y, z);

      uint32_t rgba = *reinterpret_cast<uint32_t *>(&rgba_f);
      float r = ((rgba >> 0) & 0xFF) / 255.0f;
      float g = ((rgba >> 8) & 0xFF) / 255.0f;
      float b = ((rgba >> 16) & 0xFF) / 255.0f;
      colors_private.emplace_back(r, g, b);
    }

#pragma omp critical
    {
      cloud->points_.insert(cloud->points_.end(), points_private.begin(),
                            points_private.end());
      cloud->colors_.insert(cloud->colors_.end(), colors_private.begin(),
                            colors_private.end());
    }
  }

  return cloud;
}

// Preprocess point cloud for ICP
std::shared_ptr<geometry::PointCloud>
preprocess_point_cloud(const std::shared_ptr<geometry::PointCloud> &pcd,
                       double voxel_size) {
  auto pcd_down = pcd->VoxelDownSample(voxel_size);
  pcd_down->EstimateNormals(
      geometry::KDTreeSearchParamHybrid(voxel_size * 2, 30));
  auto pcd_fpfh = pipelines::registration::ComputeFPFHFeature(
      *pcd_down, geometry::KDTreeSearchParamHybrid(voxel_size * 5, 100));
  return pcd_down;
}

// Point-to-plane ICP on two point clouds
Eigen::Matrix4d
PerformICP(std::shared_ptr<geometry::PointCloud> source,
           std::shared_ptr<geometry::PointCloud> target,
           Eigen::Matrix4d initial_guess = Eigen::Matrix4d::Identity(),
           double threshold = 0.01) {
  // Preprocess point clouds
  auto source_prep = preprocess_point_cloud(source, threshold);
  auto target_prep = preprocess_point_cloud(target, threshold);
  // Create a Tukey robust kernel with scale 0.1
  auto tukey_kernel =
      std::make_shared<open3d::pipelines::registration::TukeyLoss>(0.1);

  // Pass the shared pointer to the point-to-plane estimator
  auto estimation =
      open3d::pipelines::registration::TransformationEstimationPointToPlane(
          tukey_kernel);
  //   auto result = open3d::pipelines::registration::RegistrationColoredICP(
  //       *source_prep, *target_prep, 0.1, Eigen::Matrix4d::Identity(),
  //       open3d::pipelines::registration::TransformationEstimationForColoredICP(
  //           0.968, tukey_kernel));
  auto result = open3d::pipelines::registration::RegistrationICP(
      *source_prep, *target_prep, 0.05, initial_guess, estimation);
  return result.transformation_;
}

// Average N transformations
Eigen::Matrix4d
AverageTransformations(const std::vector<Eigen::Matrix4d> &transforms) {
  if (transforms.empty())
    return Eigen::Matrix4d::Identity();

  Eigen::Vector3d avg_t = Eigen::Vector3d::Zero();
  std::vector<Eigen::Quaterniond> quats;

  for (const auto &T : transforms) {
    avg_t += T.block<3, 1>(0, 3);
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    quats.emplace_back(Eigen::Quaterniond(R));
  }
  avg_t /= transforms.size();

  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  for (auto &q : quats) {
    Eigen::Vector4d vec(q.w(), q.x(), q.y(), q.z());
    A += vec * vec.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(A);
  Eigen::Vector4d avg_q_vec = solver.eigenvectors().col(3);
  Eigen::Quaterniond avg_q(avg_q_vec(0), avg_q_vec(1), avg_q_vec(2),
                           avg_q_vec(3));
  avg_q.normalize();

  Eigen::Matrix4d avg_T = Eigen::Matrix4d::Identity();
  avg_T.block<3, 3>(0, 0) = avg_q.toRotationMatrix();
  avg_T.block<3, 1>(0, 3) = avg_t;
  return avg_T;
}

// Save transformation to YAML
void SaveTransform(const std::string &filename, const Eigen::Matrix4d &T) {
  cv::Mat cv_T;
  cv::eigen2cv(T, cv_T);
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  fs << "transform" << cv_T;
  fs.release();
  cout << "Saved averaged transform to: " << filename << endl;
}

int main(int argc, char **argv) {
  std::string calib_file = (argc > 1) ? argv[1] : "";
  if (calib_file.empty()) {
    cerr << "Usage: " << argv[0] << " <calibration_file.yaml>" << endl;
    return -1;
  }
  cout << "Loading init calibration from: " << calib_file << endl;
  Eigen::Matrix4d T_init = LoadCalibrationYAML(calib_file);
  std::cout << "Initial guess:\n" << T_init << std::endl;

  const int N = 5; // number of first clouds to average
  Camera zed1, zed2;
  InitParameters init_params;
  init_params.camera_resolution = RESOLUTION::HD2K;
  init_params.coordinate_system = COORDINATE_SYSTEM::IMAGE;
  init_params.coordinate_units = UNIT::METER;
  init_params.depth_mode = DEPTH_MODE::NEURAL_PLUS;
  init_params.depth_maximum_distance = 2.0f;

  if (zed1.open(init_params) != ERROR_CODE::SUCCESS ||
      zed2.open(init_params) != ERROR_CODE::SUCCESS) {
    cerr << "Failed to open ZED cameras." << endl;
    return -1;
  }

  Mat pc1_mat, pc2_mat;
  RuntimeParameters runtime_params;

  std::vector<std::shared_ptr<geometry::PointCloud>> first_clouds;
  std::vector<Eigen::Matrix4d> icp_results;
  mutex cloud_mutex;

  cout << "Capturing first " << N << " frames for calibration..." << endl;
  int i = 0;
  while (i < N) {
    if (zed1.grab(runtime_params) == ERROR_CODE::SUCCESS &&
        zed2.grab(runtime_params) == ERROR_CODE::SUCCESS) {

      zed1.retrieveMeasure(pc1_mat, MEASURE::XYZRGBA);
      zed2.retrieveMeasure(pc2_mat, MEASURE::XYZRGBA);

      auto pc1 = ZEDToOpen3D(pc1_mat);
      auto pc2 = ZEDToOpen3D(pc2_mat);

      lock_guard<mutex> lock(cloud_mutex);
      if (!pc1->points_.empty() && !pc2->points_.empty()) {
        pc2->Transform(T_init);
        open3d::visualization::DrawGeometries({pc1, pc2},
                                              "Captured Point Clouds");
        // filter outliers
        pc1->RemoveStatisticalOutliers(40, 2.0);
        pc2->RemoveStatisticalOutliers(40, 2.0);
        Eigen::Matrix4d T_icp = PerformICP(pc1, pc2);
        pc2->Transform(T_icp);
        icp_results.push_back(T_icp);
        open3d::visualization::DrawGeometries({pc1, pc2},
                                              "ICP Result Point Clouds");
        std::cout << "ICP result for frame " << first_clouds.size() + 1
                  << ": \n"
                  << T_icp << std::endl;
        std::cout << "Captured frame " << i + 1 << endl;
        i++;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  Eigen::Matrix4d T_avg =
      icp_results[0]; // AverageTransformations(icp_results);
  //   std::cout << "Averaged transformation:\n" << T_avg << std::endl;
  std::cout << "Final transformation:\n" << T_avg * T_init << std::endl;
  // Stack transformation
  SaveTransform("calibration_transform.yaml", T_avg * T_init);

  zed1.close();
  zed2.close();
  return 0;
}
