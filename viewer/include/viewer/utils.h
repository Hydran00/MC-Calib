#include "yolov8_seg.h"
#include <Eigen/Dense>
#include <chrono>
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

// Load relative transformations from YAML calibration file for n cameras
vector<Eigen::Matrix4d> LoadCalibrationYAML(const string &filename, int n) {
  vector<Eigen::Matrix4d> Ts(n, Eigen::Matrix4d::Identity());
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    cerr << "Failed to open YAML file: " << filename << endl;
    return Ts;
  }

  cv::Mat pose;
  for (int i = 0; i < n; i++) {
    string cam_key = "camera_" + to_string(i);
    fs[cam_key]["camera_pose_matrix"] >> pose;
    Eigen::Matrix4d eigen_pose;
    cv::cv2eigen(pose, eigen_pose);
    Ts[i] = eigen_pose;
  }

  fs.release();
  return Ts;
}
// Convert ZED XYZRGBA to Open3D PointCloud
void SavePointClouds(const vector<shared_ptr<geometry::PointCloud>> &pcs,
                     const shared_ptr<geometry::PointCloud> &merged) {
  io::WritePointCloud("merged_point_cloud.ply", *merged);
  for (int i = 0; i < pcs.size(); i++)
    io::WritePointCloud("pc" + to_string(i) + ".ply", *pcs[i]);
  cout << "Saved first frame clouds!" << endl;
}
// Merge multiple point clouds into one
shared_ptr<geometry::PointCloud> MergePointClouds(
    const vector<shared_ptr<geometry::PointCloud>> &pcs) {
    auto merged = make_shared<geometry::PointCloud>();
    for (auto &pc : pcs) {
        merged->points_.insert(merged->points_.end(),
                               pc->points_.begin(), pc->points_.end());
        merged->colors_.insert(merged->colors_.end(),
                               pc->colors_.begin(), pc->colors_.end());
    }
    return merged;
}
// Convert ZED XYZRGBA to Open3D PointCloud using bbox+mask
shared_ptr<geometry::PointCloud> ZEDToOpen3D(sl::Mat &pointcloud,
                                             cv::Mat &image,
                                             const cv::Rect &bbox,
                                             const cv::Mat &mask) {
  int width = pointcloud.getWidth();
  int height = pointcloud.getHeight();
  auto cloud = make_shared<geometry::PointCloud>();

  float *ptr = pointcloud.getPtr<float>(MEM::CPU);
  for (int y = bbox.y; y < bbox.y + bbox.height; y++) {
    for (int x = bbox.x; x < bbox.x + bbox.width; x++) {
      if (y >= height || x >= width)
        continue;

      // skip if outside YOLO mask
      if (mask.empty() || mask.at<uchar>(y - bbox.y, x - bbox.x) == 0) {
        continue;
      }
      // set pixel color for visualization
      cv::Vec3b &pixel = image.at<cv::Vec3b>(y, x);
      pixel[2] = 255;

      int idx = (y * width + x) * 4;
      float X = ptr[idx + 0];
      float Y = ptr[idx + 1];
      float Z = ptr[idx + 2];
      float rgba_f = ptr[idx + 3];

      if (!isfinite(X) || !isfinite(Y) || !isfinite(Z))
        continue;

      cloud->points_.emplace_back(X, Y, Z);

      uint32_t rgba = *reinterpret_cast<uint32_t *>(&rgba_f);
      float r = ((rgba >> 0) & 0xFF) / 255.0f;
      float g = ((rgba >> 8) & 0xFF) / 255.0f;
      float b = ((rgba >> 16) & 0xFF) / 255.0f;
      cloud->colors_.emplace_back(r, g, b);
    }
  }

  return cloud;
}

void show_resized_img(cv::Mat &img, float scale, std::string name) {
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(), scale, scale);
  cv::imshow(name, resized);
  cv::waitKey(1);
}
