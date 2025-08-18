#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <open3d/Open3D.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

using namespace sl;
using namespace open3d;
using namespace std;

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

    auto cloud = std::make_shared<geometry::PointCloud>();
    cloud->points_.reserve(width * height);
    cloud->colors_.reserve(width * height);

    float *ptr = mat.getPtr<float>(MEM::CPU);
#pragma omp parallel
    for (int i = 0; i < width * height; i++) {
        float x = ptr[i * 4 + 0];
        float y = ptr[i * 4 + 1];
        float z = ptr[i * 4 + 2];
        float rgba_f = ptr[i * 4 + 3];

        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            continue;

        uint32_t rgba = *reinterpret_cast<uint32_t *>(&rgba_f);
        float r = ((rgba >> 0) & 0xFF) / 255.0f;
        float g = ((rgba >> 8) & 0xFF) / 255.0f;
        float b = ((rgba >> 16) & 0xFF) / 255.0f;

#pragma omp critical
        {
            cloud->points_.emplace_back(x, y, z);
            cloud->colors_.emplace_back(r, g, b);
        }
    }
    return cloud;
}

int main(int argc, char **argv) {
    string calib_file = (argc > 1) ? argv[1] : "calibration.yaml";
    cout << "Loading calibration from: " << calib_file << endl;
    Eigen::Matrix4d T = LoadCalibrationYAML(calib_file);
    cout<< "Transformation matrix:\n" << T << endl;
    Camera zed1, zed2;
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD2K;
    init_params.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP;
    init_params.coordinate_units = UNIT::METER;
    init_params.depth_mode = DEPTH_MODE::NEURAL;

    if (zed1.open(init_params) != ERROR_CODE::SUCCESS ||
        zed2.open(init_params) != ERROR_CODE::SUCCESS) {
        cerr << "Failed to open ZED cameras." << endl;
        return -1;
    }

    cout << "ZED cameras opened!" << endl;

    Mat pc1_mat, pc2_mat;
    RuntimeParameters runtime_params;

    auto cloud_combined = make_shared<geometry::PointCloud>();
    mutex cloud_mutex;

    // Background thread: grab frames and convert
    thread capture_thread([&]() {
        while (true) {
            if (zed1.grab(runtime_params) == ERROR_CODE::SUCCESS &&
                zed2.grab(runtime_params) == ERROR_CODE::SUCCESS) {

                zed1.retrieveMeasure(pc1_mat, MEASURE::XYZRGBA);
                zed2.retrieveMeasure(pc2_mat, MEASURE::XYZRGBA);

                auto pc1 = ZEDToOpen3D(pc1_mat);
                auto pc2 = ZEDToOpen3D(pc2_mat);
                pc1->Transform(T);

                // Merge
                auto merged = make_shared<geometry::PointCloud>();
                merged->points_ = pc1->points_;
                merged->colors_ = pc1->colors_;
                merged->points_.insert(merged->points_.end(),
                                       pc2->points_.begin(), pc2->points_.end());
                merged->colors_.insert(merged->colors_.end(),
                                       pc2->colors_.begin(), pc2->colors_.end());

                lock_guard<mutex> lock(cloud_mutex);
                *cloud_combined = *merged;
            }
            this_thread::sleep_for(chrono::milliseconds(10));
        }
    });

    visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Merged Point Clouds", 1280, 720);
    auto reference_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.5);
    vis.AddGeometry(reference_frame);
    vis.AddGeometry(cloud_combined);

    // Main thread: visualization
    while (true) {
        {
            lock_guard<mutex> lock(cloud_mutex);
            vis.UpdateGeometry(cloud_combined);
        }
        vis.PollEvents();
        vis.UpdateRender();
        this_thread::sleep_for(chrono::milliseconds(10));
    }

    capture_thread.join();
    zed1.close();
    zed2.close();
    vis.DestroyVisualizerWindow();
    return 0;
}
