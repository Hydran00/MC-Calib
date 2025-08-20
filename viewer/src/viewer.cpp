#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <open3d/Open3D.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <thread>
#include <mutex>
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

// Convert ZED XYZRGBA to Open3D PointCloud (same as before)
shared_ptr<geometry::PointCloud> ZEDToOpen3D(Mat &mat) {
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

            if (!isfinite(x) || !isfinite(y) || !isfinite(z))
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
            cloud->points_.insert(cloud->points_.end(),
                                  points_private.begin(), points_private.end());
            cloud->colors_.insert(cloud->colors_.end(),
                                  colors_private.begin(), colors_private.end());
        }
    }

    return cloud;
}

int main(int argc, char **argv) {
    string calib_file = (argc > 1) ? argv[1] : "calibration.yaml";
    int n_cameras = (argc > 2) ? stoi(argv[2]) : 2; // number of cameras
    cout << "Loading calibration from: " << calib_file << endl;
    auto Ts = LoadCalibrationYAML(calib_file, n_cameras);

    vector<Camera> zeds(n_cameras);
    vector<Mat> pc_mats(n_cameras);
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD2K;
    init_params.coordinate_system = COORDINATE_SYSTEM::IMAGE;
    init_params.coordinate_units = UNIT::METER;
    init_params.depth_mode = DEPTH_MODE::NEURAL_PLUS;
    init_params.depth_maximum_distance = 2.4f;

    // Open all cameras
    for (int i = 0; i < n_cameras; i++) {
        if (zeds[i].open(init_params) != ERROR_CODE::SUCCESS) {
            cerr << "Failed to open ZED camera " << i << endl;
            return -1;
        }
    }
    cout << "All ZED cameras opened!" << endl;

    RuntimeParameters runtime_params;
    auto cloud_combined = make_shared<geometry::PointCloud>();
    shared_ptr<geometry::PointCloud> latest_cloud;
    mutex cloud_mutex;

    visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Merged Point Clouds", 1280, 720);
    vis.AddGeometry(open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.5));
    vis.AddGeometry(cloud_combined);

    bool save_pc = true;
    bool first_frame = true;

    thread grab_thread([&]() {
        while (true) {
            bool all_grabbed = true;
            for (auto &zed : zeds) {
                if (zed.grab(runtime_params) != ERROR_CODE::SUCCESS)
                    all_grabbed = false;
            }
            if (!all_grabbed) continue;

            vector<shared_ptr<geometry::PointCloud>> pcs(n_cameras);
            for (int i = 0; i < n_cameras; i++) {
                zeds[i].retrieveMeasure(pc_mats[i], MEASURE::XYZRGBA);
                pcs[i] = ZEDToOpen3D(pc_mats[i]);
                pcs[i]->Transform(Ts[i]);  // transform to reference frame
            }

            auto merged = make_shared<geometry::PointCloud>();
            for (auto &pc : pcs) {
                merged->points_.insert(merged->points_.end(),
                                       pc->points_.begin(), pc->points_.end());
                merged->colors_.insert(merged->colors_.end(),
                                       pc->colors_.begin(), pc->colors_.end());
            }

            lock_guard<mutex> lock(cloud_mutex);
            latest_cloud = merged;

            if (save_pc && first_frame) {
                io::WritePointCloud("merged_point_cloud.ply", *merged);
                for (int i = 0; i < n_cameras; i++)
                    io::WritePointCloud("pc" + to_string(i) + ".ply", *pcs[i]);
                cout << "Merged point cloud saved!" << endl;
                first_frame = false;
            }

            this_thread::sleep_for(chrono::milliseconds(10));
        }
    });

    while (true) {
        {
            lock_guard<mutex> lock(cloud_mutex);
            if (latest_cloud) {
                cloud_combined->points_ = latest_cloud->points_;
                cloud_combined->colors_ = latest_cloud->colors_;
            }
        }
        vis.UpdateGeometry(cloud_combined);
        vis.PollEvents();
        vis.UpdateRender();
        this_thread::sleep_for(chrono::milliseconds(10));
    }

    grab_thread.join();
    for (auto &zed : zeds)
        zed.close();
    vis.DestroyVisualizerWindow();
    return 0;
}
