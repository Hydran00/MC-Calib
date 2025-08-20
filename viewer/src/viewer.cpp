#include "viewer/utils.h"

using namespace std;
using namespace sl;
using namespace open3d;

// ------------------ Helper Functions ------------------

vector<Camera> InitZEDCameras(int n_cameras, InitParameters &init_params) {
    vector<Camera> zeds(n_cameras);
    for (int i = 0; i < n_cameras; i++) {
        if (zeds[i].open(init_params) != ERROR_CODE::SUCCESS) {
            cerr << "Failed to open ZED camera " << i << endl;
            exit(-1);
        }
        cv::namedWindow("Camera " + to_string(i), cv::WINDOW_AUTOSIZE);
    }
    cout << "All ZED cameras opened!" << endl;
    return zeds;
}

cv::dnn::Net LoadYOLOModel(Yolov8Seg &yolov8Seg, string &model_path) {
    cv::dnn::Net net;
    if (!yolov8Seg.ReadModel(net, model_path, true)) {
        cerr << "YOLO model load failed!" << endl;
        exit(-1);
    }
    cout << "YOLOv8 model loaded successfully!" << endl;
    return net;
}

shared_ptr<geometry::PointCloud> GrabAndProcessFrame(
    Camera &zed, Mat &pc_mat, const Eigen::Matrix4d &T,
    cv::dnn::Net &net, Yolov8Seg &yolov8Seg, const string &win_name) {

    // Grab RGB
    sl::Mat sl_image;
    zed.retrieveImage(sl_image, VIEW::LEFT);
    cv::Mat cvImage(sl_image.getHeight(), sl_image.getWidth(), CV_8UC4,
                    sl_image.getPtr<sl::uchar1>(sl::MEM::CPU));
    cv::cvtColor(cvImage, cvImage, cv::COLOR_BGRA2BGR);

    // YOLO detect
    vector<OutputParams> detections;
    if (!yolov8Seg.Detect(cvImage, net, detections)) {
        cerr << "YOLO detect failed!" << endl;
        return make_shared<geometry::PointCloud>();
    }

    // Take first person detected
    cv::Rect best_bbox;
    cv::Mat best_mask;
    for (auto &det : detections) {
        if (det.id == 0) {
            best_bbox = det.box;
            best_mask = det.boxMask.clone();
            break;
        }
    }

    // Retrieve 3D points
    zed.retrieveMeasure(pc_mat, MEASURE::XYZRGBA);
    if (!best_mask.empty()) {
        cv::Mat eroded_mask;
        cv::erode(best_mask, eroded_mask,
                  cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10)));
        best_mask = eroded_mask;

        auto pc = ZEDToOpen3D(pc_mat, cvImage, best_bbox, best_mask);
        pc->Transform(T);
        show_resized_img(cvImage, 0.5, win_name);
        return pc;
    }
    return make_shared<geometry::PointCloud>();
}

// ------------------ Main ------------------

int main(int argc, char **argv) {
    string calib_file = (argc > 1) ? argv[1] : "calibration.yaml";
    int n_cameras = (argc > 2) ? stoi(argv[2]) : 2;

    cout << "Loading calibration from: " << calib_file << endl;
    auto Ts = LoadCalibrationYAML(calib_file, n_cameras);

    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD2K;
    init_params.coordinate_system = COORDINATE_SYSTEM::IMAGE;
    init_params.coordinate_units = UNIT::METER;
    init_params.depth_mode = DEPTH_MODE::NEURAL_PLUS;
    init_params.depth_maximum_distance = 2.5f;

    auto zeds = InitZEDCameras(n_cameras, init_params);
    vector<Mat> pc_mats(n_cameras);

    Yolov8Seg yolov8Seg;
    string yolo_model_path = "yolov8x-seg.onnx";
    auto net = LoadYOLOModel(yolov8Seg, yolo_model_path);

    RuntimeParameters runtime_params;
    auto cloud_combined = make_shared<geometry::PointCloud>();
    shared_ptr<geometry::PointCloud> latest_cloud;
    mutex cloud_mutex;

    visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Human Cropped Point Clouds", 1280, 720);
    vis.AddGeometry(open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.5));
    vis.AddGeometry(cloud_combined);

    bool save_pc = true;
    bool first_frame = true;

    // ------------------ Grab thread ------------------
    thread grab_thread([&]() {
        while (true) {
            bool all_grabbed = true;
            for (auto &zed : zeds) {
                if (zed.grab(runtime_params) != ERROR_CODE::SUCCESS)
                    all_grabbed = false;
            }
            if (!all_grabbed)
                continue;

            vector<shared_ptr<geometry::PointCloud>> pcs(n_cameras);
            for (int i = 0; i < n_cameras; i++)
                pcs[i] = GrabAndProcessFrame(zeds[i], pc_mats[i], Ts[i], net,
                                             yolov8Seg, "Camera " + to_string(i));

            auto merged = MergePointClouds(pcs);

            {
                lock_guard<mutex> lock(cloud_mutex);
                latest_cloud = merged;
            }

            if (save_pc && first_frame) {
                SavePointClouds(pcs, merged);
                first_frame = false;
            }

            this_thread::sleep_for(chrono::milliseconds(10));
        }
    });

    // ------------------ Visualization loop ------------------
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
