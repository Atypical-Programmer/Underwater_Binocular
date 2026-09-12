/**
 * ORB-SLAM3 stereo runner for a ZED SVO2 recording.
 *
 * The SVO is opened only as an image source.  LEFT_UNRECTIFIED_BGR and
 * RIGHT_UNRECTIFIED_BGR are passed to ORB-SLAM3, which performs the
 * rectification from the independent Calibration-derived settings file.
 */

#include <sl/Camera.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "System.h"

namespace fs = std::filesystem;

namespace {

struct Arguments {
    std::string vocabulary;
    std::string settings;
    std::string svo;
    std::string output;
    std::uint64_t max_frames = 0; // 0 means all frames.
    double image_scale = 1.0;
};

void printUsage(const char* executable)
{
    std::cerr << "Usage: " << executable
              << " <ORBvoc.txt> <ORB-SLAM3.yaml> <recording.svo2> <output_dir> [max_frames] [image_scale]"
              << std::endl;
}

Arguments parseArguments(int argc, char** argv)
{
    if (argc < 5 || argc > 7) {
        printUsage(argv[0]);
        throw std::invalid_argument("invalid argument count");
    }

    Arguments args;
    args.vocabulary = argv[1];
    args.settings = argv[2];
    args.svo = argv[3];
    args.output = argv[4];
    if (argc >= 6) {
        args.max_frames = std::stoull(argv[5]);
    }
    if (argc == 7) {
        args.image_scale = std::stod(argv[6]);
    }
    if (!std::isfinite(args.image_scale) || args.image_scale <= 0.0 || args.image_scale > 1.0) {
        throw std::invalid_argument("image_scale must be in the interval (0, 1]");
    }
    return args;
}

std::string joinPath(const fs::path& directory, const std::string& filename)
{
    return (directory / filename).string();
}

cv::Mat copyZedBgr(const sl::Mat& image)
{
    if (image.getWidth() == 0 || image.getHeight() == 0 || image.getPtr<unsigned char>(sl::MEM::CPU) == nullptr) {
        return cv::Mat();
    }

    cv::Mat view(
        static_cast<int>(image.getHeight()),
        static_cast<int>(image.getWidth()),
        CV_8UC3,
        image.getPtr<unsigned char>(sl::MEM::CPU),
        image.getStepBytes(sl::MEM::CPU));
    return view.clone();
}

void writeRunMetadata(const fs::path& output, const Arguments& args, const sl::CameraInformation& info)
{
    std::ofstream metadata(joinPath(output, "run_metadata.txt"), std::ios::trunc);
    metadata << "input_svo=" << fs::absolute(args.svo).string() << '\n';
    metadata << "orb_vocabulary=" << fs::absolute(args.vocabulary).string() << '\n';
    metadata << "orb_settings=" << fs::absolute(args.settings).string() << '\n';
    metadata << "zed_image_source=VIEW::LEFT_UNRECTIFIED_BGR,VIEW::RIGHT_UNRECTIFIED_BGR\n";
    metadata << "zed_optional_opencv_calibration_file=<not used>\n";
    metadata << "zed_embedded_calibration_for_slam=<not used>\n";
    metadata << "orbslam3_camera_model=PinHole (external Calibration/)\n";
    metadata << "resolution=" << info.camera_configuration.resolution.width << 'x'
             << info.camera_configuration.resolution.height << '\n';
    metadata << "input_image_scale=" << std::setprecision(17) << args.image_scale << '\n';
    metadata << "processed_resolution="
             << static_cast<int>(std::lround(info.camera_configuration.resolution.width * args.image_scale))
             << 'x'
             << static_cast<int>(std::lround(info.camera_configuration.resolution.height * args.image_scale))
             << '\n';
    metadata << "fps=" << info.camera_configuration.fps << '\n';
    metadata << "max_frames=" << args.max_frames << " (0=complete SVO)\n";
}

} // namespace

int main(int argc, char** argv)
{
    Arguments args;
    try {
        args = parseArguments(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Argument error: " << error.what() << std::endl;
        return 2;
    }

    const fs::path output = fs::absolute(args.output);
    fs::create_directories(output);

    sl::InitParameters init;
    init.input.setFromSVOFile(args.svo.c_str());
    init.svo_real_time_mode = false;
    init.camera_disable_self_calib = true;
    init.depth_mode = sl::DEPTH_MODE::NONE;

    sl::Camera zed;
    const sl::ERROR_CODE open_error = zed.open(init);
    if (open_error != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "Could not open SVO2: " << sl::toString(open_error) << std::endl;
        return 3;
    }

    const sl::CameraInformation camera_info = zed.getCameraInformation();
    writeRunMetadata(output, args, camera_info);

    std::ofstream tracking(joinPath(output, "tracking_log.csv"), std::ios::trunc);
    tracking << "frame,svo_position,timestamp_sec,tracking_state,pose_valid,tracked_map_points,width,height\n";

    std::cout << "Opened SVO2: " << args.svo << std::endl;
    std::cout << "Image source: LEFT_UNRECTIFIED_BGR + RIGHT_UNRECTIFIED_BGR" << std::endl;
    std::cout << "SLAM calibration: " << args.settings << std::endl;
    std::cout << "Embedded SVO calibration is not supplied to ORB-SLAM3." << std::endl;
    std::cout << "Input image scale: " << args.image_scale << std::endl;

    ORB_SLAM3::System slam(args.vocabulary, args.settings, ORB_SLAM3::System::STEREO, false);

    sl::Mat left_image;
    sl::Mat right_image;
    std::uint64_t frame = 0;
    std::uint64_t valid_poses = 0;
    bool reached_svo_end = false;
    const auto wall_start = std::chrono::steady_clock::now();

    while (args.max_frames == 0 || frame < args.max_frames) {
        const sl::ERROR_CODE grab_error = zed.grab();
        if (grab_error != sl::ERROR_CODE::SUCCESS) {
            if (grab_error == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
                reached_svo_end = true;
            } else {
                std::cerr << "SVO grab stopped at frame " << frame << ": " << sl::toString(grab_error) << std::endl;
            }
            break;
        }

        const sl::ERROR_CODE left_error = zed.retrieveImage(
            left_image, sl::VIEW::LEFT_UNRECTIFIED_BGR, sl::MEM::CPU);
        const sl::ERROR_CODE right_error = zed.retrieveImage(
            right_image, sl::VIEW::RIGHT_UNRECTIFIED_BGR, sl::MEM::CPU);
        if (left_error != sl::ERROR_CODE::SUCCESS || right_error != sl::ERROR_CODE::SUCCESS) {
            std::cerr << "Image retrieval failed at frame " << frame
                      << ": left=" << sl::toString(left_error)
                      << ", right=" << sl::toString(right_error) << std::endl;
            break;
        }

        cv::Mat left = copyZedBgr(left_image);
        cv::Mat right = copyZedBgr(right_image);
        if (left.empty() || right.empty() || left.size() != right.size()) {
            std::cerr << "Invalid stereo image pair at frame " << frame << std::endl;
            break;
        }

        if (args.image_scale != 1.0) {
            const cv::Size processed_size(
                static_cast<int>(std::lround(left.cols * args.image_scale)),
                static_cast<int>(std::lround(left.rows * args.image_scale)));
            cv::resize(left, left, processed_size, 0.0, 0.0, cv::INTER_AREA);
            cv::resize(right, right, processed_size, 0.0, 0.0, cv::INTER_AREA);
        }

        const double timestamp = static_cast<double>(
            zed.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds()) * 1e-9;
        const std::uint64_t svo_position = zed.getSVOPosition();
        slam.TrackStereo(left, right, timestamp);
        const int tracking_state = slam.GetTrackingState();
        const bool pose_valid = tracking_state == 2 || tracking_state == 5; // OK / OK_KLT
        if (pose_valid) {
            ++valid_poses;
        }

        tracking << frame << ',' << svo_position << ',' << std::setprecision(17) << timestamp << ','
                 << tracking_state << ',' << (pose_valid ? 1 : 0) << ','
                 << slam.GetTrackedMapPoints().size() << ',' << left.cols << ',' << left.rows << '\n';

        ++frame;
        if (frame % 100 == 0) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wall_start).count();
            const double rate = elapsed > 0.0 ? static_cast<double>(frame) / elapsed : 0.0;
            std::cout << "frame=" << frame << ", valid_pose=" << valid_poses
                      << ", rate=" << std::fixed << std::setprecision(2) << rate << " fps" << std::endl;
        }
    }

    tracking.flush();
    zed.close();
    slam.Shutdown();

    const bool map_points_saved = slam.SaveMapPointsXYZ(joinPath(output, "map_points_xyz.csv"));
    slam.SaveTrajectoryTUM(joinPath(output, "CameraTrajectory.txt"));
    slam.SaveKeyFrameTrajectoryTUM(joinPath(output, "KeyFrameTrajectory.txt"));

    std::ofstream summary(joinPath(output, "run_summary.txt"), std::ios::trunc);
    summary << "frames_processed=" << frame << '\n';
    summary << "valid_pose_frames=" << valid_poses << '\n';
    summary << "pose_valid_ratio=" << std::setprecision(17)
            << (frame ? static_cast<double>(valid_poses) / static_cast<double>(frame) : 0.0) << '\n';
    summary << "completed_to_svo_end=" << ((args.max_frames == 0 && reached_svo_end) ? 1 : 0) << '\n';
    summary << "map_points_saved=" << (map_points_saved ? 1 : 0) << '\n';

    std::cout << "ORB-SLAM3 stereo run finished. frames=" << frame
              << ", valid_pose=" << valid_poses << std::endl;
    return frame > 0 ? 0 : 4;
}
