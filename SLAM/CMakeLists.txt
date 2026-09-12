cmake_minimum_required(VERSION 3.20)
project(ORB_SLAM3_SVO2 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W3 /EHsc /bigobj)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX _USE_MATH_DEFINES _HAS_STD_BYTE=0)
else()
    add_compile_options(-Wall -O3)
endif()

set(ORB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/ORB_SLAM3")

find_package(OpenCV 4.4 REQUIRED)
find_package(Eigen3 CONFIG REQUIRED NO_MODULE)
find_package(Pangolin CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

if(TARGET Pangolin::pangolin)
    set(ORB_PANGOLIN_TARGET Pangolin::pangolin)
elseif(TARGET pangolin)
    set(ORB_PANGOLIN_TARGET pangolin)
else()
    set(ORB_PANGOLIN_TARGET ${Pangolin_LIBRARIES})
endif()

set(ZED_SDK_ROOT_DIR "C:/Program Files (x86)/ZED SDK.old" CACHE PATH "ZED SDK root")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" CACHE PATH "CUDA Toolkit root")
find_path(ZED_INCLUDE_DIR sl/Camera.hpp
    PATHS "${ZED_SDK_ROOT_DIR}/include"
    NO_DEFAULT_PATH)
find_library(ZED_LIBRARY NAMES sl_zed64
    PATHS "${ZED_SDK_ROOT_DIR}/lib"
    NO_DEFAULT_PATH)
find_path(CUDA_INCLUDE_DIR cuda.h
    PATHS "${CUDA_TOOLKIT_ROOT_DIR}/include"
    NO_DEFAULT_PATH)
if(NOT ZED_INCLUDE_DIR OR NOT ZED_LIBRARY OR NOT CUDA_INCLUDE_DIR)
    message(FATAL_ERROR "ZED SDK or CUDA was not found. Set ZED_SDK_ROOT_DIR and CUDA_TOOLKIT_ROOT_DIR.")
endif()

file(GLOB G2O_SOURCES CONFIGURE_DEPENDS
    "${ORB_ROOT}/Thirdparty/g2o/g2o/core/*.cpp"
    "${ORB_ROOT}/Thirdparty/g2o/g2o/types/*.cpp"
    "${ORB_ROOT}/Thirdparty/g2o/g2o/stuff/*.cpp"
    "${ORB_ROOT}/Thirdparty/g2o/g2o/stuff/*.c")
list(FILTER G2O_SOURCES EXCLUDE REGEX "_test\\.cpp$")

set(ORB_SLAM3_CORE_SOURCES
    ${ORB_ROOT}/src/System.cc
    ${ORB_ROOT}/src/Tracking.cc
    ${ORB_ROOT}/src/LocalMapping.cc
    ${ORB_ROOT}/src/LoopClosing.cc
    ${ORB_ROOT}/src/ORBextractor.cc
    ${ORB_ROOT}/src/ORBmatcher.cc
    ${ORB_ROOT}/src/FrameDrawer.cc
    ${ORB_ROOT}/src/Converter.cc
    ${ORB_ROOT}/src/MapPoint.cc
    ${ORB_ROOT}/src/KeyFrame.cc
    ${ORB_ROOT}/src/Atlas.cc
    ${ORB_ROOT}/src/Map.cc
    ${ORB_ROOT}/src/MapDrawer.cc
    ${ORB_ROOT}/src/Optimizer.cc
    ${ORB_ROOT}/src/Frame.cc
    ${ORB_ROOT}/src/KeyFrameDatabase.cc
    ${ORB_ROOT}/src/Sim3Solver.cc
    ${ORB_ROOT}/src/Viewer.cc
    ${ORB_ROOT}/src/ImuTypes.cc
    ${ORB_ROOT}/src/G2oTypes.cc
    ${ORB_ROOT}/src/CameraModels/Pinhole.cpp
    ${ORB_ROOT}/src/CameraModels/KannalaBrandt8.cpp
    ${ORB_ROOT}/src/OptimizableTypes.cpp
    ${ORB_ROOT}/src/MLPnPsolver.cpp
    ${ORB_ROOT}/src/GeometricTools.cc
    ${ORB_ROOT}/src/TwoViewReconstruction.cc
    ${ORB_ROOT}/src/Config.cc
    ${ORB_ROOT}/src/Settings.cc
    ${ORB_ROOT}/Thirdparty/DBoW2/DBoW2/BowVector.cpp
    ${ORB_ROOT}/Thirdparty/DBoW2/DBoW2/FORB.cpp
    ${ORB_ROOT}/Thirdparty/DBoW2/DBoW2/FeatureVector.cpp
    ${ORB_ROOT}/Thirdparty/DBoW2/DBoW2/ScoringObject.cpp
    ${ORB_ROOT}/Thirdparty/DBoW2/DUtils/Random.cpp
    ${ORB_ROOT}/Thirdparty/DBoW2/DUtils/Timestamp.cpp
    ${G2O_SOURCES})

add_library(ORB_SLAM3 SHARED ${ORB_SLAM3_CORE_SOURCES})
target_include_directories(ORB_SLAM3 PUBLIC
    ${ORB_ROOT}
    ${ORB_ROOT}/include
    ${ORB_ROOT}/include/CameraModels
    ${ORB_ROOT}/Thirdparty/Sophus
    ${ORB_ROOT}/Thirdparty/DBoW2
    ${ORB_ROOT}/Thirdparty/g2o
    ${ORB_ROOT}/Thirdparty/g2o/g2o
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS})
target_link_libraries(ORB_SLAM3 PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${ORB_PANGOLIN_TARGET}
    Boost::serialization
    OpenSSL::Crypto)

add_executable(svo2_stereo ${ORB_ROOT}/Examples/Stereo/svo2_stereo.cc)
target_include_directories(svo2_stereo PRIVATE ${ZED_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
target_link_libraries(svo2_stereo PRIVATE ORB_SLAM3 ${ZED_LIBRARY})

set_target_properties(ORB_SLAM3 PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_target_properties(svo2_stereo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_custom_command(TARGET svo2_stereo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ZED_SDK_ROOT_DIR}/bin/sl_zed64.dll"
        "$<TARGET_FILE_DIR:svo2_stereo>")
