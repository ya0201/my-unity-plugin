//
//  Plugin.cpp
//  MyUnityPlugin
//

#include "Plugin.pch"
#include <iostream>
#include <unistd.h>
#include <sstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"

// the size of image from camera
#define DEFAULT_IMG_SIZE_W 400
#define DEFAULT_IMG_SIZE_H 300

#define FACE_DOWNSAMPLE_RATIO 4

// Detect faces every $(this value) frames
#define FACE_DETECTION_INTERVAL_FRAMES 2


void* getVideoDevice(int camera_num) {
    cv::VideoCapture* cap = new cv::VideoCapture(camera_num);
    
    // check whether camera device is opened
    if(!cap->isOpened()) {
        // failed to open camera device
        std::cerr << "Error: Failed to open camera" << std::endl;
        exit(1);
    }
    // wait for adjusting brightness
    sleep(1);
    
    // set default image size
    cap->set(CV_CAP_PROP_FRAME_WIDTH, DEFAULT_IMG_SIZE_W);
    cap->set(CV_CAP_PROP_FRAME_HEIGHT, DEFAULT_IMG_SIZE_H);
    
    return static_cast<void*>(cap);
}

void releaseVideoDevice(void* cap) {
    auto vc = static_cast<cv::VideoCapture*>(cap);
    delete vc;
}

void* getDetectorAndPoseModel(const char* model_dat_path) {
    const std::string FACE_LANDMARK_MODEL = std::string(model_dat_path);
    
    dlib::frontal_face_detector* detector = new dlib::frontal_face_detector();
    std::istringstream sin(dlib::get_serialized_frontal_faces());
    dlib::deserialize(*detector, sin);
    
    dlib::shape_predictor* pose_model = new dlib::shape_predictor();
    dlib::deserialize(FACE_LANDMARK_MODEL) >> *pose_model;
    
    auto return_ptr = static_cast<void*>(new std::pair<dlib::frontal_face_detector*, dlib::shape_predictor*>(detector, pose_model));
    return return_ptr;
}

cv::Mat getGrayFrameFromCamera(cv::VideoCapture* vc) {
    static cv::Mat frame;
    
    vc->read(frame);
    if (frame.channels() == 4) {
        cv::cvtColor(frame, frame, CV_BGRA2GRAY);
    } else if (frame.channels() == 3) {
        cv::cvtColor(frame, frame, CV_BGR2GRAY);
    } //else if (frame.channels() == 1) {
    //cv::cvtColor(frame, frame, CV_GRAY2BGR);
    //}
//    cv::Mat* frame = new cv::Mat();
    return frame;
}

// if failed face detection, then this method returns empty rectangle
// use dlib::rectangle::is_empty() to check whether or not the return rectangle is empty
dlib::rectangle detectPrimaryFaceRect(dlib::cv_image<unsigned char> cimg_small, dlib::frontal_face_detector &detector) {
    static std::vector<dlib::rectangle> faces;
    static dlib::rectangle empty_rect = dlib::rectangle(0, 0, 0, 0);
    
    faces = detector(cimg_small);
    if(faces.size() != 1) {
        return empty_rect;
    }
    return faces[0];
}

void detect(void* cap, void* dapm, int face_rect[4], int parts_matrix[NUM_OF_PARTS][2]) {
    static dlib::rectangle primary_face_rect;
    static int frame_ctr = 0;
    static std::vector<dlib::rectangle> faces;
    
    cv::Mat frame, frame_small;
    auto vc = static_cast<cv::VideoCapture*>(cap);
    auto pair = static_cast<std::pair<dlib::frontal_face_detector*, dlib::shape_predictor*>*>(dapm);
    auto detector = *(pair->first);
    auto pose_model = *(pair->second);
    
//    vc->read(frame);
    frame = getGrayFrameFromCamera(vc);
    cv::resize(frame, frame_small, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
    
    // dlib
    {
        dlib::cv_image<unsigned char> cimg_gray(frame);
        dlib::cv_image<unsigned char> cimg_gray_small(frame_small);
        
        // Detect faces every $(interval value) frame
        // if (frame_ctr++ % FACE_DETECTION_INTERVAL_FRAMES == 0) {
        //     faces = detector(cimg_gray);
        // }
        // if (faces.size() != 1) {
        //     // no face or more than 2 faces
        //     return;
        // }

        // detect Primary Face Rectangle once
        if (primary_face_rect.is_empty() ||
            frame_ctr++ % FACE_DETECTION_INTERVAL_FRAMES == 0) {
          primary_face_rect = detectPrimaryFaceRect(cimg_gray, detector);
          if (primary_face_rect.is_empty()) return;
        }

        
        // Find the pose the face
        // Resize obtained rectangle for full resolution image.
        dlib::rectangle r(
                    (long)(primary_face_rect.left()),
                    (long)(primary_face_rect.top()),
                    (long)(primary_face_rect.right()),
                    (long)(primary_face_rect.bottom())
                    );
//        dlib::full_object_detection shape = pose_model(cimg_gray, faces[0]);
        dlib::full_object_detection shape = pose_model(cimg_gray, r);
        if (shape.num_parts() != NUM_OF_PARTS) {
            // the number of detected parts is not 68
            return;
        }
        
        auto rect = shape.get_rect();
        face_rect[0] = (int)rect.left();
        face_rect[1] = (int)rect.top();
        face_rect[2] = (int)rect.width();
        face_rect[3] = (int)rect.height();
        for (auto i = 0; i < NUM_OF_PARTS; i++) {
            auto part = shape.part(i);
                
            parts_matrix[i][0] = (int)part.x();
            parts_matrix[i][1] = (int)part.y();
        }
    }
}

void setImgSize(void* cap, int img_size_w, int img_size_h) {
    auto vc = static_cast<cv::VideoCapture*>(cap);
    vc->set(CV_CAP_PROP_FRAME_WIDTH, img_size_w);
    vc->set(CV_CAP_PROP_FRAME_HEIGHT, img_size_h);
}
