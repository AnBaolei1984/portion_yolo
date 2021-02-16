/* Copyright 2019-2025 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/
#include <boost/filesystem.hpp>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <thread>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include "yolov3.hpp"
#include "utils.hpp"

namespace fs = boost::filesystem;
using namespace std;
using time_stamp_t = time_point<steady_clock, microseconds>;

static void detect(YOLO &net, vector<cv::Mat>& images,
                                      vector<string> names, TimeStamp *ts) {
 
  string save_folder = "result_imgs";
  if (!fs::exists(save_folder)) {
    fs::create_directory(save_folder);
  }
  int x_stride_ = net.getXStride();
  int y_stride_ = net.getYStride();

  int x_times = images[0].cols / (4 * x_stride_);
  if (images[0].cols % (4 * x_stride_)) {
    x_times += 1;
  }
  int y_times = images[0].rows / (y_stride_);
  if (images[0].rows % (y_stride_)) {
    y_times += 1;
  }
  for (int y = 0; y < y_times; y++) {
    for (int x = 0; x < x_times; x++) {
      ts->save("detection overall");
      ts->save("stage 1: pre-process");
      net.preForward(images, x * 4, y);
      ts->save("stage 1: pre-process");
      ts->save("stage 2: detection  ");
      net.forward();
      ts->save("stage 2: detection  ");
      ts->save("stage 3:post-process");
      vector<vector<yolov3_DetectRect>> dets = net.postForward();
      ts->save("stage 3:post-process");
      ts->save("detection overall");

      for (size_t i = 0; i < images.size(); i++) {
        for (size_t j = 0; j < dets[i].size(); j++) {
          int x_min = dets[i][j].left;
          int x_max = dets[i][j].right;
          int y_min = dets[i][j].top;
          int y_max = dets[i][j].bot;

          std::cout << "Category: " << dets[i][j].category
            << " Score: " << dets[i][j].score << " : " << x_min <<
            "," << y_min << "," << x_max << "," << y_max << std::endl;
          cv::Rect rc;
          rc.x = x_min;
          rc.y = y_min;;
          rc.width = x_max - x_min;
          rc.height = y_max - y_min;
          cv::rectangle(images[0], rc, cv::Scalar(0, 0, 255), 2, 1, 0);
        }
      }
    }
  }
  cv::imwrite(save_folder + "/" + names[0], images[0]);
}

int main(int argc, char **argv) {
  cout.setf(ios::fixed);

  if (argc < 4) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " image <image list> <bmodel file> " << endl;
    cout << "  " << argv[0] << " video <video url>  <bmodel file> " << endl;
    exit(1);
  }

  bool is_video = false;
  if (strcmp(argv[1], "video") == 0) {
    is_video = true;
  } else if (strcmp(argv[1], "image") == 0) {
    is_video = false;
  } else {
    cout << "Wrong input type, neither image nor video." << endl;
    exit(1);
  }

  string image_list = argv[2];
  if (!is_video && !fs::exists(image_list)) {
    cout << "Cannot find input image file." << endl;
    exit(1);
  }

  string bmodel_file = argv[3];
  if (!fs::exists(bmodel_file)) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }

  YOLO net(bmodel_file);
  TimeStamp ts;
  net.enableProfile(&ts);
  char image_path[1024] = {0};
  ifstream fp_img_list(image_list);

  vector <cv::VideoCapture> caps;
  vector <string> cap_srcs;
  while(fp_img_list.getline(image_path, 1024)) {
    cv::VideoCapture cap(image_path);
    cap.set(cv::CAP_PROP_OUTPUT_YUV, 1);
    caps.push_back(cap);
    cap_srcs.push_back(image_path);
  }
  
  uint32_t batch_id = 0;
  const uint32_t run_frame_no = 50;
  uint32_t frame_id = 0;
  while(1) {
    vector<cv::Mat> batch_imgs;
    vector<string> batch_names;
    for (size_t i = 0; i < caps.size(); i++) {
      if (caps[i].isOpened()) {
        int w = int(caps[i].get(cv::CAP_PROP_FRAME_WIDTH));
        int h = int(caps[i].get(cv::CAP_PROP_FRAME_HEIGHT));
        cv::Mat img;
        ts.save("decode overall");
        ts.save("stage 0: decode");
        caps[i] >> img;
        ts.save("stage 0: decode");
        ts.save("decode overall");
        if (img.rows != h || img.cols != w) {
          break;
        }
        batch_imgs.push_back(img);
        batch_names.push_back(to_string(batch_id) + "_" +
                            to_string(i) + "_video.jpg");
        batch_id++;
      } else {
        cout << "VideoCapture " << i << " "
                   << cap_srcs[i] << " open failed!" << endl;
      }
    }
    detect(net, batch_imgs, batch_names, &ts);
    batch_imgs.clear();
    batch_names.clear();
    frame_id += 1;
    if (frame_id == run_frame_no) {
      break;
    }
  }

  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ts.calbr_basetime(base_time);
  ts.build_timeline("yolo detect");
  ts.show_summary("detect ");
  ts.clear();

  std::cout << std::endl;

  return 0;
}
