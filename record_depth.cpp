#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/cudastereo.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <time.h>
#include <sstream>
#include "popt_pp.h"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;


int main(int argc, char const *argv[])
{
  int im_width, im_height;
  char* imgs_directory;
  char* calib_file;
  static struct poptOption options[] = {
    { "img_width",'w',POPT_ARG_INT,&im_width,0,"Image width","NUM" },
    { "img_height",'h',POPT_ARG_INT,&im_height,0,"Image height","NUM" },
    { "imgs_directory",'d',POPT_ARG_STRING,&imgs_directory,0,"Directory to save images in","STR" },
    { "calib_file",'c',POPT_ARG_STRING,&calib_file,0,"Stereo calibration file","STR" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };
  
  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}
  
  char key = ' ';
  bool record = false;
  Size frameSize(static_cast<int>(im_width), static_cast<int>(im_height));
  
    //Disparity computation & filtering configurations (CPU)
  double lambda = 8000.0;
  int max_disp = 160;
  double sigma = 1.5;
  int wsize = 3;
  /*
  Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
  Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);
  left_matcher->setP1(24*wsize*wsize);
  left_matcher->setP2(96*wsize*wsize);
  left_matcher->setPreFilterCap(63);
  left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
  Ptr<DisparityWLSFilter> wls_filter = createDisparityWLSFilter(left_matcher);
  wls_filter->setLambda(lambda);
  wls_filter->setSigmaColor(sigma);
  */
	//Disparity configurations (GPU)
  Ptr<cuda::StereoBM> matcher_gpu  = cuda::createStereoBM(max_disp,wsize);
  Ptr<cuda::DisparityBilateralFilter> gpu_filter =  cv::cuda::createDisparityBilateralFilter(64,5,3);
    //Read intrinsic and extrinsic parameters
  Mat R1, R2, P1, P2, Q;
  Mat K1, K2, R;
  Mat D1, D2;
  
  cv::FileStorage fs1(calib_file, cv::FileStorage::READ);
  fs1["K1"] >> K1;
  fs1["K2"] >> K2;
  fs1["D1"] >> D1;
  fs1["D2"] >> D2;
  fs1["R"] >> R;

  fs1["R1"] >> R1;
  fs1["R2"] >> R2;
  fs1["P1"] >> P1;
  fs1["P2"] >> P2;
  fs1["Q"] >> Q;
  
  VideoCapture cap(1);
  cap.set(CV_CAP_PROP_FRAME_WIDTH,im_width);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT,im_height);
  //VideoWriter vidWrite(imgs_directory+".avi",CV_FOURCC('I','Y','U','V'),20,frameSize,true);
  Mat img, img_res, imgL, imgR, imgUL, imgUR, imgGL, imgGR;
  Mat lmapx, lmapy, rmapx, rmapy;
  Mat left_disp, right_disp, filtered_disp, filtered_disp_vis, filtered_res;
  cuda::GpuMat imgG1_gpu, imgG2_gpu, gpu_mapL1, gpu_mapL2, gpu_mapR1, gpu_mapR2;
  cuda::GpuMat imgUL_gpu, imgUR_gpu, left_disp_gpu, imgL_gpu, filt_disp_gpu;
  
  while (key != 'q') {
	  // Get next frame
    cap >> img;
    imgL = img(Rect(0,0,1920,1080));
    imgR = img(Rect(1920,0,1920,1080));
    imgL_gpu.upload(imgL);
     //Undistort and remap images
    cv::initUndistortRectifyMap(K1, D1, R1, P1, imgL.size(), CV_32F, lmapx, lmapy);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, imgR.size(), CV_32F, rmapx, rmapy);
    cv::remap(imgL, imgUL, lmapx, lmapy, cv::INTER_LINEAR);
    cv::remap(imgR, imgUR, rmapx, rmapy, cv::INTER_LINEAR);
    cvtColor(imgUL,imgGL,COLOR_BGR2GRAY);
    cvtColor(imgUR,imgGR,COLOR_BGR2GRAY);
    imgUL_gpu.upload(imgGL);
    imgUR_gpu.upload(imgGR);
    
    matcher_gpu->compute(imgUL_gpu, imgUR_gpu, left_disp_gpu);
    gpu_filter->apply(left_disp_gpu,imgL_gpu,filt_disp_gpu);
    filt_disp_gpu.download(left_disp);
      //Calculate disparity and depth
      /*
    left_matcher->compute(imgUL, imgUR,left_disp);
    right_matcher->compute(imgUR, imgUL, right_disp);
    wls_filter->filter(left_disp,imgL,filtered_disp,right_disp);
    //filtered_disp.convertTo(filtered_disp,CV_32F, 1.0/16.0, 0.0);
      */

      //Display images
    resize(img, img_res, Size(im_width/2, im_height/2));
    //resize(filtered_disp, filtered_res, Size(im_width/4, im_height/2));
    imshow("RGB-Unrectified", img_res);
    //getDisparityVis(filtered_disp,filtered_disp_vis);
    //normalize(filtered_res,filtered_disp_vis,0,255,CV_MINMAX,CV_8U);
    imshow("Filtered disparity", left_disp);
	key = cv::waitKey(5);
	
	if(key == 'r') {
		record = true;
		cout << "Recording started..\n" << endl;
	}
	if(key == 's') {
		record = false;
		cout << "Recording stopped..\n" << endl;
	}
	
	if (record)
		//vidWrite.write(img_res2);
		cout << "recording enabled" << endl;
	}
}

