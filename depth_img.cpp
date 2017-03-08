#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
#include <time.h>
#include <sstream>
#include "popt_pp.h"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

int z = 0;
Mat depth;

static void onMouseCallback(int32_t event, int32_t x, int32_t y, int, void*) {
    if (event == CV_EVENT_LBUTTONDOWN) {

		float ptr_depth = depth.at<float>(y,x);

		printf("\n Depth at clicked position is: %2.2f\n", ptr_depth);
	}
}


int main(int argc, char const *argv[])
{
  char* extension;
 // char* src_path;
  char* dst_path;
  char* calib_file;
  int im_width, im_height;
  int interval = 2, j=0;
  static struct poptOption options[] = {
    { "img_width",'w',POPT_ARG_INT,&im_width,0,"Image width","NUM" },
    { "img_height",'h',POPT_ARG_INT,&im_height,0,"Image height","NUM" },
    { "calib_file",'c',POPT_ARG_STRING,&calib_file,0,"Stereo calibration file","STR" },
  //  { "src_path", 's', POPT_ARG_STRING,&src_path,0,"Source path","STR" },
    { "dst_path", 'd', POPT_ARG_STRING,&dst_path,0,"Destination path","STR" },
    { "extension",'e',POPT_ARG_STRING,&extension,0,"Image extension","STR" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  POpt popt(NULL, argc, argv, options, 0);
  int c;
  char key = ' ';
  bool record = false;
  while((c = popt.getNextOpt()) >= 0) {}
  
  // Initialize video stream
  Size frameSize(static_cast<int>(im_width), static_cast<int>(im_height));
  VideoCapture cap(1);
  cap.set(CAP_PROP_FRAME_WIDTH,im_width);
  cap.set(CAP_PROP_FRAME_HEIGHT,im_height);
  cap.set(CAP_PROP_FPS, 30);
  Mat img, img_res, imgL, imgR, imgDisp;
  Size displaySize(1280, 720);
  
  /* TODO
   * Choose between folder with images or capture new images
   * Automatically check if folders exist, create if not
   * Check if path is valid
   * Set parameters
   */
    
  // Variables for undistortion
  Mat R1, R2, P1, P2, Q;
  Mat K1, K2, R;
  Vec3d T;
  Mat D1, D2;
  Mat lmapx, lmapy, rmapx, rmapy;
  Mat imgUL, imgUR;
  Rect validROI[2];
  FileStorage fs1(calib_file, FileStorage::READ);
  fs1["K1"] >> K1;
  fs1["K2"] >> K2;
  fs1["D1"] >> D1;
  fs1["D2"] >> D2;
  fs1["R"] >> R;
  fs1["T"] >> T;

  fs1["R1"] >> R1;
  fs1["R2"] >> R2;
  fs1["P1"] >> P1;
  fs1["P2"] >> P2;
  fs1["Q"] >> Q;
  fs1["ROI1"] >> validROI[0];
  fs1["ROI2"] >> validROI[1];
  
  //Variables for computing disparity
  int max_disp = 64;	//160
  double lambda = 8000.0;
  double sigma  = 1.5;
  double vis_mult = 1.0;
  int wsize = 11;	//3
  Mat left_for_matcher, right_for_matcher;
  Mat left_disp,right_disp;
  Mat filtered_disp, filtered_disp_vis;
  Mat xyz, binImg, locations;
  Mat reproj_img, reproj_img_rec, depth_color;
  vector<Mat> channels;
  Rect ROI;
  double min_depth, max_depth;
  float max_height;
  Point minLoc, maxLoc; 
    
  Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
  left_matcher->setP1(24*wsize);	//24*wsize*wsize
  left_matcher->setP2(96*wsize*wsize);
  left_matcher->setPreFilterCap(63);
  left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
  Ptr<DisparityWLSFilter> wls_filter;
  wls_filter = createDisparityWLSFilter(left_matcher);
  Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);
  wls_filter->setLambda(lambda);
  wls_filter->setSigmaColor(sigma);
  namedWindow( "filtered disparity", 0 );
  setMouseCallback("filtered disparity", onMouseCallback, 0);
  
  while (key != 'q') {
	  
	//Capture next frame
    cap >> img;
    resize(img, img_res, Size(im_width, im_height));
    resize(img, imgDisp, displaySize);
    imshow("IMG", imgDisp);
    imgL = img_res(Rect(0,0,1920,1080));
    imgR = img_res(Rect(1920,0,1920,1080));
	key = cv::waitKey(5);
	
    if (key == 'p') {
      z++;
      cout << "Computing and saving depth for img pair " << z << endl;
      // Save raw image
      char filename1[200], filename2[200];
      sprintf(filename1, "%s%sleft%d.%s", dst_path, "raw/left/", z, extension);
      sprintf(filename2, "%s%sright%d.%s", dst_path, "raw/right/", z, extension);
      
      imwrite(filename1, imgL);
      imwrite(filename2, imgR);
      
      // Undistort images and save them
      initUndistortRectifyMap(K1, D1, R1, P1, imgL.size(), CV_32F, lmapx, lmapy);
      initUndistortRectifyMap(K2, D2, R2, P2, imgR.size(), CV_32F, rmapx, rmapy); 
	  remap(imgL, imgUL, lmapx, lmapy, cv::INTER_LINEAR);
	  remap(imgR, imgUR, rmapx, rmapy, cv::INTER_LINEAR);
	  
	  sprintf(filename1, "%s%sleft%d.%s", dst_path, "rectified/left/", z, extension);
      sprintf(filename2, "%s%sright%d.%s", dst_path, "rectified/right/", z, extension);
      imwrite(filename1, imgUL);
      imwrite(filename2, imgUR);
      
      //Compute disparity and depth
      resize(imgUL,left_for_matcher ,Size(),0.5,0.5);
      resize(imgUR,right_for_matcher,Size(),0.5,0.5);
      
      left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
      right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
     
      wls_filter->filter(left_disp,imgUL,filtered_disp,right_disp);
      filtered_disp.convertTo(filtered_disp,CV_32F, 1.0/16.0, 0.0);

      ROI = wls_filter->getROI();
        
      threshold(filtered_disp,binImg,0,255,THRESH_BINARY);
      binImg.convertTo(binImg,CV_8UC1);
      findNonZero(binImg,locations);
      
      Point firstP = locations.at<Point>(0);
	  Size filt_size = filtered_disp.size();
      //Calculate 3D coordinates from validROI
      Rect rect(firstP.x+1,firstP.y+1,filt_size.width-(firstP.x+1),filt_size.height-(firstP.y+1));
	  reproj_img = filtered_disp(rect);
		
      resize(reproj_img,reproj_img,filt_size);
      reproj_img = reproj_img(validROI[0]);
      resize(reproj_img,reproj_img,displaySize);
      
      reprojectImageTo3D(reproj_img, xyz, Q, false, -1);
	  split(xyz,channels);
	  depth = channels[2];
		
      //Calculate min (and max depth)
      minMaxLoc(depth, &min_depth, &max_depth, &minLoc, &maxLoc, cv::noArray());
	  
	  normalize(reproj_img,filtered_disp_vis,0,255,CV_MINMAX, CV_8U);
	  applyColorMap(filtered_disp_vis,depth_color,COLORMAP_JET);
	  sprintf(filename1, "%s%sfiltered_disp%d.%s", dst_path, "depth/", z, extension);
	  imwrite(filename1,depth_color);
	  imshow("filtered disparity", depth_color);
    }
  }
  return 0;
}
