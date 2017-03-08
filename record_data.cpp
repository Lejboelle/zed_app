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
#include <ctime>
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
    { "imgs_directory",'d',POPT_ARG_STRING,&imgs_directory,0,"Directory to data in","STR" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };
  
  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}
  
  char key = ' ';
  bool record = false;
  Size frameSize(static_cast<int>(im_width), static_cast<int>(im_height));
  char timeBuf[80];
  struct tm tstruct;
  Size displaySize(1280, 720);
  
  VideoCapture cap(1);
  if (!cap.isOpened()){
	cout << "Cannot open camera object, try different number." << endl;
	return -1;
  }
	
  //Set camera specs
  cap.set(CAP_PROP_FRAME_WIDTH,im_width);
  cap.set(CAP_PROP_FRAME_HEIGHT,im_height);
  cap.set(CAP_PROP_FPS, 30);
 /* cap.set(CAP_PROP_BRIGHTNESS, 0.625);
  cap.set(CAP_PROP_CONTRAST, 0.5);
  cap.set(CAP_PROP_HUE, 0);
  cap.set(CAP_PROP_SATURATION, 0.5);
  cap.set(CAP_PROP_GAIN, 0.8);*/
  cout << cap.get(CAP_PROP_FRAME_WIDTH) << endl;
  cout << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
  cout << cap.get(CAP_PROP_FPS) << endl;
  cout << cap.get(CAP_PROP_BRIGHTNESS) << endl;
  cout << cap.get(CAP_PROP_CONTRAST) << endl;
  cout << cap.get(CAP_PROP_HUE) << endl;
  cout << cap.get(CAP_PROP_SATURATION) << endl;
  cout << cap.get(CAP_PROP_GAIN) << endl;
  
  time_t now = time(0);
  tstruct = *localtime(&now);
  namedWindow("Live", 0);
  strftime(timeBuf, sizeof(timeBuf),"%d-%m-%Y.%H:%M:%S",&tstruct);
  VideoWriter vidWrite(imgs_directory+string(timeBuf)+".avi",CV_FOURCC('I','Y','U','V'),30,frameSize,true);
  
  if (!vidWrite.isOpened()){
	cout << "Could not open video writer, select different path." << imgs_directory << endl;
	return -1;
  }
  Mat img, img_res;
  while (key != 'q') {
	  // Get next frame
	
    cap >> img;
    if (img.empty()) {
		cout << "Empty frame.." << endl;
		break;
	}

	resize(img,img_res,displaySize);
	imshow("Live", img_res);
	key = waitKey(5);
	if(key == 'r') {
		record = true;
		cout << "Recording started..\n" << endl;
	}
	if(key == 's') {
		record = false;
		cout << "Recording stopped..\n" << endl;
	}
	
	if (record)
		vidWrite.write(img);
  }
}

