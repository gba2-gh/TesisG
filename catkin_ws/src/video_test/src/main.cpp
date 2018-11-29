//Udacity HW1 Solution

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>


void rgba2hsv(const uchar4 * const h_rgbaImage, 
                            uchar4 * const d_rgbaImage,
                            uchar4 * const d_greyImage, 
                            size_t numRows, size_t numCols);

//include the definitions of the above functions for this homework
#include "HW1.cpp"

#include <opencv2/opencv.hpp>
#include "ros/ros.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>



using namespace cv;
const int max_value_H = 360/2;
const int max_value = 255;
//const String window_capture_name = "Captura HSV";
const String window_detection_name = "captura hsv";
//const String window_ero = "Video erosionado";
const String window_dil = "Video dilatadao";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

//se conoce el número de filas y columnas del cuadro con las funciones numRows y numCols

//size_t numRows() { return frame_HSV.rows; }
//size_t numCols() { return frame_HSV.cols; }


static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H-1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}

static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S-1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}

static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V-1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}





////////////////////////////////////MAIN//////////////////////////////

int main(int argc, char **argv) {

VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);
  
 uchar4        *h_rgbaImage, *d_rgbaImage, *h_hsvImage, *d_hsvImage;
  unsigned char *h_greyImage, *d_greyImage;


    namedWindow(window_detection_name);
    //    namedWindow(window_ero);
    namedWindow(window_dil);
    // Trackbars to set thresholds for HSV values
    createTrackbar("H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);

    
    Mat frame, frame_HSV, frame_threshold, frame_eroded, frame_dilated, frameRGBA;
    
    std::string output_file;


      while (true) {
      //nuevo frame para la imagen de cámara
        cap >> frame;
        if(frame.empty())
        {
            break;
        }
        // COnvertir BGR a HSV con opencv
        cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
        // Tresholding a imagen HSV con opencv
        // inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(170, 250, 250), frame_threshold);
	inRange(frame_HSV, Scalar(31, 94, 107), Scalar(170, 250, 250), frame_threshold);

	//obtención del numero de fIlas y columnas del cuadro de video
	size_t rows= frame_HSV.rows;
	size_t cols= frame_HSV.cols;    
	
	//kernel elíptico para hacer la erosión
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5), Point(2,2));
	//erosionar con opencv
	erode(frame_threshold, frame_eroded, kernel);

	//dilatar con opencv
	dilate(frame_eroded, frame_dilated, kernel);
  
  //load the image and give us our input and output pointers
  preProcess(&h_rgbaImage, &h_hsvImage, &d_rgbaImage, &d_hsvImage, frame);

  GpuTimer timer;
  timer.Start();
  //definida en student_func
  //MODIFICAR KERNEL PARA HSV, DILATACIÓN Y EROSIÓN
  
  rgba2hsv(h_rgbaImage, d_rgbaImage, d_hsvImage, numRows(), numCols());
  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //Copiar imagen de dispositivo a host
 size_t numPixels = numRows()*numCols();
 checkCudaErrors(cudaMemcpy(h_hsvImage, d_hsvImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

  //check results and output the grey image
  postProcess(output_file, h_hsvImage);

  cleanup();

  imshow(window_detection_name, frame_HSV);
	//	imshow(window_ero, frame_eroded);
    	imshow(window_dil, frame_dilated);
        char key = (char) waitKey(30);
        if (key == 'q' || key == 27)
        {
            break;
        }
    }

  return 0;
}
