//Programa para realizar una segmentación por color con CUDA C++

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include "procesar.cpp"
#include <opencv2/opencv.hpp>
#include "ros/ros.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>

//#include "color_seg.cu"
void color_seg(const uchar4 * const h_rgbaImage, 
                            uchar4 * const d_rgbaImage,
                            uchar3 * const d_hsvImage,
	                    unsigned char * const d_thresImage,
	       unsigned char * const d_erodedImage,
	       unsigned char * const d_dilatedImage,
                            size_t numRows, size_t numCols);

using namespace cv;
const int max_value_H = 360/2;
const int max_value = 255;
const String window_detection_name = "OPenCV";
const String window_ero = "Video erosionado";
const String window_dil = "Video dilatadao";
const String window_hsv = "HSV Paralell";
const String window_thres = "Threshold parallel";
const String window_ero_p = "erode parallel";
const String window_dil_p = "dilate parallel";
//trackbar para modificarvalores HSV
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

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


//se conoce el número de filas y columnas del cuadro con las funciones numRows y numCols
//size_t numRows() { return frame_HSV.rows; }
//size_t numCols() { return frame_HSV.cols; }





////////////////////////////////////MAIN//////////////////////////////

int main(int argc, char **argv) {

VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);
  
 uchar4        *h_rgbaImage, *d_rgbaImage;
 uchar3        *h_hsvImage, *d_hsvImage;
 unsigned char *h_thresImage, *d_thresImage;
 unsigned char *h_erodedImage, *d_erodedImage, *h_dilatedImage, *d_dilatedImage ;
  


    namedWindow(window_detection_name);
    namedWindow(window_ero);
    namedWindow(window_dil);
    namedWindow(window_hsv);
    namedWindow(window_thres);

    // Trackbar  thresholds HSV
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
        // Thresholding a imagen HSV con opencv //31,94,107
	// inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(170, 250, 250), frame_threshold);
	//inRange(frame_HSV, Scalar(100, 120, 115), Scalar(170, 250, 250), frame_threshold);
       	inRange(frame_HSV, Scalar(90, 120, 100), Scalar(170, 250, 250), frame_threshold);

	//obtención del numero de fIlas y columnas del cuadro de video
	size_t rows= frame_HSV.rows;
	size_t cols= frame_HSV.cols;    
	
	//kernel elíptico para hacer la erosión
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5), Point(2,2));
	//erosionar con opencv
       	erode(frame_threshold, frame_eroded, kernel);

	//dilatar con opencv
	dilate(frame_eroded, frame_dilated, kernel);
  
  //cargar imagen y entregar apuntadores input y output
	preProcess(&h_rgbaImage, &h_hsvImage, &h_thresImage, &h_erodedImage, &h_dilatedImage,  &d_rgbaImage, &d_hsvImage, &d_thresImage, &d_erodedImage, &d_dilatedImage,  frame); //procesar.cpp

	//iniciar timer
  GpuTimer timer;
  timer.Start();
  
  //definida en color_seg.cu
  //MODIFICAR KERNEL PARA HSV, DILATACIÓN Y EROSIÓN
  color_seg(h_rgbaImage, d_rgbaImage, d_hsvImage, d_thresImage, d_erodedImage, d_dilatedImage,  numRows(), numCols());
  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //Copiar imagen de dispositivo a host
 size_t numPixels = numRows()*numCols();
 checkCudaErrors(cudaMemcpy(h_hsvImage, d_hsvImage, sizeof(unsigned char) * numPixels*3, cudaMemcpyDeviceToHost));
 checkCudaErrors(cudaMemcpy(h_thresImage, d_thresImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
 checkCudaErrors(cudaMemcpy(h_erodedImage, d_erodedImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_dilatedImage, d_dilatedImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
 

  //Desplegar imagen de salida

  cv::Mat img = cv::Mat(numRows(),numCols(),CV_8UC3,(void*)h_hsvImage);
  cv::Mat imgTH = cv::Mat(numRows(),numCols(),CV_8UC1,(void*)h_thresImage);
  cv::Mat imgEro = cv::Mat(numRows(),numCols(),CV_8UC1,(void*)h_erodedImage);
  cv::Mat imgDil = cv::Mat(numRows(),numCols(),CV_8UC1,(void*)h_dilatedImage);
 //cv::Mat imgRGBA ;
 // cv::cvtColor(img, imgRGBA, CV_BGR2RGBA);

 // imshow(window_hsv, img);
  imshow(window_thres, imgTH);
  imshow(window_ero_p, imgEro);
  imshow(window_dil_p, imgDil);

  cleanup();    //procesar.cpp

  //Mostrar imagenes procesadas por OpenCV
   imshow(window_detection_name, frame_threshold);
      imshow(window_dil, frame_dilated);
	imshow(window_ero, frame_eroded);
        char key = (char) waitKey(30);
        if (key == 'q' || key == 27)
        {
            break;
        }
    }

  return 0;
}
