#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>



cv::Mat imageRGBA;
cv::Mat imageGrey;
cv::Mat imageThres;
cv::Mat imageEro;
cv::Mat imageDil;
cv::Mat imageWindowHGW;
cv::Mat imageEroHGW;
cv::Mat imageDilHGW;

uchar4        *d_rgbaImage__;
uchar3        *d_hsvImage__;
unsigned char  *d_thresImage__;
unsigned char  *d_erodedImage__;
unsigned char  *d_dilatedImage__;
unsigned char *d_window_hgw__;
unsigned char *d_erohgw__;
unsigned char *d_dilhgw__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

void preProcess(uchar4 **inputImage, uchar3 **hsvImage, unsigned char **thresImage,
		unsigned char **erodedImage, unsigned char **dilatedImage,
		unsigned char **window_hgw, unsigned char **erohgw, unsigned char **dilhgw,
                uchar4 **d_rgbaImage, uchar3 **d_hsvImage,
		unsigned char **d_thresImage,
		unsigned char **d_erodedImage, unsigned char **d_dilatedImage,
		unsigned char **d_window_hgw, unsigned char **d_erohgw, unsigned char **d_dilhgw,
		cv::Mat frame) {
  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));


  cv::cvtColor(frame, imageRGBA, CV_BGR2RGBA);

  //allocate memory for the hsv output
  imageGrey.create(frame.rows, frame.cols, CV_8UC3);
  //allocate memory for the threshold output
  imageThres.create(frame.rows, frame.cols, CV_8UC1);
  //allocate mem for erode output
  imageEro.create(frame.rows, frame.cols, CV_8UC1);
  //allocate mem for dilate output
   imageDil.create(frame.rows, frame.cols, CV_8UC1);
   //allocate mem for dilate output
   imageWindowHGW.create(frame.rows, frame.cols, CV_8UC1);
   imageEroHGW.create(frame.rows, frame.cols, CV_8UC1);
   imageDilHGW.create(frame.rows, frame.cols, CV_8UC1);

  //verify that image is continous
  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  //allocate memory on host
  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *hsvImage  = (uchar3 *)imageGrey.ptr<unsigned char>(0);
  *thresImage = (unsigned char*)imageThres.ptr<unsigned char>(0);
  *erodedImage = (unsigned char*)imageEro.ptr<unsigned char>(0);
  *dilatedImage = (unsigned char*)imageDil.ptr<unsigned char>(0);
  *window_hgw= (unsigned char*)imageWindowHGW.ptr<unsigned char>(0);
  *erohgw= (unsigned char*)imageEroHGW.ptr<unsigned char>(0);
  *dilhgw=(unsigned char*)imageDilHGW.ptr<unsigned char>(0);
  
  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_hsvImage, sizeof(uchar3) * numPixels));
  checkCudaErrors(cudaMalloc(d_thresImage, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_erodedImage, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_dilatedImage, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_window_hgw, sizeof(unsigned char) * numPixels*2));
  checkCudaErrors(cudaMalloc(d_erohgw, sizeof(unsigned char) * numPixels*2));
  checkCudaErrors(cudaMalloc(d_dilhgw, sizeof(unsigned char) * numPixels*2));
  checkCudaErrors(cudaMemset(*d_hsvImage, 0, numPixels * sizeof(uchar3))); //make sure no memory is left laying around
  cudaMemset(*d_thresImage, 0, numPixels * sizeof(unsigned char)); //make sure no memory is left laying around
  cudaMemset(*d_erodedImage, 0, numPixels * sizeof(unsigned char)); //make sure no memory is left laying around
  cudaMemset(*d_dilatedImage, 0, numPixels * sizeof(unsigned char)); //make sure no memory is left laying around

  //copy input array to the GPU
  cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

  d_rgbaImage__ = *d_rgbaImage;
  d_hsvImage__ = *d_hsvImage;
  d_thresImage__ = *d_thresImage;
  d_erodedImage__ = *d_erodedImage;
  d_dilatedImage__ = *d_dilatedImage;

  d_window_hgw__ = *d_window_hgw;
    d_erohgw__ = *d_erohgw;
  d_dilhgw__ = *d_dilhgw;
  
}



void cleanup()
{
  //cleanup
  cudaFree(d_rgbaImage__);
  cudaFree(d_hsvImage__);
  cudaFree(d_thresImage__);
  cudaFree(d_erodedImage__);
  cudaFree(d_dilatedImage__);

  cudaFree(d_window_hgw__);
  cudaFree(d_erohgw__);
  cudaFree(d_dilhgw__);
}
