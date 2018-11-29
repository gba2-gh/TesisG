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

uchar4        *d_rgbaImage__;
uchar4        *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

void preProcess(uchar4 **inputImage, uchar4 **greyImage,
                uchar4 **d_rgbaImage, uchar4**d_greyImage,
                cv::Mat frame) {
  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));


  cv::cvtColor(frame, imageRGBA, CV_BGR2RGBA);

  //allocate memory for the output
  imageGrey.create(frame.rows, frame.cols, CV_8UC1);

  //verify that image is continous
  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage  = (uchar4 *)imageGrey.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_greyImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(uchar4))); //make sure no memory is left laying around

  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file, uchar4 * data_ptr) {
  cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);

  //output the image
  // namedWindow( "Display window", CV_WINDOW_AUTOSIZE );


  imshow( "Display window", output );                   

    //waitKey(0); 
 // cv::imwrite(output_file.c_str(), output);
  //cv::imwrite("/home/gibran/Documents/gray.bmp", output);
}

void cleanup()
{
  //cleanup
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}
