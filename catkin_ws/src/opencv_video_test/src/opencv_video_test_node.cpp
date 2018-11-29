#include <opencv2/opencv.hpp>
#include "ros/ros.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "HW1.cpp"

//definido en color_seg
void prueba_cuda(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage, uchar4 * const d_hsvImage,size_t rows, size_t cols);

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
// static void on_high_H_thresh_trackbar(int, void *)
// {
//     high_H = max(high_H, low_H+1);
//     setTrackbarPos("High H", window_detection_name, high_H);
// }
static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S-1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}
// static void on_high_S_thresh_trackbar(int, void *)
// {
//     high_S = max(high_S, low_S+1);
//     setTrackbarPos("High S", window_detection_name, high_S);
// }
static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V-1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}
// static void on_high_V_thresh_trackbar(int, void *)
// {
//     high_V = max(high_V, low_V+1);
//     setTrackbarPos("High V", window_detection_name, high_V);
// }


int main(int argc, char* argv[])
{
  //abrir cámara principal
    VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);

    uchar4 *h_rgbaImage,*h_hsvImage,  *d_rgbaImage, *d_hsvImage;
    // namedWindow(window_capture_name);
    namedWindow(window_detection_name);
    //    namedWindow(window_ero);
    namedWindow(window_dil);
    // Trackbars to set thresholds for HSV values
    createTrackbar("H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    //createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
    // createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
    // createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);
    
    Mat frame, frame_HSV, frame_threshold, frame_eroded, frame_dilated, frameRGBA;

    
    
    while (true) {
      //nuevo frame para la imagen de cámara
        cap >> frame;
        if(frame.empty())
        {
            break;
        }
        // Convert from BGR to HSV colorspace
        cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
        // Detect the object based on HSV Range Values
        // inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(170, 250, 250), frame_threshold);
	inRange(frame_HSV, Scalar(31, 94, 107), Scalar(170, 250, 250), frame_threshold);

	//obtención del numero de fIlas y columnas del cuadro de video
	size_t rows= frame_HSV.rows;
	size_t cols= frame_HSV.cols;	//printf("rows=%d  \n",rows);

	//	printf("row=%zu \n",rows); //print as unsigned decimal
	
	//kernel elíptico para hacer la erosión
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5), Point(2,2));
	erode(frame_threshold, frame_eroded, kernel);

	//dilatar
	dilate(frame_eroded, frame_dilated, kernel);

	//frame to device
	//convertir de bgra a rgba
	// cv::cvtColor(frame, frameRGBA, CV_BGR2RGBA);


	preProcess(&h_rgbaImage, &h_hsvImage, &d_rgbaImage, &d_hsvImage,frame);
	cleanup();

	//cvMat a uchar
	//	*input=(uchar4 *)frameRGBA.ptr<unsigned char>(0);
	//*input = (uchar4 *)(frameRGBA.data);
	//	std::cout << "frame = " << std::endl << " " << frame << std::endl << std::endl;
	
        //const size_t pixels =rows*cols;
	uchar4 color;
	color=h_rgbaImage[1];
		printf("%u \n",color.x);

	//printf("%zu \n",pixels);
	//asignar memoria en dispositivo
	//
	//cudaMalloc((void**)d_rgbaIm, sizeof(uchar4) * pixels);

	//mandar de host a device
	//cudaMemcpy(*d_rgbaIm, *input, sizeof(uchar4) * pixels, cudaMemcpyHostToDevice);

	prueba_cuda(h_rgbaImage, d_rgbaImage, d_hsvImage, rows, cols);


	
        // Show the frames

	// imshow(window_capture_name, frame_HSV);
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
