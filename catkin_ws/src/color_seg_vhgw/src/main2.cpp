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
int indice;
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


////////////////////////////////////MAIN//////////////////////////////

int main(int argc, char **argv) {

VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);
  
 uchar4        *h_rgbaImage, *d_rgbaImage;
 uchar3        *h_hsvImage, *d_hsvImage;
 unsigned char *h_thresImage, *d_thresImage;
 unsigned char *h_erodedImage, *d_erodedImage, *h_dilatedImage, *d_dilatedImage ;
 unsigned char *h_ero_hgw, *h_eroimage_hgw, w[10]={0}, s[10]={0}, r[10]={0}, result[10]={0} ;
  


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

	//set video res
     cap.set(CV_CAP_PROP_FRAME_WIDTH,320);   // max:1280 ||  min:320   || def:640
     cap.set(CV_CAP_PROP_FRAME_HEIGHT,180);   // max:720  ||  min: 180  || def:480
      while (true) {


      //nuevo frame para la imagen de cámara
        cap >> frame;
        if(frame.empty())
        {
            break;
        }
	
	int64 t0 = cv::getTickCount();
	// GpuTimer timerop;
	// timerop.Start();
//
// Segmentación por OpenCV
//

        // Convertir BGR a HSV con opencv
        cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
        // Thresholding a imagen HSV con opencv //31,94,107
	// inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(170, 250, 250), frame_threshold);
	//inRange(frame_HSV, Scalar(100, 120, 115), Scalar(170, 250, 250), frame_threshold);
       	inRange(frame_HSV, Scalar(90, 120, 100), Scalar(170, 250, 250), frame_threshold);

	//kernel elíptico para hacer la erosión
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5), Point(2,2));
	Mat kernel2 = getStructuringElement(MORPH_RECT, Size(4,4), Point(2,2));
	//erosionar con opencv
       	erode(frame_threshold, frame_eroded, kernel2);

	//dilatar con opencv
	dilate(frame_eroded, frame_dilated, kernel2);

	// timerop.Stop();
        // int err2 = printf("Your opencv code ran in: %f msecs.\n", timerop.Elapsed());
	
	int64 t1 = cv::getTickCount();
        double secs = ((t1-t0)/cv::getTickFrequency())*1000;
	//	printf("%f\t",secs);

       
	// size_t aux=numRows();
	// printf("aux=%zu \n",aux);
  //cargar imagen y entregar apuntadores input y output
	preProcess(&h_rgbaImage, &h_hsvImage, &h_thresImage, &h_erodedImage, &h_dilatedImage, &h_ero_hgw,&h_eroimage_hgw,  &d_rgbaImage, &d_hsvImage, &d_thresImage, &d_erodedImage, &d_dilatedImage,  frame); //procesar.cpp CAMBIAR ASIGNACIÓN DE APUNTADORES

  // GpuTimer timer;   	//iniciar timer
  // timer.Start();
  
   int64 tt0 = cv::getTickCount();
  //definida en color_seg.cu
  //MODIFICAR KERNEL PARA HSV, DILATACIÓN Y EROSIÓN
  color_seg(h_rgbaImage, d_rgbaImage, d_hsvImage, d_thresImage, d_erodedImage, d_dilatedImage,  numRows(), numCols());
  // timer.Stop();        //deterner timer
    cudaDeviceSynchronize(); 

        int64 tt1 = cv::getTickCount();
        double secs2 = ((tt1-tt0)/cv::getTickFrequency())*1000;
	//	printf("%f\n",secs2);

      
  
   //  int err = printf("Your cuda code ran in: %f msecs.\n\n", timer.Elapsed());
  
  // if (err < 0) {
  //   std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
  //   exit(1);
  // }

  //Copiar imagen de dispositivo a host
 size_t numPixels = numRows()*numCols();
 cudaMemcpy(h_hsvImage, d_hsvImage, sizeof(unsigned char) * numPixels*3, cudaMemcpyDeviceToHost);
 cudaMemcpy(h_thresImage, d_thresImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
 cudaMemcpy(h_erodedImage, d_erodedImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
 cudaMemcpy(h_dilatedImage, d_dilatedImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);

 //HGW--------------------------------

                int cols = numCols();   //  320
		int rows = numRows();  //180
		int p=5;
		int apron = 2;
		int cont=0;
		int i=0;
		int indicehgw=0 ;
		int w_size=0;
		int w_cont=0;
		int indice_s=0;

		for(int x=0;x<rows;x++){
		  for(int y=0;y<cols;y++){
		    indice=y*rows + x;//x*rows + y ;

                    if(cont==0){ //agregar apron izquierdo; inicia una nueva ventana
		      
		      for(int c=0; c<apron; c++){
			h_ero_hgw[indicehgw]=0;
			// printf("%u \n", h_ero_hgw[indicehgw]);
		        indicehgw++;
		        
		        }
		      // cont++;
		    }

		         if (cont>=p){ //agregar apron derecho; finalizar ventana
		           
			   for(int c=0; c<apron; c++){
			    
			     h_ero_hgw[indicehgw]=0; //crear otro indice
			     // printf("%u \n", h_ero_hgw[indicehgw]);
		             indicehgw++;
			     
			     }
		              cont =0; //reiniciar contador de ventana

			       
		         }else{
			 
			    h_ero_hgw[indicehgw]=h_thresImage[indice];
			    // printf("%u  ", h_ero_hgw[indicehgw]);
			    // printf("%u \n", h_thresImage[indice]);
		           cont++;
		           indicehgw++;
			 
			   }

			 // if(indicehgw % w ==0){  //ventana completa
                         //   for(int i=0; i<=w;i++){
			     
		         //     s[indice_s]= h_ero_hgw[indicehgw];
			 //     printf("%u",s[indice_s]);
			 //     indice_s++;
			 //   }


			 // }
		    
		  }}
                int hgw_size = indicehgw;
		//	printf("max:%i \n",indicemax);
       //create suffix and max array
	        w_size = 2*p -1;
		indicehgw=0;
	       	//for(int i=0; i<hgw_size;i++){
		  for(int j=0; j<w_size; j++){	

		    w[j]= h_ero_hgw[indicehgw];
		    s[j]= 0;
		    r[j]= 0;
		    printf("w=%u ",w[j]);
		    indicehgw++;			 
		  }
		  
		  for(int k=0; k<=(p-1);k++){
		    for(int q=0; (k+q)<=(p-1);q++){
		      s[k]=max(w[k+q],s[k]);
		    }
		      printf(" s=%u ",s[k]);
		    }
		  
		  for(int l=(p-1); l<=(2*p-2); l++){
		    for(int m=0; (l+m)<=(2*p-2);m++){
		      r[l-(p-1)]=max(w[l+m],r[l-(p-1)]);
		    }
		    printf(" r=%u ",r[l -(p-1)]);
		    }
                  printf("\n");
		  for(int z=0; z<p;z++){
		    result[z]=max(s[z],r[z]);
		    printf(" result=%u ",result[z]);
				  
				  }
		  printf("\n");		  
		  //}

  //Desplegar imagen de salida

  cv::Mat img = cv::Mat(numRows(),numCols(),CV_8UC3,(void*)h_hsvImage);         //HSV Image
  cv::Mat imgTH = cv::Mat(numRows(),numCols(),CV_8UC1,(void*)h_thresImage);
  cv::Mat imgEro = cv::Mat(numRows(),numCols(),CV_8UC1,(void*)h_erodedImage);
  cv::Mat imgDil = cv::Mat(numRows(),numCols(),CV_8UC1,(void*)h_ero_hgw);  //Dilated Image
 //cv::Mat imgRGBA ;
 // cv::cvtColor(img, imgRGBA, CV_BGR2RGBA);

 // imshow(window_hsv, img);
 //    imshow(window_thres, imgTH);
  //     imshow(window_ero_p, imgEro);
  //	 imshow(window_dil_p, imgDil);

  cleanup();    //procesar.cpp

  //Mostrar imagenes procesadas por OpenCV
  // imshow(window_detection_name, frame_threshold);
  //  imshow(window_dil, frame_dilated);
   // 	imshow(window_ero, frame_eroded);



	char key = (char) waitKey(30);
        if (key == 'q' || key == 27)   //stop at key press
        {
            break;
        }
          }

        

  return 0;
}
