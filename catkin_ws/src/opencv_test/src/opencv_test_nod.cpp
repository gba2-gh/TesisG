#include <opencv2/opencv.hpp>
#include "ros/ros.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    cv::Mat image;
    cv::Mat image_hsv;
    Mat img_binary;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cvtColor(image, image_hsv, CV_BGR2HSV);
    inRange(image, Scalar(198,49,41), Scalar(170,250,250),img_binary);

    

    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    namedWindow( "Display windowdos", WINDOW_AUTOSIZE );
    imshow( "Display window", img_binary );// Show our image inside it.
    imshow( "Display windowdos", image_hsv );
    imshow("Display", image);
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
