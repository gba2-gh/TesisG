#include <ros/ros.h>
#include <opencv2/opencv.hpp>

/*
 * These functions are implemented in color_segmenter.cu
*/
void gpu_segment_by_color(unsigned char* bgr_image, unsigned char* segmented_image, int img_rows, int img_cols, int img_channels);
void gpu_allocate_memory(int img_rows, int img_cols, int img_channels);
void gpu_free_memory();

int main(int argc, char** argv)
{
    cv::VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH,640);   // max:1280 ||  min:320   || def:640
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);   // max:720  ||  min: 180  || def:480
    cv::Mat frame;
    cv::Mat hsv;
    cv::Mat segmented = cv::Mat(480, 640, CV_8UC1);
    gpu_allocate_memory(480,640,3);

    while(cv::waitKey(15) != 27)
    {
        cap >> frame;
        
        cv::imshow("Original", frame);
        gpu_segment_by_color(frame.data, segmented.data, 480, 640, 3);
        //cv::cvtColor(frame, hsv, CV_BGR2HSV);
        //cv::imshow("HSV Serial", hsv);
        cv::imshow("Binary", segmented);
    }
    gpu_free_memory();
    return 0;
}
