#include <opencv2/opencv.hpp>
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "std_msgs/UInt8MultiArray.h"

int main(int argc, char** argv)
{
    std::cout << "Initalizing camera-test node..." << std::endl;
    ros::init(argc, argv, "camera_test");
    ros::NodeHandle n;
    ros::Publisher pubImage = n.advertise<sensor_msgs::Image>("/minirobot/hardware/image", 1);
    ros::Publisher pubJpeg  = n.advertise<std_msgs::UInt8MultiArray>("/minirobot/hardware/img_compressed", 1);
    ros::Rate loop(60);

    cv::VideoCapture cap(0);
    if(!cap.isOpened())
    {
        std::cout << "Cannot open camera ... :'(" << std::endl;
        return -1;
    }
    
    sensor_msgs::Image msgImage;
    std_msgs::UInt8MultiArray msgCompressed;
    
    //cv::waitKey(100);
    //cv::Mat firstFrame;
    //cap >> firstFrame;
    //std::cout<<"Rows: "<<firstFrame.rows<<"  Cols: "<<firstFrame.cols <<"  ElemSize: "<<firstFrame.elemSize()<< std::endl;
    //int imageSize = firstFrame.rows*firstFrame.cols*firstFrame.elemSize();

    msgImage.header.frame_id = "camera_link";
    msgImage.data.resize(320*240*3);
    msgImage.height = 240;
    msgImage.width  = 320;
    std::vector<int> compressionParams(2);
    std::vector<uchar> compressedBuff;
    compressionParams[0] = cv::IMWRITE_JPEG_QUALITY;
    compressionParams[1] = 95;

    while(ros::ok() && cv::waitKey(15))
    {
        cv::Mat frame;
        cv::Mat downsampledFrame;
        cap >> frame;
        cv::resize(frame, downsampledFrame, cv::Size(320,240));
        
        msgImage.header.stamp = ros::Time::now();
        memcpy(msgImage.data.data(), downsampledFrame.data, 320*240*3);
        
        cv::imencode(".jpg", downsampledFrame, msgCompressed.data, compressionParams);
        
        pubImage.publish(msgImage);
        pubJpeg.publish(msgCompressed);
        ros::spinOnce();
    }
    return 0;
}
