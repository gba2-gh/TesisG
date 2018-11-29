#include <opencv2/opencv.hpp>
#include "ros/ros.h"
#include "sensor_msgs/Image.h"

cv::Mat img;
cv::Mat img_grayscale;
int imgSize;

void callbackImage(const sensor_msgs::Image::ConstPtr& msg)
{
    memcpy(img.data, msg->data.data(), imgSize);
    cv::imshow("Image", img);
    cv::cvtColor(img, img_grayscale, CV_BGR2HSV);
    cv::imshow("Grayscale", img_grayscale);
}

int main(int argc, char** argv)
{
    std::cout << "Initalizing color-tracker node..." << std::endl;
    ros::init(argc, argv, "color_tracker");
    ros::NodeHandle n;
    ros::Subscriber subImage = n.subscribe("/minirobot/hardware/image", 1, callbackImage);
    ros::Rate loop(30);

    sensor_msgs::Image firstImg = *ros::topic::waitForMessage<sensor_msgs::Image>("/minirobot/hardware/image", ros::Duration(3));

    try
    {
        img = cv::Mat(firstImg.height, firstImg.width, CV_8UC3);
        imgSize = img.rows*img.cols*img.elemSize();
    }
    catch(...)
    {
        std::cout << "Cannot get image by subscribing topic... :'(" << std::endl;
        return -1;
    }

    while(ros::ok() && cv::waitKey(15) != 27)
    {
        ros::spinOnce();
        loop.sleep();
    }
}
