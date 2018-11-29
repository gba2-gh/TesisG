#!/usr/bin/env python

import RPi.GPIO as GPIO
import rospy
import sys
import os
import socket
from subprocess import call
from std_msgs.msg import Float32MultiArray

def callback(sensors):
    global battery
    battery = sensors.data[14]/102.4

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(3, GPIO.OUT)
    GPIO.setup(5, GPIO.IN)
    led_value = False
    GPIO.output(3, led_value)

    rospy.init_node('status_checker')
    freq = 4;

    os.system('rosrun rosserial_python serial_node.py _port:=/dev/ttyACM0 _baud:=500000 &')
    os.system('rosrun camera_test camera_test_node &')
    try:
        rospy.wait_for_message('/minirobot/hardware/sensors', Float32MultiArray, timeout=60)
    except rospy.ROSException:
        freq = 8;

    rospy.Subscriber('/minirobot/hardware/sensors', Float32MultiArray, callback)
    rateNormal  = rospy.Rate(freq)
    rateLowBatt = rospy.Rate(20)
    btn = 0
    global battery
    battery = 0
    
    while not rospy.is_shutdown():
        led_value = not led_value
        GPIO.output(3, led_value)
        if GPIO.input(5) == 0:
            btn = btn + 1
            if btn == freq:
                call('halt', shell=False)
        else:
            btn = 0
        if battery > 7.6:
            rateNormal.sleep()
        else:
            rateLowBatt.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
            
