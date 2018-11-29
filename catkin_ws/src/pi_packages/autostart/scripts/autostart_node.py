#!/usr/bin/env python

import RPi.GPIO as GPIO
import rospy
import sys
import os
import socket
import time
from subprocess import call
from std_msgs.msg import Float32MultiArray

def callback(sensors):
    global battery
    global watchdog 
    battery = sensors.data[14]/102.4
    watchdog = 0

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(3, GPIO.OUT)
    GPIO.setup(5, GPIO.IN)
    led_value = False
    GPIO.output(3, led_value)

    rospy.init_node('status_checker')
    os.system('rosrun rosserial_python serial_node.py _port:=/dev/ttyACM0 _baud:=500000 &')
    os.system('rosrun camera_test camera_test_node &')
    rospy.Subscriber('/minirobot/hardware/sensors', Float32MultiArray, callback)
    rate = rospy.Rate(30)

    counter = 0
    btn = 0
    led_period = 15

    global battery
    global watchdog 
    watchdog = 0
    battery = 0

    
    while not rospy.is_shutdown():
        #Keep led blinking
        counter = counter + 1
        if counter > led_period:
            counter = 0
            led_value = not led_value
            GPIO.output(3, led_value)

        #Check if push-button has been pressed (at least one second)
        if GPIO.input(5) == 0:
            btn = btn + 1
            if btn == 30:
                call('halt', shell=False)
        else:
            btn = 0

        #Change blinking period when battery is low
        if battery > 7.6:
            led_period = 15
        else:
            led_period = 3

        watchdog = watchdog + 1
        if watchdog == 60:
            watchdog = 0
            battery = 0
            os.system('rosnode kill serial_node')
            os.system('rosrun rosserial_python serial_node.py _port:=/dev/ttyACM0 _baud:=500000 &')
            time.sleep(5)

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
            
