#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import numpy as np
import time

import cv2
import tf2_ros
import tf.transformations

#ROS Imports
import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from collections import namedtuple
Pose = namedtuple('Pose', ['x', 'y', 'yaw'])



class WallFollow:
    def __init__(self):
        self.rad2deg = 1.0/ math.pi * 180
        # Read the Wall-Following controller paramters form params.yaml
        # ...
        self.lidarscan_topic = rospy.get_param('~scan_topic')
        self.odom_topic = rospy.get_param('~odom_topic')
        self.drive_topic = rospy.get_param('~nav_topic')

        #car property
        self.max_speed = rospy.get_param('~max_speed')
        self.max_steering_angle = rospy.get_param('~max_steering_angle')
        #self.shaft_length = rospy.get_param('~l_cg2front')  #l
        self.shaft_length = 0.272

        #lidar values
        self.t_prev = rospy.get_time()
        self.max_lidar_range = rospy.get_param('~scan_range')
        self.scan_beams = rospy.get_param('~scan_beams')
        self.scan_increment = (2*math.pi)/self.scan_beams

        # Subscrbie to LiDAR scan Wheel Odometry topics. This is to read the LiDAR scan data and vehicle actual velocity
        rospy.Subscriber(self.lidarscan_topic, LaserScan, self.lidar_callback,queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback,queue_size=1)


        # Create a publisher for the Drive topic
        self.drive_pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=1)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.stamp = rospy.Time.now()
        self.drive_msg.header.frame_id = "base_link"  # Optional but good practice

        
        #car feedback
        self.vs = 0.0
        self.robot_pose = Pose(0.0, 0.0, 0.0)

        #design parameters 
        #wall
        
        self.al_index = rospy.get_param('~al_index')
        self.bl_index = rospy.get_param('~bl_index')
        self.ar_index = rospy.get_param('~ar_index')
        self.br_index = rospy.get_param('~br_index')

        self.al_angle = float(self.al_index)/self.scan_beams *2*math.pi  #convert angle index to rad
        self.bl_angle = float(self.bl_index)/self.scan_beams *2*math.pi
        self.ar_angle = float(self.ar_index)/self.scan_beams *2*math.pi
        self.br_angle = float(self.br_index)/self.scan_beams *2*math.pi

        self.theta_l = abs(self.al_angle - self.bl_angle) #theta_l
        self.theta_r = abs(self.ar_angle - self.br_angle)
        
        self.al = 0.0
        self.bl = 0.0
        self.ar = 0.0
        self.br = 0.0

        self.dl = 0.0 #distence to left wall
        self.dr = 0.0
        self.dlr = 0.0  #diatence between left and right wall
        self.target_dlr = 0.0 #deaired target dlr value
        self.err_dlr = 0.0
        self.rate_dlr = 0.0
        self.accel_dlr = 0.0

        self.beta_l = 0.0
        self.beta_r = 0.0
        self.alpha_l = 0.0
        self.alpha_r = 0.0
        

        #obj viewing
        self.viewing_index_start = rospy.get_param('~viewing_index_start')
        self.viewing_index_end = rospy.get_param('~viewing_index_end')
        self.viewing_angle = (self.viewing_index_start - self.viewing_index_end)/2
        
        self.stop_distence = rospy.get_param('~stop_distence')
        self.decay_constant = rospy.get_param('~decay_constant')
        self.obj_distence = 0


        #control
        self.target_str_angle = 0.0
        self.target_vs = rospy.get_param('~normal_speed')
        self.cmd_vs = 0.0

        #pid
        self.kp = 3.0
        self.kd = 3.0
        self.ka = 1.5

        self.count = 0 

        #self.log_data = []  # List to store logged data
        

     # The LiDAR callback function is where you read LiDAR scan data as it becomes availble and compute the vehile veloicty and steering angle commands
    
    def lidar_callback(self, data):      
        car_angle = self.robot_pose.yaw * self.rad2deg
        #print("\nindex",self.count,"yaw_angle",car_angle,"==============>")
      # Exttract the parameters of two walls on the left and right side of the vehicles. Referrring to Fig. 1 in the lab instructions, these are al, bl, thethal, ... 
      # ...
        self.al = np.clip(data.ranges[self.al_index],1e-2,self.max_lidar_range) #if data exceed range, stay as original value
        self.bl = np.clip(data.ranges[self.bl_index],1e-2,self.max_lidar_range)
        self.ar = np.clip(data.ranges[self.ar_index],1e-2,self.max_lidar_range)
        self.br = np.clip(data.ranges[self.br_index],1e-2,self.max_lidar_range)
        
        self.beta_l =  math.atan2((self.al * math.cos(self.theta_l) - self.bl) , (self.al * math.sin(self.theta_l))) 
        self.beta_r =  math.atan2((self.ar * math.cos(self.theta_r) - self.br) , (self.ar * math.sin(self.theta_r))) 

        self.alpha_l =  (-self.beta_l) + ((3.0 * math.pi) / 2.0) - self.bl_angle 
        self.alpha_r =  self.beta_r + (math.pi /2.0) - self.br_angle
	print("al",self.alpha_l,"ar",self.alpha_r)
        

        self.dl = abs(self.bl * math.cos(self.beta_l)) # Distance to Left Wall
        self.dr = abs(self.br * math.cos(self.beta_r)) # Diostance to Right Wall

        self.dlr = self.dl - self.dr # current distance between 2 wall
        self.err_dlr = self.dead_band((self.dlr - self.target_dlr),0.001)     # error distance - for centering
        self.rate_dlr =(-self.vs*math.sin(self.alpha_l) - self.vs*math.sin(self.alpha_r))
	print("dlr",self.vs)


        
      # Compute the steering angle command to maintain the vehicle in the middle of left and and right walls
      # ...  

        control_term = ((-self.kp) * self.err_dlr) - (self.kd * self.rate_dlr) 
	
        denom = (self.vs**2) * (math.cos(self.alpha_r) + math.cos(self.alpha_l))
        
        self.target_str_angle = math.atan2(-self.shaft_length*control_term , denom) # /(math.pi/2.0) * self.max_steering_angle
	      
        self.target_str_angle = np.clip(self.target_str_angle, -self.max_steering_angle, self.max_steering_angle)
      
	
        self.count += 1
        

      # Find the closest obstacle point within a narrow viewing angle in front of the vehicle and compute the vehicle velocity command accordingly
      #  ... 
        self.obj_distence =  min(data.ranges[self.viewing_index_start:self.viewing_index_end])
        self.cmd_vs = self.target_vs*(1-math.exp(-max(self.obj_distence-self.stop_distence,0)/self.decay_constant))
        self.cmd_vs = np.clip(self.cmd_vs, 0.0, self.max_speed)

	
	

        #turning_penalty = 1 - 0.5 * (abs(self.target_str_angle) / self.max_steering_angle)  
        #self.cmd_vs = self.cmd_vs * turning_penalty
      # Publish steering angle and velocity commnads to the Drive topic
      # ...




        self.drive_msg.header.stamp = rospy.Time.now()
        self.drive_msg.drive.speed = self.cmd_vs 
        self.drive_msg.drive.steering_angle = self.target_str_angle
        self.drive_pub.publish(self.drive_msg)
        

    # The Odometry callback reads the actual vehicle velocity from VESC. 
    
    def odom_callback(self, odom_msg):
        # update current speed
        self.vs = odom_msg.twist.twist.linear.x #just vx

        # Extract position data
        x = odom_msg.pose.pose.position.x 
        y = odom_msg.pose.pose.position.y
        z = odom_msg.pose.pose.position.z

        # Extract orientation (as a quaternion)
        orientation_q = odom_msg.pose.pose.orientation

        # Convert the quaternion to Euler angles
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
        
        self.robot_pose = Pose(x,y,yaw)

    def valid_reading(self,reading,fallback):
      if(reading > self.max_lidar_range):
          return fallback
      else:
          return reading

    def dead_band(self, value, dead_band_value):
      if abs(value) < dead_band_value:
          return 0.0
      else:

          return value
    

         
def main(args):
    rospy.init_node("WallFollow_node", anonymous=True)
    wf = WallFollow()
    rospy.sleep(0.1)
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)

