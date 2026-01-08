#!/usr/bin/env python

import numpy as np
import sys
import cv2
import time
import rospy
import tf2_ros
import math
import tf.transformations


from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from collections import namedtuple
Pose = namedtuple('Pose', ['x', 'y', 'yaw'])

class OccupancyGridMap:
    def __init__(self):
        # Topics & Subs, Pubs
        # Read paramters form params.yaml

        self.lidarscan_topic = rospy.get_param('~scan_topic')
        self.odom_topic = rospy.get_param('~odom_topic')
        #self.lidarscan_topic = "/scan2"
        #self.odom_topic = "/odom"
   

        self.t_prev = rospy.get_time()
        self.max_lidar_range = rospy.get_param('~scan_range')
        self.scan_beams = rospy.get_param('~scan_beams')
        self.scan_increment = (2*math.pi)/self.scan_beams


        # Read Occupancy grid map parameters from params.yaml
        self.map_topic = rospy.get_param('~occ_map_topic')
        self.p_occ = rospy.get_param('~p_occ')
        self.p_free = rospy.get_param('~p_free')
        self.map_width = rospy.get_param('~map_width')
        self.map_height = rospy.get_param('~map_height')
        self.map_res = rospy.get_param('~map_res')
        self.object_size = rospy.get_param('~object_size')
        
        # Set origin of the map to the center of the map
        self.map_origin_x = - (self.map_width * self.map_res) / 2.0
        self.map_origin_y = - (self.map_height * self.map_res) / 2.0


        # Initialize the map meta info in the Occupancy Grid Message, e.g., frame_id, stamp, resolution, width, height, etc.
        # Create the OccupancyGrid message and assign map data
        self.map_occ_grid_msg = OccupancyGrid()
        self.map_occ_grid_msg.header.frame_id = 'odom' #rospy.get_param('~map_frame')
        self.map_occ_grid_msg.header.stamp = rospy.Time.now()
        self.map_occ_grid_msg.info.resolution = self.map_res
        self.map_occ_grid_msg.info.width = self.map_width
        self.map_occ_grid_msg.info.height = self.map_height
        self.map_occ_grid_msg.info.origin.position.x = self.map_origin_x
        self.map_occ_grid_msg.info.origin.position.y = self.map_origin_y
        self.map_occ_grid_msg.info.origin.position.z = 0
        self.map_occ_grid_msg.info.origin.orientation.x = 0
        self.map_occ_grid_msg.info.origin.orientation.y = 0
        self.map_occ_grid_msg.info.origin.orientation.z = 0
        self.map_occ_grid_msg.info.origin.orientation.w = 1

        #init log_odd list and msg data
        self.log_odd_list = [0]*self.map_width*self.map_height
        self.log_ratio_list = [1]*self.map_width*self.map_height
        self.map_occ_grid_msg.data = [-1] * (self.map_width * self.map_height)

        # Initialize the robot pose
        #robot_pose = (0.0, 0.0, 0.0)
        self.robot_pose = Pose(0.0, 0.0, 0.0)
    
        #Subscribe to Lidar scan and odomery topics with corresponding lidar_callback() and odometry_callback() functions 
        self.lidar_sub = rospy.Subscriber(self.lidarscan_topic, LaserScan, self.lidar_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        
        # Create a publisher for the Occupancy Grid Map
        self.map_pub = rospy.Publisher(self.map_topic, OccupancyGrid, queue_size=1)

        rospy.loginfo("Occupancy Grid Map Node Online")


    # lidar_callback () uses the current LiDAR scan and Wheel Odometry data to uddate and publish the Grid Occupancy map 
    def lidar_callback(self, data):      

        # Compute log-odds update values for occupancy and free measurements
        ## TODO
        self.log_probability_occ = self.p_occ / (1 - self.p_occ)
        self.log_probability_fre = self.p_free / (1 - self.p_free)
        self.log_odds_occ = math.log(self.log_probability_occ)
        self.log_odds_free = math.log(self.log_probability_fre)



        lidar_map_x = self.robot_pose.x + 0.1288*math.cos(self.robot_pose.yaw)
        lidar_map_y = self.robot_pose.y + 0.1288*math.sin(self.robot_pose.yaw)
        lidar_map_angle = self.robot_pose.yaw + math.pi  # + math.pi since mounted backward

        #for each cell in the map
        for i in range(self.map_width):
             for j in range(self.map_height):
                    
                    cell_index = i + self.map_height * j  #index of cell in 1D arrayy

                    #calculate cell
                    cell_map_x  = self.map_origin_x + (i + 0.5)*self.map_res    # the cell x value respect to odom
                    cell_map_y  = self.map_origin_y + (j + 0.5)*self.map_res
                    
                    dx = cell_map_x  - lidar_map_x
                    dy = cell_map_y  - lidar_map_y
                    cell_lidar_distance = math.sqrt(dx**2 + dy**2)

                    cell_lidar_angle = (math.atan2(dy,dx) - lidar_map_angle) % (2*math.pi) # caculate angle between cell and lidar and round the result
                    scan_index = int(cell_lidar_angle / self.scan_increment)   #the point in the 720 scan point 

                    # calculate detected object 
                    obj_lidar_distance = data.ranges[scan_index] # get and clip range data
                    obj_lidar_angle = scan_index * self.scan_increment
                    
                    if((abs(cell_lidar_angle - obj_lidar_angle) < self.scan_increment)):
                        if(obj_lidar_distance > self.max_lidar_range):
                            cell_log_odd = 0

                        else:
                            if (abs(cell_lidar_distance - obj_lidar_distance) < self.map_res * math.sqrt(2)):                      
                                cell_log_odd = self.log_odds_occ #valid obj reading

                            elif(cell_lidar_distance < obj_lidar_distance):
                                cell_log_odd = self.log_odds_free

                            else:
                                cell_log_odd = 0
                        
                    else:
                        cell_log_odd = 0
                        
                    self.log_odd_list[cell_index] += cell_log_odd
                    self.log_odd_list[cell_index] = np.clip(self.log_odd_list[cell_index], -100, 100)

                    cell_log_odd = self.log_odd_list[cell_index]
                    
                    cell_prob = ((1 - (1 / (1 + np.exp(cell_log_odd)))))  #cacuate probablity of the cell
                    if(cell_prob <= 0.2):
                        self.map_occ_grid_msg.data[cell_index] = 0
                    elif(cell_prob >= 0.8):
                        self.map_occ_grid_msg.data[cell_index] = 100
                    else:
                        self.map_occ_grid_msg.data[cell_index] = -1
                            

               
                         
                    

        # Publish to map topic
        self.map_occ_grid_msg.header.stamp = rospy.Time.now()
        self.map_pub.publish(self.map_occ_grid_msg)

    # odom_callback() retrives the wheel odometry data from the publsihed odom_msg
    def odom_callback(self, odom_msg):
        
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

        # Log the retrieved odometry information
        # rospy.loginfo("Wheel Odometry - Position: x=%f, y=%f, z=%f | Yaw: %f", x, y, z, yaw)

def main(args):
    rospy.init_node("occupancygridmap", anonymous=True)
    OccupancyGridMap()
    rospy.sleep(0.1)
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)

