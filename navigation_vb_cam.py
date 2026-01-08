#!/usr/bin/env python
from __future__ import print_function
from lib2to3.pytree import Node
import sys
import math
from tokenize import Double
import numpy as np
import time
import quadprog

from  numpy import array, dot
from quadprog import solve_qp

#ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from sensor_msgs.msg import CameraInfo
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

import sensor_msgs.msg
import cv_bridge
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg


from collections import namedtuple

Pose = namedtuple('Pose', ['x', 'y', 'yaw'])

class GapBarrier:
    def __init__(self):
        
        #Topics & Subs, Pubs
        # Read the algorithm parameter paramters form params.yaml
        self.lidarscan_topic = rospy.get_param('~scan_topic')
        self.odom_topic = rospy.get_param('~odom_topic')
        self.drive_topic = rospy.get_param('~nav_topic')

        #car property
        self.max_speed = rospy.get_param('~max_speed')
        self.normal_speed = rospy.get_param('~normal_speed')
        self.max_steering_angle = rospy.get_param('~max_steering_angle')
        self.shaft_length = 0.272  #rospy.get_param('~l_cg2front')  #l

        #lidar values
        self.t_prev = rospy.get_time()
        self.max_lidar_range = rospy.get_param('~scan_range')
        self.scan_beams = rospy.get_param('~scan_beams')
        self.scan_increment = (2*math.pi)/self.scan_beams

        # Add your subscribers for LiDAR scan and Odomotery here
        # ...
        rospy.Subscriber(self.lidarscan_topic, LaserScan, self.lidar_callback,queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback,queue_size=1)
        
        # Add your publisher for Drive topic here
        #...
        # Create a publisher for the Drive topic
        self.drive_pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=1)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.stamp = rospy.Time.now()
        self.drive_msg.header.frame_id = "base_link"  


        self.rad2deg = 1.0 / math.pi * 180
        self.deg2rad = 1.0 / 180.0 * math.pi
        self.index2rad = 1.0/self.scan_beams * (2.0*math.pi)
        self.rad2index = 1.0 / (2.0*math.pi) * self.scan_beams
        self.deg2index = 1.0 / 360.0 * self.scan_beams
        self.index2deg = 1.0 / self.scan_beams * 360.0
	    

        #walls 
        self.wr_start_index = rospy.get_param('~br_index') #br ar al bl
        self.wr_end_index = rospy.get_param('~ar_index')
        self.wl_start_index = rospy.get_param('~al_index') 
        self.wl_end_index = rospy.get_param('~bl_index')

        self.beam_a_index_shift = rospy.get_param('~beam_a_index_shift')
        self.beam_b_index_shift = rospy.get_param('~beam_b_index_shift')

        self.beta_l = 0.0
        self.beta_r = 0.0
        self.alpha_l = 0.0
        self.alpha_r = 0.0

        self.dl = 0.0 #distence to left wall
        self.dr = 0.0
        self.dlr = 0.0  #diatence between left and right wall
        self.target_dlr = 0.0 #deaired target dlr value
        self.err_dlr = 0.0
        self.rate_dlr = 0.0
        self.accel_dlr = 0.0

        #pid
        self.kp = 6.0
        self.kd = 4.0

        #control
        self.target_str_angle = 0.0
        self.target_vs = rospy.get_param('~normal_speed')
        self.cmd_vs = 0.0

        # Initialize varables as needed 
        #...
        #car feedback
        self.vs = 0.0
        self.robot_pose = Pose(0.0, 0.0, 0.0)
        
        #fov values
        self.fov_angle = rospy.get_param('~fov_angle')
        self.fov_index_range = int(self.fov_angle*self.deg2index)
        self.fov_start_index = int((self.scan_beams/2.0) - (self.fov_index_range/2.0))
        self.fov_end_index   = int((self.scan_beams/2.0) + (self.fov_index_range/2.0))
        
        self.fov_safe_distance = rospy.get_param('~fov_safe_distance')
        self.tgt_direction = 0.0 #target angle in rad

        #obj viewing
        self.viewing_angle = rospy.get_param('~viewing_angle')
        self.viewing_index = int(self.viewing_angle * self.deg2index)
        self.viewing_index_start = int(self.scan_beams/2 - self.viewing_index/2)
        self.viewing_index_end = int(self.scan_beams/2 + self.viewing_index/2)
        
        self.close_distance = rospy.get_param('~close_distence')
        self.stop_distence = rospy.get_param('~stop_distence')
        self.decay_constant = rospy.get_param('~decay_constant')
        self.obj_distance = 0

        self.tgt_marker_pub = rospy.Publisher('/best_travel_angle_marker', Marker, queue_size=1)
        self.left_wall_pub = rospy.Publisher('/left_wall_marker', Marker, queue_size=1)
        self.right_wall_pub = rospy.Publisher('/right_wall_marker', Marker, queue_size=1)


        #cam obj view
        self.ground_threshold  = rospy.get_param('~ground_threshold')

        #cam
        self.depth_topic = rospy.get_param('~depth_image_topic')
        self.camera_info = rospy.get_param('~camera_info_topic')
        rospy.Subscriber(self.depth_topic, Image, self.image_callback, queue_size=1)
        rospy.Subscriber(self.camera_info, CameraInfo, self.camera_info_callback, queue_size=1)

        self.cam_obj_distance = None

        #cam pub
        # Create a publisher for the Drive topic
        self.cam_pub = rospy.Publisher('/obstacle_points', PointCloud2, queue_size=1)
        
        # Initialize the CvBridge once
        self.bridge = CvBridge()

        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.cam_width = 0.0
        self.cam_height = 0.0

        #other
        self.count = 0.0

    # process LiDAR by considering returns within range and set the unsafe value to zero and calc free length and index
    def process_lidar(self, ranges):
        max_free_len = 0
        max_free_len_start = 0
        current_free_len = 0
        current_free_len_start = 0

        ranges =  list(ranges[self.fov_start_index : self.fov_end_index])
        ranges = np.clip(ranges, 0.0,self.max_lidar_range)

        for i in range(len(ranges)):
            ranges[i] = self.dead_band(ranges[i], self.fov_safe_distance)
            
            if(ranges[i] > 0):
                current_free_len += 1
            else:            
                if (current_free_len > max_free_len):
                    max_free_len = current_free_len
                    max_free_len_start = current_free_len_start

                current_free_len_start = i+1
                current_free_len = 0

        if (current_free_len > max_free_len):
                    max_free_len = current_free_len
                    max_free_len_start = current_free_len_start    
        print(self.fov_end_index, max_free_len, max_free_len_start)
        return ranges, max_free_len, max_free_len_start
    
    #function to find the best direction of travel
    def find_best_direction(self, processed_ranges, max_free_len, max_free_len_start):
        numerator_sum = 0.0
        denominator_sum = 0.0
        distence = 0.0
        angle = 0.0

        for i in range(max_free_len):
            index = max_free_len_start + i

            distence = processed_ranges[index]
            angle = (index + self.fov_start_index)*self.index2rad

            numerator_sum +=  distence * angle
            denominator_sum += distence

        if denominator_sum == 0:
            return 0.0
        else:
            tgt_angle = numerator_sum / denominator_sum
            return tgt_angle


    def preprocess_wall_obstacles(self,tgt_direction,ranges):
        tgt_index = int(tgt_direction*self.rad2index)

        self.wr_start_index = (tgt_index - self.beam_b_index_shift)% self.scan_beams
        self.wr_end_index = (tgt_index - self.beam_a_index_shift)% self.scan_beams
        self.wl_start_index = (tgt_index + self.beam_a_index_shift)% self.scan_beams
        self.wl_end_index = (tgt_index + self.beam_b_index_shift)% self.scan_beams

        if self.wl_start_index < self.wl_end_index:
            left_obstacles = ranges[self.wl_start_index : self.wl_end_index]
        else:
            left_obstacles = ranges[self.wl_start_index :] + ranges[: self.wl_end_index]

        if self.wr_start_index < self.wr_end_index:
            right_obstacles = ranges[self.wr_start_index : self.wr_end_index]
        else:
            right_obstacles = ranges[self.wr_start_index :] + ranges[: self.wr_end_index]
        
        
        return left_obstacles,right_obstacles


   # Optional function to set up and solve the optimization problem for parallel virtual barriers 
    def getWalls(self, left_obstacles, right_obstacles, wl0, wr0, alpha):
        epsilon = 1e-3
        G = np.diag([1.0, 1.0, 1e-4])
        a = np.zeros(3)

        left_pts = []
        for i in range(len(left_obstacles)):
            left_angle = (self.wl_start_index+i)*self.index2rad - math.pi #relative to upper vertical axis
            y = left_obstacles[i]*math.sin(left_angle)
            x = left_obstacles[i]*math.cos(left_angle)
            left_pts.append([-x, -y, -1])

        C_left = np.array(left_pts) 
        b_left = np.ones(len(left_pts))

        right_pts = []
        for j in range(len(right_obstacles)): 
            right_angle = (self.wr_start_index+j)*self.index2rad - math.pi/2.0  #relative to right horizontal axis
            y = -right_obstacles[j]*math.cos(right_angle)
            x = right_obstacles[j]*math.sin(right_angle)
            right_pts.append([x, y, 1])

        C_right = np.array(right_pts) 
        b_right = np.ones(len(right_pts))

        C_b = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0]
        ])
        b_b = np.array([-1 + epsilon, -1 + epsilon])

        # Combine all constraints:
        C_total = np.vstack([C_right,C_left, C_b])
        b_total = np.concatenate((b_right, b_left, b_b))
        
        solution = solve_qp(G, a, C_total.T, b_total,0)[0]
        
        w = solution[0:2]
        b = solution[2]

        w_r = w / (b - 1.0 + epsilon)  # added a small esp to avoid division by zero
        w_l = w / (b + 1.0 + epsilon)

        # Compute the distance to the walls
        d_l = 1.0 / math.sqrt(w_l[0]**2 + w_l[1]**2)
        d_r = 1.0 / math.sqrt(w_r[0]**2 + w_r[1]**2)
        
        return w_l, w_r, d_l, d_r

    def obj_viewing_control(self,data):
        #self.obj_distance =  min(data.ranges[self.viewing_index_start:self.viewing_index_end])

        lidar_distance = min(data.ranges[self.viewing_index_start:self.viewing_index_end])

        if (self.cam_obj_distance != None):
            self.obj_distance = min(lidar_distance, self.cam_obj_distance)
        else:
            self.obj_distance = lidar_distance

        self.cmd_vs = self.target_vs*(1-math.exp(-max(self.obj_distance-self.stop_distence,0)/self.decay_constant))
        self.cmd_vs = np.clip(self.cmd_vs, 0.0, self.max_speed)

        if(self.obj_distance < self.close_distance):
            scale = 1 + (self.close_distance - self.obj_distance) / self.close_distance
            scale = min(scale, 3.0)  # limit to 3x amplification (tune this as needed)
            self.target_str_angle = np.clip(self.target_str_angle * scale, -self.max_steering_angle, self.max_steering_angle)
            if(self.cmd_vs>0.1):
                self.cmd_vs = max(self.cmd_vs * self.obj_distance / self.close_distance,0.1)
        return 1
    
    # This function is called whenever a new set of LiDAR data is received; bulk of your controller implementation should go here 
    def lidar_callback(self, data):      
        processed_ranges, max_free_len, max_free_len_start = self.process_lidar(data.ranges)
        self.tgt_direction = self.find_best_direction(processed_ranges, max_free_len, max_free_len_start)

        self.publish_best_travel_marker(self.tgt_direction)

        left_obstacles,right_obstacles = self.preprocess_wall_obstacles(self.tgt_direction,data.ranges)
        w_l,w_r,d_l,d_r = self.getWalls(left_obstacles,right_obstacles, self.wl_start_index, self.wr_start_index , 0)
        
        #print("left_obstacles: ", left_obstacles)
        #print("right_obstacles: ", right_obstacles)
        
         # compute the distance to the walls    
        
        
        # Normalize the wall vectors
        self.wl_norm = w_l / np.linalg.norm(w_l)
        self.wr_norm = w_r / np.linalg.norm(w_r)


        left_marker = self.publish_wall_marker(self.wl_norm, d_l, "left_wall", 1, (0.0, 1.0, 0.0))   # Green
        right_marker = self.publish_wall_marker(self.wr_norm, d_r, "right_wall", 2, (0.0, 0.0, 1.0))  # Blue

        self.left_wall_pub.publish(left_marker)
        self.right_wall_pub.publish(right_marker)

        vehicle_velocity = np.array([self.vs, 0])

        # Computer derivative of the distance to the walls
        self.dl_dot = np.dot(vehicle_velocity, self.wl_norm) 
        self.dr_dot = np.dot(vehicle_velocity, self.wr_norm)
        
        self.alpha_l =  math.acos(np.dot(np.array([0, -1]), self.wl_norm))
        self.alpha_r =  math.acos(np.dot(np.array([0, 1]), self.wr_norm))

        self.cos_alpha_l = np.dot(np.array([0, -1]), self.wl_norm)
        self.cos_alpha_r = np.dot(np.array([0, 1]), self.wr_norm)
        
        self.dlr = (d_l) - (d_r) # current distance between 2 wall
        self.err_dlr = (self.dlr - self.target_dlr)     # error distance - for centering
        #self.rate_dlr =(-self.vs*math.sin(self.alpha_l) - self.vs*math.sin(self.alpha_r))
        self.rate_dlr = self.dl_dot - self.dr_dot
        self.err_dlr = self.dead_band(self.err_dlr, 0.03)

        control_term = ((-self.kp) * self.err_dlr) - (self.kd * self.rate_dlr) 
        #denom = (self.vs**2) * (math.cos(self.alpha_r) + math.cos(self.alpha_l))
        denom = (self.vs**2) * (self.cos_alpha_r + self.cos_alpha_l) + 1e-3 #avoid division by zero)

        self.target_str_angle = math.atan2(-self.shaft_length*control_term , denom) #/(math.pi/2) * self.max_steering_angle  *1.3   
        self.target_str_angle = np.clip(self.target_str_angle, -self.max_steering_angle, self.max_steering_angle)
        #print("steering_angle",(self.target_str_angle*self.rad2deg))
        
        #obj_viewing ctrl
        self.obj_viewing_control(data)

        #
        self.target_str_angle = np.clip(self.target_str_angle, -self.max_steering_angle, self.max_steering_angle)
        if(self.cmd_vs > 0.1):
            self.cmd_vs = np.clip(self.cmd_vs, 0.4, self.max_speed)
        else:
            self.cmd_vs = 0.0
        

        # Publish the steering and speed commands to the drive topic
        self.drive_msg.header.stamp = rospy.Time.now()
        self.drive_msg.drive.speed = self.cmd_vs
        self.drive_msg.drive.steering_angle = self.target_str_angle
        self.drive_pub.publish(self.drive_msg)

        print("\n",self.count,"speed {:.3f},target_str_angle: {:.3f}, target_vs: {:.3f} , obj_distance: {:.3f}".format(self.vs,self.target_str_angle, self.cmd_vs,self.obj_distance))

        #print("alpha_l: {:.3f}, alpha_r: {:.3f},best dir: {:.3f}".format(self.alpha_l, self.alpha_r,self.tgt_direction))
        print("d_l:{:.3f}, d_r:{:.3f}, dlr{:.3f}, dl_dot:{:.3f} ".format(d_l, d_r,self.dlr, self.dl_dot))
        #print("wl_norm: , wr_norm: ,  w_l: , w_r: \n",self.wl_norm, self.wr_norm,w_l, w_r)

        self.count = self.count + 1


    # Odometry callback 
    def odom_callback(self, odom_msg):
        # update current speed
         self.vs = odom_msg.twist.twist.linear.x

    def image_callback(self, img_msg):
        self.bridge = CvBridge()
        # Convert the ROS image message to a CV2 image
        depth_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        depth_array = np.array(depth_image, dtype=np.float32) /1000.0

        x_indices = np.arange(self.cam_width)
        y_indices = np.arange(self.cam_height)
        X, Y = np.meshgrid(x_indices, y_indices)

        #x_3d = (X - self.cx) * depth_array / self.fx
        #y_3d = (Y - self.cy) * depth_array / self.fy
#
        #point_cloud = np.dstack((x_3d, y_3d, depth_array)) # Create a 3D point cloud
        ## Flatten the point cloud to a 2D array 
        #point_cloud = point_cloud.reshape(-1, 3)
#
        stride = 14  # down sampling rate

        # Subsample X, Y, and depth
        X_ds = X[::stride, ::stride]
        Y_ds = Y[::stride, ::stride]
        depth_ds = depth_array[::stride, ::stride]
        #print("depth_ds shape: ", depth_ds)


        x_3d = (X_ds - self.cx) * depth_ds / self.fx
        y_3d = (Y_ds - self.cy) * depth_ds / self.fy

        point_cloud_ds = np.dstack((x_3d, y_3d, depth_ds))
        point_cloud_ds = point_cloud_ds.reshape(-1, 3)
        
        valid_mask = point_cloud_ds[:, 2] > 0
        point_cloud_ds = point_cloud_ds[valid_mask]

        #np.savetxt("cloud.csv", point_cloud_ds, delimiter=",", header="X,Y,Z", comments="")

        transformed_points = self.transform_camera_to_base_link(point_cloud_ds)
        
        
        #filtering the ground
        #height clip
        z = transformed_points[:, 2]
        z_mask = (z > 0.02) #not_ground_mask
        # Side clip
        y = transformed_points[:, 1]
        y_mask = (y > -2.0) & (y < 2.0)

        mask = z_mask & y_mask
        obstacle_points = transformed_points[mask]

        if obstacle_points.shape[0] > 0:
            self.cam_obj_distance = np.min(obstacle_points[:, 0])
        else:
            self.cam_obj_distance = None

        print(self.cam_obj_distance)

        self.publish_pointcloud(obstacle_points)

        #points_lidar_plane = transformed_points[:, [0, 1]]


        #np.savetxt("cloud_baselink.csv", points_lidar_plane, delimiter=",", header="X,Y", comments="")


    def dead_band(self, value, dead_band_value):
      if abs(value) < dead_band_value:
          #print("value %.3f" % (value),"cliped to 0")
          return 0.0
      else:
          #print("value %.3f" % (value),"unchanged")
          return value
    
    def transform_camera_to_base_link(self, points_cam):
        # Extract individual axes
        x_cam = points_cam[:, 0]  # right
        y_cam = points_cam[:, 1]  # down
        z_cam = points_cam[:, 2]  # forward

        # Convert to base_link axes
        x_base = z_cam
        y_base = -x_cam
        z_base = -y_cam

        # Translate to base_link origin
        translation = np.array([0.28303, 0.0, 0.086])  # meters
        points_base = np.stack((x_base, y_base, z_base), axis=-1) + translation

        return points_base   
    def camera_info_callback(self, camera_info_msg):
        # Process camera info at the beginning of initialization
        # Extract camera intrinsic parameters
        # self.camera_info = camera_info_msg
        self.fx = camera_info_msg.K[0]
        self.fy = camera_info_msg.K[4]
        self.cx = camera_info_msg.K[2]
        self.cy = camera_info_msg.K[5]
        self.cam_width = camera_info_msg.width
        self.cam_height = camera_info_msg.height
    
    def publish_pointcloud(self, points):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "base_link"  # or whatever frame you're using

        # Must be float32 (RViz expects that)
        pc2_data = pc2.create_cloud_xyz32(header, points.astype(np.float32))

        self.cam_pub.publish(pc2_data)
          

    def publish_best_travel_marker(self, angle, frame_id="base_link"):
        """
        Publishes an arrow marker in RViz pointing in the direction of the best travel angle.
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "travel_angle"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        angle = angle - math.pi
        # Arrow start and end points
        marker.points = [Point(), Point()]
        marker.points[0].x = 0.0  # Start at origin
        marker.points[0].y = 0.0
        marker.points[0].z = 0.0

        length = 1.0  # Arrow length (meters)
        marker.points[1].x = length * math.cos(angle)
        marker.points[1].y = length * math.sin(angle)
        marker.points[1].z = 0.0

        # Arrow appearance
        marker.scale.x = 0.05  # shaft diameter
        marker.scale.y = 0.1   # head diameter
        marker.scale.z = 0.1   # head length

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Alpha = visible

        marker.lifetime = rospy.Duration(0.5)  # Auto-delete after 0.5s unless updated
        self.tgt_marker_pub.publish(marker)

    def publish_wall_marker(self, normal_vec, distance, ns, marker_id, color, frame_id="base_link"):
        """
        Publishes a wall marker (as a line) in RViz using wall normal vector and distance.
        - normal_vec: unit vector normal to the wall (2D)
        - distance: distance from origin to wall
        - ns: namespace for RViz marker
        - marker_id: unique ID for this marker
        - color: (r, g, b)
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Line width
        marker.scale.x = 0.05

        # Wall points (extend wall line in both directions)
        perp_vec = np.array([-normal_vec[1], normal_vec[0]])  # perpendicular direction
        wall_center = distance * np.array(normal_vec)

        p1 = wall_center + perp_vec * 2.0  # 2m left
        p2 = wall_center - perp_vec * 2.0  # 2m right

        marker.points.append(Point(x=p1[0], y=p1[1], z=0.0))
        marker.points.append(Point(x=p2[0], y=p2[1], z=0.0))

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(0.5)  # so it disappears if not updated
        return marker




def main(args):
    rospy.init_node("GapWallFollow_node", anonymous=True)
    wf = GapBarrier()
    rospy.sleep(0.1)
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)

