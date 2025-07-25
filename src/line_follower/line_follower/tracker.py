#!/usr/bin/env python3


import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from line_interfaces.msg import Line
import tf_transformations as tft


#############
# CONSTANTS #
#############
_RATE = 10  # (Hz) rate for rospy.rate
_MAX_SPEED = 1.5  # (m/s)
_MAX_CLIMB_RATE = 1.0  # m/s
_MAX_ROTATION_RATE = 5.0  # rad/s
IMAGE_HEIGHT = 960
IMAGE_WIDTH = 1280
# Center of the image frame. We will treat this as the center of mass of the drone
CENTER = np.array([IMAGE_WIDTH//2, IMAGE_HEIGHT//2])
EXTEND = 300  # Number of pixels forward to extrapolate the line
KP_X = 0.015
KP_Y = 0.015
KP_Z_W = 0.05  # Proportional gains for x, y, and angular velocity control
DISPLAY = True


#########################
# COORDINATE TRANSFORMS #
#########################
class CoordTransforms():

    def __init__(self):
        """
        Variable Notation:
            - v__x: vector expressed in "x" frame
            - q_x_y: quaternion of "x" frame with relative to "y" frame
            - p_x_y__z: position of "x" frame relative to "y" frame expressed in "z" coordinates
            - v_x_y__z: velocity of "x" frame with relative to "y" frame expressed in "z" coordinates
            - R_x2y: rotation matrix that maps vector represented in frame "x" to representation in frame "y" (right-multiply column vec)

        Frame Subscripts:
            - m = marker frame (x-right, y-up, z-out when looking at marker)
            - dc = downward-facing camera (if expressed in the body frame)
            - fc = forward-facing camera
            - bu = body up frame (x-forward, y-left, z-up, similar to ENU)
            - bd = body down frame (x-forward, y-right, z-down, similar to NED)
            - lenu = local East-North-Up world frame ("local" implies that it may not be aligned with east and north, but z is up)
            - lned = local North-East-Down world frame ("local" implies that it may not be aligned with north and east, but z is down)
        Rotation matrix:
            R = np.array([[       3x3     0.0]
                          [    rotation   0.0]
                          [     matrix    0.0]
                          [0.0, 0.0, 0.0, 0.0]])

            [[ x']      [[       3x3     0.0]  [[ x ]
             [ y']  =    [    rotation   0.0]   [ y ]
             [ z']       [     matrix    0.0]   [ z ]
             [0.0]]      [0.0, 0.0, 0.0, 0.0]]  [0.0]]
        """

        # Reference frames
        self.COORDINATE_FRAMES = {'lenu', 'lned', 'bu', 'bd', 'dc', 'fc'}

        self.WORLD_FRAMES = {'lenu', 'lned'}

        self.BODY_FRAMES = {'bu', 'bd', 'dc', 'fc'}

        self.STATIC_TRANSFORMS = {'R_lenu2lenu',
                                  'R_lenu2lned',

                                  'R_lned2lenu',
                                  'R_lned2lned',

                                  'R_bu2bu',
                                  'R_bu2bd',
                                  'R_bu2dc',
                                  'R_bu2fc',

                                  'R_bd2bu',
                                  'R_bd2bd',
                                  'R_bd2dc',
                                  'R_bd2fc',

                                  'R_dc2bu',
                                  'R_dc2bd',
                                  'R_dc2dc',
                                  'R_dc2fc',
                                  'R_dc2lned',

                                  'R_fc2bu',
                                  'R_fc2bd',
                                  'R_fc2dc',
                                  'R_fc2fc'
                                  }

        self.R_dc2bd = np.array([
            [0.0, -1.0, 0.0, 0.0],  # bd.x = -dc.y
            [1.0, 0.0, 0.0, 0.0],  # bd.y = dc.x
            [0.0, 0.0, 1.0, 0.0],  # bd.z = dc.z
            [0.0, 0.0, 0.0, 0.0]
        ])

        self.R_dc2lned = np.array([
            [1.0, 0.0, 0.0, 0.0],  # lned.x = dc.x
            [0.0, 1.0, 0.0, 0.0],  # lned.y = dc.y
            [0.0, 0.0, 1.0, 0.0],  # lned.z = dc.z
            [0.0, 0.0, 0.0, 0.0]
        ])

    def static_transform(self, v__fin, fin, fout):
        """
        Given a vector expressed in frame fin, returns the same vector expressed in fout.

            Args:
                - v__fin: 3D vector, (x, y, z), represented in fin coordinates
                - fin: string describing input coordinate frame
                - fout: string describing output coordinate frame

            Returns
                - v__fout: a vector, (x, y, z) represent in fout coordinates
        """
        # Check if fin is a valid coordinate frame
        if fin not in self.COORDINATE_FRAMES:
            raise AttributeError(
                '{} is not a valid coordinate frame'.format(fin))

        # Check if fout is a valid coordinate frame
        if fout not in self.COORDINATE_FRAMES:
            raise AttributeError(
                '{} is not a valid coordinate frame'.format(fout))

        # Check for a static transformation exists between the two frames
        R_str = 'R_{}2{}'.format(fin, fout)
        if R_str not in self.STATIC_TRANSFORMS:
            raise AttributeError(
                'No static transform exists from {} to {}.'.format(fin, fout))

        # v4__'' are 4x1 np.array representation of the vector v__''
        # Create a 4x1 np.array representation of v__fin for matrix multiplication
        v4__fin = np.array([[v__fin[0]],
                            [v__fin[1]],
                            [v__fin[2]],
                            [0.0]])

        # Get rotation matrix
        R_fin2fout = getattr(self, R_str)

        # Perform transformation from v__fin to v__fout
        v4__fout = np.dot(R_fin2fout, v4__fin)

        return (v4__fout[0, 0], v4__fout[1, 0], v4__fout[2, 0])


class LineController(Node):
    def __init__(self) -> None:
        super().__init__('line_controller')

        # Create CoordTransforms instance
        self.coord_transforms = CoordTransforms()

        # Store initial ground plane offset
        self.initial_d_x = None
        self.initial_d_y = None

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.line_sub = self.create_subscription(
            Line, '/line/param', self.line_sub_cb, 1)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = -3.0

        # Linear setpoint velocities in downward camera frame
        self.vx__dc = 0.0
        self.vy__dc = 0.0
        self.vz__dc = 0.0

        # Yaw setpoint velocities in downward camera frame
        self.wz__dc = 0.0

        # Quaternion representing the rotation of the drone's body frame in the NED frame. initiallize to identity quaternion
        self.quat_bu_lenu = (0, 0, 0, 1)

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard OffboardControlModecontrol mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        """Callback function for the timer."""

        self.publish_offboard_control_heartbeat_signal()

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        self.offboard_setpoint_counter += 1

    def get_ground_plane_distances(self, pixel_u, pixel_v):
        horizontal_fov_rad = 1.74
        image_width = 1280
        image_height = 960
        camera_height_meters = 3.0

        fx = image_width / (2 * math.tan(horizontal_fov_rad / 2))
        vertical_fov_rad = 2 * \
            math.atan(math.tan(horizontal_fov_rad / 2)
                      * (image_height / image_width))
        fy = image_height / (2 * math.tan(vertical_fov_rad / 2))

        cx = (image_width - 1) / 2
        cy = (image_height - 1) / 2

        # These are the horizontal distances on the ground plane
        # relative to the point directly below the camera.
        distance_x_on_ground = (pixel_u - cx) * camera_height_meters / fx
        distance_y_on_ground = (pixel_v - cy) * camera_height_meters / fy

        return distance_x_on_ground, distance_y_on_ground

    def line_sub_cb(self, param):
        """
        Callback function which is called when a new message of type Line is received by self.line_sub.
        """
        # Delay line following until drone is near takeoff height
        altitude = self.vehicle_local_position.z
        print(f'Altitude: {altitude}')
        if altitude is None or abs(altitude - self.takeoff_height) > 1:
            msg = TrajectorySetpoint()
            msg.position = [0, 0, self.takeoff_height]
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_publisher.publish(msg)
            return

        # Extract line parameters
        x, y, vx, vy = param.x, param.y, param.vx, param.vy
        # line_point = np.array([x, y])
        # line_dir = np.array([vx, vy])

        # Project a target point far along the detected line
        # target = line_point + 100 * line_dir

        # d_x, d_y = self.get_ground_plane_distances(*target)
        
        if vx == 0 or vy == 0:
            msg = TrajectorySetpoint()
            msg.velocity = [0, 0, None]
            msg.position = [None, None, self.takeoff_height]
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_publisher.publish(msg)

        vel = np.array([vx, vy, 0])
        norm = np.linalg.norm(vel)
        if norm == 0:
            return
        vel_norm = vel / norm
        vel_norm *= _MAX_SPEED
        
        print(f'VEL: {vel_norm}')
                
        msg = TrajectorySetpoint()
        msg.velocity = [-vel_norm[0], -vel_norm[1], None]
        msg.position = [None, None, self.takeoff_height]
        msg.yaw = math.atan2(y, x)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
                
def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = LineController()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
