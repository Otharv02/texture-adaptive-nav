import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleCommand, OffboardControlMode, TrajectorySetpoint

class OffboardControl(Node):
    def __init__(self):
        super().__init__('offboard_control')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )


        # Publisher 
        self.cmd_pub_ = self.create_publisher(VehicleCommand,
                                              "/fmu/in/vehicle_command",
                                              qos_profile)
        
        self.offbaord_pub_ = self.create_publisher(OffboardControlMode,
                                                   "/fmu/in/offboard_control_mode",
                                                   qos_profile)
        
        self.trajectory_pub_ = self.create_publisher(TrajectorySetpoint,
                                                     "/fmu/in/trajectory_setpoint",
                                                     qos_profile)


        self.timer_  = self.create_timer(0.05, self.offboard_heatbeat)
        self.counter = 0

    def offboard_heatbeat(self):

        msg = OffboardControlMode()

        msg.position     = True
        msg.velocity     = False
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False

        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        self.offbaord_pub_.publish(msg)

        self.hover()

        # State Machine 
        if self.counter == 50:
            self.set_offboard_cmd()
            
        elif self.counter == 60:
            self.arm()

        elif self.counter >260:
            self.fly_forward()

        elif self.counter > 60:
            self.hover()

        
        self.counter += 1
    
    def fill_command(self, msg):
        msg.target_system    = 1     # Flight Controller
        msg.target_component = 1     # Main Autopilot
        msg.source_system    = 1     # Source: This Node
        msg.source_component = 1     # Source: This Node Component 
        msg.from_external    = True  # Command from Offboard Computer

        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)

    
    def set_offboard_cmd(self):
        msg = VehicleCommand()

        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1  = 1.0
        msg.param2  = 6.0

        self.fill_command(msg)
        self.cmd_pub_.publish(msg)
        self.get_logger().info("Switching to OFFBOARD mode")


    def arm(self):
        msg = VehicleCommand()

        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1  = 1.0

        self.fill_command(msg)
        self.cmd_pub_.publish(msg)
        self.get_logger().info("Arming vehicle")

    def hover(self):
        msg = TrajectorySetpoint()

        msg.position = [0.0, 0.0, -2.0] # 2 meters 
        msg.yaw      = 0.0


        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        self.trajectory_pub_.publish(msg)

    def fly_forward(self):
        msg = TrajectorySetpoint()

        msg.position = [10.0, 0.0, -2.0] # 10 meters forward
        msg.yaw      = 0.0


        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        self.trajectory_pub_.publish(msg)




def main(args=None):
    rclpy.init(args=args)
    node = OffboardControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()

