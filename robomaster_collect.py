from robomaster_control_nodes import RoboMasterCollect
import rclpy

def main():
    rclpy.init()
    node = RoboMasterCollect()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()