from robomaster_control_nodes import RoboMasterEval
import rclpy

def main():
    rclpy.init()
    node = RoboMasterEval()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()