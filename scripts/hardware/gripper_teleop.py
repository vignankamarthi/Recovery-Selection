#!/usr/bin/env python3
"""Gamepad gripper control for the Kinova 2F-85 via the ros2_kortex gripper action.

Pairs with teleop_twist_joy (which drives the arm) to give a full drive+grip teleop for collection.
Subscribes /joy and, on a button press (rising edge), sends a GripperCommand goal to
/robotiq_gripper_controller/gripper_cmd:
    A (button 0)  -> CLOSE
    B (button 1)  -> OPEN
Position 0.0 = open, ~0.8 = closed for the 2F-85.

Run:  ros2 run ... no; just: python3 gripper_teleop.py   (after sourcing /opt/ros/humble/setup.bash)
"""
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Joy
from control_msgs.action import GripperCommand

CLOSE_BTN = 0      # A
OPEN_BTN = 1       # B
CLOSED_POS = 0.8
OPEN_POS = 0.0
MAX_EFFORT = 35.0   # of a 0-50 joint-effort ceiling; 35 backs off from max for safety (2026-08-11)


class GripperTeleop(Node):
    def __init__(self):
        super().__init__("gripper_teleop")
        self._client = ActionClient(self, GripperCommand, "/robotiq_gripper_controller/gripper_cmd")
        self.create_subscription(Joy, "/joy", self._on_joy, 10)
        self._prev = {CLOSE_BTN: False, OPEN_BTN: False}
        self.get_logger().info("gripper_teleop up: A (btn 0) = CLOSE, B (btn 1) = OPEN")

    def _on_joy(self, msg: Joy):
        for btn, pos, name in ((CLOSE_BTN, CLOSED_POS, "CLOSE"), (OPEN_BTN, OPEN_POS, "OPEN")):
            pressed = btn < len(msg.buttons) and msg.buttons[btn] == 1
            if pressed and not self._prev[btn]:
                self._send(pos, name)
            self._prev[btn] = pressed

    def _send(self, position: float, name: str):
        if not self._client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("gripper action server not available")
            return
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = MAX_EFFORT
        self._client.send_goal_async(goal)
        self.get_logger().info(f"gripper {name} -> {position}")


def main():
    rclpy.init()
    node = GripperTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
