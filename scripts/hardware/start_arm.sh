#!/usr/bin/env bash
# Self-healing launcher for the Kinova Gen3 ros2_kortex driver.
#
# WHY: the kortex real-time cyclic loop throws
#   terminate: "timeout detected: BaseCyclicClient::RefreshFeedback"
# and ABORTS ros2_control_node (exit -6) when a control cycle misses its deadline under load on a
# non-RT kernel. Worse, each crash leaves the ARM IN A FAULT (internal_fault=true, blocks ALL motion)
# and brings the controllers back with joint_trajectory_controller active instead of twist_controller.
# So a bare relaunch looks alive but silently refuses to move.
#
# THIS SCRIPT, every launch (including auto-respawns):
#   1) runs the driver with REAL-TIME scheduling (chrt SCHED_FIFO; needs rtprio ulimit, see
#      /etc/security/limits.d/99-ros2-realtime.conf),
#   2) once controllers are up, RESETS THE FAULT (/fault_controller/reset_fault, type
#      example_interfaces/srv/Trigger -- NOT std_srvs) and ACTIVATES twist_controller,
#   3) MONITORS ros2_control_node and respawns the whole stack within seconds if it dies.
# Also reduce competing CPU load while driving (stop camera_live_preview.py); keep the arm wired.
#
# NOTE: no `set -u`; the ROS setup script references unset vars and would abort under nounset.
source /opt/ros/humble/setup.bash
ROBOT_IP="${ROBOT_IP:-192.168.1.10}"
RT_PRIO="${RT_PRIO:-80}"

PREFIX=""
if [ "$(ulimit -r)" -ge "$RT_PRIO" ] 2>/dev/null && command -v chrt >/dev/null; then
  PREFIX="chrt -f $RT_PRIO"
  echo "[start_arm] real-time scheduling ON (SCHED_FIFO $RT_PRIO)"
else
  echo "[start_arm] WARN: rtprio ulimit < $RT_PRIO, running WITHOUT real-time priority (respawn still active)"
fi

make_ready() {   # runs in background after each launch: wait for controllers, clear fault, activate twist
  for _ in $(seq 1 40); do
    ros2 control list_controllers 2>/dev/null | grep -q twist_controller && break
    sleep 1
  done
  sleep 2
  # pin ros2_control_node to a dedicated core so its RT loop is not preempted (reduces RefreshFeedback
  # timeouts on a non-RT kernel; a lowlatency/PREEMPT_RT kernel is the robust fix, see HARDWARE.md)
  CN="$(pgrep -f ros2_control_node | head -1)"
  [ -n "$CN" ] && taskset -cp "${PIN_CORE:-7}" "$CN" >/dev/null 2>&1 && echo "[start_arm] pinned ros2_control_node to core ${PIN_CORE:-7}"
  ros2 service call /fault_controller/reset_fault example_interfaces/srv/Trigger >/dev/null 2>&1
  ros2 control switch_controllers --deactivate joint_trajectory_controller --activate twist_controller >/dev/null 2>&1
  echo "[start_arm] ready: fault cleared + twist_controller active + control node pinned"
}

while true; do
  echo "[start_arm] launching kortex driver (robot_ip=$ROBOT_IP) at $(date)"
  $PREFIX ros2 launch kortex_bringup gen3.launch.py \
    robot_ip:="$ROBOT_IP" gripper:=robotiq_2f_85 \
    use_internal_bus_gripper_comm:=true launch_rviz:=false &
  LAUNCH_PID=$!

  make_ready &

  # wait for the control node to appear, then watch it; respawn the moment it dies
  for _ in $(seq 1 40); do pgrep -f ros2_control_node >/dev/null && break; sleep 1; done
  while pgrep -f ros2_control_node >/dev/null; do sleep 2; done

  echo "[start_arm] ros2_control_node died -> respawning in 3s"
  kill "$LAUNCH_PID" 2>/dev/null
  pkill -f gen3.launch 2>/dev/null
  sleep 3
done
