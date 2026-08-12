"""Patch the installed robotiq 2f-85 xacro macro to accept the newer isaac_* params.

kortex_description's load_gripper passes isaac_joint_commands and isaac_joint_states, which the
installed (older) robotiq_description macro does not declare. Add both as accept-and-ignore params.
On real hardware with the internal bus the gripper ros2_control block is skipped, so the values are
irrelevant. Idempotent, backs up once. Run as root (writes under /opt/ros).
"""
import shutil

F = "/opt/ros/humble/share/robotiq_description/urdf/robotiq_2f_85_macro.urdf.xacro"
NEEDED = ["isaac_joint_commands", "isaac_joint_states"]

s = open(F).read()
missing = [p for p in NEEDED if p not in s]
if not missing:
    print("all params present, nothing to do")
    raise SystemExit(0)

# closing '">' of the macro params attribute, the first one after the com_port default
close = s.index('">', s.index("com_port:=/dev/ttyUSB0"))
inject = "".join(f"\n        {p}:=false" for p in missing)
s = s[:close] + inject + s[close:]

import os
if not os.path.exists(F + ".bak"):
    shutil.copy2(F, F + ".bak")
open(F, "w").write(s)
print("added:", ", ".join(missing))
