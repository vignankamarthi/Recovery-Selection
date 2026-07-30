"""Backend-agnostic vision reads for the overhead camera (Part 1).

Pure-numpy image functions with no MuJoCo, no torch, and no ROS. In simulation the label-visibility
signal comes from ground-truth segmentation (`sim/world.overhead_label_visibility`); on hardware there
is no segmentation, so the same signal must be read from the real overhead RGB frame. This package holds
that read, so it plugs into the hardware `SceneOracle` the moment the camera is up, and it can be
validated against the sim's ground truth on rendered frames.
"""
