import mujoco
import os

urdf_file = "FirstRobot\\robot.urdf"
mjcf_model = mujoco.MjModel.from_xml_path(urdf_file)

output_xml = "robot_converted.xml"

mujoco.mj_saveLastXML(output_xml, mjcf_model)

print(f"Model saved as {output_xml}")

#nano ~/.bashrc
#source ~/.bashrc
#onshape-to-robot FirstRobot