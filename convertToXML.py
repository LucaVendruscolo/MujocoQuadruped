import mujoco
import os

# Load the URDF file
urdf_file = "FirstRobot\\robot.urdf"
mjcf_model = mujoco.MjModel.from_xml_path(urdf_file)

# Specify the output XML file path
output_xml = "robot_converted.xml"

# Save the model to an MJCF XML file using mj_saveLastXML
mujoco.mj_saveLastXML(output_xml, mjcf_model)

print(f"Model saved as {output_xml}")

#nano ~/.bashrc
#source ~/.bashrc
#onshape-to-robot FirstRobot