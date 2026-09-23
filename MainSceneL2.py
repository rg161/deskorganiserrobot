
import swift
import spatialmath as sm
import spatialgeometry as geometry
from roboticstoolbox import models
from Robots.KukaKR6 import KukaKR6
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#ignore future warnings in terminal

env = swift.Swift()
env.launch(realtime=True)

table = geometry.Cuboid(
    scale=[1.2, 0.8, 0.05],
    pose=sm.SE3(0, 0, 0.4),
    color=(0.6, 0.4, 0.2, 1)
)
env.add(table)

#ur3e placeholder for KuraKR6
robot = KukaKR6()
robot.q = robot.qz
robot.base = sm.SE3(-0.5, -0.3, 0.4)
env.add(robot)



#loop to keep running
while True:
    try:
        env.step(0.05)
    except AttributeError:
        pass