
import swift
import spatialmath as sm
import spatialgeometry as geometry
from roboticstoolbox import models
import time

env = swift.Swift()
env.launch(realtime=True)

table = geometry.Cuboid(
    scale=[1.2, 0.8, 0.05],
    pose=sm.SE3(0, 0, 0.4),
    color=(0.6, 0.4, 0.2, 1)
)
env.add(table)

#ur3e placeholder for FanucLR
robot = models.UR3()
robot.q = robot.qr
robot.base = sm.SE3(0, -0.3, 0.4)
env.add(robot)



#loop to keep running
while True:
    try:
        env.step(0.05)
    except AttributeError:
        pass