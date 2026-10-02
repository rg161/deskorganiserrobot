
import swift
import keyboard
import spatialmath as sm
import spatialgeometry as geometry
from roboticstoolbox import models, jtraj
from Robots.KukaKR6 import KukaKR6
import numpy as np
import spatialmath as sm
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#ignore future warnings in terminal

env = swift.Swift()
env.launch(realtime=True)


#Table
tabletop_h = 0.425
table = geometry.Cuboid(
    scale=[1.2, 0.8, 0.05],
    pose=sm.SE3(0, 0, 0.4),
    color=(0.6, 0.4, 0.2, 1)
)
env.add(table)

#Pen Tray
tray_pos = [-0.2, -0.3, 0.425]
pen_tray = geometry.Cuboid(scale=[0.1, 0.1, 0.03],
                        pose=sm.SE3(tray_pos),
                        color=(0.2, 0.2, 0.2, 1))
env.add(pen_tray)

robot = KukaKR6()
robot.q = [0, np.pi/2, -np.pi/2, 0, 0, 0]  
robot.base = sm.SE3(-0.5, -0.3, 0.425)

def rotation_to_align_z(direction):

    z_axis = np.array([0, 0, 1])
    if np.allclose(direction, z_axis):
        return np.eye(3)
    elif np.allclose(direction, -z_axis):
        return sm.SO3.Rx(np.pi).R
    v = np.cross(z_axis, direction)
    c = np.dot(z_axis, direction)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


#Use cylinders to represent links as DHrobot will not display on Swift.
link_poses = robot.fkine_all(robot.q)
cylinders = []
radius = 0.02

for i in range(len(link_poses) - 1):
    p1 = link_poses[i].t
    p2 = link_poses[i + 1].t
    length = np.linalg.norm(p2 - p1)

    if length < 1e-6:
        cylinders.append(None)
        continue

    midpoint = (p1 + p2) / 2
    direction = (p2 - p1) / length
    rot = rotation_to_align_z(direction)
    pose = sm.SE3.Rt(rot, midpoint)

    cyl = geometry.Cylinder(radius=radius, length=length, pose=pose,
                             color=(0.8, 0.2, 0.1, 1))
    env.add(cyl)
    cylinders.append(cyl)


def update_cylinders():
    poses = robot.fkine_all(robot.q)
    for i, cyl in enumerate(cylinders):
        if cyl is None:
            continue
        p1 = poses[i].t
        p2 = poses[i + 1].t
        length = np.linalg.norm(p2 - p1)
        if length < 1e-6:
            continue
        midpoint = (p1 + p2) / 2
        direction = (p2 - p1) / length
        rot = rotation_to_align_z(direction)
        cyl.T = sm.SE3.Rt(rot, midpoint)

def move_to_pose(target_pos, steps = 50):
    #Find IK solution for target_pos
    #animate to target_pos (jtraj)
    sol = robot.ikine_LM(target_pos, q0=robot.q, mask=[1, 1, 1, 0, 0, 0])
    if not sol.success:
        print("IK solution not found for target position.")
        return False

    traj = jtraj(robot.q, sol.q, steps)
    for q in traj.q:
        robot.q = q
        update_cylinders()
        try:
            env.step(0.05)
        except AttributeError:
            pass
        time.sleep(0.05)
    return True

def pen_init(position, color):
    #create/add pen at position
    pen = geometry.Cylinder(radius = 0.005, length = 0.15, pose = sm.SE3(position) * sm.SE3.Rx(np.pi / 2), color = color)
    env.add(pen)
    return pen

def pen_pick_and_place(pen, pen_pos):
    og_pos = robot.fkine(robot.q)

    above_pen = sm.SE3(pen_pos[0], pen_pos[1], pen_pos[2] + 0.15) * sm.SE3.Rx(np.pi)
    at_pen = sm.SE3(pen_pos[0], pen_pos[1], pen_pos[2] + 0.02) * sm.SE3.Rx(np.pi)

    above_tray = sm.SE3(tray_pos[0], tray_pos[1], tray_pos[2] + 0.6) * sm.SE3.Rx(np.pi)
    at_tray = sm.SE3(tray_pos[0], tray_pos[1], tray_pos[2] + 0.45) * sm.SE3.Rx(np.pi)

    print(f"Pen at {pen_pos}, moving to pick up.")
    if not move_to_pose(above_pen):
        print("Failed to move above pen.")
        return False
    if not move_to_pose(at_pen):
        print("Failed to move to pen.")
        return False
    move_to_pose(above_pen)
    got = True

    move_to_pose(above_tray)
    if got:
        pen.T = robot.fkine(robot.q) * sm.SE3(0, 0, -0.075)

    move_to_pose(at_tray)
    if got:
        pen.T = sm.SE3(tray_pos)
        got = False

    move_to_pose(og_pos)
    return True

#COLOURED PENS!!!
pens = {
    "r": {"position": [0.2, -0.15, tabletop_h + 0.015], "color": (0.9, 0.1, 0.1, 1), "name": "Red Pen"},
    "g": {"position": [0.1, 0.05, tabletop_h + 0.015], "color": (0.1, 0.8, 0.1, 1), "name": "Green Pen"}, 
    "b": {"position": [0.3, 0, tabletop_h + 0.015], "color": (0.1, 0.1, 0.9, 1), "name": "Blue Pen"},  
}
for key, pen_info in pens.items():
    pen_info["object"] = pen_init(pen_info["position"], pen_info["color"])
    pen_info["picked"] = False

print("Press 'r', 'g' or 'b' to pick up the matching coloured pen.")
print("Press 'q' to quit.")

#main loop
running = True
while running:
    for key, pen_info in pens.items():
        if keyboard.is_pressed(key) and not pen_info["picked"]:
            print(f"Picking up {pen_info['name']}")
            pen_info["picked"] = True
            pen_pick_and_place(pen_info["object"], pen_info["position"])

    if keyboard.is_pressed('q'):
        print("Quitting...")
        running = False

    try:  
        env.step(0.05)
    except AttributeError:
        pass
    time.sleep(0.05)
