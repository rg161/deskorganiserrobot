
import swift
from pynput import keyboard as pynput_keyboard
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
tray_pos = [0.02, -0.15, 0.44]
pen_tray = geometry.Cuboid(scale=[0.1, 0.1, 0.03],
                        pose=sm.SE3(tray_pos),
                        color=(0.2, 0.2, 0.2, 1))
env.add(pen_tray)

pens_tray_pos = {
    "r": [tray_pos[0] - 0.03, tray_pos[1], tray_pos[2]],
    "g": [tray_pos[0], tray_pos[1], tray_pos[2]],
    "b": [tray_pos[0] + 0.03, tray_pos[1], tray_pos[2]]
}


robot = KukaKR6()
robot.q = [0, np.pi/2, -np.pi/2, 0, 0, 0]  
robot.base = sm.SE3(-0.5, -0.3, 0.425)

#tested config that keeps elbow at certan angles
#avoids arm going under the table for deposit and pickup
ElbowUp_seed = [2.21916751, -2.95703805, -0.1733883, -1.30056441, 1.36605535, 0.26050853]

def rotation_to_align_z(direction):

    z_axis = np.array([0, 0, 1])
    direction = direction / np.linalg.norm(direction)
    c = np.dot(z_axis, direction)

    if c > 1-1e-6:
        return np.eye(3)
    if c < -1+1e-6:
        return sm.SO3.Rx(np.pi).R
    v = np.cross(z_axis, direction)
    vx = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
    rot = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    U, _, Vt = np.linalg.svd(rot)
    rot = U @ Vt
    if np.linalg.det(rot) < 0:
        U[:, -1] *= -1
        rot = U @ Vt
        rot = U @ Vt
    return rot
#to avoid 'valueerror: expected SO3 or rotation matrix'

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
    

    if not np.all(np.isfinite(rot)) or not np.allclose(rot @ rot.T, np.eye(3), atol=1e-3):
    
        cylinders.append(None)
        continue
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
        
        if not np.all(np.isfinite(rot)) or not np.allclose(rot @ rot.T, np.eye(3), atol=1e-3):
            continue

        cyl.T = sm.SE3.Rt(rot, midpoint)

def move_to_pose(target_pos, steps = 50, carry_object = None, carry_offset = None):
    #Find IK solution for target_pos
    #animate to target_pos (jtraj)
    #use elbowup config to keep arm above table and avoid collisions
    #assign a set seed configuration for the elbow up position
    mask = [1, 1, 1, 0, 0, 0]  # Only consider position for IK
    IK_seed = 42

    def no_dip(q):
        return min(p.t[2] for p in robot.fkine_all(q)) >= tabletop_h - 0.002

    sol_a = robot.ikine_LM(target_pos, q0 = ElbowUp_seed, mask=mask, seed = IK_seed)
    if sol_a.success and no_dip(sol_a.q):
        q_target = sol_a.q
    else:
        sol_b = robot.ikine_LM(target_pos, q0=robot.q, mask=mask, seed = IK_seed)
        if sol_b.success and no_dip(sol_b.q):
            q_target = sol_b.q
        elif sol_a.success:
            q_target = sol_a.q
        elif sol_b.success:
            q_target = sol_b.q
        else:
            fallback_q0 = [0,np.pi/2, -np.pi/2, 0, 0, 0]
            sol_c = robot.ikine_LM(target_pos, q0=fallback_q0, mask=mask, seed = IK_seed)
            if not sol_c.success:
                print("IK solution not found for target position.")
                return False
            q_target = sol_c.q


    traj = jtraj(robot.q, q_target, steps)
    for q in traj.q:
        robot.q = q
        update_cylinders()
        if carry_object is not None:
            carry_object.T = robot.fkine(robot.q) * carry_offset
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

def pen_pick_and_place(pen, pen_pos, drop_pos, clearance = 0.2):
    og_pos = robot.fkine(robot.q)

    transit = sm.SE3(pen_pos[0], pen_pos[1], 0.75) * sm.SE3.Rx(np.pi)  # high, safe waypoint

    above_pen = sm.SE3(pen_pos[0], pen_pos[1], pen_pos[2] + 0.3) * sm.SE3.Rx(np.pi)
    at_pen = sm.SE3(pen_pos[0], pen_pos[1], pen_pos[2] + clearance) * sm.SE3.Rx(np.pi)

    above_tray = sm.SE3(drop_pos[0], drop_pos[1], drop_pos[2] + 0.2) * sm.SE3.Rx(np.pi)
    at_tray = sm.SE3(drop_pos[0], drop_pos[1], drop_pos[2] + 0.05) * sm.SE3.Rx(np.pi)


    print(f"Pen at {pen_pos}, moving to pick up.")

    if not move_to_pose(transit):
        print("Failed to move to transit position.")
        return False
    if not move_to_pose(above_pen):
        print("Failed to move above pen.")
        return False
    if not move_to_pose(at_pen):
        print("Failed to move to pen.")
        return False
    

    carry_offset = sm.SE3(0, 0, -0.075)  # Offset to carry the pen correctly
    move_to_pose(above_pen, carry_object = pen, carry_offset = carry_offset)
    move_to_pose(above_tray, carry_object = pen, carry_offset = carry_offset)
    move_to_pose(at_tray, carry_object = pen, carry_offset = carry_offset)
    

    pen.T = sm.SE3(drop_pos)
    move_to_pose(og_pos)
    return True

#COLOURED PENS!!!
pens = {
    "r": {"position": [0.1, -0.3, tabletop_h + 0.015], "color": (0.9, 0.1, 0.1, 1), "name": "Red Pen", "clearance": 0.02},
    "g": {"position": [-0.5, 0.25, tabletop_h + 0.015], "color": (0.1, 0.8, 0.1, 1), "name": "Green Pen", "clearance": 0.025},
    "b": {"position": [0.0, 0.1, tabletop_h + 0.015], "color": (0.1, 0.1, 0.9, 1), "name": "Blue Pen", "clearance": 0.02},
}
for key, pen_info in pens.items():
    pen_info["object"] = pen_init(pen_info["position"], pen_info["color"])
    pen_info["picked"] = False

print("Press Enter to collect all pens into the tray.")
print("Press 'q' to quit.")


def collect_all_pens():
    for key, pen_info in pens.items():
        if pen_info["picked"]:
            continue
        print(f"Picking up {pen_info['name']}")
        pen_pick_and_place(pen_info["object"], pen_info["position"], pens_tray_pos[key], clearance = pen_info["clearance"])
        pen_info["picked"] = True
    print("All pens have been picked up.")


#use pynput for keyboard inputs
pressed_keys = set()
enter_pressed = False

def on_press(key):
    global enter_pressed
    if key == pynput_keyboard.Key.enter:
        enter_pressed = True
        return
    try:
        pressed_keys.add(key.char)
    except AttributeError:  
        pass    
def on_release(key):
    try:
        pressed_keys.discard(key.char)
    except AttributeError:  
        pass

key_listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
key_listener.start()

#main loop
            
collection_complete = False            
running = True
while running:
    if enter_pressed and not collection_complete:
        enter_pressed = False
        collect_all_pens() 
        collection_complete = True
    
    if 'q' in pressed_keys:
        print("Quitting...")
        running = False

    try:  
        env.step(0.05)
    except AttributeError:
        pass
    time.sleep(0.05)

key_listener.stop()
   
