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
from ir_support.robots import UR3e
warnings.filterwarnings("ignore", category=FutureWarning)
#ignore future warnings in terminal

env = swift.Swift()
env.launch(realtime=True)

"""
CONTROLS

enter = KUKA pen collection
r/g/b = UR3e pen delivery
space = E-Stop (engage/disengage)
a = reset (after E-Stop is disengaged)
s = resume (after resume)
z = person enters/leaves danger area
o = object placed in robots planned path
q = quit

"""

#Table
tabletop_h = 0.425
table = geometry.Cuboid(
    scale=[1.2, 0.8, 0.05],
    pose=sm.SE3(0, 0, 0.4),
    color=(0.6, 0.4, 0.2, 1)
)
env.add(table)

#UR3e robot
ur3e = UR3e()
ur3e.base = sm.SE3(-0.4, 0.05, 0.425)
ur3e.add_to_env(env)
UR3E_HOME_Q =[0, 0, 0, 0, 0, 0]
ur3e.q = UR3E_HOME_Q  # Initial joint configuration

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


#Safety System!!
#initiating safety objects
HIDE = sm.SE3(0, 0, -5) #put hidden objects below the floor unless called up 

#Equipment models
def add_box(scale, pos, color):
    box = geometry.Cuboid(scale = scale, pose = sm.SE3(pos[0], pos[1], pos[2]), color = color)
    env.add(box)
    return box

#table legs
for lx in (-0.55, 0.55):
    for ly in (-0.35, 0.35):
        add_box([0.05, 0.05, 0.375], (lx, ly, 0.1875), (0.45, 0.3, 0.15, 1))

#Back guard wall
#(placed out of reach of KUKA collision y = -0.82, z = 0.94)
BACK_WALL_Y = -0.93
add_box([1.2, 0.02, 1.1], (0, BACK_WALL_Y, 0.55), (0.7, 0.7, 0.75, 0.4))

#side guard wall (level with back wall and table edge)
SIDE_WALL_X = 0.61
SIDE_WALL_Y = 0.43
side_l = SIDE_WALL_Y - SIDE_WALL_X
add_box([0.02, side_l, 1.1], (SIDE_WALL_X, (SIDE_WALL_Y + BACK_WALL_Y) / 2, 0.55), (0.7, 0.7, 0.75, 0.4))

#example green front curtain
CURTAIN_POSE = sm.SE3(0, 0.43, tabletop_h + 0.30)
curtain_safe = add_box([1.2, 0.005, 0.5], (0, 0.43, tabletop_h + 0.30), (0.1, 0.9, 0.1, 0.35)) #green
curtain_unsafe = add_box([1.2, 0.005, 0.5], (0, 0.43, tabletop_h + 0.30), (0.95, 0.1, 0.1, 0.5)) #red
curtain_unsafe.T = HIDE

#E-STOP button
add_box([0.08, 0.08, 0.06], (0.5, 0.34, tabletop_h + 0.03), (0.25, 0.25, 0.25, 1))
estop_button = geometry.Cylinder(radius = 0.03, length = 0.04, pose = sm.SE3(0.5, 0.34, tabletop_h + 0.08), color = (0.9, 0.05, 0.05, 1))
env.add(estop_button)

#Warning Light
pole = geometry.Cylinder(radius = 0.01, length = 0.25, pose = sm.SE3(0.55, -0.34, tabletop_h + 0.125), color = (0.15, 0.15, 0.15, 1))
env.add(pole)
POLE_POSE = sm.SE3(0.55, -0.34, tabletop_h + 0.28)
lights = {
    "green": geometry.Sphere(radius = 0.03, pose = HIDE, color = (0.1, 0.9, 0.1, 1)),
    "yellow": geometry.Sphere(radius = 0.03, pose = HIDE, color = (1.0, 0.65, 0.0, 1)),
    "red": geometry.Sphere(radius = 0.03, pose = HIDE, color = (0.95, 0.1, 0.1, 1)),
}
for light in lights.values():
    env.add(light)

#Collision Obstacle
COLLISION_OBSTACLE = [0.1, 0.1, 0.15]
obstacle = geometry.Cuboid(scale = COLLISION_OBSTACLE, pose = HIDE, color = (1.0, 0.5, 0.0, 1))
env.add(obstacle)






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
    print("Press 'r' for the red pen, 'g' for the green pen or 'b' for the blue pen to have it delivered.")

### UR3E SECTION!

HANDOFF_POS = [-0.65, 0.2, 0.6] #right front of the desk

EE_DOWN = sm.SE3.Rx(np.pi)  # End-effector pointing downwards
GRIPPER_OFFSET = sm.SE3(0, 0, 0.05) # Offset to hold the pen correctly

#fix to avoid under tabletop
def ur3e_path_clear(q_start, q_end, steps = 50):
    #true if no joint dips below the tabletop while over the table
    for q in jtraj(q_start, q_end, steps).q:
        for p in ur3e.fkine_all(q):
            x, y, z = p.t
            if abs(x) < 0.6 and abs(y) < 0.4 and z < tabletop_h - 0.002:
                return False
    return True

def ur3e_move(target, steps = 50, carry_object = None):
    mask = [1, 1, 1, 1, 1, 0]
    rng = np.random.default_rng(0)
    seeds = [ur3e.q, np.zeros(6), np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])]
    seeds += [ur3e.q + rng.uniform(-1, 1, 6) for _ in range(6)]

    q_target = None
    first = None
    for s in seeds:
        sol = ur3e.ikine_LM(target, q0 = s, mask = mask, seed = 42)
        if not sol.success:
            continue
        if first is None:
            first = sol.q
        if ur3e_path_clear(ur3e.q, sol.q, steps):
            q_target = sol.q
            break
    if q_target is None:
        if first is None:
            print("UR3e IK failed for", np.round(target.t, 3))
            return False
        print("Warning: every UR3e solution goes under the table", np.round(target.t, 3))
        q_target = first


    #fix to stop ur3e spinning into kuka
    #spin opp direction
    q_target = np.array(q_target, dtype = float)
    q_target[0] = (q_target[0] + np.pi) % (2 * np.pi) - np.pi
    if q_target[0] > 0:
        q_target[0] -= 2 * np.pi



    for q in jtraj(ur3e.q, q_target, steps).q:
        ur3e.q = q
        if carry_object is not None:
            carry_object.T = ur3e.fkine(ur3e.q) * GRIPPER_OFFSET
        try:
            env.step(0.05)
        except AttributeError:
            pass
        time.sleep(0.05)
    return True

def ur3e_return_home(steps = 50):
    for q in jtraj(ur3e.q, UR3E_HOME_Q, steps).q:
        ur3e.q = q
        try:
            env.step(0.05)
        except AttributeError:
            pass
        time.sleep(0.05)

def deliver_pen(key):
    info = pens[key]
    if not info["picked"]:
        print(f"{info['name']} is not in the tray yet.")
        return False
    if info.get("delivered"):
        print(f"{info['name']} has already been delivered.")
        return False

    pen = info["object"]
    slot = pens_tray_pos[key]
    above = sm.SE3(slot[0], slot[1], slot[2] + 0.25) * EE_DOWN
    get = sm.SE3(slot[0], slot[1], slot[2] + 0.05) * EE_DOWN
    out = sm.SE3(*HANDOFF_POS) * EE_DOWN

#delivery process
    print(f"delivering {info['name']}")
    if not (ur3e_move(above) and ur3e_move(get)):
        ur3e_return_home()
        return False

    ur3e_move(above, carry_object = pen)
    ur3e_move(out, carry_object = pen)

    time.sleep(2) #time for user to take pen
    pen.T = sm.SE3(0, 0, -5) #pen gone
    info["delivered"] = True
    ur3e_return_home()
    return True




#use pynput for keyboard inputs
pressed_keys = set()
enter_pressed = False

delivery_needed = []

def on_press(key):

    global enter_pressed
    if key == pynput_keyboard.Key.enter:
        enter_pressed = True
        return
    try:
        pressed_keys.add(key.char)

        if key.char in ("r", "g", "b"):
            delivery_needed.append(key.char)
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

    if delivery_needed and collection_complete:
        deliver_pen(delivery_needed.pop(0))
        #check if pen r, g, b are needed
    
    if 'q' in pressed_keys:
        print("Quitting...")
        running = False

    try:  
        env.step(0.05)
    except AttributeError:
        pass
    time.sleep(0.05)

key_listener.stop()
   
