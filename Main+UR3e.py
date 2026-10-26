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
from ir_support_extra_parts.parts import part_mesh
warnings.filterwarnings("ignore", category=FutureWarning)
#ignore future warnings in terminal

env = swift.Swift()
env.launch(realtime=True)
#keep simulator running

"""
CONTROLS

enter = KUKA pen collection
r/g/b = UR3e pen delivery
space = E-Stop (engage/disengage)
a = reset (after E-Stop is disengaged)
s = resume (after reset)
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

#UR3e robot (from ir_support)
ur3e = UR3e()
ur3e.base = sm.SE3(-0.4, 0.05, 0.425)
ur3e.add_to_env(env)
UR3E_HOME_Q =[0, 0, 0, 0, 0, 0]
#arm returns here after each delivery
ur3e.q = UR3E_HOME_Q  # Initial joint configuration

#Pen Tray (hand off point between the two robots)
tray_pos = [0.02, -0.15, 0.44]
pen_tray = geometry.Cuboid(scale=[0.1, 0.1, 0.03],
                        pose=sm.SE3(tray_pos),
                        color=(0.2, 0.2, 0.2, 1))
env.add(pen_tray)

pens_tray_pos = {
    "r": [tray_pos[0] - 0.03, tray_pos[1], tray_pos[2]],    #left
    "g": [tray_pos[0], tray_pos[1], tray_pos[2]],           #middle
    "b": [tray_pos[0] + 0.03, tray_pos[1], tray_pos[2]]     #right
}


robot = KukaKR6() #KUKA KR6 R900 sixx using researched DH Parameters from robot website
robot.q = [0, np.pi/2, -np.pi/2, 0, 0, 0]  
# initial position
robot.base = sm.SE3(-0.5, -0.3, 0.425) #right back corner

#tested config that keeps elbow at certan angles
#avoids arm going under the table for deposit and pickup
ElbowUp_seed = [2.21916751, -2.95703805, -0.1733883, -1.30056441, 1.36605535, 0.26050853]
#joint angle that avoids going under the table

#creates the rotation that directs a cylinders z-axis along the link direction towards the next one
def rotation_to_align_z(direction):

    z_axis = np.array([0, 0, 1])
    direction = direction / np.linalg.norm(direction)
    #converts to direction to unit vector
    c = np.dot(z_axis, direction)
    #cos of angle between original up facing and link direction

    if c > 1-1e-6:
        return np.eye(3)
        #if c close to 1 then no rotation needed
    if c < -1+1e-6:
        #if c close to -1 flip by 180 deg
        return sm.SO3.Rx(np.pi).R
    v = np.cross(z_axis, direction)
    vx = np.array([[0, -v[2], v[1]], #v as a 3x3 matrix
                     [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
    rot = np.eye(3) + vx + vx @ vx * (1 / (1 + c)) #rotation formula turns one vector to another
    U, _, Vt = np.linalg.svd(rot)
    rot = U @ Vt
    if np.linalg.det(rot) < 0:
        U[:, -1] *= -1
        rot = U @ Vt
        rot = U @ Vt
        #splits into parts, rebuilds the nearest perfect rotation
    return rot
#to avoid 'valueerror: expected SO3 or rotation matrix'



#Use cylinders to represent links as DHrobot will not display on Swift.
#DH model, only defines joint frames
#swift cant draw 3D shape, sub with cylinder estimates
#Toolbox had KR5 but DH parameters didn't align
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
                            color=(1.0, 0.5, 0.0, 1))
    
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

#textured floor (wood planks made from alternating strips)
FLOOR_COLORS = [(0.55, 0.40, 0.25, 1), (0.48, 0.34, 0.20, 1)]
for i in range(-10, 10):
    add_box([3.0, 0.15, 0.01], (0, i * 0.15 + 0.075, 0.0), FLOOR_COLORS[i % 2])

#table legs
for lx in (-0.55, 0.55):
    for ly in (-0.35, 0.35):
        add_box([0.05, 0.05, 0.375], (lx, ly, 0.1875), (0.45, 0.3, 0.15, 1))

#Back guard wall
#(placed out of reach of KUKA collision y = -0.82, z = 0.94)
#extended along -x to just past the KUKA's start-pose reach (x = -0.92)
BACK_WALL_Y = -0.93
RIGHT_WALL_X = -1.02
SIDE_WALL_X = 0.61
SIDE_WALL_Y = 0.43
side_l = SIDE_WALL_Y - BACK_WALL_Y
add_box([SIDE_WALL_X - RIGHT_WALL_X, 0.02, 1.1], ((SIDE_WALL_X + RIGHT_WALL_X) / 2, BACK_WALL_Y, 0.55), (0.7, 0.7, 0.75, 0.4))

#side guard walls (level with back wall, at the table edge and just past the KUKA's reach)
for wx in (SIDE_WALL_X, RIGHT_WALL_X):
    add_box([0.02, side_l, 1.1], (wx, (SIDE_WALL_Y + BACK_WALL_Y) / 2, 0.55), (0.7, 0.7, 0.75, 0.4))

caution = part_mesh("CautionSign")
caution.scale = [0.5, 0.5, 0.5]
caution.T = sm.SE3(-0.81, 0.35, 0) * sm.SE3.Rz(np.pi / 2)
env.add(caution)

#example green front curtain
CURTAIN_POSE = sm.SE3(0, 0.43, tabletop_h + 0.250)
curtain_safe = add_box([1.2, 0.005, 0.5], (0, 0.43, tabletop_h + 0.30), (0.1, 0.9, 0.1, 0.35)) #green
curtain_unsafe = add_box([1.2, 0.005, 0.5], (0, 0.43, tabletop_h + 0.30), (0.95, 0.1, 0.1, 0.5)) #red
curtain_unsafe.T = HIDE

#E-STOP button
add_box([0.08, 0.08, 0.06], (0.5, 0.34, tabletop_h + 0.03), (0.25, 0.25, 0.25, 1))
estop_button = geometry.Cylinder(radius = 0.03, length = 0.04, pose = sm.SE3(0.5, 0.34, tabletop_h + 0.08), color = (0.9, 0.05, 0.05, 1))
ESTOP_UP = sm.SE3(0.5, 0.34, tabletop_h + 0.08)
ESTOP_DOWN = sm.SE3(0.5, 0.34, tabletop_h + 0.05)
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
obstacle = geometry.Cuboid(scale = COLLISION_OBSTACLE, pose = HIDE, color = (0.6, 0.1, 0.8, 1.0))
env.add(obstacle)


#SAFETY SYSTEM STATE
 
#RUNNING - normal operation

#SAFETY_STOP - danger zone entered or collision detected

#ESTOP - E-stop engaged

#AWAITING_RESET - E-stop released, awaiting 'a'

#AWAITING_RESUME - 'a' pressed, awaiting 's'

state = "RUNNING"
stop_reason = ""
estop = False
in_zone = False
obstacle_in = False
obstacle_pos = np.zeros(3)

COLLISION_MARGIN = 0.05 #stop before collision
FUTURE_FRAMES = 15 #check 15 frames ahead the path


active_motion = None
 
def start_motion(robot_obj, qs):
    global active_motion
    active_motion = {"robot": robot_obj, "q": qs, "idx": 0}
 
def end_motion():
    global active_motion
    active_motion = None
 
def read_physical_estop():
    return False
 
def set_state(new_state, reason = ""):
    global state, stop_reason
    if new_state != state or reason != stop_reason:
        state, stop_reason = new_state, reason
        print(f"[SAFETY] {state}" + (f" - {reason}" if reason else ""))
 
def sample_links(poses, n = 6):
    P = [p.t for p in poses]
    pts = []
    for a, b in zip(P[:-1], P[1:]):
        for t in np.linspace(0, 1, n):
            pts.append(a + (b - a) * t)
    return np.array(pts)
 
def predicted_collision():
    #True if the next part of the robot's planned path hits the obstacle
    if not obstacle_in or active_motion is None:
        return False
    half = np.array(COLLISION_OBSTACLE) / 2 + COLLISION_MARGIN
    qs = active_motion["q"]
    start = active_motion["idx"]
    for k in range(start, min(start + FUTURE_FRAMES, len(qs)), 2):
        pts = sample_links(active_motion["robot"].fkine_all(qs[k]), 5)
        if np.any(np.all(np.abs(pts - obstacle_pos) <= half, axis = 1)):
            return True
    return False
 
def toggle_obstacle():
    global obstacle_in, obstacle_pos
    if obstacle_in:
        obstacle_in = False
        obstacle.T = HIDE
        print("[SENSOR] Obstacle removed")
    elif active_motion is None:
        print("[SENSOR] No motion in progress - start motion first, then press 'o' to drop an obstacle in its path")
    else:
        qs = active_motion["q"]
        k = min(active_motion["idx"] + 25, len(qs) - 1)
        pos = np.array(active_motion["robot"].fkine(qs[k]).t)
        #never drop the obstacle on top of the robot - it must be ahead in the path
        half = np.array(COLLISION_OBSTACLE) / 2 + COLLISION_MARGIN
        now = sample_links(active_motion["robot"].fkine_all(qs[active_motion["idx"]]), 5)
        if np.any(np.all(np.abs(now - pos) <= half, axis = 1)):
            print("[SENSOR] Too close to the robot to place the obstacle - wait, then try again")
            return
        obstacle_pos = pos
        obstacle.T = sm.SE3(obstacle_pos[0], obstacle_pos[1], obstacle_pos[2])
        obstacle_in = True
        print("[SENSOR] Obstacle placed in the planned path")
 
def update_safety(events):
    global estop, in_zone
 
    #1. inputs that change the world
    for ev in events:
        if ev == "estop_toggle":
            estop = not estop
        elif ev == "zone":
            in_zone = not in_zone
            print("[SENSOR] Person entered the danger zone" if in_zone else "[SENSOR] danger zone clear")
        elif ev == "obstacle":
            toggle_obstacle()
 
    # 2. E-Stop prioritised - dont move until released/reset
    estop_in = estop or read_physical_estop()
    if estop_in:
        set_state("ESTOP", "E-Stop pressed")
    elif state == "ESTOP":
        set_state("AWAITING_RESET", "E-Stop released - press 'a' to reset")
 
    # 3. safety operator actions (needs reset and resume)
    #print reasonings for clarity
    for ev in events:
        if ev == "reset":
            if estop_in:
                print("[SAFETY] Release the E-Stop before resetting")
            elif state == "AWAITING_RESET":
                set_state("AWAITING_RESUME", "reset acknowledged - press 's' to resume")
            else:
                print("[SAFETY] Nothing to reset")
        elif ev == "resume":
            if state == "AWAITING_RESET":
                print("[SAFETY] Reset first (press 'a')")
            elif state != "AWAITING_RESUME":
                print(f"[SAFETY] Cannot resume from {state}")
            elif in_zone:
                print("[SAFETY] Cannot resume - unsafe zone still occupied")
            elif predicted_collision():
                print("[SAFETY] Cannot resume - planned path is still blocked")
            else:
                set_state("RUNNING")
 
    # 4. safety stops from the safety sensors
    if state == "RUNNING":
        if in_zone:
            set_state("SAFETY_STOP", "person in danger zone")
        elif predicted_collision():
            set_state("SAFETY_STOP", "collision predicted in planned path")
    elif state == "SAFETY_STOP":
        #safety stops clear by themselves once the hazard is gone
        if stop_reason.startswith("person") and not in_zone:
            set_state("RUNNING", "zone clear - resuming")
        elif stop_reason.startswith("collision") and not predicted_collision():
            set_state("RUNNING", "path clear - resuming")
 
equip_signature = None
 
def update_equipment():
    # warning light colour and light-curtain beam follow the safety state
    global equip_signature
    if state == "RUNNING":
        light = "green"
    elif state in ("SAFETY_STOP", "AWAITING_RESUME"):
        light = "yellow"
    else:
        light = "red"
    pressed = estop or read_physical_estop()
    signature = (light, in_zone, pressed) #checks the colour, if anything is triggered, and button press
    if signature == equip_signature:
        return
    equip_signature = signature
    for name, obj in lights.items():
        obj.T = POLE_POSE if name == light else HIDE
    curtain_safe.T = HIDE if in_zone else CURTAIN_POSE
    curtain_unsafe.T = CURTAIN_POSE if in_zone else HIDE
    estop_button.T = ESTOP_DOWN if pressed else ESTOP_UP
 



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
    start_motion(robot, traj.q)
    for i, q in enumerate(traj.q):
        active_motion["idx"] = i
        robot.q = q
        update_cylinders()
        if carry_object is not None:
            carry_object.T = robot.fkine(robot.q) * carry_offset
        yield
    end_motion()
    return True



def pen_init(position, color):
    #create/add pen at position
    pen = geometry.Cylinder(radius = 0.005, length = 0.15, pose = sm.SE3(position) * sm.SE3.Rx(np.pi / 2), color = color)
    env.add(pen)
    return pen

#yield from -> run motions frame by frame to allow for stoppage
def pen_pick_and_place(pen, pen_pos, drop_pos, clearance = 0.2):
    
    og_pos = robot.fkine(robot.q)

    transit = sm.SE3(pen_pos[0], pen_pos[1], 0.75) * sm.SE3.Rx(np.pi)  # high, safe waypoint

    above_pen = sm.SE3(pen_pos[0], pen_pos[1], pen_pos[2] + 0.3) * sm.SE3.Rx(np.pi)
    at_pen = sm.SE3(pen_pos[0], pen_pos[1], pen_pos[2] + clearance) * sm.SE3.Rx(np.pi)

    above_tray = sm.SE3(drop_pos[0], drop_pos[1], drop_pos[2] + 0.2) * sm.SE3.Rx(np.pi)
    at_tray = sm.SE3(drop_pos[0], drop_pos[1], drop_pos[2] + 0.05) * sm.SE3.Rx(np.pi)


    print(f"Pen at {pen_pos}, moving to pick up.")

    if not (yield from move_to_pose(transit)):
        print("Failed to move to transit position.")
        return False
    if not (yield from move_to_pose(above_pen)):
        print("Failed to move above pen.")
        return False
    if not (yield from move_to_pose(at_pen)):
        print("Failed to move to pen.")
        return False

    carry_offset = sm.SE3(0, 0, -0.075)  # Offset to carry the pen correctly
    yield from move_to_pose(above_pen, carry_object = pen, carry_offset = carry_offset)
    yield from move_to_pose(above_tray, carry_object = pen, carry_offset = carry_offset)
    yield from move_to_pose(at_tray, carry_object = pen, carry_offset = carry_offset)

    pen.T = sm.SE3(drop_pos)
    yield from move_to_pose(og_pos)
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
print("Safety controls:")
print("  space = E-Stop (engage/disengage)")
print("  a     = reset (after E-Stop is released)")
print("  s     = resume (after reset)")
print("  z     = person enters/leaves the danger area")
print("  o     = place an object in the robot's planned path")
print("  q     = quit")

collection_complete = False
def collect_all_pens():
    global collection_complete
    for key, pen_info in pens.items():
        if pen_info["picked"]:
            continue
        print(f"Picking up {pen_info['name']}")
        yield from pen_pick_and_place(pen_info["object"], pen_info["position"], pens_tray_pos[key], clearance = pen_info["clearance"])
        pen_info["picked"] = True
        collection_complete = True
    print("All pens have been picked up.")
    print("Press 'r' for the red pen, 'g' for the green pen or 'b' for the blue pen to have it delivered.")

### UR3E SECTION!

HANDOFF_POS = [-0.65, 0.2, 0.6] #right front of the desk
HANDOFF_HOLD_FRAMES = 40

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



    traj = jtraj(ur3e.q, q_target, steps)
    start_motion(ur3e, traj.q)
    for i, q in enumerate(traj.q):
        active_motion["idx"] = i
        ur3e.q = q
        if carry_object is not None:
            carry_object.T = ur3e.fkine(ur3e.q) * GRIPPER_OFFSET
        yield
    end_motion()
    return True

def ur3e_return_home(steps = 50):
    traj = jtraj(ur3e.q, UR3E_HOME_Q, steps)
    start_motion(ur3e, traj.q)
    for i, q in enumerate(traj.q):
        active_motion["idx"] = i
        ur3e.q = q
        yield
    end_motion()


#RMRC
def ur3e_rmrc_z(z_target, speed = 0.08, dt = 0.05, carry_object = None):
    """RMRC: move the tool in a straight vertical line at a set speed
    each step: joint speeds = pinv(Jacobian) * wanted hand velocity, then q = q + qdot * dt"""
    dz = z_target - ur3e.fkine(ur3e.q).t[2]
    n = int(np.ceil(abs(dz) / (speed * dt)))
    if n == 0:
        return True
    v = np.array([0, 0, np.sign(dz) * speed, 0, 0, 0]) #wanted hand velocity (vx vy vz wx wy wz), angular = 0 keeps the tool pointing down
    q = np.array(ur3e.q, dtype = float)
    path = []
    for _ in range(n):
        qd = np.linalg.pinv(ur3e.jacob0(q)) @ v #pinv copes with near-singular poses
        q = q + qd * dt
        path.append(q.copy())
    start_motion(ur3e, path) #share the path so the collision look-ahead can see it
    for i, q in enumerate(path):
        active_motion["idx"] = i
        ur3e.q = q
        if carry_object is not None:
            carry_object.T = ur3e.fkine(ur3e.q) * GRIPPER_OFFSET
        yield
    end_motion()
    return True

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
    ok = yield from ur3e_move(above)
    if ok:
        ok = yield from ur3e_rmrc_z(get.t[2]) #RMRC: straight down onto the pen
    if not ok:
        yield from ur3e_return_home()
        return False

    yield from ur3e_rmrc_z(above.t[2], carry_object = pen) #RMRC: straight back up with the pen
    yield from ur3e_move(out, carry_object = pen)

    for _ in range(HANDOFF_HOLD_FRAMES): #time for user to take pen
        yield
    pen.T = sm.SE3(0, 0, -5) #pen gone
    info["delivered"] = True
    yield from ur3e_return_home()
    return True




#use pynput for keyboard inputs
pressed_keys = set()
enter_pressed = False

delivery_needed = []
key_events = []
KEY_EVENTS = {"a": "reset", "s": "resume", "z": "zone", "o": "obstacle"}

def on_press(key):
    global enter_pressed
    if key == pynput_keyboard.Key.enter:
        enter_pressed = True
        #RUN KUKA
        return
    if key == pynput_keyboard.Key.space:
        key_events.append("estop_toggle")
        #TRIGGER ESTOP
        return
    try:
        pressed_keys.add(key.char)
        if key.char in ("r", "g", "b"):
            delivery_needed.append(key.char)
            #RUN UR3E
        elif key.char in KEY_EVENTS:
            key_events.append(KEY_EVENTS[key.char])
            #RUN a, s, z, or o
    except AttributeError:
        pass
def on_release(key):
    try:
        pressed_keys.discard(key.char)
    except AttributeError:
        pass


key_listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
key_listener.start()


#Adding GUI on swift for e-stop simulation
env.add(swift.Button(lambda *a: key_events.append("estop_toggle"), desc = "E-Stop (engage/disengage)"))
env.add(swift.Button(lambda *a: key_events.append("reset"), desc = "RESET"))
env.add(swift.Button(lambda *a: key_events.append("resume"), desc = "RESUME"))

status_label = swift.Label("STATE: RUNNING")
env.add(status_label)
last_status = ""

def update_gui():
    global last_status
    text = f"STATE: {state}"
    if stop_reason:
        text += f" - {stop_reason}"
    if in_zone and "person" not in stop_reason:
        text += f" | WARNING: Person in danger zone."
    if obstacle_in and "collision" not in stop_reason:
        text += f" | WARNING: Obstacle in path."
    if text != last_status:
        last_status = text
        status_label.desc = text
    



#adjusted MAIN LOOP for safety system inputs
            
active_task = None
running = True
while running:
    events = []
    while key_events:
        events.append(key_events.pop(0))
    update_safety(events)

    if enter_pressed:
        enter_pressed = False
        if state != "RUNNING":
            print(f"[SAFETY] Cannot start - system is {state}")
        elif active_task is None and not collection_complete:
            #collect pens (kuka)
            active_task = collect_all_pens()

    if state == "RUNNING":
        if active_task is None and collection_complete and delivery_needed:
            active_task = deliver_pen(delivery_needed.pop(0))
            #Deliver pens to user (ur3e)
        if active_task is not None:
            try:
                next(active_task)
            except StopIteration:
                active_task = None

    update_equipment()
    update_gui()

    if 'q' in pressed_keys:
        print("Quitting...")
        running = False
        #quit program

    try:
        env.step(0.05)
    except AttributeError:
        pass
    time.sleep(0.05)

key_listener.stop()