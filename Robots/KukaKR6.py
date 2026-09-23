import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH

class KukaKR6(DHRobot):
    #standard DH parameters for KUKA KR6 R900 sixx
    def __init__(self):

        dg = np.pi/180

        L = [
            RevoluteDH(a = 0, alpha = 0, d = 0, qlim = [-170 * dg, 170 * dg]),                  #Joint 1
            RevoluteDH(a = 0, alpha = -np.pi/2, d = 0, qlim = [-190 * dg, 45 * dg]),            #Joint 2
            RevoluteDH(a = 0.455, alpha = 0, d = 0, qlim = [-120 * dg, 156 * dg]),               #Joint 3
            RevoluteDH(a = 0.035, alpha = -np.pi/2, d = 0.42, qlim = [-185 * dg, 185 * dg]),    #Joint 4
            RevoluteDH(a = 0, alpha = -np.pi/2, d = 0, qlim = [-120 * dg, 120 * dg]),           #Joint 5
            RevoluteDH(a = 0, alpha = np.pi/2, d = 0.08, qlim = [-350 * dg, 350 * dg]),         #Joint 6
        ]
        super().__init__(L, name = 'Kuka_KR6_R900_sixx')
        #start from home position (all zeros)
        self.qz = np.zeros(6)
        self.q=self.qz

if __name__ == "__main__":
    robot = KukaKR6()
    print(robot)
#test
    print("Forward kinematics at q = 0:")
    print(robot.fkine(robot.qz))
