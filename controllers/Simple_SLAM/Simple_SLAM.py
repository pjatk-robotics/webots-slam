"""Simple keyboard controlled Graph SLAM example"""
import math
import numpy as np
from numpy import inf
from controller import Robot, Keyboard
import matplotlib.pyplot as plt

# ------------------ global variables -----------------------------

CRUISING_SPEED = 5.0
TURN_SPEED = 1
TIME_STEP = 0
TIME_STEP_SECONDS = 0.0
WHEEL_UNIT = 0.033
WHEEL_SEPARATION = 0.178 #the distance is given as 16cm on website, but in webots it's 17.8
MAPPING_RATE = 3000 #frequency of expanding map in miliseconds
ESTIMATED_ERROR_STEP = 1.00015 #estimated compounding divergence from odometry per timestep
MAPPING_STEPS = 0 #initialized as MAPPING_RATE/TIME_STEP after acquiring timestep value
LIDAR_RANGE = 3.5

ICP_Iter = [0] #iterator for generation of ICP step images

# ------------------ classes --------------------------------------
class PD_controller: #PD regulator for drive control
        def __init__(self, p, d, sampling_period, target=0.0):
            self.target = target
            self.response = 0.0
            self.old_error = 0.0
            self.p = p
            self.d = d
            self.sampling_period = sampling_period
            
        def process_measurement(self, measurement):
            error = self.target - measurement
            derivative = (error - self.old_error)/self.sampling_period
            self.old_error = error
            self.response = self.p*error + self.d*derivative
            
            return self.response
        
        def reset(self):
            self.target = 0.0
            self.response = 0.0
            self.old_error = 0.0
            
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

class MotorController: #wrapper for individual motors
    def __init__(self, pd):
        self.pd = pd
        self.motor = None
        self.enabled = False
        self.velocity = 0.0
        
    def enable(self, robot, timestep, name):
        if not self.enabled:
            self.motor = robot.getDevice(name)
            self.motor.setPosition(float('inf'))
            self.motor.setVelocity(0.0)
            self.enabled = True
            
    def update(self):
        self.velocity += self.pd.process_measurement(self.motor.getVelocity())
        self.motor.setVelocity(self.velocity)
        
    def set_target(self, target):
        self.pd.target = target
        
    def emergency_brake(self):
        self.motor.setVelocity(0.0)
        self.pd.reset()
        print("Emergency brake")
        
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
        
class Drive_Module: #movement control 
    def __init__(self,  p, d, timestep):
        self.enabled = False
        self.left_motor = MotorController(PD_controller(p, d, timestep))
        self.right_motor = MotorController(PD_controller(p, d, timestep))
        self.commands = {
            ord('W'): (CRUISING_SPEED, CRUISING_SPEED),
            ord('S'): (-CRUISING_SPEED, -CRUISING_SPEED),
            ord('A'): (-TURN_SPEED, TURN_SPEED),
            ord('D'): (TURN_SPEED, -TURN_SPEED),
            ord('E'): (0.0, 0.0)
        } 

        
    def enable(self, robot, timestep, left_motor_name, right_motor_name):
        if not self.enabled:
            self.left_motor.enable(robot, timestep, left_motor_name)
            self.right_motor.enable(robot, timestep, right_motor_name)
            self.enabled = True
        
    def update(self):
        self.left_motor.update()
        self.right_motor.update()
        
    def emergency_brake(self):
        self.left_motor.emergency_brake()
        self.right_motor.emergency_brake()
        
    def KeyAction(self, key): 
        if key in self.commands.keys():
            self.left_motor.set_target(self.commands[key][0])
            self.right_motor.set_target(self.commands[key][1])
            
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
                
class Odometry_Module: #simple location from encoders
    def __init__(self):
        self.enabled = False
        self.left_sensor = None
        self.right_sensor = None
        self.pos = [0.0, 0.0] #x, y
        self.angle = 0.0
        self.left_sensor_last = 0.0
        self.right_sensor_last = 0.0
        self.distance = 0.0
        self.total_rotation = 0.0
        
    def enable(self, robot, timestep, left_sensor_name,
               right_sensor_name):
        if not self.enabled:
            self.left_sensor = robot.getDevice(left_sensor_name)
            self.right_sensor = robot.getDevice(right_sensor_name)
            self.left_sensor.enable(timestep)
            self.right_sensor.enable(timestep)
            self.enabled = True
        
    def update(self):
        new_sensor_left = self.left_sensor.getValue()
        new_sensor_right = self.right_sensor.getValue()
        
        delta_l = (self.left_sensor_last - new_sensor_left) * WHEEL_UNIT
        delta_r = (self.right_sensor_last - new_sensor_right) * WHEEL_UNIT
        
        linear_v = (delta_l + delta_r)/2 
        angular_v = (delta_l - delta_r)/WHEEL_SEPARATION 
        

        self.angle += (angular_v)
        self.angle = self.angle % (math.pi*2)
        self.pos[0] += (linear_v * math.cos(self.angle))
        self.pos[1] += (linear_v * math.sin(self.angle))
        
        self.distance += abs(linear_v)
        self.total_rotation += abs(angular_v)
        
        self.left_sensor_last = new_sensor_left
        self.right_sensor_last = new_sensor_right
        

    def getPosition(self):
        return [self.pos[0], self.pos[1], self.angle]

    def setPosition(self, pos, angle):
        self.pos = pos
        self.angle = angle
    
    def getDistanceTravelled(self):
        ret = [self.distance, self.total_rotation]
        return ret
    
    def resetStep(self):
        self.distance = 0
        self.total_rotation = 0
    
    
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    
class SLAM_Module: #lidar mapping + localization
    def __init__(self, odometry):
        self.enabled = False
        self.graph = []
        self.divergence = 1
        self.last = None;
        self.odometry_module = odometry
        self.map_iterator = 0
        
        self.aligned = 'False'

        
        
    def enable(self, robot, lidar, TIME_STEP):
        if not self.enabled:
            self.odometry_module.enable(robot, TIME_STEP,
                                   'left wheel sensor', 'right wheel sensor')
            self.lidar = robot.getDevice(lidar)
            self.lidar.enable(TIME_STEP)
            self.lidar.enablePointCloud()
            self.enabled = True

        
    def getPointCloud(self, pos): #creating point cloud from sensor range image
        sensor_image = self.lidar.getRangeImage() 
        obstacle_points = []
        for i in range(len(sensor_image)):
            angle = pos[2] + math.radians(360-i)
            if not sensor_image[i] == inf:
                x = sensor_image[i]*math.cos(angle) + pos[0]
                y = sensor_image[i]*math.sin(angle) + pos[1]
                w = 0.95               
                if sensor_image[i] < 0.5:
                    w = 1.0-0.015/sensor_image[i]
                obstacle_points.append(np.array([x, y, w]))
                
        return obstacle_points
        
    
        
    def updateGraph(self):       
        #if the distance travelled is less than 5cm 
        #consider robot to be stationary and do not update graph
        dist = self.odometry_module.getDistanceTravelled()
        if dist[0] < 0.05:
            return 
        self.odometry_module.resetStep()
        ICP_Iter[0] = 0
                
        pos = np.array(self.odometry_module.getPosition())
        points = self.getPointCloud(pos)
        
        #estimate postion divergence from odometry 
        last_divergence = (1+(((ESTIMATED_ERROR_STEP ** MAPPING_STEPS)-1)
                              *(dist[0]+dist[1])))
        divergence = self.divergence*last_divergence
        
        new_node = Node(pos, points, divergence)
        self.aligned = False
        if self.last is not None:
            new_node.addRelation(self.last, 1/last_divergence)
            
            
        if len(new_node.points) > 60:
            near_nodes = self.findNearNodes(new_node, divergence)
            sortNodesDiv(near_nodes, 0, len(near_nodes) -1) 
            #sort nodes starting from lowest divergence
    
            for node in near_nodes:
                if (len(node.points) <= 60):
                    continue
                relative_divergence = new_node.divergence - node.divergence
                new_point_cloud = [] #points with matching counterparts
                old_point_cloud = [] #preexisting points with matching counterparts
                weights = []
                lasterror = 99999.0 #initial match quality for icp
                errr = 9999.0
                last_len = 1
                target_cloud = None
                base_cloud = None
                total_rotation = None
                total_translation = None
                if relative_divergence > 0:
                    target_cloud = new_node.getCopyCloud()
                    base_cloud = node.getCopyCloud()
                else:
                    target_cloud = node.getCopyCloud()
                    base_cloud = new_node.getCopyCloud()
                    
                while errr < lasterror*0.98: #every update must be at least a 2% improvement
                    new_point_cloud = []
                    old_point_cloud = []
                    lasterror = errr
                    errr = 0
                    for point in target_cloud:
                        nearest = None 
                        d = 999999 #initial nearest distance
                        for tniop in base_cloud:
                             if (d > #current lowest distance
                                 distancesq(point[:2], tniop[:2]) < #distance between curent points
                                 relative_divergence**2-0.000025):
                                 nearest = tniop #assign current point as nearest
                                 d = distancesq(point[:2], tniop[:2]) #assign current distance as lowest

                        if nearest is not None:
                            new_point_cloud.append(point[:2])
                            old_point_cloud.append(nearest[:2])
                            weights.append(point[2])
                            errr += d
                    if len(new_point_cloud) > 60: 
                        last_len = len(new_point_cloud)
                        delta, R = SLAM_Module.alignPointClouds(np.array(new_point_cloud),
                                                    np.array(old_point_cloud),
                                                    np.array(weights),
                                                    self.map_iterator)
                        Node.realignCloud(target_cloud, R, delta)
                        if total_rotation is None:
                             total_rotation = R
                        else:
                             total_rotation = R @ total_rotation
                        if total_translation is None:
                            total_translation = delta
                        else:
                            total_translation = delta + total_translation
                    
                    else:#end of ICP loop
                        lasterror = 0
                        errr = 999999999999999.0
                        
                if total_rotation is not None and total_translation is not None:
                    final_err = errr
                    if final_err/last_len < 0.001 and final_err < 0.15:
                        self.aligned = 'True'
                        if relative_divergence > 0:
                            new_node.realign(total_rotation, total_translation,
                                             1-final_err, node, relative_divergence, 
                                             new_node)
                        else:
                            node.realign(total_rotation, total_translation,
                                         1-final_err, new_node, relative_divergence,
                                         new_node)

        self.divergence = new_node.divergence
        self.odometry_module.setPosition(new_node.position, new_node.rotation)
        self.last = new_node
        self.graph.append(new_node)
        self.showMap()
        self.map_iterator += 1


    def findNearNodes(self, current_node, divergence):
        distance = (LIDAR_RANGE+(divergence-1)) ** 2
        highx = current_node.position[0]+LIDAR_RANGE+((divergence-1)*1.3)
        lowx = current_node.position[0]-LIDAR_RANGE-((divergence-1)*1.3)
        highy = current_node.position[1]+LIDAR_RANGE+((divergence-1)*1.3)
        lowy = current_node.position[1]-LIDAR_RANGE-((divergence-1)*1.3)
        
        near_nodes = []
        
        for node in self.graph:
            if (highx > node.position[0] > lowx and
                highy > node.position[1] > lowy):
                if (distancesq(node.position, current_node.position) <
                    distance):
                            near_nodes.append(node)
        return near_nodes         

  
    def showMap(self, display = False):
        allx = []
        ally = []
        pastx = []
        pasty = []
        pastdiv = []
        pos = self.odometry_module.getPosition()
        for loc in self.graph:
            for point in loc.points:
                allx.append(point[0])
                ally.append(point[1])
            pastx.append(loc.position[0])
            pasty.append(loc.position[1])
            pastdiv.append((loc.divergence-1)*1000+1)
        
        fig = plt.figure(facecolor=(0.11, 0.1, 0.12))
        plt.xlim([-10.2, 2.2])
        plt.ylim([-7.2, 4.2])
        plt.plot(pastx, pasty, linewidth=2, c='deepskyblue') #traveled path
        plt.scatter(allx, ally, s=1, c='y') #found obstacles
        plt.scatter(pastx, pasty, s=pastdiv, c='c') #past scan locations
        #current position
        plt.scatter(pos[0], pos[1], s=max((self.divergence-1)*1200, 10), c='m') 
        #if last location and position fix was succesful
        plt.title(f'Alignment Fixed: {self.aligned}', fontdict={'color':'white'}) 
        ax = plt.gca()
        ax.set_facecolor((0.21, 0.2, 0.22))
        for spine in ax.spines.values():
            spine.set_color('w')
        ax.xaxis.label.set_color('w')
        ax.tick_params(axis='x', colors='w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='y', colors='w')
        
        print(f'graph: {len(self.graph)}')
        if display:
            plt.show()
        else:
            plt.savefig(f'../../maps/{self.map_iterator}_steps.png')
            plt.close(fig)

     # Point Cloud Alignment for ICP
    def alignPointClouds(sensor_cloud, map_cloud, sensor_weights, map_iterator):

        sensor_center = np.array([0.0, 0.0])
        map_center = np.array([0.0, 0.0])
        
        #estimate and overlay center of mass
        for i in range(len(sensor_cloud)-1):
            sensor_center += np.array(sensor_cloud[i])*sensor_weights[i]
            map_center += np.array(map_cloud[i])*sensor_weights[i]
            
        sensor_center /= sum(sensor_weights)
        map_center /= sum(sensor_weights)
            
       
        H=np.array([[0.0, 0.0],[0.0, 0.0]])
        
        for i in range(len(sensor_cloud)-1):
            H += (np.multiply((sensor_cloud[i]-sensor_center),
                             (map_cloud[i]-map_center)[:, np.newaxis]))*sensor_weights[i]
            
        #Using SVD to estimate optimal rotation
        U, D, VT = np.linalg.svd(H)
        R = np.transpose(VT) @ np.transpose(U)
        delta = map_center - R @ sensor_center
        
        #preparing point clouds for visual output
        c = np.copy(sensor_cloud)
        a = np.rot90(map_cloud)
        b = np.rot90(sensor_cloud)
        for i in range(len(c)):
           c[i] = R @ c[i] + delta
        c = np.rot90(c)

        #matched point clouds
        fig1 = plt.figure()
        plt.scatter(a[1], a[0], c='red', s=3)
        plt.scatter(b[1], b[0], c='green', s=2)
        plt.title("Before Alignment")

        plt.savefig(f'../../maps/ICP/{map_iterator}_{ICP_Iter[0]}_1_Base.png')
        plt.close(fig1)
        
        #matched point clouds with matches marked
        fig2 = plt.figure()
        plt.plot([a[1], b[1]], [a[0], b[0]], 'k-', linewidth=1)
        plt.scatter(a[1], a[0], c='red', s=3)
        plt.scatter(b[1], b[0], c='green', s=2)
        plt.title("Matches")
        plt.savefig(f'../../maps/ICP/{map_iterator}_{ICP_Iter[0]}_2_Links.png')
        plt.close(fig2)
        
        #point clouds after alignment
        fig3 = plt.figure()
        plt.scatter(a[1], a[0], c='red', s=3)
        plt.scatter(c[1], c[0], c='green', s=2)
        plt.title("After Alignment")
        plt.savefig(f'../../maps/ICP/{map_iterator}_{ICP_Iter[0]}_3_Fix.png')
        plt.close(fig3)
        
        ICP_Iter[0]+=1
        
        return delta, R
        
     
           
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

class Node:
    def __init__(self, position, points, divergence):
        self.adjusted = False
        self.position = np.array(position[:2])
        self.rotation = position[2]
        self.points = np.array(points)
        self.divergence = divergence
        self.relations = {}
        self.last_adjusted = None;
        
    def addRelation(self, relation, certainty):
        self.relations[relation] = certainty;
        if self not in relation.relations.keys():
            relation.addRelation(self, certainty)
            
    def getCopyCloud(self):
        return np.copy(self.points)
            
    def realign(self, rotation, translation, certainty, node, delta_div, adjust_by):
        self.adjusted = True
        t_value = (translation[0]**2+translation[1]**2)
        rot_value = np.arcsin(rotation[1][0])
        
        self.position = rotation @ self.position[:2] + translation
        self.rotation = (self.rotation + np.arcsin(rotation[1][0]))%(2*math.pi)
        self.last_adjusted = adjust_by
    
        Node.realignCloud(self.points, rotation, translation)
        self.divergence = self.divergence - (delta_div*certainty)

        for r_node in self.relations.keys():
            if r_node is not node:
                r_certainty = self.relations[r_node]
                if (r_node.adjusted == False and r_node.last_adjusted is not adjust_by
                                        #if adjustment translation is at least ~3mm
                                        and (t_value*r_certainty > 0.00001 
                                        #or adjustment rotation is at least 0.05 deg
                                        or abs(rot_value*r_certainty) > 0.00175)): 
                    #modified rotation matrix
                    mod_rotation = np.array([[np.cos(rot_value*r_certainty), 
                                              np.sin(rot_value*r_certainty)],
                                             [-1*np.sin(rot_value*r_certainty),
                                              np.cos(rot_value*r_certainty)]])
                    
                    r_node.realign(mod_rotation,
                                   translation*r_certainty,
                                   certainty*r_certainty,
                                   self,
                                   delta_div*r_certainty,
                                   adjust_by)
        if node in self.relations.keys():
            rel_value = 1-(1-self.relations[node])*(1-certainty**2)
            self.relations[node] = rel_value
            node.relations[self] = rel_value
        else:
            self.addRelation(node, certainty)
        self.adjusted = False
                
    def realignCloud(points, rotation, translation):
        for i in range(len(points)):
            points[i] = np.concatenate((rotation @ points[i][:2] + translation,
                                       np.array([points[i][2]])))
            
        
    
    

# ------------------ utilty functions -----------------------------
def sortNodesDiv(graph, low, high): #QuickSort nodes by divergence
    if low < high:
        pivot = partitionQS(graph, low, high)
        sortNodesDiv(graph, low, pivot-1)
        sortNodesDiv(graph, pivot+1, high)
        
def partitionQS(graph, low, high):
    pivot = graph[high]
    i = low - 1
    for j in range(low, high):
        if(graph[j].divergence) <= pivot.divergence:
            i += 1
            graph[i], graph[j] = graph[j], graph[i]
    graph[i+1], graph[j] = graph[j], graph[i+1]
    return i+1

def distancesq(a, b): #simple squared distance for comparsions
        return ((a[0]-b[0]) ** 2) + ((a[1]-b[1]) ** 2)

# ------------------ main loop ------------------------------------
#initializing robot and keyboard control
robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())
keyboard = Keyboard()
keyboard.enable(TIME_STEP)
TIME_STEP_SECONDS = TIME_STEP/1000
MAPPING_STEPS = MAPPING_RATE/TIME_STEP

#initializing motor control
drive_module = Drive_Module(0.3, 0.005, TIME_STEP_SECONDS)
drive_module.enable(robot, TIME_STEP, 'left wheel motor', 'right wheel motor')
    
#initializing mapper
slam_module = SLAM_Module(Odometry_Module())
slam_module.enable(robot, 'LDS-01', TIME_STEP)

#mapping iterator
map_iterator = 0

#main loop
while robot.step(TIME_STEP) != -1:
    key = keyboard.getKey()
    #motor controls
    if key == ord('Q'):
        drive_module.emergency_brake()
    else:
        drive_module.KeyAction(key)
        
    drive_module.update()
    
    #SLAM
    slam_module.odometry_module.update()
    
    map_iterator = map_iterator + TIME_STEP
    if map_iterator > MAPPING_RATE:
        map_iterator = map_iterator % MAPPING_RATE
        slam_module.updateGraph()
        
    #debug
    if key == ord('R'):
        slam_module.showMap(True)
    if key == ord('Y'):
        print(f'{slam_module.divergence}\n'
              f'{slam_module.odometry_module.getPosition()}')

    if key == ord('P'):
        print(slam_module.lidar.getRangeImage())
    if key == ord('M'):
        slam_module.showMap(display=True)