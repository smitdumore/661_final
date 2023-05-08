import numpy as np
import math
import cv2
import time
import resource

# set hard limit of 2 seconds of CPU time
hard_limit = 120
resource.setrlimit(resource.RLIMIT_CPU, (hard_limit, hard_limit))

tab_width = 600
tab_height = 200

def get_inquality_obstacles(x, y, clearance):
    if (x >= (tab_width - clearance)) or (y >= (tab_height - clearance)) or (x <= clearance) or (y <= clearance) or\
       ((y >= 75 - clearance) and (x <= (165 + clearance)) and (x >= (150 - clearance)) and (y <= tab_height)) or\
       ((y <= (125 + clearance)) and (x >= (250 - clearance)) and (y >= 0) and (x <= (265 + clearance))) or\
        (((x-400)**2 + (y-110)**2) < (60 + clearance)**2):
            return True
    return False

# def get_inquality_obstacles(x, y, clearance):
#     if (x >= (tab_width - clearance)) or (y >= (tab_height - clearance)) or (x <= clearance) or (y <= clearance):
#             return True
#     return False

class Node:
    def __init__(self, x, y, parent=None, cost = np.inf):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost

class RRT:
    def __init__(self, start, goal, clearance):

        self.start_node = start
        self.goal_node = goal
        self.clearance = clearance
        self.goal_tolerance = 15

        self.steer_len = 30
        self.rewire_rad = 60
        
        self.tree = None
        self.path = []
        self.flag = False
        self.colo = 100
        

    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + abs(node1.y - node2.y)**2)

    def random_node(self):
        rnd = Node(np.random.randint(0, 600), np.random.randint(0, 200))
        return rnd

    def gen_biased_nodes(self, major_axis, focal_dist, origin, rot):
        
        if major_axis == float('inf'):
            return self.random_node()
        
        x = np.random.random()
        y = np.random.random()

        minor_axis = [major_axis / 2.0, math.sqrt(abs(major_axis ** 2 - focal_dist ** 2)) / 4.0, 
            math.sqrt(abs(major_axis ** 2 - focal_dist ** 2)) / 4.0]

        r_array = np.array([[minor_axis[0], 0.0, 0.0], 
                            [0.0, minor_axis[1], 0.0], 
                            [0.0, 0.0, minor_axis[2]]])

        
        new_node = (y * math.cos(2 * math.pi * x / y),
                y * math.sin(2 * math.pi * x / y))

        threeD_pt = np.array([[new_node[0]], [new_node[1]], [0]])

        term_1 = np.matmul(rot, r_array)
        term_2 = np.matmul(term_1, threeD_pt)

        new_point = term_2 + origin

        return Node(new_point[0][0], new_point[1][0])
    
    def get_path_len(self, path):
        path = np.array(path)
        path_diff = np.diff(path, axis=0)
        path_len = np.sum(np.sqrt(np.sum(path_diff ** 2, axis=1)))
        return int(path_len)

    def nearest(self, node):
        nearest_node = None
        nearest_dist = np.inf
        
        for n in self.tree:
            dist = self.distance(n, node)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_node = n
                
        if(nearest_dist == np.inf):
            return self.start_node
        
        return nearest_node

    def steer(self, from_node, to_node):

        new_node = Node(from_node.x, from_node.y)
        
        dist = self.distance(from_node, to_node)

        if dist > self.steer_len:
            theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
            new_node.x = int(from_node.x + self.steer_len * math.cos(theta))
            new_node.y = int(from_node.y + self.steer_len * math.sin(theta))
            new_node.cost = self.steer_len
        else:
            new_node.x = int(to_node.x)
            new_node.y = int(to_node.y)
            new_node.cost = dist

        return new_node
    
    def near_nodes(self, node, rad):
        
        near_nodes = []

        # node_list is the tree itself
        for n in self.tree:
            if self.distance(n, node) < rad:
                near_nodes.append(n)
        return near_nodes

    def rewire(self, node, img):
        
        # see if new node can be a new parent to already exsisting node

        near_nodes = self.near_nodes(node, self.rewire_rad)

        for near_node in near_nodes:
        
            if near_node == node.parent:
                continue

            if(self.check_collision(near_node, node)):
                continue
            
            new_cost = node.cost + self.distance(node, near_node)
            
            if new_cost < near_node.cost:
                near_node.parent = node
                near_node.cost = new_cost
                ## erase
                cv2.line(img, (node.x, node.y),(node.parent.x, node.parent.y),(0, 0, 0), 1)

                ## draw new
                cv2.line(img, (near_node.x, near_node.y),(node.parent.x, node.parent.y),(255, 0, 0), 1)

    def func(self, a1):
        orientation_ellipse = math.atan2(a1[1], a1[0])  
        
        _, _, vh = np.linalg.svd(a1.T)
        
        c = vh.T @ np.diag([1.0, 1.0, np.linalg.det(vh)]) @ vh

        return orientation_ellipse, c

    def search(self):

        self.tree = []
        
        start_node = self.start_node
        goal_node = self.goal_node
        
        self.tree.append(start_node)

        ### CANVAS ###
        img = np.zeros((200, 600, 3), dtype=np.uint8)
        ### CANVAS ###

        cv2.circle(img, (start_node.x, start_node.y), 5, (255, 0, 0), -1)
        cv2.circle(img, (goal_node.x, goal_node.y), 5, (0, 255, 0), -1)  

        for i in range(600):
            for j in range(200):
                if(get_inquality_obstacles(i,j,self.clearance)):
                    cv2.circle(img, (i, j), 1, (0, 255, 255), -1)

        major_axis = float('inf')
        
        origin = np.array([[(self.start_node.x + self.goal_node.x) / 2.0],
                             [(self.start_node.y + self.goal_node.y) / 2.0], [0]])
        
        minor_axis = math.hypot(self.start_node.x - self.goal_node.y, self.start_node.y - self.goal_node.y)
        
        a1 = np.array([[(self.goal_node.x - self.start_node.x) / minor_axis], [(self.goal_node.y - self.start_node.y) / minor_axis], [0]])
        
        orientation_ellipse, c = self.func(a1)

        ##################
        #### RRT LOOP ####
        ##################
        for i in range(3000):

            rand_node = self.gen_biased_nodes(major_axis, minor_axis, origin, c)

            if get_inquality_obstacles(rand_node.x, rand_node.y, self.clearance):
                continue
            
            nearest_node = self.nearest(rand_node)
            
            new_node = self.steer(nearest_node, rand_node)

            if get_inquality_obstacles(new_node.x, new_node.y, self.clearance):
                continue
            
            if not self.check_collision(nearest_node, new_node):
                #print("showing rand node")

                new_node.parent = nearest_node
                
                #cv2.line(img, (nearest_node.x, nearest_node.y),(new_node.x, new_node.y),(255, 0, 0), 1)
                cv2.circle(img, (new_node.x, new_node.y), 1, (0, 0, 255), -1)
                cv2.imshow("RRT Tree", img)
                cv2.waitKey(10)

                near_nodes = self.near_nodes(new_node, self.steer_len)
                
                # find cheapest parent for new node
                node_with_min_cost = nearest_node
                min_cost = nearest_node.cost + self.distance(nearest_node,new_node)
                
                for near_node in near_nodes:
                    
                    if self.check_collision(near_node, new_node):
                        continue

                    if near_node.cost + self.distance(near_node,new_node) < min_cost:
                        node_with_min_cost = near_node
                        min_cost = near_node.cost + self.distance(near_node, new_node)
                
                new_node.parent = node_with_min_cost
                new_node.cost = min_cost

                self.tree.append(new_node)
                
                self.rewire(new_node, img)
            
            else:

                new_node.parent = None
                continue


            if self.distance(self.goal_node, new_node) <= self.goal_tolerance:
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + self.distance(new_node, goal_node)
                self.tree.append(goal_node)
                print("FOUNDDDDDDDDDDDDDD")

                temp_path = self.extract_path(goal_node)
                temp_path_len = self.get_path_len(temp_path)

                if major_axis != float('inf'):
                    print("plotting ellipse...................")
                    
                    for i in range(600):
                        for j in range(200):
                            if(get_inquality_obstacles(i,j,self.clearance)):
                                cv2.circle(img, (i, j), 1, (0, 255, 255), -1)

                    self.draw_region(img, origin, major_axis, minor_axis, orientation_ellipse)

                if temp_path_len < major_axis:

                    self.path = temp_path
                    
                    major_axis = abs(temp_path_len)
                    
                    self.flag = True
                    print("path found, finding optimal")
                    print("path visualising")

                    self.colo += 50

                    for i in range(self.path.shape[0] - 1):
                        cv2.line(img, (int(self.path[i, 0]), int(self.path[i, 1])),
                        (int(self.path[i + 1, 0]), int(self.path[i + 1, 1])),
                        (0, 50 + self.colo, 0), 2)

                    img_new = np.zeros((200, 600, 3), dtype=np.uint8)

                    for i in range(600):
                        for j in range(200):
                            if(get_inquality_obstacles(i,j,self.clearance)):
                                cv2.circle(img_new, (i, j), 1, (0, 255, 255), -1)

                    for i in range(self.path.shape[0] - 1):
                        cv2.line(img_new, (int(self.path[i, 0]), int(self.path[i, 1])),
                        (int(self.path[i + 1, 0]), int(self.path[i + 1, 1])),
                        (0, 255, 0), 2)

                    cv2.imshow("RRT Tree new", img_new)
                    cv2.waitKey(10)

                    
                    cv2.imshow("RRT Tree", img)
                    cv2.waitKey(10)

        
        ## for ends

        if(self.flag):
            cv2.destroyAllWindows()

        return self.path
    
    def rotation_matrix(self, theta):
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])
        
        return rotation_matrix

    def draw_region(self, canvas, origin, major_axis, focal_dis, orientation_ellipse):

        minor_axis = abs(major_axis ** 2 - focal_dis ** 2) / 4.0

        a = math.sqrt(minor_axis) / 2.0

        b = major_axis / 2.0

        angle = math.pi / 2.0 - orientation_ellipse

        cx = int(origin[0])
        cy = int(origin[1])

        t = []
        x = []
        y = []

        for i in range(int(20 * math.pi)):
            x.append(a * math.cos(i/10.0))
            y.append(b * math.sin(i/10.0))
            

        fx = []
        fy = []

        for i in range(len(x)):
            fx.append(x[i] * math.cos(angle) - y[i] * math.sin(angle))
            fy.append(x[i] * math.sin(angle) + y[i] * math.cos(angle))

        px = []
        py = []

        for i in range(len(fx)):
            px.append(int(round(fx[i] + cx)))
            py.append(int(round(fy[i] + cy)))

        pts = []

        for i in range(len(px)):
            pts.append((px[i], py[i]))

        pts = np.array(pts)

        cv2.polylines(canvas, [pts], True, (255, 255, 255))


        
    def check_collision(self, nearest_node, new_node):
        """
        Check for collision between two nodes: nearest_node and new_node
        """
        x1, y1 = nearest_node.x, nearest_node.y
        x2, y2 = new_node.x, new_node.y

        dx = math.ceil(x2 - x1)
        dy = math.ceil(y2 - y1)
        
        steps = 10
        
        x_step = dx // steps
        y_step = dy // steps

        # Check for collision with each point on the line segment
        for i in range(int(steps) + 1):

            x = math.ceil(x1 + i * x_step)
            y = math.ceil(y1 + i * y_step)

            # Check if the point collides with any obstacles
            if get_inquality_obstacles(x, y, self.clearance):
                return True

        return False  # No collision

    def extract_path(self, goal_node):

        print("extracting path")
        path = []
        node = goal_node
        visited = set()  # set of visited nodes
        while node is not None and node.parent is not None:
            if node in visited:
                print("Error: Found a loop in the graph!")
                return self.path
                # return IRRT_main()
            visited.add(node)
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])  # add last node
        path.reverse()
        return np.array(path)

######## CHANGE VARIABLE NAMES

def IRRT_main():

    start_x = int(50)
    start_y = int(100)

    goal_x = int(500)
    goal_y = int(100)

    clearance = int(15)

    if get_inquality_obstacles(start_x, start_y, clearance):
        print("Start in obstacle, exit")
        return []
    
    if get_inquality_obstacles(goal_x, goal_y, clearance):
        print("Goal in obstacle, exit")
        return []
    
    start_node = Node(start_x, start_y, None, 0)
    goal_node = Node(goal_x, goal_y, None, 0)

    obj = RRT(start_node, goal_node, clearance)
    path = obj.search()

    return path


#print(IRRT_main())
