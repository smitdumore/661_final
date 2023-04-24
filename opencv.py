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
    if (((x-400)**2 + (y-110)**2) < (10 + clearance)**2) or\
        ((x-200)**2 + (y-110)**2) < (10 + clearance)**2 or\
        (x >= (tab_width - clearance)) or (y >= (tab_height - clearance)) or (x <= clearance) or (y <= clearance):
            return True
    
    return False

class Node:
    def __init__(self, x, y, parent=None, cost = np.inf):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost

class RRT:
    def __init__(self, start, goal, clearance, expand_dis= 30,
                 goal_sample_rate=10, max_iter=2000):

        self.start_node = start
        self.goal_node = goal
        self.expand_dis = expand_dis
        self.rewire_rad = 60
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.clearance = clearance
        self.goal_tolerance = 10
        self.obstacle_list = []
        self.node_list = None
        self.path = []
        self.flag = False
        self.colo = 100
        

    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + abs(node1.y - node2.y)**2)

    def random_node(self):
        rnd = Node(np.random.randint(0, 600), np.random.randint(0, 200))
        return rnd


    def informed_sample(self, c_max, c_min, x_center, c):
        if c_max < float('inf'):
            if c_min >= c_max:
                return self.random_node()
            r = [c_max / 2.0, math.sqrt(c_max ** 2 - c_min ** 2) / 2.0,
                 math.sqrt(c_max ** 2 - c_min ** 2) / 2.0]
            rl = np.diag(r)
            x_ball = self.sample_unit_ball()
            rnd = np.dot(np.dot(c, rl), x_ball) + x_center
            rnd = [rnd[(0, 0)], rnd[(1, 0)]]
            rnd_n = Node(rnd[0], rnd[1])
        else:
            rnd_n = self.random_node()

        return rnd_n

    def get_path_len(self, path):
        path_len = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            path_len += math.hypot(node1_x - node2_x, node1_y - node2_y)

        return path_len


    def nearest(self, node):
        nearest_node = None
        nearest_dist = np.inf
        
        for n in self.node_list:
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

        if dist > self.expand_dis:
            theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
            new_node.x = int(from_node.x + self.expand_dis * math.cos(theta))
            new_node.y = int(from_node.y + self.expand_dis * math.sin(theta))
            new_node.cost = self.expand_dis
        else:
            new_node.x = int(to_node.x)
            new_node.y = int(to_node.y)
            new_node.cost = dist

        return new_node
    
    def sample_unit_ball(self):
        a = np.random.random()
        b = np.random.random()

        if b < a:
            a, b = b, a

        sample = (b * math.cos(2 * math.pi * a / b),
                  b * math.sin(2 * math.pi * a / b))
        return np.array([[sample[0]], [sample[1]], [0]])

    def near_nodes(self, node, rad):
        
        near_nodes = []

        # node_list is the tree itself
        for n in self.node_list:
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



    def search(self):

        self.node_list = []
        
        start_node = self.start_node
        goal_node = self.goal_node
        
        self.node_list.append(start_node)

        ### CANVAS ###
        img = np.zeros((200, 600, 3), dtype=np.uint8)
        ### CANVAS ###

        cv2.circle(img, (start_node.x, start_node.y), 5, (255, 0, 0), -1)
        cv2.circle(img, (goal_node.x, goal_node.y), 5, (0, 255, 0), -1)  

        for i in range(600):
            for j in range(200):
                if(get_inquality_obstacles(i,j,self.clearance)):
                    cv2.circle(img, (i, j), 1, (0, 255, 255), -1)

        c_best = float('inf')
        c_min = math.hypot(self.start_node.x - self.goal_node.y,
                           self.start_node.y - self.goal_node.y)
        
        x_center = np.array([[(self.start_node.x + self.goal_node.x) / 2.0],
                             [(self.start_node.y + self.goal_node.y) / 2.0], [0]])
        
        a1 = np.array([[(self.goal_node.x - self.start_node.x) / c_min],
                       [(self.goal_node.y - self.start_node.y) / c_min], [0]])


        e_theta = math.atan2(a1[1], a1[0])
        # first column of identity matrix transposed
        id1_t = np.array([1.0, 0.0, 0.0]).reshape(1, 3)
        m = a1 @ id1_t
        u, s, vh = np.linalg.svd(m, True, True)
        c = u @ np.diag(
            [1.0, 1.0,
             np.linalg.det(u) * np.linalg.det(np.transpose(vh))]) @ vh

        ##################
        #### RRT LOOP ####
        ##################
        for i in range(self.max_iter):
            
            rand_node = self.informed_sample(c_best, c_min, x_center, c)

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
                #time.sleep(2)
                cv2.imshow("RRT Tree", img)
                cv2.waitKey(10)

                near_nodes = self.near_nodes(new_node, self.expand_dis)
                
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

                self.node_list.append(new_node)
                
                self.rewire(new_node, img)
            
            else:

                new_node.parent = None
                continue


            #print(self.distance(self.goal_node, new_node))

            if self.distance(self.goal_node, new_node) <= self.goal_tolerance:
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + self.distance(new_node, goal_node)
                self.node_list.append(goal_node)
                print("FOUNDDDDDDDDDDDDDD")

                temp_path = self.extract_path(goal_node)
                temp_path_len = self.get_path_len(temp_path)

                if c_best != float('inf'):
                    print("plotting ellipse...................")
                    self.plot_ellipse(img, x_center, c_best, c_min, e_theta)

                if temp_path_len < c_best:
                    self.path = temp_path
                    c_best = temp_path_len
                    self.flag = True
                    print("path found, finding optimal")
                    print("path visualising")

                    self.colo += 50

                    for i in range(self.path.shape[0] - 1):
                        cv2.line(img, (int(self.path[i, 0]), int(self.path[i, 1])),
                        (int(self.path[i + 1, 0]), int(self.path[i + 1, 1])),
                        (0, 50 + self.colo, 0), 2)
                    
                    cv2.imshow("RRT Tree", img)
                    cv2.waitKey(10)


                #return


        if(self.flag):
            cv2.imshow("RRT Tree", img)
            cv2.waitKey(100)

        return self.path
    
    def rot_mat_2d(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])
    
    def plot_ellipse(self,canvas, x_center, c_best, c_min, e_theta):

        a = math.sqrt(c_best ** 2 - c_min ** 2) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - e_theta
        cx = int(x_center[0])
        cy = int(x_center[1])
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        fx = self.rot_mat_2d(-angle) @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        # assuming 'canvas' is your OpenCV image, use cv2.polylines to draw the ellipse
        pts = np.array(list(zip(px, py)), np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(canvas,[pts],True,(0,255,255))



        
    def check_collision(self, nearest_node, new_node):
        """
        Check for collision between two nodes: nearest_node and new_node
        """
        x1, y1 = nearest_node.x, nearest_node.y
        x2, y2 = new_node.x, new_node.y

        # Discretize the line segment into points
        #dx = abs(x2 - x1)
        #dy = abs(y2 - y1)

        dx = math.ceil(x2 - x1)
        dy = math.ceil(y2 - y1)
        
        steps = 10
        
        x_step = dx // steps
        y_step = dy // steps

        # Check for collision with each point on the line segment
        for i in range(int(steps) + 1):
            #x = int(x1 + i * x_step)
            #y = int(y1 + i * y_step)

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
                return np.array([])
            visited.add(node)
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])  # add last node
        path.reverse()
        return np.array(path)



def IRRT_main():

    start_x = int(50)
    start_y = int(100)

    goal_x = int(500)
    goal_y = int(100)

    clearance = int(5 + 10)

    start = np.array([start_x, 200 - start_y])
    goal = np.array([goal_x, 200 - goal_y])

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

    

print(IRRT_main())