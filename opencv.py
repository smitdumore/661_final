import numpy as np
import math
import cv2

tab_width = 600
tab_height = 200


def get_inquality_obstacles(x, y, clearance):
    if (x >= (tab_width - clearance)) or (y >= (tab_height - clearance)) or (x <= clearance) or (y <= clearance) or\
       ((y >= 75 - clearance) and (x <= (165 + clearance)) and (x >= (150 - clearance)) and (y <= tab_height)) or\
       ((y <= (125 + clearance)) and (x >= (250 - clearance)) and (y >= 0) and (x <= (265 + clearance))) or\
        (((x-400)**2 + (y-110)**2) < (60 + clearance)**2):
            return True
    
    return False

class Node:
    def __init__(self, x, y, parent=None, cost = np.inf):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost

class RRT:
    def __init__(self, start, goal, expand_dis= 50,
                 goal_sample_rate=10, max_iter=6000):

        self.start_node = start
        self.goal_node = goal
        self.expand_dis = expand_dis
        self.rewire_rad = 100
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = []
        self.node_list = None

    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + abs(node1.y - node2.y)**2)

    def random_node(self):
        rnd = Node(np.random.randint(0, 600), np.random.randint(0, 200))
        return rnd

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

        new_node.parent = from_node
        
        return new_node

    def near_nodes(self, node, rad):
        
        near_nodes = []

        # node_list is the tree itself
        for n in self.node_list:
            if self.distance(n, node) < rad:
                near_nodes.append(n)
        return near_nodes

    def rewire(self, node):
        
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

    def search(self):

        self.node_list = []
        
        start_node = self.start_node
        goal_node = self.goal_node
        
        self.node_list.append(start_node)

        img = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.circle(img, (start_node.x, start_node.y), 5, (255, 0, 0), -1)
        cv2.circle(img, (goal_node.x, goal_node.y), 5, (0, 255, 0), -1)  

        for i in range(600):
            for j in range(200):
                if(get_inquality_obstacles(i,j,15)):
                    cv2.circle(img, (i, j), 1, (0, 255, 255), -1)


        for i in range(self.max_iter):
            
            rand_node = self.random_node()

            if get_inquality_obstacles(rand_node.x, rand_node.y, 15):
                continue
            
            nearest_node = self.nearest(rand_node)
            
            new_node = self.steer(nearest_node, rand_node)
            
            if not self.check_collision(nearest_node, new_node):
                #print("showing rand node")

                cv2.circle(img, (new_node.x, new_node.y), 1, (0, 0, 255), -1)
                cv2.imshow("RRT Tree", img)
                cv2.waitKey(10)

                near_nodes = self.near_nodes(new_node, self.expand_dis)
                
                # find cheapest parent for new node
                node_with_min_cost = nearest_node
                min_cost = nearest_node.cost + self.distance(nearest_node,new_node)
                
                for near_node in near_nodes:
                    if near_node.cost + self.distance(near_node,new_node) < min_cost:
                        node_with_min_cost = near_node
                        min_cost = near_node.cost + self.distance(near_node, new_node)
                
                new_node.parent = node_with_min_cost
                new_node.cost = min_cost

                self.node_list.append(new_node)
                
                self.rewire(new_node)
            
            else:

                new_node.parent = None


            #print(self.distance(self.goal_node, new_node))

            if self.distance(self.goal_node, new_node) <= 5:
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + self.distance(new_node, goal_node)
                self.node_list.append(goal_node)
                print("FOUNDDDDDDDDDDDDDD")
                path = self.extract_path(goal_node)

                for i in range(path.shape[0] - 1):
                    cv2.line(img, (int(path[i, 0]), int(path[i, 1])),
                     (int(path[i + 1, 0]), int(path[i + 1, 1])),
                     (0, 255, 0), 2)
                    
                cv2.imshow("RRT Tree", img)
                cv2.waitKey(100000)

                return



        return None
        
    def check_collision(self, nearest_node, new_node):
        """
        Check for collision between two nodes: nearest_node and new_node
        """
        x1, y1 = nearest_node.x, nearest_node.y
        x2, y2 = new_node.x, new_node.y

        # Discretize the line segment into points
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(dx, dy)

        if(steps == 0):
            return True

        x_step = dx / steps
        y_step = dy / steps

        # Check for collision with each point on the line segment
        for i in range(int(steps) + 1):
            x = int(x1 + i * x_step)
            y = int(y1 + i * y_step)

            # Check if the point collides with any obstacles
            if get_inquality_obstacles(x, y, 15):
                return True

        return False  # No collision


    def extract_path(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.reverse()
        return np.array(path)


def main():

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
    obj = RRT(start_node, goal_node)

    path = obj.search()

    

main()