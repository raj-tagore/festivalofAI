import cv2
import numpy as np

ROWS = 10 
COLS = 10

decimal_maze = [
    [9, 8, 8, 12, 11, 12, 9, 8, 8, 14],
    [5, 1, 0, 4, 9, 4, 5, 3, 2, 12],
    [5, 5, 3, 4, 1, 4, 1, 14, 9, 4],
    [5, 1, 12, 1, 0, 0, 4, 15, 1, 6],
    [5, 3, 0, 2, 2, 4, 3, 10, 0, 12],
    [1, 12, 3, 8, 12, 1, 8, 8, 2, 4],
    [5, 5, 9, 4, 5, 3, 0, 0, 8, 6],
    [1, 6, 1, 6, 3, 12, 3, 2, 4, 15],
    [5, 9, 4, 9, 8, 0, 8, 12, 3, 4],
    [3, 2, 2, 2, 2, 6, 3, 2, 10, 6]
]

binary_maze = [[bin(num)[2:].zfill(4) for num in row] for row in decimal_maze]

class Node:
    def __init__(self, maze_coordinates):
        self.pixel_coordinates = None
        self.maze_coordinates = maze_coordinates
        self.parent = None
        self.children = []
        self.visited = False
        
    def heuristic(self, goal):
        return abs(self.maze_coordinates[0] - goal[0]) + abs(self.maze_coordinates[1] - goal[1])
    
maze_nodes = [[Node((i, j)) for j in range(COLS)] for i in range(ROWS)]
start_node = maze_nodes[0][4]
goal_node = maze_nodes[9][5]
ai_path = []

cell_size = 0
start_coordinates = (0, 0)

def print_maze(maze):
    for row in maze:
        print(' '.join(row))

# draws the maze in the frame and configures the children of each node
def process_maze(frame, binary_maze, maze_nodes): 
    
    global cell_size, start_coordinates, ROWS, COLS
    
    cell_size = int(frame.shape[0]/13)
    start_coordinates = (int(frame.shape[1]/2 - COLS * cell_size/2), int(frame.shape[0]/2 - ROWS * cell_size/2))

    for i in range(ROWS):
        for j in range(COLS):
            
            cell = binary_maze[i][j]
            node = maze_nodes[i][j]
            x = start_coordinates[0] + j * cell_size
            y = start_coordinates[1] + i * cell_size
            node.pixel_coordinates = (int(x + cell_size/2), int(y + cell_size/2))

            if cell[0] == '1': #Top wall
                cv2.line(frame, (x, y), (x + cell_size, y), (0, 255, 255), 2)
            else:
                node.children.append(maze_nodes[i - 1][j])
            if cell[1] == '1': #Right wall
                cv2.line(frame, (x + cell_size, y), (x + cell_size, y + cell_size), (0, 255, 255), 2)
            else:
                node.children.append(maze_nodes[i][j + 1])
            if cell[2] == '1': #Bottom wall
                cv2.line(frame, (x, y + cell_size), (x + cell_size, y + cell_size), (0, 255, 255), 2)
            else:
                node.children.append(maze_nodes[i + 1][j])
            if cell[3] == '1': #Left wall
                cv2.line(frame, (x, y), (x, y + cell_size), (0, 255, 255), 2)
            else:
                node.children.append(maze_nodes[i][j - 1])
    return frame

# detects the aruco markers and sets the start and goal nodes
def detect_aruco_markers(frame):
    global maze_nodes, start_node, goal_node, ROWS, COLS
    maze_nodes = [[Node((i, j)) for j in range(COLS)] for i in range(ROWS)]
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    
    if ids is not None:
        for _ in range(len(ids)):
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
    
        for marker_id, corner in zip(ids, corners):
            top_left, top_right, bottom_right, bottom_left = corner[0][0], corner[0][1], corner[0][2], corner[0][3]
            center = (int((top_left[0] + bottom_right[0])/2), int((top_left[1] + bottom_right[1])/2))
            angle = np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0]) * 180/np.pi
            global start_coordinates, cell_size
            u = int((center[0] - start_coordinates[0])//(cell_size))
            v = int((center[1] - start_coordinates[1])//(cell_size))
            u=0 if u<0 else u
            v=0 if v<0 else v
            u=9 if u>9 else u
            v=9 if v>9 else v
        
            if marker_id[0] == 0:
                start_node = maze_nodes[v][u]
                
            if marker_id[0] == 1:
                goal_node = maze_nodes[v][u]
            
    return frame

# finds the path from the start node to the goal node
def update_ai_path(start, goal):
    open_list = [start]
    start.visited = True
    while open_list:
        current_node = open_list.pop(0)
        if current_node == goal:
            path = []
            while current_node:
                path.append(current_node)
                current_node = current_node.parent
            global ai_path
            ai_path = path[::-1]
        if current_node is not None:
            for child in current_node.children:
                if not child.visited:
                    child.parent = current_node
                    child.visited = True
                    open_list.append(child)

# draws the path in the frame
def draw_path(frame):
    global ai_path
    for i in range(len(ai_path) - 1):
        cv2.line(frame, ai_path[i].pixel_coordinates, ai_path[i + 1].pixel_coordinates, (0, 255, 0), 2)
    return frame
    

if __name__ == "__main__":
    
    cap = cv2.VideoCapture(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_aruco_markers(frame)
        frame = process_maze(frame, binary_maze, maze_nodes)
        update_ai_path(start_node, goal_node)
        frame = draw_path(frame)
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) == ord('q'):
            breakpoint
            
    cap.release()
    cv2.destroyAllWindows()