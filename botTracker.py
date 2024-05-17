import cv2
import cv2.aruco as aruco
import numpy as np

ROWS = 10 #Maze dimensions
COLS = 10

# Set all to 1111 (all walls) to avoid errors
def create_maze(rows, cols):
    return [['1111' for _ in range(cols)] for _ in range(rows)]

maze = [[0 for _ in range(COLS)] for _ in range(ROWS)]
maze = create_maze(ROWS, COLS)

#Row 0
maze[0][0] = '1001'
maze[0][1] = '1010'
maze[0][2] = '1000'
maze[0][3] = '1100'
maze[0][4] = '0001'
maze[0][5] = '1010'
maze[0][6] = '1000'
maze[0][7] = '1000'
maze[0][8] = '1000'
maze[0][9] = '1100'

#Row 1
maze[1][0] = '0011'
maze[1][1] = '1100'
maze[1][2] = '0111'
maze[1][3] = '0011'
maze[1][4] = '0110'
maze[1][5] = '1011'
maze[1][6] = '0110'
maze[1][7] = '0001'
maze[1][8] = '0110'
maze[1][9] = '0101'

#Row 2
maze[2][0] = '1001'
maze[2][1] = '0010'
maze[2][2] = '1110'
maze[2][3] = '1101'
maze[2][4] = '1011'
maze[2][5] = '1100'
maze[2][6] = '1011'
maze[2][7] = '0110'
maze[2][8] = '1001'
maze[2][9] = '0100'

#Row 3
maze[3][0] = '0001'
maze[3][1] = '1000'
maze[3][2] = '1110'
maze[3][3] = '0011'
maze[3][4] = '1100'
maze[3][5] = '0101'
maze[3][6] = '1001'
maze[3][7] = '1110'
maze[3][8] = '0101'
maze[3][9] = '0111'

#Row 4
maze[4][0] = '0101'
maze[4][1] = '0011'
maze[4][2] = '1010'
maze[4][3] = '1100'
maze[4][4] = '0101'
maze[4][5] = '0011'
maze[4][6] = '0010'
maze[4][7] = '1100'
maze[4][8] = '0011'
maze[4][9] = '1100'

#Row 5
maze[5][0] = '0101'
maze[5][1] = '1011'
maze[5][2] = '1010'
maze[5][3] = '0110'
maze[5][4] = '0101'
maze[5][5] = '1001'
maze[5][6] = '1010'
maze[5][7] = '0100'
maze[5][8] = '1101'
maze[5][9] = '0101'

#Row 6
maze[6][0] = '0011'
maze[6][1] = '1010'
maze[6][2] = '1100'
maze[6][3] = '1001'
maze[6][4] = '0110'
maze[6][5] = '0101'
maze[6][6] = '1101'
maze[6][7] = '0011'
maze[6][8] = '0100'
maze[6][9] = '0101'

#Row 7
maze[7][0] = '1001'
maze[7][1] = '1000'
maze[7][2] = '0010'
maze[7][3] = '0110'
maze[7][4] = '1001'
maze[7][5] = '0110'
maze[7][6] = '0001'
maze[7][7] = '1100'
maze[7][8] = '0101'
maze[7][9] = '0101'

#Row 8
maze[8][0] = '0111'
maze[8][1] = '0001'
maze[8][2] = '1100'
maze[8][3] = '1101'
maze[8][4] = '0101'
maze[8][5] = '1011'
maze[8][6] = '0100'
maze[8][7] = '0001'
maze[8][8] = '0110'
maze[8][9] = '0101'

#Row 9
maze[9][0] = '1011'
maze[9][1] = '0110'
maze[9][2] = '0011'
maze[9][3] = '0110'
maze[9][4] = '0011'
maze[9][5] = '1100'
maze[9][6] = '0111'
maze[9][7] = '0011'
maze[9][8] = '1010'
maze[9][9] = '0110'


# Test maze has been made correctly
def print_maze(maze):
    for row in maze:
        print(' '.join(row))

def draw_maze(frame, maze, cell_size=50): #draw lines for 1s
    maze_height = len(maze)
    maze_width = len(maze[0])

    for i in range(maze_height):
        for j in range(maze_width):
            cell = maze[i][j]
            x = j * cell_size
            y = i * cell_size

            if cell[0] == '1': #Top wall
                cv2.line(frame, (x, y), (x + cell_size, y), (0, 0, 0), 2)
            if cell[1] == '1': #Right wall
                cv2.line(frame, (x + cell_size, y), (x + cell_size, y + cell_size), (0, 0, 0), 2)
            if cell[2] == '1': #Bottom wall
                cv2.line(frame, (x, y + cell_size), (x + cell_size, y + cell_size), (0, 0, 0), 2)
            if cell[3] == '1': #Left wall
                cv2.line(frame, (x, y), (x, y + cell_size), (0, 0, 0), 2)
    
def botTracker():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250) #For generating markers
    parameters = cv2.aruco.DetectorParameters() #Parameters for detection

    # Camera matrix and distortion coefficients
    camera_matrix = np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read() #Each frame

        if not ret:
            print("Error: Failed to capture image.")
            break

        draw_maze(frame, maze) #overlay maze onto feed

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale for reading markers

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters) #Detect markers

        if ids is not None and len(ids) > 0: #Marker detected
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs) #Find position of marker
        
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1) # Draw axis for each marker
                cv2.aruco.drawDetectedMarkers(frame, corners) #Outline
            
                position = f"Position: x={tvec[0][0]:.2f}, y={tvec[0][1]:.2f}, z={tvec[0][2]:.2f}" #Display info
                orientation = f"Orientation: rvec={rvec[0][0]:.2f}, {rvec[0][1]:.2f}, {rvec[0][2]:.2f}"
                cv2.putText(frame, position, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, orientation, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No markers detected", (1100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Camera Feed', frame) #Show frame

        if cv2.waitKey(1) & 0xFF == ord('q'): #q to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print_maze(maze)
    botTracker()