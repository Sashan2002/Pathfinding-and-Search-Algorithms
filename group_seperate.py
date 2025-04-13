# Import required libraries
import numpy as np  # For numerical operations and random number generation
import matplotlib.pyplot as plt  # For visualizing the grid and paths
import heapq  # For maintaining the priority queue in A* algorithm
import time  # For calculating execution time
from collections import deque  # For implementing the queue in BFS

# ---------- Configuration ----------
GRID_SIZE = 7  # Size of the grid (7x7)
OBSTACLE_PROBABILITY = 0.1  # Probability of placing an obstacle at any cell
START_NODE = (0, 0) # Starting node of the path
GOAL_NODE = (6, 6)  # Goal node of the path
ALLOWED_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right moves allowed
np.random.seed(42)  # Seed for random number generator to ensure reproducibility

# ---------- Grid Setup ----------
def create_grid():
    """
    Creates a 2D grid with randomly placed obstacles (represented as 1s) 
    except at the start and goal positions.
    """
    # Initialize grid with all 0s
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Add obstacle based on probability if it's not the start or goal node
            if (i, j) not in [START_NODE, GOAL_NODE] and np.random.rand() < OBSTACLE_PROBABILITY:
                grid[i, j] = 1
    return grid

# ---------- Helper Functions ----------
def is_valid(grid, x, y):
    """
    Checks if the given position (x, y) is inside the grid boundaries and not an obstacle.
    """
    
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x, y] == 0

def manhattan_distance(a, b):
    """
    Computes the Manhattan distance between two points a and b.
    """
    
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, start, goal):
    """
    Reconstructs the path from the start node to the goal node using the came_from dictionary.
    """
    
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])  # Trace path back from goal to start
    path.reverse()  # Reverse to get path from start to goal
    return path

# ---------- A* Algorithm ----------
def a_star(grid, start, goal):
    """
    Implements the A* pathfinding algorithm.
    """
    
    open_set = [(0, start)]  # Priority queue for exploring nodes, sorted by f_cost
    came_from = {}  # To store the path
    g_cost = {start: 0}  # Cost from start to a given node
    f_cost = {start: manhattan_distance(start, goal)}   # Estimated cost from start to goal via current node

    while open_set:
        _, current = heapq.heappop(open_set)  # Pop node with lowest f_cost

        if current == goal:
            return reconstruct_path(came_from, start, goal)  # Goal reached

        for dx, dy in ALLOWED_MOVES:
            neighbor = (current[0] + dx, current[1] + dy)
            if not is_valid(grid, neighbor[0], neighbor[1]):
                continue
            
            # Calculate the tentative g_cost
            tentative_g = g_cost[current] + 1

            # If this path to neighbor is better than any previous one
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g
                f_cost[neighbor] = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_set, (f_cost[neighbor], neighbor))  # Add neighbor to the priority queue

    return None  # Return None if no path found

# ---------- BFS Algorithm ----------
def bfs(grid, start, goal):
    """
    Implements the Breadth-First Search (BFS) pathfinding algorithm.
    """
    
    queue = deque([start])  # Initialize queue with the start node
    came_from = {}  # To store the path
    visited = set([start])  # Set to keep track of visited nodes

    while queue:
        current = queue.popleft()  # Remove node from front of the queue

        if current == goal:
            return reconstruct_path(came_from, start, goal)  # Goal reached

        for dx, dy in ALLOWED_MOVES:
            neighbor = (current[0] + dx, current[1] + dy)
            if is_valid(grid, neighbor[0], neighbor[1]) and neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)  # Add neighbor to the queue for further exploration

    return None  # Return None if no path found


# ---------- Visualization ----------
def draw_grid(grid, path=None, title="Pathfinding"):
    """
    Visualizes the grid and the path using matplotlib.
    """
    
    plt.figure(figsize=(7, 7))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] == 1:
                plt.scatter(y, GRID_SIZE - x - 1, c='red', s=200)  # Obstacle


    # Plot the start and goal positions
    plt.scatter(START_NODE[1], GRID_SIZE - START_NODE[0] - 1, c='green', s=200, label='Start')
    plt.scatter(GOAL_NODE[1], GRID_SIZE - GOAL_NODE[0] - 1, c='blue', s=200, label='Goal')

    # Plot the path if one exists
    if path:
        for x, y in path:
            plt.scatter(y, GRID_SIZE - x - 1, c='yellow', s=100)

    plt.title(title)
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))
    plt.grid(True)
    plt.legend()
    plt.show()

# ---------- Main Function ----------
def main():
    """
    Main function to run the pathfinding simulation using A* and BFS.
    """

    # Generate random grid with obstacles
    grid = create_grid()
    print("Generated Grid (0: free, 1: obstacle):\n", grid)

    # ----- Run A* Search-----
    print("\n Running A* Search...")
    start_time = time.time()  # Start timing A*
    path_a_star = a_star(grid, START_NODE, GOAL_NODE)
    time_a_star = time.time() - start_time  # End timing A*

    print(f"A* Execution Time: {time_a_star:.4f} seconds")
    if path_a_star:
        print("A* Path Found:")
        print(path_a_star)
        draw_grid(grid, path_a_star, "A* Pathfinding")
    else:
        print("A* → No valid path found.")
        draw_grid(grid, title="A* - No Path Found")

    # ----- Run BFS Search-----
    print("\n Running BFS...")
    start_time = time.time()   # Start timing BFS
    path_bfs = bfs(grid, START_NODE, GOAL_NODE)
    time_bfs = time.time() - start_time  # End timing BFS

    print(f"BFS Execution Time: {time_bfs:.4f} seconds")
    if path_bfs:
        print("BFS Path Found:")
        print(path_bfs)
        draw_grid(grid, path_bfs, "BFS Pathfinding")
    else:
        print("BFS → No valid path found.")
        draw_grid(grid, title="BFS - No Path Found")


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
