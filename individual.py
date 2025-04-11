# Import necessary libraries
import numpy as np  # For grid creation and handling matrices
import matplotlib.pyplot as plt  # For visualizing the grid and paths
import heapq  # For implementing priority queues in A* and Dijkstra
import time  # For measuring execution time

# Define the grid size and key points
GRID_SIZE = 7
START = (0, 6)  # Starting position on the grid
GOAL = (6, 0)  # Goal position on the grid
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right directions

# Function to generate the grid with random obstacles
def generate_grid(size, obstacle_chance=0.2):
    """
    Generate a square grid of given size filled with 0s (walkable) and 1s (obstacles).
    Obstacles are placed randomly with a certain probability.
    
    Parameters:
        size (int): Size of the square grid.
        obstacle_chance (float): Probability of placing an obstacle in a cell.
    
    Returns:
        np.ndarray: The generated grid.
    """

    grid = np.zeros((size, size), dtype=int)  # Create a grid filled with zeros (empty cells)
    for x in range(size):
        for y in range(size):
            # Randomly place obstacles (1s), but not at START or GOAL
            if (x, y) != START and (x, y) != GOAL and np.random.rand() < obstacle_chance:
                grid[x][y] = 1
    return grid


# Function to check whether a cell is valid (within grid and not an obstacle)
def is_valid(grid, x, y):
    """
    Check if a given position is within grid bounds and not an obstacle.
    
    Parameters:
        grid (np.ndarray): The grid map.
        x (int): Row index.
        y (int): Column index.
    
    Returns:
        bool: True if the position is valid and not blocked.
    """
    
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x][y] == 0

# Heuristic function for A* (Manhattan distance)
def heuristic(a, b):
    """
    Compute the Manhattan distance heuristic between two points.
    
    Parameters:
        a (tuple): First point (x, y).
        b (tuple): Second point (x, y).
    
    Returns:
        int: Manhattan distance.
    """

    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Function to reconstruct the path from the goal to the start
def reconstruct_path(came_from, current):
    """
    Reconstruct the path from start to goal using the 'came_from' map.
    
    Parameters:
        came_from (dict): Dictionary containing the path traversal history.
        current (tuple): Goal node.
    
    Returns:
        list: List of tuples representing the path from start to goal.
    """
    
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()  # Reverse to get path from start to goal
    return path

# A* pathfinding algorithm
def a_star(grid, start, goal):
    """
    Perform A* search to find the shortest path from start to goal.
    
    Parameters:
        grid (np.ndarray): The grid map.
        start (tuple): Start position.
        goal (tuple): Goal position.
    
    Returns:
        list or None: The path if found, otherwise None.
    """
    
    open_set = []
    heapq.heappush(open_set, (0, start))  # Push the start node with priority 0
    came_from = {}  # Tracks the path
    g_score = {start: 0}  # Cost from start to current node
    f_score = {start: heuristic(start, goal)}  # Estimated total cost (g + h)

    while open_set:
        _, current = heapq.heappop(open_set)  # Get node with lowest f_score

        if current == goal:
            return reconstruct_path(came_from, current)  # Path found

        # Check all neighbors
        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)
            if not is_valid(grid, neighbor[0], neighbor[1]):
                continue  # Skip invalid neighbors

            temp_g = g_score[current] + 1
            if neighbor not in g_score or temp_g < g_score[neighbor]:
                # Better path found to neighbor
                came_from[neighbor] = current
                g_score[neighbor] = temp_g
                f_score[neighbor] = temp_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None  # No path found

# Breadth-First Search (BFS) algorithm
def bfs(grid, start, goal):
    """
    Perform Breadth-First Search to find the shortest path from start to goal.
    
    Parameters:
        grid (np.ndarray): The grid map.
        start (tuple): Start position.
        goal (tuple): Goal position.
    
    Returns:
        list or None: The path if found, otherwise None.
    """
    
    from collections import deque
    queue = deque()
    queue.append((start, [start]))  # Add start node and path to queue
    visited = set()
    visited.add(start)

    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path  # Return the complete path


        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)
            if is_valid(grid, neighbor[0], neighbor[1]) and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))  # Add neighbor with updated path
    return None  # No path found

# Dijkstra's algorithm for shortest path
def dijkstra(grid, start, goal):
    """
    Perform Dijkstra's algorithm to find the shortest path from start to goal.
    
    Parameters:
        grid (np.ndarray): The grid map.
        start (tuple): Start position.
        goal (tuple): Goal position.
    
    Returns:
        list or None: The path if found, otherwise None.
    """
    
    pq = [(0, start)]  # Priority queue with starting point and cost 0
    came_from = {}  # Track path history
    cost = {start: 0}  # Distance from start to node


    while pq:
        curr_cost, current = heapq.heappop(pq)

        if current == goal:
            return reconstruct_path(came_from, current)  # Return shortest path

        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)
            if is_valid(grid, neighbor[0], neighbor[1]):
                new_cost = cost[current] + 1
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(pq, (new_cost, neighbor))  # Add with updated cost
    return None  # No path found

# Function to visualize the grid and the path
def visualize(grid, path, title):
    """
    Visualize the grid with obstacles, start, goal, and the path found.
    
    Parameters:
        grid (np.ndarray): The grid map.
        path (list): The path to visualize.
        title (str): Title of the plot.
    """
    
    plt.figure(figsize=(6, 6))

    # Draw obstacles
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x][y] == 1:
                plt.scatter(y, GRID_SIZE - 1 - x, c='black', s=200)  # Obstacle

    # Draw path if exists
    if path:
        for x, y in path:
            plt.scatter(y, GRID_SIZE - 1 - x, c='yellow', s=100)

    # Mark the start and goal
    plt.scatter(START[1], GRID_SIZE - 1 - START[0], c='blue', s=200, label='Start')
    plt.scatter(GOAL[1], GRID_SIZE - 1 - GOAL[0], c='green', s=200, label='Goal')
    plt.grid(True)
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))
    plt.legend()
    plt.title(title)
    plt.show()

# Main function to execute all algorithms and display results
def main():
    """
    Main function to generate a grid and execute A*, BFS, and Dijkstra's algorithms.
    Also measures time and visualizes the path found by each algorithm.
    """
    
    # Generate random grid
    grid = generate_grid(GRID_SIZE)
    print("Grid:\n", grid)

    # Run A* Algorithm
    start = time.time()
    path_astar = a_star(grid, START, GOAL)
    end = time.time()
    print("A* Time: {:.4f}s".format(end - start))
    if path_astar:
        print("A* Path:", path_astar)
        
    else:
        print("A* No path found.")
    visualize(grid, path_astar, "A* Search")

    # Run BFS Algorithm
    start = time.time()
    path_bfs = bfs(grid, START, GOAL)
    end = time.time()
    print("BFS Time: {:.4f}s".format(end - start))
    if path_bfs:
        print("BFS Path:", path_bfs)
        
    else:
        print("BFS No path found.")
    visualize(grid, path_bfs, "Breadth-First Search")

    # Run Dijkstra's Algorithm
    start = time.time()
    path_dijkstra = dijkstra(grid, START, GOAL)
    end = time.time()
    print("Dijkstra Time: {:.4f}s".format(end - start))
    if path_dijkstra:
        print("Dijkstra Path:", path_dijkstra)
        
    else:
        print("Dijkstra No path found.")
    visualize(grid, path_dijkstra, "Dijkstra's Algorithm")
    
# Entry point of the script
if __name__ == "__main__":
    main()
