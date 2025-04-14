import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
from collections import deque

# ---------- Configuration ----------
GRID_SIZE = 7
OBSTACLE_PROBABILITY = 0.1
START_NODE = (0, 0)
GOAL_NODE = (6, 6)
ALLOWED_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
np.random.seed(42)  # For consistent results during testing

# ---------- Grid Setup ----------
def create_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) not in [START_NODE, GOAL_NODE] and np.random.rand() < OBSTACLE_PROBABILITY:
                grid[i, j] = 1  # Mark as obstacle
    return grid

# ---------- Helper Functions ----------
def is_valid(grid, x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x, y] == 0

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path

# ---------- A* Search Algorithm ----------
def a_star(grid, start, goal):
    open_set = [(0, start)]
    came_from = {}
    g_cost = {start: 0}
    f_cost = {start: manhattan_distance(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for dx, dy in ALLOWED_MOVES:
            neighbor = (current[0] + dx, current[1] + dy)
            if not is_valid(grid, neighbor[0], neighbor[1]):
                continue

            tentative_g = g_cost[current] + 1

            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g
                f_cost[neighbor] = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_set, (f_cost[neighbor], neighbor))

    return None  # No path found

# ---------- Breadth-First Search Algorithm ----------
def bfs(grid, start, goal):
    queue = deque([start])
    came_from = {}
    visited = set()
    visited.add(start)

    while queue:
        current = queue.popleft()

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for dx, dy in ALLOWED_MOVES:
            neighbor = (current[0] + dx, current[1] + dy)
            if is_valid(grid, neighbor[0], neighbor[1]) and neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return None

# ---------- Visualization ----------
def draw_both_paths(grid, path_a_star=None, path_bfs=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    titles = ['A* Search Path', 'Breadth-First Search Path']
    paths = [path_a_star, path_bfs]

    for ax, title, path in zip(axes, titles, paths):
        ax.set_title(title)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if grid[x, y] == 1:
                    ax.scatter(y, GRID_SIZE - x - 1, c='red', s=200)  # Obstacles

        # Draw start and goal
        ax.scatter(START_NODE[1], GRID_SIZE - START_NODE[0] - 1, c='green', s=200, label='Start')
        ax.scatter(GOAL_NODE[1], GRID_SIZE - GOAL_NODE[0] - 1, c='blue', s=200, label='Goal')

        if path:
            for x, y in path:
                ax.scatter(y, GRID_SIZE - x - 1, c='yellow', s=100)

        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

# ---------- Main Function ----------
def main():
    grid = create_grid()
    print("Generated Grid (0: free, 1: obstacle):\n", grid)

    # A* Search
    start_time = time.time()
    path_a_star = a_star(grid, START_NODE, GOAL_NODE)
    time_a_star = time.time() - start_time

    # BFS Search
    start_time = time.time()
    path_bfs = bfs(grid, START_NODE, GOAL_NODE)
    time_bfs = time.time() - start_time

    print(f"\nA* Execution Time: {time_a_star:.4f} seconds")
    if path_a_star:
        print("A* Shortest Path:", path_a_star)
    else:
        print("A* → No valid path found.")

    print(f"\nBFS Execution Time: {time_bfs:.4f} seconds")
    if path_bfs:
        print("BFS Shortest Path:", path_bfs)
    else:
        print("BFS → No valid path found.")

    draw_both_paths(grid, path_a_star, path_bfs)

if __name__ == "__main__":
    main()
