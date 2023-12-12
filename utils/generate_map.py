import os
import sys
import random
import argparse
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.distance import distance

def is_valid_move(map, visited, row, col):
    """Check if a move is valid: within the map boundaries, not an obstacle, and not visited."""
    return (0 <= row < len(map)) and (0 <= col < len(map[0])) and (map[row][col] == 0) and not visited[row][col]

def path_exists(map):
    """Uses BFS to find if there is a path from the top-left corner to the bottom-right corner in a given map."""
    if map[0][0] == 1 or map[-1][-1] == 1:
        # No path if the map is empty or start/end is blocked
        return False

    rows, cols = len(map), len(map[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    queue = deque([(0, 0)])
    visited[0][0] = True

    while queue:
        x, y = queue.popleft()

        # Check if reached bottom-right corner
        if x == rows - 1 and y == cols - 1:
            return True

        # Check all adjacent cells
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if is_valid_move(map, visited, new_x, new_y):
                queue.append((new_x, new_y))
                visited[new_x][new_y] = True

    return False


def generate_maps_with_distance(target_distance, map_size, init_num_walls=8, max_iterations=1000, file_name='maps.py'):
    '''Generates two maps with the specified distance between them.'''
    length, width = map_size
    map1 = np.zeros((length, width), dtype=int)
    all_locs = [(i, j) for i in range(length) for j in range(width)]
    all_locs.remove((0, 0))
    all_locs.remove((length-1, width-1))
    
    path_e = False
    while not path_e:
        for i, j in random.sample(all_locs, init_num_walls):
            map1[i][j] = 1
        path_e = path_exists(map1)
        if not path_e:
            map1 = np.zeros((length, width), dtype=int) # revert the changes
    
    map2 = map1.copy()
    maps = [map1, map2]
    sucess = False

    for i in range(max_iterations):
        current_distance = distance(map1, map2)
        if current_distance == target_distance:
            sucess = True
            break
        
        diff_map = np.bitwise_xor(map1, map2) # 1 where two maps differ
        path_e = False
        map_index = random.choice([0, 1])
        
        if current_distance > target_distance:
            locs = np.argwhere(diff_map == 1) # locations of dissimilarties between two maps
        else:
            locs = np.argwhere(diff_map == 0) # locations of similarities between two maps
        
        # Remove top-left and bottom-right corners from the options
        locs = [loc for loc in locs if not (tuple(loc) == (0, 0) or tuple(loc) == (length-1, width-1))]
      
        if locs:  # Check if locs is not empty after removing corners
            # TODO: it might stuck in the loop for ever
            while not path_e:
                i, j = random.choice(locs)
                maps[map_index][i][j] = 1 - maps[map_index][i][j]
                path_e = path_exists(maps[map_index])  
                if not path_e:
                    maps[map_index][i][j] = 1 - maps[map_index][i][j] # revert the change
        else:
            continue
        
    if sucess:    
        with open(file_name, 'w') as f:
            f.write(f'map1 = {map1.tolist()}\n')
            f.write(f'map2 = {map2.tolist()}\n')
    else:
        raise Exception(f'Failed to find maps with distance {target_distance} within {max_iterations} iterations')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate maps with distance")

    parser.add_argument("--target_distance", type=int, default=1, help="target distance between two maps")
    parser.add_argument("--map_size", type=int, nargs=2, default=[8, 8], help="size of the map")
    parser.add_argument("--max_iterations", type=int, default=1000, help="maximum number of iterations to find the maps")
    parser.add_argument("--file_name", type=str, default="maps.py", help="file name to store the maps")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--init_num_walls", type=int, default=8, help="number of walls in the initial map")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    generate_maps_with_distance(args.target_distance, map_size=args.map_size, init_num_walls=args.init_num_walls, max_iterations=args.max_iterations, file_name=args.file_name)
        