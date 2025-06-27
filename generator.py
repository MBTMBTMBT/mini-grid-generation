import random
from collections import deque
from typing import Dict, Tuple

import numpy as np


def generate_single_env(
    height: int,
    width: int,
    mode: str,
    wall_prob: float = 0.3,
    key_prob: float = 0.15,
    door_prob: float = 0.75,
) -> np.ndarray:
    """Dispatch to either the random generator or the path‑and‑branch generator."""
    if mode == "random":
        return random_generate(height, width, wall_prob, key_prob, door_prob)

    path_len = int((height - 2) * (width - 2) * (1 - wall_prob))
    branch_num = random.randint(height // 2, width)
    branch_len = random.randint(height // 3, height - 2)
    return generate_path_branch(
        height,
        width,
        path_len,
        branch_num=branch_num,
        branch_length=branch_len,
        key_prob=key_prob,
        door_prob=door_prob,
    )


def random_generate(
    height: int,
    width: int,
    wall_ratio: float = 0.1,
    key_ratio: float = 0.05,
    door_ratio: float = 0.05,
) -> np.ndarray:
    """Fill the interior with floor, then randomly sprinkle walls/keys/doors."""
    # Tile codes: 1 = floor, 2 = wall, 4 = door, 5 = key, 8 = goal
    grid = 2 * np.ones((height, width), np.uint8)
    grid[1:-1, 1:-1] = 1

    # Place goal
    gy, gx = random.randint(1, height - 2), random.randint(1, width - 2)
    grid[gy, gx] = 8

    empty = np.argwhere(grid == 1)
    np.random.shuffle(empty)

    num_walls = int(len(empty) * wall_ratio)
    num_keys = int(num_walls * key_ratio)
    num_doors = int(num_walls * door_ratio)

    # Keys
    for y, x in empty[:num_keys]:
        grid[y, x] = 5
    # Doors
    for y, x in empty[num_keys : num_keys + num_doors]:
        grid[y, x] = 4
    # Extra walls
    for y, x in empty[num_keys + num_doors : num_keys + num_doors + num_walls]:
        grid[y, x] = 2

    return grid


def generate_path_branch(
    height: int,
    width: int,
    path_length: int = 50,
    branch_num: int = 5,
    branch_length: int = 7,
    key_prob: float = 0.0,
    door_prob: float = 0.0,
) -> np.ndarray:
    """Grow a random self‑avoiding walk with side branches, then sprinkle keys/doors."""
    grid = 2 * np.ones((height, width), int)

    def inside(y: int, x: int) -> bool:
        return 1 <= y < height - 1 and 1 <= x < width - 1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    y, x = random.randint(1, height - 2), random.randint(1, width - 2)
    goal = (y, x)
    grid[y, x] = 1
    path = [(y, x)]

    for _ in range(path_length):
        random.shuffle(directions)
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if inside(ny, nx) and grid[ny, nx] == 2:
                y, x = ny, nx
                grid[y, x] = 1
                path.append((y, x))
                break

 
    for _ in range(branch_num):
        if not path:
            break
        sy, sx = random.choice(path)
        for _ in range(branch_length):
            random.shuffle(directions)
            for dy, dx in directions:
                ny, nx = sy + dy, sx + dx
                if inside(ny, nx) and grid[ny, nx] == 2:
                    grid[ny, nx] = 1
                    sy, sx = ny, nx
                    path.append((sy, sx))
                    break

    path_no_goal = [p for p in path if p != goal]
    random.shuffle(path_no_goal)
    nk = int(len(path_no_goal) * key_prob)
    nd = int(len(path_no_goal) * door_prob)

    for y, x in path_no_goal[:nk]:
        grid[y, x] = 5  # key
    for y, x in path_no_goal[nk : nk + nd]:
        grid[y, x] = 4  # door

    grid[goal] = 8  # goal tile
    return grid


def add_start_next_to_key(
    grid: np.ndarray,
    start_value: int = 3,
    key_value: int = 5,
    floor_value: int = 1,
) -> Tuple[np.ndarray, bool]:
    """Place a start tile adjacent to any key.

    Returns the modified grid and a flag indicating success.
    """
    height, width = grid.shape
    keys = [tuple(p) for p in np.argwhere(grid == key_value)]
    random.shuffle(keys)

    for ky, kx in keys:
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            sy, sx = ky + dy, kx + dx
            if 0 <= sy < height and 0 <= sx < width and grid[sy, sx] == floor_value:
                new_grid = grid.copy()
                new_grid[sy, sx] = start_value
                return new_grid, True
    return grid, False



def compute_descriptors(grid: np.ndarray) -> np.ndarray:
    """Return [empty‑floor ratio, normalised longest path length]."""
    empty_ratio = np.mean(grid == 1)
    norm_path = estimate_path_length(grid) / (grid.shape[0] + grid.shape[1])
    return np.array([empty_ratio, norm_path])


def estimate_path_length(grid: np.ndarray) -> int:
    """Breadth‑first search for the longest geodesic distance over floor tiles."""
    h, w = grid.shape
    floor_pts = np.argwhere(grid == 1)
    if len(floor_pts) < 2:
        return 0

    max_dist = 0
    for idx in range(len(floor_pts)):
        visited = np.zeros_like(grid, bool)
        q = deque([(tuple(floor_pts[idx]), 0)])
        visited[tuple(floor_pts[idx])] = True
        while q:
            (y, x), d = q.popleft()
            max_dist = max(max_dist, d)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 1 and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append(((ny, nx), d + 1))
    return max_dist



def _bfs(grid: np.ndarray, starts: np.ndarray, valid) -> set:
    h, w = grid.shape
    q = deque([tuple(starts[0])])
    reached: set = set()
    while q:
        y, x = q.popleft()
        if (y, x) in reached:
            continue
        reached.add((y, x))
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and valid(grid[ny, nx]):
                q.append((ny, nx))
    return reached


def is_reachable(grid: np.ndarray, require_key_door: bool = False) -> bool:
    """Return True if all non‑wall tiles are mutually reachable (and key‑door path is valid if requested)."""
    h, w = grid.shape
    non_walls = np.argwhere(grid != 2)
    if len(non_walls) == 0 or not np.any(grid == 8):
        return False

    if not require_key_door:
        reached = _bfs(grid, non_walls, lambda v: v != 2)
        return len(reached) == len(non_walls)

    # --- key‑door logic -------------------------------------------------
    KEY, DOOR, WALL = 5, 4, 2

    keys = np.argwhere(grid == KEY)
    if len(keys) == 0:
        return False

    reachable_no_door = _bfs(grid, non_walls, lambda v: v != WALL and v != DOOR)
    if not any(tuple(k) in reachable_no_door for k in keys):
        return False

    reachable_after_key = _bfs(grid, keys, lambda v: v != WALL)
    return len(reachable_after_key) == len(non_walls)




def generate_envs_dataset(
    rows: int,
    cols: int,
    num_maps: int,
    wall_p_range: Tuple[float, float] = (0.2, 0.5),
    door_p_range: Tuple[float, float] = (0.075, 0.15),
    key_p_range: Tuple[float, float] = (0.1, 0.3),
    max_tries: int = int(1e6),
    random_gen_max: int = int(2e4),
    start_point_flag: bool = False,
) -> Dict[int, np.ndarray]:
    """Generate *num_maps* mini‑grid environments satisfying reachability constraints.

    The generator keeps sampling random parameters and maps until the archive
    reaches *num_maps* unique entries or *max_tries* attempts are exhausted.
    """
    archive: Dict[int, np.ndarray] = {}
    attempts = 0
    key_door_required = key_p_range[1] > 0 or door_p_range[1] > 0

    while attempts < max_tries and len(archive) < num_maps:
        wall_p = random.uniform(*wall_p_range)
        door_p = random.uniform(*door_p_range)
        key_p = random.uniform(door_p, key_p_range[1])

        # Try both fast random generator and fallback path‑branch generator
        for _ in range(int(random_gen_max)):
            attempts += 1
            grid = generate_single_env(rows, cols, "random", wall_p, key_p, door_p)
            if is_reachable(grid, key_door_required):
                break
        else:
            # Fallback – structured generator
            grid = generate_single_env(rows, cols, "path", wall_p, key_p, door_p)
            attempts += 1

        # Optionally place a start tile next to a key
        if start_point_flag:
            grid, ok = add_start_next_to_key(grid, start_value=3, key_value=5)
            if not ok:
                continue

        archive[len(archive)] = grid

    return archive


if __name__ == "__main__":
    envs = generate_envs_dataset(15, 15, 3, start_point_flag=True)
    for idx, g in envs.items():
        print(f"\nMap {idx}\n", g)
