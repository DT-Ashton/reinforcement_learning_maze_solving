from collections import deque


def bfs_shortest_path(env) -> list | None:
    """BFS from (0,0) to (size-1, size-1) on env._base_grid.
    Returns path as list of (r,c) cells (including start and goal), or None if unsolvable.
    Path length in steps = len(path) - 1.
    """
    size = env.size
    grid = env._base_grid
    start = (0, 0)
    goal = (size - 1, size - 1)

    parent = {start: None}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            path = []
            cur = (r, c)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < size and 0 <= nc < size
                    and grid[2 * r + 1 + dr, 2 * c + 1 + dc] == 1
                    and (nr, nc) not in parent):
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    return None
