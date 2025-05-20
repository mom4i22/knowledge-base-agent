from heapq import heappush, heappop
from typing import Tuple, List, Dict, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt


class Predicate:
    def __init__(self, name: str, args: Tuple):
        self.name = name
        self.args = args

    def __repr__(self):
        return f'{self.name}{self.args}'


class KnowledgeBase:
    def __init__(self):
        self.facts: Dict[str, List[Predicate]] = {}

    def add(self, fact: Predicate):
        self.facts.setdefault(fact.name, []).append(fact)

    def fetch(self, name: str) -> List[Predicate]:
        return self.facts.get(name, [])


def unify(a, b, theta=None):
    if theta is None:
        theta = {}
    if a == b:
        return theta
    return None


def build_maze(obstacles: List[Tuple[int, int]], rows=5, cols=6) -> KnowledgeBase:
    kb = KnowledgeBase()
    for r in range(rows):
        for c in range(cols):
            kb.add(Predicate('Cell', (r, c)))
    for cell in obstacles:
        kb.add(Predicate('Obstacle', cell))
    kb.add(Predicate('Start', (0, 0)))
    kb.add(Predicate('Goal', (4, 5)))
    for r in range(rows):
        for c in range(cols):
            if (r, c) in obstacles:
                continue
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < rows and 0 <= c2 < cols and (r2, c2) not in obstacles:
                    kb.add(Predicate('Move', (r, c, r2, c2)))
    return kb


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_search_with_visual(
        kb: KnowledgeBase,
        start: Tuple[int, int] = (0, 0),
        goal: Tuple[int, int] = (4, 5),
        rows: int = 5,
        cols: int = 6,
        obstacles: List[Tuple[int, int]] = []
) -> Optional[List[Tuple[int, int]]]:
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.invert_yaxis()
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True)

    for (r, c) in obstacles:
        ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='red'))

    ax.add_patch(patches.Rectangle((start[1] - 0.5, start[0] - 0.5), 1, 1, facecolor='green', label='Start'))
    ax.add_patch(patches.Rectangle((goal[1] - 0.5, goal[0] - 0.5), 1, 1, facecolor='green', label='Goal'))

    plt.ion()
    plt.show()

    open_set = []
    heappush(open_set, (manhattan(start, goal), 0, start, [start]))
    closed = set()

    while open_set:
        f, g, current, path = heappop(open_set)
        if current in closed:
            continue

        ax.scatter(current[1], current[0], marker='.', color='black')
        plt.pause(0.1)

        if current == goal:
            for node in path:
                ax.scatter(node[1], node[0], marker='x')
                plt.pause(0.05)
            plt.ioff()
            plt.show()
            return path

        closed.add(current)

        for move in kb.fetch('Move'):
            if (move.args[0], move.args[1]) == current:
                neigh = (move.args[2], move.args[3])
                if neigh in closed:
                    continue
                g2 = g + 1
                f2 = g2 + manhattan(neigh, goal)
                heappush(open_set, (f2, g2, neigh, path + [neigh]))

    plt.ioff()
    plt.show()
    return None


if __name__ == '__main__':
    obstacles = [(0, 1), (2, 1), (3, 1), (2, 3), (3, 4), (4, 4)]
    maze = build_maze(obstacles)
    path = astar_search_with_visual(
        maze,
        start=(0, 0),
        goal=(4, 5),
        rows=5,
        cols=6,
        obstacles=obstacles
    )
    if path:
        print('Found path:', path)
    else:
        print('No path found.')
