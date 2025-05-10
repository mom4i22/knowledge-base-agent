from heapq import heappush, heappop
from typing import Tuple, List, Dict, Optional

# --- 2.1. Basic KB structures and unification -----------------

class Predicate:
    def __init__(self, name: str, args: Tuple):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"{self.name}{self.args}"

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
    # For this maze problem we only need ground unification
    return None

def build_maze(obstacles: List[Tuple[int,int]],
                  rows=5, cols=6) -> KnowledgeBase:
    kb = KnowledgeBase()
    for r in range(rows):
        for c in range(cols):
            kb.add(Predicate("Cell", (r,c)))
    for cell in obstacles:
        kb.add(Predicate("Obstacle", cell))
    kb.add(Predicate("Start", (0,0)))
    kb.add(Predicate("Goal", (4,5)))

    # Adjacency and Move facts
    for r in range(rows):
        for c in range(cols):
            if (r,c) in obstacles:
                continue
            for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                r2, c2 = r+dr, c+dc
                if 0 <= r2 < rows and 0 <= c2 < cols and (r2,c2) not in obstacles:
                    kb.add(Predicate("Move", (r,c,r2,c2)))
    return kb

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar_search(kb: KnowledgeBase,
                 start=(0,0), goal=(4,5)
                ) -> Optional[List[Tuple[int,int]]]:
    open_set = []
    heappush(open_set, (manhattan(start, goal), 0, start, [start]))
    closed = set()

    while open_set:
        f, g, current, path = heappop(open_set)
        if current == goal:
            return path
        if current in closed:
            continue
        closed.add(current)

        # Expand neighbours
        for move in kb.fetch("Move"):
            if (move.args[0], move.args[1]) == current:
                neigh = (move.args[2], move.args[3])
                if neigh in closed:
                    continue
                g2 = g + 1
                f2 = g2 + manhattan(neigh, goal)
                heappush(open_set, (f2, g2, neigh, path + [neigh]))
    return None

if __name__ == "__main__":
    obstacles = [(0,1),(2,1),(3,1),(2,3),(3,4),(4,4)]
    maze = build_maze(obstacles)
    path = astar_search(maze)
    if path:
        print("Found path:", path)
    else:
        print("No path found.")
