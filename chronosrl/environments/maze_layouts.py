"""Maze layouts shared by the ant and drone maze environments.

Kept free of simulator imports so every pixi environment can load them.
"""

R = "r"
G = "g"

LAYOUTS = {

    "u_maze": [
        [1, 1, 1, 1, 1],
        [1, R, G, G, 1],
        [1, 1, 1, G, 1],
        [1, G, G, G, 1],
        [1, 1, 1, 1, 1],
    ],
    
    "u4_maze": [
        [1, 1, 1, 1, 1],
        [1, G, G, G, 1],
        [1, R, 1, G, 1],
        [1, 1, 1, G, 1],
        [1, G, 1, G, 1],
        [1, G, G, G, 1],
        [1, 1, 1, 1, 1],
    ],
    "u5_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, G, G, G, G, G, G, 1],
        [1, R, 1, 1, 1, 1, G, 1],
        [1, 1, 1, 1, 1, 1, G, 1],
        [1, G, 1, 1, 1, 1, G, 1],
        [1, G, G, G, G, G, G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "big_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, R, G, 1, 1, G, G, 1],
        [1, G, G, 1, G, G, G, 1],
        [1, 1, G, G, G, 1, 1, 1],
        [1, G, G, 1, G, G, G, 1],
        [1, G, 1, G, G, 1, G, 1],
        [1, G, G, G, 1, G, G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "hardest_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, R, G, G, G, 1, G, G, G, G, G, 1],
        [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
        [1, G, G, G, G, G, G, 1, G, G, G, 1],
        [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
        [1, G, G, 1, G, 1, G, G, G, G, G, 1],
        [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
        [1, G, G, 1, G, G, G, 1, G, G, G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
}
