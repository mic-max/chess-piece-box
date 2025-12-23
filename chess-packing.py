# Algorithms:
# - Greedy:
#   - First-Fit: Place objects one by one in the first available position within the bounding box regardless of how it might affect subsequent placements.
#     - First-Fit-Decreasing: Same as first fit but sort the items into decreasing order. (Kings first?)
#     - "there always exists at least one ordering of items that allows first-fit to produce an optimal solution"
#   - Best-Fit: Place objects in the position that maximizes the available space around the object. This approach aims to minimize the "void" space left by each object.
# - Linear Programming: Formulate my packing problem as a linear program. The goal is to minimize the bounding box dimensions. While respecting the constraints of object dimensions and non-overlapping.
# - Heuristics
#   - Simulated Annealing: Start with an initial random placement and iteratively move objects around, accepting changes that decreased the bounding box volume.

# CELING HEIGHT domain is [king height, king height * 2]
# an upper bound of  could be used? since at that height there is no chance of a vertical overlap between any pieces.
# so the optimal positioning would just be two of the best 2d circle packing stacked on top of each other.

# flamechart this to find where time is being spent


# class Piece():
#     cylinders = [
#         [[0, 55, 26], [0, 36, 170]],
#         [[0, 70, 33], [0, 52, 190]],
#         [[0, 70, 110], [-79, 16, 224], [20, 36, 229], [56, 22, 193], [-30, 52, 266]], # the last cylinder can be excluded from the base circles
#         [[0, 70, 35], [0, 45, 148], [0, 36, 232], [0, 16, 251]],
#         [[0, 87, 38], [0, 60, 216], [0, 44, 240], [0, 34, 293]],
#         [[0, 87, 38], [0, 60, 301], [0, 25, 347]],
#     ]

#     def __init__(self, x, y, piece_id, is_white):
#         self.overlaps = False
#         self.x = x
#         self.y = y
#         self.piece_id = piece_id
#         self.base_circles = []
#         self.is_white = is_white

#     def compute_base_circles(self):
#         pass

#     def colour_string(self):
#         if self.is_white:
#             return "[.8, .8, .8]"
#         return "[.3, .3, .3]"


# Variables: Every items (x, y) position
# Constraints: 1) Items cannot overlap. 2) Items fit inside the container.
# Objective: Minimize bounding box volume

"""
If given a rectangle of 1661 x 371. I could consider each position in that grid as a possible place for each item.
So that becomes 1661 * 371 = 616,231 positions. And since each rectangle must house 17 items it becomes 10,475,927.
Some starting positions can be eliminated immediately. A border on the inside of the rectangle that is as thick as the items radius will be an illegal start position.
If we take the pawn, the smallest item it has a radius of 55.
That means the range of x values it can have is 55 to 1661-55 = 1606.
And y range would be 55 to 371-55 = 316. 
So the new possible locations is reduced from 616,231 to 507,496 which ~82%.
The savings for larger pieces will be even larger.
Reducing the total positions to around 8.6 million instead of 10.4 million.
"""

# NUM_PAWNS = 8
# NUM_ROOKS = 2
# NUM_KNIGHTS = 2
# NUM_BISHOPS = 2
# NUM_QUEENS = 2
# NUM_KINGS = 1

# rough dimensions: 17.1" x 3.8" x 3.5"
# I want to make the box wider instead of it being so long.

# import pandas as pd
# import matplotlib.pyplot as plt
# import circlify

# df = pd.DataFrame({
#     'Name': ['Q1', 'Q2', 'K', 'N1', 'N2', 'B1', 'B2', 'R1', 'R2', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'],
#     'Value': [87, 87, 87, 70, 70, 70, 70, 70, 70, 55, 55, 55, 55, 55, 55, 55, 55]
# })
# # compute circle positions
# circles = circlify.circlify(
#     df['Value'].tolist(),
#     show_enclosure=False,
#     target_enclosure=circlify.Circle(x=0, y=0, r=1)
# )

# # reverse the order of the circles to match the order of data
# circles = circles[::-1]
# # Create just a figure and only one subplot
# fig, ax = plt.subplots(figsize=(10, 10))

# # Title
# ax.set_title('Basic circular packing')

# # Remove axes
# ax.axis('off')

# # Find axis boundaries
# lim = max(
#     max(
#         abs(circle.x) + circle.r,
#         abs(circle.y) + circle.r,
#     )
#     for circle in circles
# )
# plt.xlim(-lim, lim)
# plt.ylim(-lim, lim)

# # list of labels
# labels = df['Name']

# # print circles
# for circle, label in zip(circles, labels):
#     x, y, r = circle
#     ax.add_patch(plt.Circle((x, y), r, alpha=0.2, linewidth=2))
#     plt.annotate(
#         label,
#         (x, y),
#         va='center',
#         ha='center'
#     )


# # Compute bounding box coordinates
# min_x = min(circle.x - circle.r for circle in circles)
# max_x = max(circle.x + circle.r for circle in circles)
# min_y = min(circle.y - circle.r for circle in circles)
# max_y = max(circle.y + circle.r for circle in circles)

# # Draw the rectangle using a matplotlib Rectangle patch
# from matplotlib.patches import Rectangle

# width = max_x - min_x
# height = max_y - min_y

# rect = Rectangle(
#     (min_x, min_y),  # bottom-left corner
#     width,
#     height,
#     linewidth=2,
#     edgecolor='red',
#     facecolor='none'
# )

# ax.add_patch(rect)

# print(max_x - min_x, "by", max_y - min_y)

# plt.show()

# TODO: What is the volume of each piece.

# The unit is ~ 0.01"
# Integer arithmetic should be faster?

# TODO: allow knights to rotate in the z axis.

# TODO: bounding_box: could loop over only all the pieces base circles rather than each of all the pieces cylinders.
    # cuts it down from 98 cylinders to 4*4 for the knights then every other piece has 1 base circle equaling: 30+16 = 46.
    # cuts in half!
    # TODO: if both pieces being checked for collision are both already overlapping, don't do any work.

    # TODO: pre-compute base circles and save them to the chess piece.

from copy import deepcopy
from itertools import combinations, product
from dataclasses import dataclass
from random import uniform, randint, random, getrandbits
from typing import List, Tuple
import math

@dataclass
class Piece:
    x: float
    y: float
    cylinders: List[Tuple[float, float, float]]
    overlaps: bool
    is_white: bool

def get_circles(piece: Piece) -> List[Tuple[float, float, float]]:
    return [(piece.x, xoff + piece.y, r) for xoff, r, _ in piece.cylinders]

def circles_overlap(a, b):
    ax, ay, ar = a
    bx, by, br = b
    dx = ax - bx
    dy = ay - by
    r_sum = ar + br
    return dx * dx + dy * dy < r_sum * r_sum

def print_cylinder(x, y, z, h, r):
    result = "\n"
    result += f"  translate([{x}, {y}, {z}])\n"
    result += f"  cylinder(h={h}, r={r});\n"
    return result

def any_overlaps(pieces: List[Piece]) -> bool:
    for a, b in combinations(pieces, 2):
        for ca, cb in product(get_circles(a), get_circles(b)):
            if circles_overlap(ca, cb):
                return True
    return False

def any_vertical_overlaps(pieces: List[List[Piece]]) -> bool:
    for a in pieces[0]:
        for ax, ay, ar in get_circles(a):
            for b in pieces[1]:
                for bx, by, br in get_circles(b):
                    if circles_overlap((ax, ay, ar), (bx, by, br)):
                        wz = max(h for _, r, h in a.cylinders if r == ar)
                        bz = CEIL - max(h for _, r, h in b.cylinders if r == br)
                        if wz >= bz:
                            return True
    return False

def mark_overlaps(pieces: List[Piece]) -> None:
    for a, b in combinations(pieces, 2):
        for ca, cb in product(get_circles(a), get_circles(b)):
            if circles_overlap(ca, cb):
                a.overlaps = b.overlaps = True
                break

def check_vertical_conflicts(pieces: List[List[Piece]]) -> None:
    for a in pieces[0]:
        for ax, ay, ar in get_circles(a):
            for b in pieces[1]:
                for bx, by, br in get_circles(b):
                    if circles_overlap((ax, ay, ar), (bx, by, br)):
                        az = max(h for _, r, h in a.cylinders if r == ar)
                        bz = CEIL - max(h for _, r, h in b.cylinders if r == br)
                        if az >= bz:
                            a.overlaps = b.overlaps = True

def write_piece_scad(f, piece: Piece):
    color = "[.8, .1, .1]" if piece.overlaps else ("[.8, .8, .8]" if piece.is_white else "[.3, .3, .3]")
    f.write(f"color({color})\n")
    f.write("union(){")
    for xoff, r, h in piece.cylinders:
        z = 0 if piece.is_white else CEIL - h
        f.write(print_cylinder(piece.x, xoff + piece.y, z, h, r))
    f.write("}\n\n")

def bounding_box(pieces: List[Piece]) -> Tuple[float, float, float]:
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for piece in pieces:
        for cx, cy, r in get_circles(piece):
            min_x = min(min_x, cx - r)
            max_x = max(max_x, cx + r)
            min_y = min(min_y, cy - r)
            max_y = max(max_y, cy + r)
    return (max_x - min_x, max_y - min_y, CEIL)

def simulated_annealing(
        pieces: List[List[Piece]],
        iterations:int=10000,
        initial_temp:float=100.0,
        cooling_rate:float=0.995,
        move_scale:float=2.0
    ):

    def random_move(piece: Piece):
        dx = uniform(-move_scale, move_scale)
        dy = uniform(-move_scale, move_scale)
        return piece.x + dx, piece.y + dy


    def evaluate(pieces: List[List[Piece]]) -> float:
        if any_overlaps(pieces[0]) or any_overlaps(pieces[1]) or any_vertical_overlaps(pieces):
            return float('inf')
        x, y, z = bounding_box(pieces[0] + pieces[1])
        return x * y * z

    best_pieces = deepcopy(pieces)
    best_score = evaluate(best_pieces)
    current_pieces = deepcopy(pieces)
    current_score = best_score
    temp = initial_temp
    
    for i in range(iterations):
        trial_pieces = deepcopy(current_pieces)
        idx = getrandbits(1)
        jdx = randint(0, len(trial_pieces[idx]) - 1)
        old_x = trial_pieces[idx][jdx].x
        old_y = trial_pieces[idx][jdx].y
        new_x, new_y = random_move(trial_pieces[idx][jdx])
        trial_pieces[idx][jdx].x = new_x
        trial_pieces[idx][jdx].y = new_y

        trial_score = evaluate(trial_pieces)

        # will this work, because sometimes I think to achieve a better score two or more pieces might have to move at the same time?
        if trial_score < current_score or random() < math.exp((current_score - trial_score) / temp):
            current_pieces = trial_pieces
            current_score = trial_score
            if current_score < best_score:
                best_score = current_score
                best_pieces = deepcopy(current_pieces)
        
        temp *= cooling_rate

        if i % 500 == 0:
            with open("chess-packing.scad", "w") as f:
                for p in best_pieces[0] + best_pieces[1]:
                    write_piece_scad(f, p)
                if show_box:
                    f.write("}\n")
            print(f"Step {i}, Temp={temp:.3f}, Best Volume={best_score:.2f}")

    return best_pieces



cylinders = {
    "pawn": [[0, 55, 26], [0, 36, 170]],
    "rook": [[0, 70, 33], [0, 52, 190]],
    "knight": [[0, 70, 110], [-79, 16, 224], [-30, 52, 266], [20, 36, 229], [56, 22, 193]], # the circle (52, 266) can be excluded from being a base-circle
    "bishop": [[0, 70, 35], [0, 45, 148], [0, 36, 232], [0, 16, 251]],
    "queen": [[0, 87, 38], [0, 60, 216], [0, 44, 240], [0, 34, 293]],
    "king": [[0, 87, 38], [0, 60, 301], [0, 25, 347]],
}

CEIL = 347 # [347, ] pawn + rook = 360. pawns don't interfere vertically with rooks
S = 155
# S = 158, CEIL = 347. Bounding box: 1509.2 x 379.4 x 347 --- Volume: 198688896.56
# Bounding box: 1467.4 x 376.8 x 347 Volume: 191861963.04000002

# how can I identify which pieces are touching the bounding box?
# they should be a different colour. since those are the first place i should try to optimize.

if __name__ == '__main__':
    pieces = [
        # "K": [(8.1*S, S)]
        # "Q": [(), ()]
        # "B": [(), ()]
        # "N": [(), ()]
        # "R": [(), ()]
        # "P": [(0*S, 0), (), (), (), (), (), (), ()]
        [
            Piece(0*S, 0, cylinders["pawn"], False, True),
            Piece(1*S, 0, cylinders["pawn"], False, True),
            Piece(1.8*S, 0, cylinders["pawn"], False, True),
            Piece(3.1*S, 0, cylinders["pawn"], False, True),
            Piece(3.9*S, 0, cylinders["pawn"], False, True),
            Piece(5.1*S, 0, cylinders["pawn"], False, True),
            Piece(5.9*S, 0, cylinders["pawn"], False, True),
            Piece(7.1*S, 0, cylinders["pawn"], False, True),

            Piece(0.95*S, 0.85*S, cylinders["rook"], False, True),
            Piece(0*S, S, cylinders["knight"], False, True),
            Piece(3*S, 0.9*S, cylinders["bishop"], False, True),
            Piece(1.975*S, 1.1*S, cylinders["queen"], False, True),
            Piece(4*S, 1.1*S, cylinders["queen"], False, True),
            Piece(5.1*S, 0.9*S, cylinders["bishop"], False, True),
            Piece(6.05*S, S, cylinders["knight"], False, True),
            Piece(7*S, 0.9*S, cylinders["rook"], False, True),
            Piece(8.1*S, S, cylinders["king"], False, True),
        ],
        [
            Piece(1.6*S, 1.6*S, cylinders["pawn"], False, False),
            Piece(2.5*S, 1.5*S, cylinders["pawn"], False, False),
            Piece(3.4*S, 1.4*S, cylinders["pawn"], False, False),
            Piece(4.6*S, 1.4*S, cylinders["pawn"], False, False),
            Piece(5.4*S, 1.4*S, cylinders["pawn"], False, False),
            Piece(7.46*S, 0.3*S, cylinders["pawn"], False, False),
            Piece(7.5*S, 1.4*S, cylinders["pawn"], False, False),
            Piece(6.5*S, 1.4*S, cylinders["pawn"], False, False),

            Piece(0.5*S, 0.3*S, cylinders["rook"], False, False),
            Piece(3.5*S, 0.5*S, cylinders["rook"], False, False),

            Piece(2.5*S, 0.37*S, cylinders["knight"], False, False),
            Piece(4.5*S, 0.4*S, cylinders["knight"], False, False),

            Piece(1.4*S, 0.5*S, cylinders["bishop"], False, False),
            Piece(5.5*S, 0.4*S, cylinders["bishop"], False, False),

            Piece(0.6*S, 1.5*S, cylinders["queen"], False, False),
            Piece(6.5*S, 0.2*S, cylinders["queen"], False, False),
            Piece(8.4*S, 0.23*S, cylinders["king"], False, False),
        ]
    ]

    mark_overlaps(pieces[0])
    mark_overlaps(pieces[1])
    check_vertical_conflicts(pieces)
    
    x, y, z = bounding_box(pieces[0] + pieces[1])
    print(f"Bounding box: {x} x {y} x {z}")
    print(f"Volume: {x * y * z}")

    show_box = False
    optimal_pieces = simulated_annealing(pieces, iterations=50_000)

    with open("chess-packing.scad", "w") as f:
        # TODO: don't hard code the translation.
        if show_box:
            f.write("difference() {\n")
            f.write(f"translate([{-95}, {-55}, {0.1}])\n")
            f.write(f"cube([{x + 0.1}, {y + 0.1}, {z - 0.2}])\n")
        for p in optimal_pieces[0] + optimal_pieces[1]:
            write_piece_scad(f, p)
        if show_box:
            f.write("}\n")


