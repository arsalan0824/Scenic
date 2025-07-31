#ethan's code + arsalan
"""
Generate a room for the i-roomba create vacuum
"""
from scenic.core.external_params import VerifaiRange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from vacuum_lib import *
from itertools import combinations
no_verifai = True
param verifaiSamplerType = 'mab'
dnev = 1.59576912 # double normal ev, |[-dnev, dnev]| has the same ev as |N(0, 1)|
dpi = 6.28318531

## Scene Layout ##

# Create room region and set it as the workspace
room_region = RectangularRegion(0 @ 0, 0, 5.09, 5.09)
workspace = Workspace(room_region)

# Create floor and walls
floor = new Floor
wall_offset = floor.width/2 + 0.04/2 + 1e-4
right_wall = new Wall at (wall_offset, 0, 0.25), facing toward floor
left_wall = new Wall at (-wall_offset, 0, 0.25), facing toward floor
front_wall = new Wall at (0, wall_offset, 0.25), facing toward floor
back_wall = new Wall at (0, -wall_offset, 0.25), facing toward floor

if no_verifai:
    # Place vacuum on floor
    ego = new Vacuum on floor
    record (ego.x, ego.y, ego.z) as EgoPosition

    # Create a "safe zone" around the vacuum so that it does not start stuck
    safe_zone = CircularRegion(ego.position, radius=1)

    # # Create a dining room region where we will place dining room furniture
    dining_room_region = RectangularRegion(1.25 @ 0, 0, 5, 5).difference(safe_zone)

    dining_table = new DiningTable on floor, facing Range(0, 360 deg), with size .1
    record (dining_table.x, dining_table.y, dining_table.z) as DiningTablePosition

    chair_1 = new DiningChair on floor, behind dining_table by -.1, 
                    facing toward dining_table, with regionContainedIn dining_room_region
    record (chair_1.x, chair_1.y, chair_1.z) as chair_1Position

    chair_3 = new DiningChair on floor, left of dining_table by -.1, 
                    facing toward dining_table, with regionContainedIn dining_room_region
    record (chair_3.x, chair_3.y, chair_3.z) as chair_3Position

    #Add some noise to the positions and yaw of the chairs around the table
    mutate chair_1, chair_3

    # Create a living room region where we will place living room furniture
    living_room_region = RectangularRegion(-1.25 @ 0, 0, 5, 5).difference(safe_zone)

    couch = new Couch on floor, ahead of left_wall by 0.335, 
            facing away from left_wall, with regionContainedIn living_room_region
    record (couch.x, couch.y, counch.z) as couchPosition
    coffee_table = new CoffeeTable on floor, ahead of couch by 0.336, 
            facing away from couch, with regionContainedIn living_room_region
    record (coffee_table.x, coffee_table.y, coffee_table.z) as couchPosition

    # # Add some noise to the positions of the couch and coffee table
    mutate couch, coffee_table

    # Spawn some toys
    for _ in range(3):
        new Toy on floor
else:
    # Place vacuum on floor
    ego = new Vacuum at (VerifaiRange(-2.5, 2.5) @ VerifaiRange(-2.5, 2.5)), on floor
    record (ego.x, ego.y, ego.z) as EgoPosition

    # Create a "safe zone" around the vacuum so that it does not start stuck
    safe_zone = CircularRegion(ego.position, radius=1)

    # # Create a dining room region where we will place dining room furniture
    dining_room_region = RectangularRegion(1.25 @ 0, 0, 5, 5).difference(safe_zone)

    dining_table = new DiningTable on floor, at (VerifaiRange(-2.5, 2.5) @ VerifaiRange(-2.5, 2.5)), with size .1
    ideal_pos = new OrientedPoint behind dining_table by -0.1, facing toward dining_table
    chair_1 = new DiningChair on floor, 
                facing VerifaiRange((-dnev*10) deg, (dnev*10) deg) relative to ideal_pos,
                at ideal_pos offset by (new VerifaiRange(-dnev*.05, dnev*.05) @ new VerifaiRange(-dnev*.05, dnev*.05)),
                with regionContainedIn dining_room_region     
    ideal_pos = new OrientedPoint left of dining_table by -0.1, facing toward dining_table
    chair_3 = new DiningChair on floor, 
                facing VerifaiRange((-dnev*10) deg, (dnev*10) deg) relative to ideal_pos,
                at ideal_pos offset by (VerifaiRange(-dnev*.05, dnev*.05) @ VerifaiRange(-dnev*.05, dnev*.05)), 
                with regionContainedIn dining_room_region

    # # Create a living room region where we will place living room furniture
    living_room_region = RectangularRegion(-1.25 @ 0, 0, 5, 5).difference(safe_zone)
    ideal_pos = new OrientedPoint ahead of left_wall by 0.335, facing away from left_wall
    couch = new Couch on floor, facing VerifaiRange((-dnev*5) deg, (dnev*5) deg) relative to ideal_pos, with regionContainedIn living_room_region,
                at ideal_pos offset by (VerifaiRange(-dnev*.05, dnev*.05) @ VerifaiRange(-dnev*.5, dnev*.5))
    ideal_pos = new OrientedPoint ahead of couch by 0.336, facing away from couch
    coffee_table = new CoffeeTable on floor, facing VerifaiRange((-dnev*5) deg, (dnev*5) deg) relative to ideal_pos, with regionContainedIn living_room_region,
                at ideal_pos offset by (VerifaiRange(-dnev*.05, dnev*.05) @ VerifaiRange(-dnev*.05, dnev*.05))
    # print ("Coffee Table (x, y):", coffee_table.x, coffee_table.y)

    new Toy on floor, at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10)
    new Toy on floor, at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10)
    new Toy on floor, at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10)


room_size = 5.09
half_size = room_size / 2

objects = [
    {"name": "dining_table", "x": dining_table.x, "y": dining_table.y, "w": 0.4, "h": 0.4},
    {"name": "chair_1", "x": chair_1.x, "y": chair_3.Y, "w": 0.2, "h": 0.2},
    {"name": "chair_3", "x": chair_3.x, "y": chair_3.Y, "w": 0.2, "h": 0.2},
    {"name": "couch", "x": couch.x, "y": 0.couch.y, "w": 0.7, "h": 0.3},
    {"name": "coffee_table", "x": coffee_table.x, "y": coffee_table.y, "w": 0.4, "h": 0.2},
    {"name": "toy_1", "x": toy_1.x, "y": toy_1.y, "w": 0.1, "h": 0.1},
    {"name": "toy_2", "x": toy_2.x, "y": toy_3.y, "w": 0.1, "h": 0.1},
    {"name": "toy_3", "x": toy_3.x, "y": toy_3.y, "w": 0.1, "h": 0.1},
]

# Set up the figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-half_size, half_size)
ax.set_ylim(-half_size, half_size)
ax.set_aspect('equal')
ax.set_title("Room Object Placement Grid")
ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Draw the room boundary
room_rect = patches.Rectangle((-half_size, -half_size), room_size, room_size,
                              linewidth=1, edgecolor='black', facecolor='none')
ax.add_patch(room_rect)

# Plot all objects as black rectangles
for obj in objects:
    rect = patches.Rectangle((obj["x"] - obj["w"]/2, obj["y"] - obj["h"]/2),
                             obj["w"], obj["h"],
                             linewidth=1, edgecolor='black', facecolor='black', label=obj["name"])
    ax.add_patch(rect)
    ax.text(obj["x"], obj["y"], obj["name"], color='white',
            ha='center', va='center', fontsize=6)

plt.show()

# import random

# big_objects = [
#     lambda: new DiningTable at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#             with size 0.1,
#             facing Range(0, 360 deg),

#     lambda: new Couch at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#                 facing Range(0, 360 deg),

#     lambda: new CoffeeTable at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#                 facing Range(0, 360 deg),
# ]
# small_objects = [
#     lambda: new DiningChair at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#                 facing Range(0, 360 deg),

#     lambda: new DiningChair
#                 at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#                 facing Range(0, 360 deg),

#     lambda: new BlockToy at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#             facing Range(0, 360 deg),

#     lambda: new BlockToy at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#             facing Range(0, 360 deg),
# ]

# chosen_big_objects = random.sample(big_objects, 2)
# chosen_small_objects =  random.sample(small_objects, 2)

# walls = [right_wall, left_wall, front_wall, back_wall]
# objects = [ego]
# for obj in chosen_big_objects:
#     objects.append(obj())
# for obj in chosen_small_objects:
#     objects.append(obj())

# wallDist = 0.335
# objDist = 0.1
# require (
#   all(not (a.occupiedSpace._bufferOverapproximate(wallDist,1).intersects(b.occupiedSpace)) for a in walls for b in objects)
#   and
#   all(not (a.occupiedSpace._bufferOverapproximate(objDist,1).intersects(b.occupiedSpace)) for a, b in combinations(objects, 2))
# )