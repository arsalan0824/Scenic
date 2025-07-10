"""
Generate a room for the i-roomba create vacuum
"""

from vacuum_lib import *
from itertools import combinations


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

# Place vacuum on floor
ego = new Vacuum on floor
record (ego.x, ego.y, ego.z) as EgoPosition

import random

big_objects = [
    lambda: new DiningTable on floor,
            with size 0.1,
            facing Range(0, 360 deg),

    lambda: new Couch on floor,
                facing Range(0, 360 deg),

    lambda: new CoffeeTable on floor,
                facing Range(0, 360 deg),
]
small_objects = [
    lambda: new DiningChair on floor,
                facing Range(0, 360 deg),

    lambda: new DiningChair
                on floor,
                facing Range(0, 360 deg),

    lambda: new BlockToy on floor,
            facing Range(0, 360 deg),

    lambda: new BlockToy on floor,
            facing Range(0, 360 deg),
]

chosen_big_objects = random.sample(big_objects, 2)
chosen_small_objects =  random.sample(small_objects, 2)

walls = [right_wall, left_wall, front_wall, back_wall]
objects = [ego]
for obj in chosen_big_objects:
    objects.append(obj())
for obj in chosen_small_objects:
    objects.append(obj())

wallDist = 0.335
objDist = 0.1
require (
  all(not (a.occupiedSpace._bufferOverapproximate(wallDist,1).intersects(b.occupiedSpace)) for a in walls for b in objects)
  and
  all(not (a.occupiedSpace._bufferOverapproximate(objDist,1).intersects(b.occupiedSpace)) for a, b in combinations(objects, 2))
)

# Create a "safe zone" around the vacuum so that it does not start stuck
# safe_zone = CircularRegion(ego.position, radius=1)

# # Create a dining room region where we will place dining room furniture
# dining_room_region = RectangularRegion(1.25 @ 0, 0, 5, 5).difference(safe_zone)

# dining_table = new DiningTable on floor, facing Range(0, 360 deg), with size .1

# chair_1 = new DiningChair on floor,#behind dining_table by -.1,
#                 facing toward dining_table
# chair_2 = new DiningChair on chair_1,
#                 facing toward dining_table, with regionContainedIn dining_room_region
# chair_3 = new DiningChair on floor,#left of dining_table by -.1,
#                 facing toward dining_table, with regionContainedIn dining_room_region
# fallen_orientation = Uniform((0, -90 deg, 0), (0, 90 deg, 0), (0, 0, -90 deg), (0, 0, 90 deg))

# chair_4 = new DiningChair contained in dining_room_region, facing fallen_orientation,
#                 on floor, with baseOffset(0,0,-0.2)

# Add some noise to the positions and yaw of the chairs around the table
# mutate chair_1

# # Create a living room region where we will place living room furniture
# living_room_region = RectangularRegion(-1.25 @ 0, 0, 5, 5).difference(safe_zone)

# couch = new Couch on floor, ahead of left_wall by 0.335,
#            facing away from left_wall

# coffee_table = new CoffeeTable on floor, ahead of couch by 0.336,
#           facing away from couch

# # # Add some noise to the positions of the couch and coffee table
# mutate couch, coffee_table

# toy_stack = new BlockToy on floor
# toy_stack = new BlockToy on toy_stack
# toy_stack = new BlockToy on toy_stack

# # Spawn some toys
# for _ in range(globalParameters.numToys):
#     new Toy on floor

# ## Simulation Setup ##
# #terminate after globalParameters.duration * 500 seconds
# record (ego.x, ego.y) as VacuumPosition
#Need to implement monitors here to ensure early stopping for given cases

