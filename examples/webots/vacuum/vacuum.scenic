"""
Generate a room for the i-roomba create vacuum
"""
from scipy.stats import qmc
from vacuum_lib import *
import matplotlib.pyplot as plt

# Declare 'clip_range' as a parameter for this scenario.
# Its value will be set by the 'params' dictionary in scenic_supervisor.py.
param clip_range = 0.3 # Initial default value, will be overridden.

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

#------------------------------------------------------------------------
#halton
def generate_halton_sequence(num_points: int, base: int) -> np.ndarray:
    """Generate a Halton sequence."""
    sampler = qmc.Halton(d=2, scramble=False)
    points = sampler.random(num_points)
    # Scale points to the desired range
    scaled_points = (points * 5) - 2.5  # Scale to [-2.5, 2.5]
    return scaled_points

def get_spawn_points(num_points: int) -> np.ndarray:
    return generate_halton_sequence(num_points, base=2)

def show_spawn_points(points: np.ndarray):
    plt.figure(figsize=(6,6))
    plt.scatter(points[:, 0], points[:, 1], color='blue')
    plt.title('Spawn Points on 5.09 x 5.09 Grid')
    plt.xlim(-2.545, 2.545)  # half of 5.09 in both directions
    plt.ylim(-2.545, 2.545)
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Generate and print points
spawn_points = get_spawn_points(20)
print("First 20 spawn points (x, y):")
for i, point in enumerate(spawn_points):
    print(f"Point {i+1}: x={point[0]:.2f}, y={point[1]:.2f}")

# Show popup with points plotted
show_spawn_points(spawn_points)

#------------------------------------------------------------------------
# Place vacuum on floor
ego = new Vacuum on floor
ego.clip_range = globalParameters.clip_range

record (ego.x, ego.y, ego.z) as EgoPosition

# Create a "safe zone" around the vacuum so that it does not start stuck
safe_zone = CircularRegion(ego.position, radius=1)

# Create a dining room region where we will place dining room furniture
dining_room_region = RectangularRegion(1.25 @ 0, 0, 2.5, 5).difference(safe_zone)

# (Commented out furniture sections remain as they were in your original code)
#dining_table = new DiningTable contained in dining_room_region, on floor, facing Range(0, 360 deg), with size .1
#chair_1 = new DiningChair behind dining_table by -0.1, on floor,
#                #facing toward dining_table, with regionContainedIn dining_room_region
# chair_2 = new DiningChair on chair_1,
#                # facing toward dining_table, with regionContainedIn dining_room_region
# chair_3 = new DiningChair left of dining_table by -0.1, on floor,
#                # facing toward dining_table, with regionContainedIn dining_room_region
#fallen_orientation = Uniform((0, -90 deg, 0), (0, 90 deg, 0), (0, 0, -90 deg), (0, 0, 90 deg))
# chair_4 = new DiningChair contained in dining_room_region, facing fallen_orientation,
#                # on floor, with baseOffset(0,0,-0.2)
# # Add some noise to the positions and yaw of the chairs around the table
# mutate chair_1, chair_2, chair_3
# # Create a living room region where we will place living room furniture
# living_room_region = RectangularRegion(-1.25 @ 0, 0, 2.5, 5).difference(safe_zone)
# couch = new Couch ahead of left_wall by 0.335,
#                # on floor, facing away from left_wall
# coffee_table = new CoffeeTable ahead of couch by 0.336,
#                # on floor, facing away from couch
# # Add some noise to the positions of the couch and coffee table
# mutate couch, coffee_table
# toy_stack = new BlockToy on floor
# toy_stack = new BlockToy on toy_stack
# toy_stack = new BlockToy on toy_stack
# # Spawn some toys
# for _ in range(globalParameters.numToys):
#    new Toy on floor

## Simulation Setup ##
# #terminate after globalParameters.duration * 500 seconds
# record (ego.x, ego.y) as VacuumPosition
#Need to implement monitors here to ensure early stopping for given cases