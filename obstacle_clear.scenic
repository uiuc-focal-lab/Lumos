# 1. Set map and model
param map = localPath('../assets/maps/CARLA/Town01.xodr')
param carla_map = 'Town01'
model scenic.simulators.carla.model
param weather = 'ClearNoon'

import random

# 2. Define constants
EGO_BLUEPRINT = "vehicle.lincoln.mkz_2017"
vehicle_models = [
    'vehicle.audi.a2',
    'vehicle.audi.etron',
    'vehicle.audi.tt',
    'vehicle.bmw.grandtourer',
    'vehicle.chevrolet.impala',
    'vehicle.citroen.c3',
    'vehicle.dodge.charger_2020',
    'vehicle.dodge.charger_police',
    'vehicle.dodge.charger_police_2020',
    'vehicle.ford.crown',
    'vehicle.ford.mustang',
    'vehicle.jeep.wrangler_rubicon',
    'vehicle.lincoln.mkz_2020',
    'vehicle.mercedes.coupe',
    'vehicle.mercedes.coupe_2020',
    'vehicle.micro.microlino',
    'vehicle.mini.cooper_s',
    'vehicle.mini.cooper_s_2021',
    'vehicle.nissan.micra',
    'vehicle.nissan.patrol',
    'vehicle.nissan.patrol_2021',
    'vehicle.seat.leon',
    'vehicle.tesla.model3',
    'vehicle.toyota.prius',
    'vehicle.carlamotors.carlacola',
    'vehicle.carlamotors.european_hgv',
    'vehicle.carlamotors.firetruck',
    'vehicle.tesla.cybertruck',
    'vehicle.ford.ambulance',
    'vehicle.mercedes.sprinter',
    'vehicle.volkswagen.t2',
    'vehicle.volkswagen.t2_2021',
    'vehicle.mitsubishi.fusorosa',
    'vehicle.harley-davidson.low_rider',
    'vehicle.kawasaki.ninja',
    'vehicle.vespa.zx125',
    'vehicle.yamaha.yzf',
    'vehicle.bh.crossbike',
    'vehicle.diamondback.century',
    'vehicle.gazelle.omafiets'
]

OTHER_BLUEPRINT = random.choice(vehicle_models)
EGO_INITIAL_SPEED = 0 # For a static scene

# 3. Define the road and lane
roads = Uniform(*network.roads)
lane = Uniform(*roads.lanes)

# 5. Define Behaviors
behavior EgoBehavior():
    take SetSpeedAction(EGO_INITIAL_SPEED)

# 4. Set the scene
ego = new Car on lane.centerline,  # ego is placed on the centerline of the chosen lane
    with blueprint EGO_BLUEPRINT,
    with behavior EgoBehavior()

import random
if random.random() < 0.5:
    target_object = new Car visible, ahead of ego by Uniform(1, 2), with behavior EgoBehavior(), with blueprint OTHER_BLUEPRINT
else:
    target_object = new Barrier visible, ahead of ego by Uniform(1, 2)

terminate after 125 seconds