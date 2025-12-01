param map = localPath('../assets/maps/CARLA/Town05.xodr')
param carla_map = 'Town05'
model scenic.simulators.carla.model
param weather = 'ClearNoon'

EGO_BLUEPRINT = "vehicle.lincoln.mkz_2017"

intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))

egoInitLane = Uniform(*intersection.incomingLanes)

egoSpawnPt = new OrientedPoint in egoInitLane.centerline


ego = new Car at egoSpawnPt,
      with blueprint EGO_BLUEPRINT,
      with behavior EgoBehavior()

require (distance to intersection) < 20