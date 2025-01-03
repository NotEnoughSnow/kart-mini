import random

import pymunk

from core.sim.maps.map_manager import AbstractMap as abs_map


class RandomPoint(abs_map):

    def __init__(self, spawn_range, fixed_goal):
        self.spawn_range = spawn_range

        if fixed_goal is not None:
            self.random = True
            self.fixed_point_spawn = fixed_goal
        else:
            self.random = False

        self.initial_pos = self.wc
        self.missing_walls_flag = True
        self.missing_sectors_flag = True

    def init_track(self, space, world_center):
        self.space = space
        self.wc = world_center.copy()



    def reset(self, playerShapes):

        for item in self.space.shapes:
            if item not in playerShapes:
                self.space.remove(item)

        position = self.wc

        angle = 0

        self.missing_walls_flag = True
        self.missing_sectors_flag = True

        return None, angle, position

    def create_walls(self):

        self.missing_walls_flag = False

        return []

    def create_goals(self, mode="random"):
        if self.random:
            goal_x = random.uniform(-self.spawn_range / 2, self.spawn_range / 2)
            goal_y = random.uniform(-self.spawn_range / 2, self.spawn_range / 2)

            random_goal = [self.wc[0] + goal_x, self.wc[1] + goal_y]

        if not self.random:
            random_goal = [self.wc[0] + self.fixed_point_spawn[0], self.wc[1] + self.fixed_point_spawn[1]]


        # random_goal = [600, 600]

        # FIXME make maps
        # sectors
        sensor_bodies = self.space.static_body

        static_sector_lines = []
        sector_midpoints = []

        # static_sector_lines.append(pymunk.Segment(sensor_bodies, goal[0], goal[1], 0.0))
        shape = pymunk.Segment(sensor_bodies, random_goal, random_goal, 5.0)
        shape.color = (255, 255, 0, 255)
        static_sector_lines.append(shape)
        sector_midpoints.append(random_goal)

        # FIXME use np.average ?
        # FIXME midpoints

        for i in range(len(static_sector_lines)):
            static_sector_lines[i].elasticity = 0
            static_sector_lines[i].friction = 1
            static_sector_lines[i].sensor = True

        for i in range(len(static_sector_lines)):
            static_sector_lines[i].collision_type = i + 2
            static_sector_lines[i].filter = pymunk.ShapeFilter(categories=0x10)

        self.missing_sectors_flag = False

        return static_sector_lines, sector_midpoints
