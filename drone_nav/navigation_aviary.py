import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gymnasium import spaces


class NavigationAviary(BaseRLAviary):

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL,
                 target_pos=None,
                 obstacles=None,
                 randomize: bool = True,
                 num_obstacles_range=(2, 4),
                 obstacle_radius_range=(0.12, 0.45),
                 obstacle_height_range=(0.3, 2.2),
                 target_bounds=((1.4, 2.4), (1.4, 2.4), (0.4, 1.8)),
                 obstacle_xy_bounds=((0.5, 2.0), (0.5, 2.0)),
                 min_clearance: float = 0.35,
                 wind_std: float = 0.0,
                 episode_len_sec: float = 15.0,
                 bounds: float = 3.0,
                 goal_radius: float = 0.22,
                 proximity_range: float = 0.8,
                 drone_radius: float = 0.06,
                 ceiling: float = 2.2,
                 tall_prob: float = 0.5):

        self._randomize = randomize
        self._num_obstacles_range = num_obstacles_range
        self._obstacle_radius_range = obstacle_radius_range
        self._obstacle_height_range = obstacle_height_range
        self._target_bounds = target_bounds
        self._obstacle_xy_bounds = obstacle_xy_bounds
        self._min_clearance = min_clearance

        self.OBSTACLE_IDS = []

        self.WIND_STD = wind_std
        self.EPISODE_LEN_SEC = episode_len_sec
        self.BOUNDS = bounds
        self.GOAL_RADIUS = goal_radius
        self.PROXIMITY_RANGE = proximity_range
        self.DRONE_RADIUS = drone_radius
        self.CEILING = ceiling          # flight ceiling (m): can't just fly over tall obstacles
        self._tall_prob = tall_prob     # fraction of obstacles that are floor-to-ceiling pillars
        self._prev_dist_to_target = None
        self._prev_phi = 0.0
        self._collided = False
        self._reached_goal = False

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 1.0]])
        self._start_xyzs = np.array(initial_xyzs, dtype=float)

        if randomize:
            self._sample_scene(self._start_xyzs[0][:2])
        else:
            self.TARGET_POS = np.array([2.0, 2.0, 1.5]) if target_pos is None else np.array(target_pos, dtype=float)
            if obstacles is None:
                obstacles = [
                    {"pos": [0.7, 0.6, 1.0], "radius": 0.25, "height": 1.6},
                    {"pos": [1.3, 1.4, 1.2], "radius": 0.20, "height": 0.8},
                    {"pos": [1.8, 0.9, 0.6], "radius": 0.35, "height": 0.5},
                ]
            self.NAV_OBSTACLES = obstacles

        super().__init__(drone_model=drone_model,
                          num_drones=1,
                          initial_xyzs=initial_xyzs,
                          initial_rpys=initial_rpys,
                          physics=physics,
                          pyb_freq=pyb_freq,
                          ctrl_freq=ctrl_freq,
                          gui=gui,
                          record=record,
                          obs=obs,
                          act=act)


    #  Domain randomization: sample a fresh scene (target + obstacles)
    def _sample_scene(self, start_xy):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self._target_bounds

        # 1) Sample target, rejecting points too close to the start
        target = None
        for _ in range(100):
            candidate = np.array([
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax),
                np.random.uniform(zmin, zmax),
            ])
            if np.linalg.norm(candidate[:2] - start_xy) > 4 * self._min_clearance:
                target = candidate
                break
        self.TARGET_POS = target if target is not None else candidate

        # 2) Sample a random number of cylindrical obstacles
        n_obst = np.random.randint(self._num_obstacles_range[0], self._num_obstacles_range[1] + 1)
        (oxmin, oxmax), (oymin, oymax) = self._obstacle_xy_bounds
        obstacles = []
        attempts = 0
        while len(obstacles) < n_obst and attempts < 200:
            attempts += 1
            radius = np.random.uniform(*self._obstacle_radius_range)
            if np.random.rand() < self._tall_prob:
                height = self.CEILING  # pillar: must be flown AROUND
            else:
                height = np.random.uniform(self._obstacle_height_range[0], 0.9)  # low: can go OVER
            xy = np.array([np.random.uniform(oxmin, oxmax), np.random.uniform(oymin, oymax)])

            if np.linalg.norm(xy - start_xy) < self._min_clearance + radius:
                continue
            if np.linalg.norm(xy - self.TARGET_POS[:2]) < self._min_clearance + radius:
                continue
            if any(np.linalg.norm(xy - np.array(o["pos"][:2])) < self._min_clearance + radius + o["radius"]
                   for o in obstacles):
                continue

            obstacles.append({"pos": [xy[0], xy[1], height / 2.0], "radius": radius, "height": height})

        self.NAV_OBSTACLES = obstacles


    #  Obstacles (rendered as vertical cylinders)
    def _addObstacles(self):
        self.OBSTACLE_IDS = []
        for obst in self.NAV_OBSTACLES:
            pos = obst["pos"]
            radius = obst["radius"]
            height = obst["height"]
            visual_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                             rgbaColor=[0.85, 0.1, 0.1, 1.0],
                                             physicsClientId=self.CLIENT)
            collision_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height,
                                                   physicsClientId=self.CLIENT)
            body_id = p.createMultiBody(baseMass=0,  # static obstacle
                                         baseCollisionShapeIndex=collision_id,
                                         baseVisualShapeIndex=visual_id,
                                         basePosition=pos,
                                         physicsClientId=self.CLIENT)
            self.OBSTACLE_IDS.append(body_id)

        # Small visual marker for the goal (no collision shape -> doesn't
        # interfere with physics or with the collision check).
        goal_visual = p.createVisualShape(p.GEOM_SPHERE, radius=self.GOAL_RADIUS,
                                           rgbaColor=[0.1, 0.9, 0.1, 0.5],
                                           physicsClientId=self.CLIENT)
        p.createMultiBody(baseMass=0,
                           baseVisualShapeIndex=goal_visual,
                           basePosition=self.TARGET_POS.tolist(),
                           physicsClientId=self.CLIENT)


    #  Reset
    def reset(self, seed: int = None, options: dict = None):
        if seed is not None:
            np.random.seed(seed)
        if self._randomize:
            self._sample_scene(np.array(self._start_xyzs[0][:2]))
        obs, info = super().reset(seed=seed, options=options)
        state = self._getDroneStateVector(0)
        self._prev_dist_to_target = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        self._collided = False
        self._reached_goal = False
        return obs, info


    #  Nondeterministic transition: random wind disturbance
    def step(self, action):
        if self.WIND_STD > 0:
            drone_id = self.DRONE_IDS[0]
            wind_force = np.random.normal(0.0, self.WIND_STD, size=3)
            wind_force[2] *= 0.3  # wind mostly horizontal, weaker vertical component
            p.applyExternalForce(objectUniqueId=drone_id,
                                  linkIndex=-1,
                                  forceObj=wind_force.tolist(),
                                  posObj=[0, 0, 0],
                                  flags=p.LINK_FRAME,
                                  physicsClientId=self.CLIENT)
        return super().step(action)


    #  Helpers
    def _nearest_obstacle_info(self, pos):
        """Returns (distance_to_surface, relative_vector_to_center) for the
        closest obstacle, using a cheap analytic point-to-cylinder distance
        (rounded-corner approximation: combines horizontal and vertical
        clearance instead of a true capsule sdf, which is enough for reward
        shaping / observations)."""
        best_dist = np.inf
        best_vec = np.zeros(3)
        if not self.NAV_OBSTACLES:
            return best_dist, best_vec
        for obst in self.NAV_OBSTACLES:
            center = np.array(obst["pos"])
            radius = obst["radius"]
            half_h = obst["height"] / 2.0

            horiz_vec = pos[:2] - center[:2]
            horiz_dist = np.linalg.norm(horiz_vec)
            dx = max(horiz_dist - radius, 0.0)
            dz = max(abs(pos[2] - center[2]) - half_h, 0.0)
            dist_to_surface = float(np.hypot(dx, dz)) - self.DRONE_RADIUS

            if dist_to_surface < best_dist:
                best_dist = dist_to_surface
                best_vec = center - pos
        return best_dist, best_vec


    #  Observation space: kinematic state + goal-relative + obstacle-relative
    def _observationSpace(self):
        base_space = super()._observationSpace()  # Box, shape (1, 12 + 4*buffer)
        extra_low = np.array([[-np.inf] * 7])   # rel_vec_to_goal(3) + rel_vec_to_obst(3) + dist_to_obst(1) -> 7
        extra_high = np.array([[np.inf] * 7])
        low = np.hstack([base_space.low, extra_low])
        high = np.hstack([base_space.high, extra_high])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        base_obs = super()._computeObs()  # shape (1, 12 + 4*buffer)
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rel_goal = self.TARGET_POS - pos
        obst_dist, obst_vec = self._nearest_obstacle_info(pos)
        if not np.isfinite(obst_dist):
            obst_dist = 10.0  # no obstacles this episode -> "far away" placeholder
        extra = np.hstack([rel_goal, obst_vec, [obst_dist]]).astype('float32').reshape(1, 7)
        return np.hstack([base_obs, extra]).astype('float32')


    def _phi(self, dist):
        closeness = max(0.0, 1.0 - dist / self.PROXIMITY_RANGE)
        return 3.0 * closeness ** 2


    #  Reward
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        dist_to_target = float(np.linalg.norm(self.TARGET_POS - pos))
        obst_dist, _ = self._nearest_obstacle_info(pos)

        # 1) Progress shaping: positive when getting closer to the goal
        progress = self._prev_dist_to_target - dist_to_target
        reward = 10.0 * progress
        self._prev_dist_to_target = dist_to_target
        self._prev_dist_to_target = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        self._prev_phi = self._phi(self._prev_dist_to_target)
        # 1b) continuous proximity bonus, based on absolute distance (not delta)
        #     This is what actually pulls the drone the last
        #     bit of the way in, instead of letting it "coast" near the goal
        #     with a near-zero delta signal. Shaped as a smooth, increasingly
        #     steep funnel: 0 at PROXIMITY_RANGE, growing as dist -> 0
        # 1b) potential-based shaping: reward only for getting closer,
        #     so hovering near the goal earns nothing
        phi_new = self._phi(dist_to_target)
        reward += 0.99 * phi_new - self._prev_phi
        self._prev_phi = phi_new

        # 2) Small step penalty, encourages efficient (short) paths.
        reward -= 0.01

        # 3) Obstacle proximity penalty (soft, grows as drone gets close)
        if obst_dist < 0.3:
            reward -= 0.5 * (0.3 - obst_dist) / 0.3

        # 4) Hard collision penalty + episode ends
        if obst_dist < 0.0:
            self._collided = True
            reward -= 50.0

        # 4b) Soft penalty for hugging the ceiling (stops "just climb above everything")
        if pos[2] > self.CEILING - 0.2:
            reward -= 0.5

        # 5) Goal bonus
        if dist_to_target < self.GOAL_RADIUS:
            self._reached_goal = True
            reward += 100.0

        return reward


    #  Termination / truncation
    def _computeTerminated(self):
        return bool(self._collided or self._reached_goal)

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        out_of_bounds = (abs(pos[0]) > self.BOUNDS or abs(pos[1]) > self.BOUNDS
                          or pos[2] > self.CEILING or pos[2] < 0.02)
        too_tilted = (abs(rpy[0]) > 0.6 or abs(rpy[1]) > 0.6)
        timed_out = self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC
        return bool(out_of_bounds or too_tilted or timed_out)

    def _computeInfo(self):
        return {
            "collided": self._collided,
            "reached_goal": self._reached_goal,
            "dist_to_target": self._prev_dist_to_target,
        }