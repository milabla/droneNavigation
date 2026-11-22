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
                 episode_len_sec: float = 30.0,
                 bounds: float = 3.0,
                 goal_radius: float = 0.22,
                 proximity_range: float = 0.8,
                 drone_radius: float = 0.06,
                 ceiling: float = 2.2,
                 tall_prob: float = 0.5,
                 proximity_penalty: float = 0.1,
                 trunc_penalty_base: float = 10.0,
                 trunc_penalty_per_m: float = 5.0,
                 collision_penalty: float = 30.0,
                 num_obs_tracked: int = 3,
                 difficulty: float = 1.0,
                 num_horizontal_range=(1, 2),
                 horizontal_length_range=(2.0, 3.0),
                 horizontal_radius_range=(0.06, 0.12),
                 horizontal_z_bands=((0.25, 0.5), (0.8, 1.3), (1.6, 1.95))):

        # Horizontal bars: random count, length, axis (x or y) and height band (low / middle / high).
        # num_horizontal_range=(0, 0) disables them and gives the old vertical-only scenes.
        self._num_horizontal_range = num_horizontal_range
        self._horizontal_length_range = horizontal_length_range
        self._horizontal_radius_range = horizontal_radius_range
        self._horizontal_z_bands = horizontal_z_bands

        self._randomize = randomize
        self._num_obstacles_range = num_obstacles_range
        self._obstacle_radius_range = obstacle_radius_range
        self._obstacle_height_range = obstacle_height_range
        self._target_bounds = target_bounds
        self._obstacle_xy_bounds = obstacle_xy_bounds
        self._min_clearance = min_clearance
        self._difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.NUM_OBS_TRACKED = num_obs_tracked
        self.COLLISION_PENALTY = collision_penalty

        self.OBSTACLE_IDS = []

        self.WIND_STD = wind_std
        self.EPISODE_LEN_SEC = episode_len_sec
        self.BOUNDS = bounds
        self.GOAL_RADIUS = goal_radius
        self.PROXIMITY_RANGE = proximity_range
        self.DRONE_RADIUS = drone_radius
        self.CEILING = ceiling
        self._tall_prob = tall_prob
        self.PROXIMITY_PENALTY = proximity_penalty
        self.TRUNC_PENALTY_BASE = trunc_penalty_base
        self.TRUNC_PENALTY_PER_M = trunc_penalty_per_m
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

    #  Curriculum control (callable through VecEnv.env_method)
    def set_difficulty(self, d: float):
        self._difficulty = float(np.clip(d, 0.0, 1.0))

    def get_difficulty(self) -> float:
        return self._difficulty

    #  Domain randomization: sample a fresh scene (target + obstacles)
    def _sample_scene(self, start_xy):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self._target_bounds
        d = self._difficulty
        scale = 0.4 + 0.6 * d   # easy: target pulled toward the start; hard: full distance
        start3 = np.array([start_xy[0], start_xy[1], self._start_xyzs[0][2]])

        # Sample target, rejecting points too close to the start
        target = None
        candidate = None
        for _ in range(100):
            full = np.array([
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax),
                np.random.uniform(zmin, zmax),
            ])
            candidate = start3 + (full - start3) * scale
            if np.linalg.norm(candidate[:2] - start_xy) > 4 * self._min_clearance * scale:
                target = candidate
                break
        self.TARGET_POS = target if target is not None else candidate

        # Sample cylindrical obstacles, count scales with difficulty (0 -> none)
        lo, hi = self._num_obstacles_range
        hi_eff = int(round(hi * d))
        lo_eff = min(lo, hi_eff)
        n_obst = np.random.randint(lo_eff, hi_eff + 1)
        (oxmin, oxmax), (oymin, oymax) = self._obstacle_xy_bounds
        obstacles = []
        attempts = 0
        while len(obstacles) < n_obst and attempts < 200:
            attempts += 1
            radius = np.random.uniform(*self._obstacle_radius_range)
            if np.random.rand() < self._tall_prob:
                height = self.CEILING
            else:
                height = np.random.uniform(self._obstacle_height_range[0], 0.9)
            xy = np.array([np.random.uniform(oxmin, oxmax), np.random.uniform(oymin, oymax)])

            if np.linalg.norm(xy - start_xy) < self._min_clearance + radius:
                continue
            if np.linalg.norm(xy - self.TARGET_POS[:2]) < self._min_clearance + radius:
                continue
            if any(np.linalg.norm(xy - np.array(o["pos"][:2])) < self._min_clearance + radius + o["radius"]
                   for o in obstacles):
                continue

            obstacles.append({"pos": [xy[0], xy[1], height / 2.0], "radius": radius, "height": height})

        # Horizontal bars (axis 0 = along x, axis 1 = along y). "height" holds the bar length.
        # Count scales with difficulty like the vertical obstacles.
        h_lo, h_hi = self._num_horizontal_range
        if h_hi > 0:
            h_hi_eff = int(round(h_hi * d))
            h_lo_eff = min(h_lo, h_hi_eff)
            n_horiz = np.random.randint(h_lo_eff, h_hi_eff + 1)
            start3_full = np.array(self._start_xyzs[0], dtype=float)
            handled = 0          # bars placed or given up on
            band = None
            fails = 0
            while handled < n_horiz:
                # Pick the height band once per bar and keep trying positions inside it, so rejections
                # do not bias the scene toward whichever band is easiest to place.
                if band is None:
                    band = self._horizontal_z_bands[np.random.randint(len(self._horizontal_z_bands))]
                    fails = 0
                radius = np.random.uniform(*self._horizontal_radius_range)
                length = np.random.uniform(*self._horizontal_length_range)
                z = np.random.uniform(*band)
                axis = int(np.random.randint(2))
                cx = np.random.uniform(oxmin, oxmax)
                cy = np.random.uniform(oymin, oymax)
                bar = {"pos": [cx, cy, z], "radius": radius, "height": length, "axis": axis}

                # Check clearance along the whole bar (points sampled along its axis)
                ok = True
                for t in np.linspace(-0.5, 0.5, 11):
                    point = np.array([cx, cy, z])
                    point[axis] += t * length
                    if np.linalg.norm(point - start3_full) < self._min_clearance + radius:
                        ok = False
                        break
                    if np.linalg.norm(point - self.TARGET_POS) < self._min_clearance + radius:
                        ok = False
                        break
                    if any(self._surface_dist(point, o) < radius + 0.15 for o in obstacles):
                        ok = False
                        break
                if not ok:
                    fails += 1
                    if fails >= 300:     # no room for a bar in this band: give up on it
                        band = None
                        handled += 1
                    continue

                obstacles.append(bar)
                band = None
                handled += 1

        self.NAV_OBSTACLES = obstacles

    #  Obstacles
    def _addObstacles(self):
        self.OBSTACLE_IDS = []
        for obst in self.NAV_OBSTACLES:
            pos = obst["pos"]
            radius = obst["radius"]
            height = obst["height"]
            axis = obst.get("axis", 2)
            if axis == 0:      # bar along x: rotate the cylinder's z axis onto x
                orn = p.getQuaternionFromEuler([0.0, np.pi / 2, 0.0])
            elif axis == 1:    # bar along y
                orn = p.getQuaternionFromEuler([np.pi / 2, 0.0, 0.0])
            else:              # vertical pillar
                orn = [0.0, 0.0, 0.0, 1.0]
            color = [0.85, 0.1, 0.1, 1.0] if axis == 2 else [0.95, 0.5, 0.1, 1.0]
            visual_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                             rgbaColor=color,
                                             physicsClientId=self.CLIENT)
            collision_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height,
                                                   physicsClientId=self.CLIENT)
            body_id = p.createMultiBody(baseMass=0,
                                         baseCollisionShapeIndex=collision_id,
                                         baseVisualShapeIndex=visual_id,
                                         basePosition=pos,
                                         baseOrientation=orn,
                                         physicsClientId=self.CLIENT)
            self.OBSTACLE_IDS.append(body_id)

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
        self._prev_phi = self._phi(self._prev_dist_to_target)
        self._collided = False
        self._reached_goal = False
        return obs, info

    #  Nondeterministic transition: random wind disturbance
    def step(self, action):
        if self.WIND_STD > 0:
            drone_id = self.DRONE_IDS[0]
            wind_force = np.random.normal(0.0, self.WIND_STD, size=3)
            wind_force[2] *= 0.3
            p.applyExternalForce(objectUniqueId=drone_id,
                                  linkIndex=-1,
                                  forceObj=wind_force.tolist(),
                                  posObj=[0, 0, 0],
                                  flags=p.LINK_FRAME,
                                  physicsClientId=self.CLIENT)
        obs, reward, terminated, truncated, info = super().step(action)
        if truncated and not terminated:
            reward -= self.TRUNC_PENALTY_BASE + self.TRUNC_PENALTY_PER_M * info["dist_to_target"]
        return obs, reward, terminated, truncated, info

    #  Helpers
    @staticmethod
    def _surface_dist(pos, obst):
        # Distance from pos to the surface of a finite cylinder (0 if inside).
        # axis 2 = vertical pillar (default), axis 0 / 1 = horizontal bar along x / y.
        center = np.array(obst["pos"])
        axis = obst.get("axis", 2)
        radial_idx = [i for i in range(3) if i != axis]
        radial = np.linalg.norm(np.asarray(pos)[radial_idx] - center[radial_idx])
        along = abs(pos[axis] - center[axis])
        dr = max(radial - obst["radius"], 0.0)
        da = max(along - obst["height"] / 2.0, 0.0)
        return float(np.hypot(dr, da))

    def _obstacle_infos(self, pos):
        # List of (distance_to_surface, vector_to_center) for every obstacle, nearest first.
        infos = []
        for obst in self.NAV_OBSTACLES:
            center = np.array(obst["pos"])
            infos.append((self._surface_dist(pos, obst) - self.DRONE_RADIUS, center - pos))
        infos.sort(key=lambda t: t[0])
        return infos

    def _nearest_obstacle_info(self, pos):
        infos = self._obstacle_infos(pos)
        if not infos:
            return np.inf, np.zeros(3)
        return infos[0]

    #  Observation space: kinematic state + goal-relative + K nearest obstacles
    def _observationSpace(self):
        base_space = super()._observationSpace()
        n_extra = 3 + 4 * self.NUM_OBS_TRACKED
        extra_low = np.array([[-np.inf] * n_extra])
        extra_high = np.array([[np.inf] * n_extra])
        low = np.hstack([base_space.low, extra_low])
        high = np.hstack([base_space.high, extra_high])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        base_obs = super()._computeObs()
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rel_goal = self.TARGET_POS - pos
        infos = self._obstacle_infos(pos)[:self.NUM_OBS_TRACKED]
        parts = [rel_goal]
        for dist, vec in infos:
            parts.append(np.hstack([vec, [dist]]))
        for _ in range(self.NUM_OBS_TRACKED - len(infos)):
            parts.append(np.array([0.0, 0.0, 0.0, 10.0]))   # "no obstacle" placeholder: far away
        extra = np.hstack(parts).astype('float32').reshape(1, -1)
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

        progress = self._prev_dist_to_target - dist_to_target
        reward = 10.0 * progress
        self._prev_dist_to_target = dist_to_target
        phi_new = self._phi(dist_to_target)
        reward += 0.99 * phi_new - self._prev_phi
        self._prev_phi = phi_new

        reward -= 0.01

        if obst_dist < 0.3:
            reward -= self.PROXIMITY_PENALTY * (0.3 - obst_dist) / 0.3

        if obst_dist < 0.0:
            self._collided = True
            reward -= self.COLLISION_PENALTY

        if pos[2] > self.CEILING - 0.2:
            reward -= 0.5

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
            "is_success": self._reached_goal,
            "dist_to_target": self._prev_dist_to_target,
        }
