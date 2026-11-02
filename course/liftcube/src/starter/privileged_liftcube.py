"""Explicit object-truth teacher; shares the visual controller's servo guards.

Development candidate, not a visual baseline or an accepted demonstration source.
The caller supplies a cube-in-base reader; object truth never enters Observation.
"""
import numpy as np

from .scripted_liftcube import SteadyBiasScriptedLiftCube, _array


def read_cube_in_base(env):
    from scipy.spatial.transform import Rotation
    raw = env._unwrapped
    robot = raw.scene['robot']
    index = list(robot.body_names).index('base')
    position = robot.data.body_pos_w[0, index].detach().cpu().numpy()
    quat = robot.data.body_quat_w[0, index].detach().cpu().numpy()
    cube = raw.scene['cube'].data.root_pos_w[0].detach().cpu().numpy()
    return Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix().T @ (cube-position)


class PrivilegedLiftCube(SteadyBiasScriptedLiftCube):
    def __init__(self, calibration, kinematics, read_cube=None):
        if calibration.get('grasp_presets') or calibration.get('plan_while_opening'):
            raise ValueError('Teacher uses calibrated fixed waypoints, not visual grasp presets')
        self.read_cube = read_cube
        super().__init__(calibration, kinematics)

    def reset(self):
        super().reset()
        self.anchor = None

    def _point(self):
        if not callable(self.read_cube):
            raise ValueError('Explicit privileged cube reader is required')
        point = _array(self.read_cube(), (3,), 'privileged cube in base')
        # The cube center may be below the TCP workspace's safe lower height.
        # Actual TCP waypoint heights still pass the inherited _solve guard.
        if np.any(point[:2] < self.c['workspace'][0][:2]) or np.any(point[:2] > self.c['workspace'][1][:2]):
            raise ValueError('Privileged target outside calibrated XY workspace')
        return point

    def _localize(self, image):
        point = self._point()
        self.stable = self.stable+1 if self.anchor is not None and np.linalg.norm(point-self.anchor) <= .003 else 1
        self.anchor = point.copy()
        return point if self.stable >= 3 else None

    def _check_target(self, image):
        if self.anchor is None or np.linalg.norm(self._point()-self.anchor) > .005:
            raise ValueError('Privileged target moved before descent')

    def act(self, observation, deterministic=False):
        action = super().act(observation, deterministic)
        self.telemetry.update(privileged_teacher=True,
            cube_base=None if self.anchor is None else self.anchor.tolist())
        return action
