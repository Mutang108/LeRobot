"""Front-RGB rule baseline. Calibration is required; no simulator state is read."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from starter.interfaces.action import ActionSpec, degrees_to_motor, motor_to_degrees, validate_action


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _rotation(value, name):
    result = _array(value, (3, 3), name)
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(result), 1):
        raise ValueError(f"{name} must be a proper rotation")
    return result


def validate_calibration(data):
    """Validate numbers, not the truth of an operator's calibration evidence."""
    c = dict(data)
    if c.get("schema") != "so101-front-rgb-rule-v1" or not c.get("evidence"):
        raise ValueError("Calibration needs schema and evidence reference")
    if c.get("image_shape") not in ([64,64,3], [128,128,3], [256,256,3]):
        raise ValueError("Calibrate the configured square front RGB (64, 128 or 256)")
    c["K"] = _array(c["K"], (3, 3), "K")
    if min(c["K"][0, 0], c["K"][1, 1]) <= 0 or not np.allclose(c["K"][2], [0, 0, 1]):
        raise ValueError("Invalid pinhole intrinsics")
    c["camera_to_base"] = _array(c["camera_to_base"], (4, 4), "camera_to_base")
    _rotation(c["camera_to_base"][:3, :3], "camera rotation")
    if not np.allclose(c["camera_to_base"][3], [0, 0, 0, 1]):
        raise ValueError("Invalid camera transform")
    c["grasp_rotation"] = _rotation(c["grasp_rotation"], "grasp_rotation")
    c["lift_rotation"] = _rotation(c.get("lift_rotation", c["grasp_rotation"]), "lift_rotation")
    if not isinstance(c.get("gravity_compensation", False), bool):
        raise ValueError("gravity_compensation must be boolean")
    if not isinstance(c.get("contact_close", False), bool):
        raise ValueError("contact_close must be boolean")
    if not isinstance(c.get("plan_while_opening", False), bool):
        raise ValueError("plan_while_opening must be boolean")
    if c.get("plan_while_opening") and not c.get("grasp_presets"):
        raise ValueError("plan_while_opening requires geometric grasp_presets")
    c["joint_sign"] = _array(c["joint_sign"], (5,), "joint_sign")
    if not np.isin(c["joint_sign"], [-1, 1]).all():
        raise ValueError("joint_sign must contain only -1/+1")
    c["joint_offset_deg"] = _array(c["joint_offset_deg"], (5,), "joint_offset_deg")
    c["grasp_xy_offset"] = _array(c["grasp_xy_offset"], (2,), "grasp_xy_offset")
    c["workspace"] = _array(c["workspace"], (2, 3), "workspace")
    if not (c["workspace"][0] < c["workspace"][1]).all():
        raise ValueError("Workspace lower bounds must be below upper bounds")
    for name in ("visible_plane_z", "grasp_z", "hover_z", "lift_z", "open_motor", "close_motor"):
        c[name] = float(c[name])
        if not np.isfinite(c[name]):
            raise ValueError(f"Invalid {name}")
    if not (0 <= c["close_motor"] < c["open_motor"] <= 100):
        raise ValueError("Expected 0 <= close_motor < open_motor <= 100")
    if not (c["workspace"][0, 2] <= c["grasp_z"] < c["hover_z"] <= c["lift_z"] <= c["workspace"][1, 2]):
        raise ValueError("Grasp/hover/lift heights must be ordered inside workspace")
    names = c.get("joint_names", [])
    if len(names) != 5 or len(set(names)) != 5 or not all(isinstance(n, str) and n for n in names):
        raise ValueError("Provide five ordered URDF arm joint names")
    if not c.get("tcp_frame"):
        raise ValueError("Provide the calibrated TCP frame")
    return c


def load_calibration(path):
    path = Path(path)
    c = validate_calibration(json.loads(path.read_text(encoding="utf-8")))
    backend = c.get("kinematics_backend", "urdf")
    if backend not in ("urdf", "joint_chain"):
        raise ValueError("Unknown kinematics_backend")
    key = "chain" if backend == "joint_chain" else "urdf"
    source = (path.parent / c[key+"_path"]).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != c[key+"_sha256"]:
        raise ValueError(f"{key} SHA256 mismatch")
    c[key+"_path"] = str(source)
    if backend == "joint_chain":
        c["tcp_offset_m"] = _array(c["tcp_offset_m"], (3,), "tcp_offset_m")
        if not np.array_equal(c["joint_sign"], np.ones(5)) or np.any(c["joint_offset_deg"] != 0):
            raise ValueError("joint_chain uses native USD axes and zero offsets")
    return c


def red_centroid(image, expected_shape=(64,64,3)):
    """One isolated red component; reject ambiguity instead of guessing a target."""
    image = np.asarray(image)
    if image.shape != tuple(expected_shape) or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 front RGB {tuple(expected_shape)}")
    height, width = image.shape[:2]
    r, g, b = image.astype(float).transpose(2, 0, 1)
    mask = (r >= 80) & (r > 1.5 * g) & (r > 1.5 * b) & ((r - np.maximum(g, b)) >= 35)
    components = []
    for y, x in np.argwhere(mask):
        if not mask[y, x]:
            continue
        stack, pixels = [(int(y), int(x))], []
        mask[y, x] = False
        while stack:
            yy, xx = stack.pop()
            pixels.append((xx, yy))
            for ny, nx in ((yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx]:
                    mask[ny, nx] = False
                    stack.append((ny, nx))
        if len(pixels) >= 3 * height * width / 4096:
            points = np.asarray(pixels)
            if (len(points) > height*width/4 or np.any(points[:,0] == 0)
                    or np.any(points[:,0] == width-1) or np.any(points[:,1] == 0)
                    or np.any(points[:,1] == height-1)):
                raise ValueError("Red region is oversized or touches image boundary")
            components.append(points.mean(axis=0))
    if len(components) != 1:
        raise ValueError(f"Expected one red cube, found {len(components)} components")
    return components[0]


def pixel_to_base(pixel, c):
    """Intersect an optical-camera ray with the calibrated visible object plane."""
    ray = c["camera_to_base"][:3, :3] @ np.linalg.solve(c["K"], np.r_[pixel, 1.0])
    origin = c["camera_to_base"][:3, 3]
    if abs(ray[2]) < 1e-8:
        raise ValueError("Camera ray parallel to object plane")
    distance = (c["visible_plane_z"] - origin[2]) / ray[2]
    if distance <= 0:
        raise ValueError("Object plane behind camera")
    point = origin + distance * ray
    point[:2] += c["grasp_xy_offset"]
    return point


def rotation_step(current, target, max_angle=.02):
    """Shortest SO(3) step; includes the otherwise ambiguous half-turn case."""
    relative = current.T @ target
    angle = float(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1)))
    if angle <= max_angle:
        return target.copy(), angle
    if np.pi-angle < 1e-5:
        _, vectors = np.linalg.eigh((relative+relative.T)/2)
        axis = vectors[:, -1]
    else:
        axis = np.array([relative[2,1]-relative[1,2], relative[0,2]-relative[2,0],
                         relative[1,0]-relative[0,1]])
        axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    step = np.eye(3)+np.sin(max_angle)*skew+(1-np.cos(max_angle))*(skew@skew)
    return current @ step, angle


class ScriptedLiftCube:
    """Open -> localize -> hover -> descend -> close -> lift -> hold.

    Failure latches a hold of the last command until the native episode ends.
    Holding is not a hardware emergency stop. No object pose or success input.
    """
    action_spec = ActionSpec(frequency_hz=30.0)

    def __init__(self, calibration, kinematics):
        self.c = validate_calibration(calibration)
        self.kin = kinematics
        self.reset()

    def reset(self):
        self.phase = "open"
        self.failure_reason = None
        self.last = None
        self.target = None
        self.pixel = None
        self.stable = self.ticks = self.settled = 0
        self.telemetry = {}
        self.grasp_motor = self.c["close_motor"]
        self.previous_jaw = None
        self.contact_stable = 0
        self.jaw_blocked = False

    def _urdf(self, motor):
        return motor_to_degrees(motor)[:5] * self.c["joint_sign"] + self.c["joint_offset_deg"]

    def _pose(self, motor):
        pose = _array(self.kin.forward_kinematics(self._urdf(motor)), (4, 4), "FK")
        return pose

    def _inside(self, xyz):
        if np.any(xyz < self.c["workspace"][0]) or np.any(xyz > self.c["workspace"][1]):
            raise ValueError("TCP target outside calibrated workspace")

    def _solve(self, xyz, measured, grip, rotation=None):
        self._inside(xyz)
        pose = np.eye(4)
        if rotation is None:
            rotation = self.c["lift_rotation"] if xyz[2] >= self.c["lift_z"]-.0001 else self.c["grasp_rotation"]
        pose[:3, :3] = rotation
        pose[:3, 3] = xyz
        q = self._urdf(measured)
        # Repeated calls also work with LeRobot v0.4.2's one-iteration solver.
        for _ in range(20):
            q = _array(self.kin.inverse_kinematics(q, pose, orientation_weight=1.0), (5,), "IK")
            actual = _array(self.kin.forward_kinematics(q), (4, 4), "FK")
            angle = np.arccos(np.clip((np.trace(actual[:3, :3].T @ pose[:3, :3])-1)/2, -1, 1))
            if np.linalg.norm(actual[:3, 3] - xyz) <= .003 and angle <= .15:
                break
        else:
            raise ValueError("IK residual exceeds 3mm / 0.15rad")
        degrees = motor_to_degrees(measured).copy()
        degrees[:5] = (q - self.c["joint_offset_deg"]) / self.c["joint_sign"]
        result = degrees_to_motor(degrees)
        result[5] = grip
        return validate_action(result, self.action_spec, clip=False)

    def _transition(self, phase):
        self.phase, self.ticks, self.settled = phase, 0, 0

    def act(self, observation, deterministic=False):
        # Deliberately do not inspect metadata, padded dq/EE, or previous labels.
        measured = _array(observation.proprio[:6], (6,), "measured motor positions")
        # Sensor feedback is not a requested target: contact/solver compliance
        # may put it slightly beyond a stop. Keep raw feedback; validate every
        # outgoing target below and retain measured-state tracking protection.
        if self.last is None:
            self.last = measured.copy()
        requested = self.last.copy()
        try:
            if self.failure_reason is None:
                requested = self._step(observation.image, measured)
        except ValueError as exc:
            self.failure_reason = str(exc)
            self.phase = "failed"
        # 30 motor-units/s; cap command lead to 8 units against actual feedback.
        delta = requested - self.last
        # One common fraction preserves the IK joint-space direction.
        sent = self.last + delta * min(1.0, 1.0 / max(float(np.max(np.abs(delta))), 1e-9))
        if self.failure_reason is None and np.any(np.abs(sent[:5] - measured[:5]) > 8):
            self.failure_reason, self.phase = "Arm tracking lag exceeds 8 motor units", "failed"
            sent = self.last.copy()
        self.last = validate_action(sent, self.action_spec, clip=False).copy()
        self.telemetry = {"phase": self.phase, "failure_reason": self.failure_reason,
                          "pixel": None if self.pixel is None else self.pixel.tolist(),
                          "target_xyz": None if self.target is None else self.target.tolist(),
                          "requested_motor": requested.tolist(), "command_motor": self.last.tolist(),
                          "measured_motor": measured.tolist(), "jaw_contact_proxy": self.jaw_blocked,
                          "grasp_motor": self.grasp_motor}
        return self.last.copy()

    def _localize(self, image):
        pixel = red_centroid(image, self.c["image_shape"])
        scale = self.c["image_shape"][0]/64
        self.stable = self.stable + 1 if self.pixel is not None and np.linalg.norm(pixel-self.pixel) <= 2*scale else 1
        self.pixel = pixel
        return pixel_to_base(pixel, self.c) if self.stable >= 3 else None

    def _check_target(self, image):
        pixel = red_centroid(image, self.c["image_shape"])
        if np.linalg.norm(pixel-self.pixel) > 3*self.c["image_shape"][0]/64:
            raise ValueError("Red target shifted before descent; reacquire next episode")

    def _step(self, image, q):
        self.ticks += 1
        c = self.c
        if self.phase not in ("hold", "failed") and self.ticks > 150:
            raise ValueError("Stage timeout (5 policy seconds)")
        goal = self.last.copy()
        if self.phase == "open":
            pose = self._pose(q)
            self._inside(pose[:3, 3])
            if pose[2, 3] < c["hover_z"] - .003:
                raise ValueError("Reset TCP is below safe hover height; no blind recovery motion")
            goal[5] = c["open_motor"]
            self.settled = self.settled + 1 if abs(q[5]-goal[5]) < 2 else 0
            if self.settled >= 3:
                self._transition("localize")
        elif self.phase == "localize":
            xyz = self._localize(image)
            if xyz is not None:
                # Fail before descending if any complete waypoint is unreachable.
                for z, rotation in ((c["hover_z"], c["grasp_rotation"]),
                                    (c["grasp_z"], c["grasp_rotation"]),
                                    (c["lift_z"], c["lift_rotation"])):
                    self._solve(np.r_[xyz[:2], z], q, c["open_motor"], rotation=rotation)
                self.target = np.r_[xyz[:2], c["hover_z"]]
                self._transition("hover")
        elif self.phase in ("hover", "descend", "lift"):
            if self.phase == "hover":
                self._check_target(image)
            current_pose = self._pose(q)
            current = current_pose[:3, 3]
            self._inside(current)
            delta = self.target - current
            # Advance the reference from the last sent target, not lagging
            # feedback: otherwise gravity sag repeatedly drags the path down.
            reference = self._pose(self.last)
            reference_delta = self.target - reference[:3, 3]
            intermediate = reference[:3, 3] + reference_delta * min(
                1.0, .004 / max(np.linalg.norm(reference_delta), 1e-9))
            grip = self.grasp_motor if self.phase == "lift" else c["open_motor"]
            target_rotation = c["lift_rotation"] if self.phase == "lift" else c["grasp_rotation"]
            orientation, _ = rotation_step(reference[:3, :3], target_rotation)
            _, remaining_angle = rotation_step(current_pose[:3, :3], target_rotation)
            goal = self._solve(intermediate, self.last, grip, rotation=orientation)
            reached = (np.linalg.norm(delta) <= .003 and remaining_angle <= .15
                       and np.max(np.abs(goal[:5]-q[:5])) <= 2)
            self.settled = self.settled + 1 if reached else 0
            if self.settled >= 3:
                if self.phase == "hover":
                    self.target[2] = c["grasp_z"]
                    self._transition("descend")
                elif self.phase == "descend":
                    self._transition("close")
                else:
                    self._transition("hold")  # Not a declaration of grasp success.
        elif self.phase == "close":
            goal[5] = c["close_motor"]
            if c.get("contact_close", False):
                if not self.jaw_blocked:
                    blocked = (self.previous_jaw is not None and abs(q[5]-self.previous_jaw) < .1
                               and q[5]-self.last[5] >= 1.5)
                    self.contact_stable = self.contact_stable+1 if blocked else 0
                    self.previous_jaw = float(q[5])
                    # Position-feedback proxy, not a force/contact sensor. Limit
                    # jaw lead instead of driving harder against an obstruction.
                    goal[5] = max(c["close_motor"], float(q[5])-2.)
                    if self.contact_stable >= 3:
                        self.grasp_motor = float(self.last[5])
                        self.jaw_blocked = True
                if self.jaw_blocked:
                    goal[5] = self.grasp_motor
            # Contact can prevent reaching the commanded jaw position. Dwell only
            # after full-close or the optional feedback proxy, not stage entry.
            ready = (self.jaw_blocked or
                     abs(self.last[5]-c["close_motor"]) < 1e-4)
            self.settled = self.settled + 1 if ready else 0
            if self.settled >= 15:
                self.target[2] = c["lift_z"]
                self._transition("lift")
        return goal


class SteadyBiasScriptedLiftCube(ScriptedLiftCube):
    """Optional bounded steady-state correction; only RGB/q feedback is used.

    Nominal path and sent target stay separate. Integrate only at a stationary
    reference, never on the jaw; bounds, slew and tracking guards still apply.
    """
    def reset(self):
        super().reset()
        self.bias = np.zeros(5)
        self.previous_nominal = self.sent = None
        self.stationary = 0

    def act(self, observation, deterministic=False):
        measured = np.asarray(observation.proprio[:6], dtype=float)
        if self.sent is None:
            self.sent = measured.copy()
        nominal = super().act(observation, deterministic=deterministic)
        if self.failure_reason is None:
            stable = (self.previous_nominal is not None
                      and np.max(np.abs(nominal[:5]-self.previous_nominal[:5])) <= .02
                      and self.phase in ('hover', 'descend', 'lift'))
            self.stationary = self.stationary+1 if stable else 0
            if self.stationary >= 3:
                self.bias = np.clip(self.bias+.1*(nominal[:5]-measured[:5]), -3, 3)
            corrected = nominal.copy()
            corrected[:5] += self.bias
            try:
                validate_action(corrected, self.action_spec, clip=False)
                delta = corrected-self.sent
                proposed = self.sent+delta*min(1., 1./max(float(np.max(np.abs(delta))), 1e-9))
                if np.any(np.abs(proposed[:5]-measured[:5]) > 8):
                    raise ValueError('Compensated target tracking lag exceeds 8 motor units')
                self.sent = validate_action(proposed, self.action_spec, clip=False).copy()
            except ValueError as exc:
                self.failure_reason, self.phase = str(exc), 'failed'
        self.previous_nominal = nominal.copy()
        self.telemetry.update(phase=self.phase, failure_reason=self.failure_reason,
            nominal_motor=nominal.tolist(), steady_bias_motor=self.bias.tolist(),
            command_motor=self.sent.tolist())
        return self.sent.copy()
