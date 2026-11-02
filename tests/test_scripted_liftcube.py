"""Synthetic CPU checks; neither physics nor calibration accuracy is established."""
import importlib.util
from pathlib import Path
import sys
import unittest
import io
import json

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ("numpy", "torch"))
if AVAILABLE:
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "course/liftcube/src"))
    from starter.scripted_liftcube import ScriptedLiftCube, SteadyBiasScriptedLiftCube, red_centroid, pixel_to_base, validate_calibration, rotation_step
    from starter.interfaces.action import degrees_to_motor
    from starter.interfaces.observation import Observation


@unittest.skipUnless(AVAILABLE, "Requires course NumPy/Torch (no simulator)")
class ScriptedLiftCubeTests(unittest.TestCase):
    def calibration(self):
        return dict(schema="so101-front-rgb-rule-v1", evidence="synthetic test only",
            image_shape=[64, 64, 3], K=[[50, 0, 32], [0, 50, 32], [0, 0, 1]],
            camera_to_base=[[1,0,0,0],[0,-1,0,0],[0,0,-1,1],[0,0,0,1]],
            grasp_rotation=np.eye(3).tolist(), joint_sign=[1]*5, joint_offset_deg=[0]*5,
            grasp_xy_offset=[0,0], workspace=[[-.25,-.25,.01],[.25,.25,.3]],
            visible_plane_z=.09, grasp_z=.09, hover_z=.13, lift_z=.17,
            open_motor=100, close_motor=0, joint_names=list("abcde"), tcp_frame="tcp")

    def image(self):
        image = np.zeros((64,64,3), np.uint8)
        image[30:35,30:35,0] = 255
        return image

    def kin(self):
        class Kinematics:
            def forward_kinematics(self, q):
                pose = np.eye(4)
                pose[:3,3] = np.asarray(q[:3])*.001 + [0,0,.15]
                return pose

            def inverse_kinematics(self, q, pose, **kwargs):
                result = q.copy()
                result[:3] = (pose[:3,3]-[0,0,.15])*1000
                return result
        return Kinematics()

    def obs(self, q, image=None):
        # Poison non-permitted fields: a change here must not affect policy output.
        return Observation(self.image() if image is None else image,
            np.r_[q, np.full(15, 999)], metadata={"object_pose": "must never read"})

    def initial(self):
        q = degrees_to_motor(np.array([0,0,30,0,0,0], dtype=float))
        q[5] = 100
        return q

    def test_detection_and_ambiguity(self):
        np.testing.assert_allclose(red_centroid(self.image()), [32,32])
        for image in (np.zeros((64,64,3), np.uint8), np.zeros((32,32,3), np.uint8)):
            with self.assertRaises(ValueError):
                red_centroid(image)
        image = self.image()
        image[5:8,5:8,0] = 255
        with self.assertRaisesRegex(ValueError, "2 components"):
            red_centroid(image)

    def test_configured_resolution_keeps_projection_and_detection_contract(self):
        old=validate_calibration(self.calibration())
        for side in (128,256):
            scale=side/64
            c=self.calibration();c['image_shape']=[side,side,3]
            K=np.array(c['K'],float);K[:2,:2]*=scale
            K[:2,2]=(K[:2,2]+.5)*scale-.5;c['K']=K
            c=validate_calibration(c)
            image=np.repeat(np.repeat(self.image(),int(scale),axis=0),int(scale),axis=1)
            pixel=red_centroid(image,c['image_shape'])
            np.testing.assert_allclose(pixel_to_base(pixel,c),pixel_to_base([32,32],old),atol=1e-8)
            with self.assertRaisesRegex(ValueError,'Expected'):
                red_centroid(image)
            image[:]=0;image[5:7,5:7,0]=255
            with self.assertRaisesRegex(ValueError,'0 components'):
                red_centroid(image,c['image_shape'])

    def test_rotation_steps_are_bounded_and_reach_half_turn(self):
        for target in (np.eye(3), np.diag([-1.,-1.,1.])):
            current = np.eye(3)
            for _ in range(160):
                following, remaining = rotation_step(current, target)
                angle = np.arccos(np.clip((np.trace(current.T@following)-1)/2,-1,1))
                self.assertLessEqual(angle, .020001)
                np.testing.assert_allclose(following.T@following, np.eye(3), atol=1e-6)
                current = following
            np.testing.assert_allclose(current,target,atol=1e-6)

    def test_position_arrival_does_not_skip_orientation(self):
        c=self.calibration()
        c['grasp_rotation']=np.diag([-1.,-1.,1.])
        p=ScriptedLiftCube(c,self.kin())
        q=self.initial()
        p.last=q.copy(); p.phase='descend'; p.target=p._pose(q)[:3,3]
        for _ in range(4):
            p.act(self.obs(q))
        self.assertEqual(p.phase,'descend')

    def test_slew_preserves_joint_direction(self):
        p=ScriptedLiftCube(self.calibration(),self.kin())
        q=self.initial(); p.last=q.copy()
        requested=q.copy(); requested[:2]+=[4,2]
        p._step=lambda image, measured: requested
        sent=p.act(self.obs(q))
        np.testing.assert_allclose(sent[:2]-q[:2],[1,.5],atol=1e-6)

    def test_reference_does_not_follow_sagging_feedback(self):
        p=ScriptedLiftCube(self.calibration(),self.kin())
        commanded=self.initial()
        measured=commanded.copy(); measured[2]-=5
        p.last=commanded.copy(); p.phase='descend'
        p.target=p._pose(commanded)[:3,3].copy()
        before=p.last.copy()
        p.act(self.obs(measured))
        np.testing.assert_allclose(p.last,before,atol=1e-5)
        self.assertEqual(p.phase,'descend')  # Actual position has not arrived.

    def test_feedback_is_not_validated_as_outgoing_target(self):
        p=ScriptedLiftCube(self.calibration(),self.kin())
        command=self.initial(); command[3]=100
        measured=command.copy(); measured[3]+=.0006
        p.last=command.copy(); p.phase='hold'
        sent=p.act(self.obs(measured))
        np.testing.assert_array_equal(sent,command)
        self.assertEqual(p.phase,'hold')
        self.assertGreater(p.telemetry['measured_motor'][3],100)

    def test_projection_and_invalid_calibration(self):
        c = validate_calibration(self.calibration())
        np.testing.assert_allclose(pixel_to_base([32,32], c), [0,0,.09])
        np.testing.assert_allclose(pixel_to_base([33,32], c), [.0182,0,.09])
        for key, value in (("K", np.zeros((3,3))), ("joint_sign", [0]*5),
                           ("grasp_rotation", np.ones((3,3))), ("lift_z", float("nan"))):
            data = self.calibration(); data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_calibration(data)
        c["visible_plane_z"] = 2
        with self.assertRaisesRegex(ValueError, "behind"):
            pixel_to_base([32,32], c)

    def test_lift_uses_separate_rotation_with_bounded_interpolation(self):
        c=self.calibration(); c['lift_rotation']=np.diag([-1.,-1.,1.])
        p=ScriptedLiftCube(c,self.kin()); q=self.initial()
        p.last=q.copy(); p.phase='lift'; p.target=p._pose(q)[:3,3].copy()
        captured=[]
        p._solve=lambda xyz, measured, grip, rotation=None: captured.append(rotation.copy()) or measured.copy()
        p.act(self.obs(q))
        angle=np.arccos(np.clip((np.trace(captured[0])-1)/2,-1,1))
        self.assertAlmostEqual(angle,.02,places=6)
        self.assertEqual(p.phase,'lift')

    def test_steady_bias_is_bounded_and_nominal_reference_does_not_drift(self):
        p=SteadyBiasScriptedLiftCube(self.calibration(),self.kin())
        nominal=self.initial(); p.last=nominal.copy(); p.phase='hover'
        p._step=lambda image,q: nominal.copy()
        bias=np.array([0,1.77,1.57,.28,0,0]); q=nominal+bias
        for _ in range(80):
            sent=p.act(self.obs(q)); q=sent+bias
        np.testing.assert_allclose(p.last,nominal,atol=1e-6)
        np.testing.assert_allclose(q[:5],nominal[:5],atol=.001)
        self.assertLessEqual(np.max(np.abs(p.bias)),3)
        p.reset(); self.assertIsNone(p.sent)
        np.testing.assert_array_equal(p.bias,np.zeros(5))

    def test_invalid_steady_bias_holds_instead_of_clipping(self):
        p=SteadyBiasScriptedLiftCube(self.calibration(),self.kin())
        nominal=self.initial(); nominal[0]=99
        p.last=nominal.copy(); p.phase='hover'; p._step=lambda image,q: nominal.copy()
        q=nominal.copy(); q[0]=97
        for _ in range(20):
            before=p.sent.copy() if p.sent is not None else q.copy()
            sent=p.act(self.obs(q))
            if p.failure_reason:
                np.testing.assert_array_equal(sent,before); break
        self.assertEqual(p.phase,'failed')

    def test_contact_close_uses_jaw_feedback_and_keeps_grasp_target(self):
        c=self.calibration();c['contact_close']=True
        p=SteadyBiasScriptedLiftCube(c,self.kin())
        q=self.initial();q[5]=30
        p.last=q.copy();p.phase='close';p.target=np.array([0,0,c['grasp_z']])
        for _ in range(100):
            before=p.sent.copy() if p.sent is not None else q.copy()
            sent=p.act(self.obs(q))
            self.assertLessEqual(np.max(np.abs(sent-before)),1.00001)
            q=sent.copy();q[5]=max(12,q[5])  # Synthetic obstruction, no force API.
            if p.phase=='lift':break
        self.assertEqual(p.phase,'lift',p.failure_reason)
        self.assertTrue(p.jaw_blocked)
        self.assertAlmostEqual(p.grasp_motor,10)
        p._solve=lambda xyz,measured,grip,rotation=None: np.r_[measured[:5],grip]
        self.assertAlmostEqual(p.act(self.obs(q))[5],10)
        p.reset()
        self.assertFalse(p.jaw_blocked)
        self.assertIsNone(p.previous_jaw)
        self.assertEqual(p.grasp_motor,c['close_motor'])

    def test_contact_close_without_obstruction_can_finish_empty(self):
        c=self.calibration();c['contact_close']=True
        p=ScriptedLiftCube(c,self.kin());q=self.initial();q[5]=30
        p.last=q.copy();p.phase='close';p.target=np.array([0,0,c['grasp_z']])
        for _ in range(100):
            before=q.copy();q=p.act(self.obs(q))
            self.assertLessEqual(np.max(np.abs(q-before)),1.00001)
            if p.phase=='lift':break
        self.assertEqual(p.phase,'lift',p.failure_reason)
        self.assertFalse(p.jaw_blocked)  # Proxy never declares success.
        self.assertEqual(p.grasp_motor,0)
        c['contact_close']='yes'
        with self.assertRaisesRegex(ValueError,'boolean'):
            validate_calibration(c)

    def test_complete_synthetic_sequence_and_reset(self):
        p = ScriptedLiftCube(self.calibration(), self.kin())
        q = self.initial()
        phases = set()
        for _ in range(320):
            old = q.copy()
            q = p.act(self.obs(q))
            self.assertLessEqual(float(np.max(np.abs(q-old))), 1.0001)
            phases.add(p.phase)
            if p.phase == "hold":
                break
        self.assertEqual(p.phase, "hold", p.failure_reason)
        self.assertTrue({"localize","hover","descend","close","lift","hold"} <= phases)
        self.assertNotIn("success", p.telemetry)
        p.reset()
        self.assertEqual(p.phase, "open")
        self.assertIsNone(p.target)
        self.assertIsNone(p.last)

    def test_missing_cube_latches_hold_not_success(self):
        p = ScriptedLiftCube(self.calibration(), self.kin())
        q = self.initial()
        for _ in range(4):
            q = p.act(self.obs(q, np.zeros((64,64,3), np.uint8)))
        self.assertEqual(p.phase, "failed")
        np.testing.assert_array_equal(q, p.act(self.obs(q)))

    def test_privileged_teacher_is_explicit_and_does_not_need_red_pixels(self):
        from starter.privileged_liftcube import PrivilegedLiftCube
        p = PrivilegedLiftCube(self.calibration(), self.kin(), read_cube=lambda: np.array([0, 0, -.01]))
        q = self.initial()
        blank = np.zeros((64, 64, 3), np.uint8)
        for _ in range(320):
            q = p.act(self.obs(q, blank))
            if p.phase == 'hold':
                break
        self.assertEqual(p.phase, 'hold', p.failure_reason)
        self.assertTrue(p.telemetry['privileged_teacher'])
        self.assertNotIn('success', p.telemetry)
        p.reset()
        self.assertIsNone(p.anchor)
        missing = PrivilegedLiftCube(self.calibration(), self.kin())
        for _ in range(6):
            missing.act(self.obs(self.initial(), blank))
        self.assertIn('reader', missing.failure_reason)

    def test_out_of_range_ik_and_lag_rejected(self):
        p = ScriptedLiftCube(self.calibration(), self.kin())
        with self.assertRaises(ValueError):
            p._solve(np.array([.2,0,.15]), self.initial(), 100)
        p.last = self.initial()
        p.last[0] += 10
        p.phase = "hold"
        p.act(self.obs(self.initial()))
        self.assertEqual(p.phase, "failed")
        self.assertIn("lag", p.failure_reason)

    def test_unreachable_ik_does_not_clip(self):
        kin = self.kin()
        kin.inverse_kinematics = lambda q, pose, **kw: q
        p = ScriptedLiftCube(self.calibration(), kin)
        with self.assertRaisesRegex(ValueError, "residual"):
            p._solve(np.array([.05,0,.15]), self.initial(), 100)

    def test_padding_and_metadata_do_not_influence_actions(self):
        a = ScriptedLiftCube(self.calibration(), self.kin())
        b = ScriptedLiftCube(self.calibration(), self.kin())
        q = self.initial()
        for _ in range(12):
            clean = Observation(self.image(), np.r_[q, np.zeros(15)])
            qa, qb = a.act(clean), b.act(self.obs(q))
            np.testing.assert_array_equal(qa, qb)
            q = qa

    def test_low_initial_pose_refuses_motion(self):
        p = ScriptedLiftCube(self.calibration(), self.kin())
        q = degrees_to_motor(np.array([0,0,-90,0,0,0], dtype=float))
        np.testing.assert_array_equal(p.act(self.obs(q)), q)
        self.assertIn("below", p.failure_reason)

    def test_evaluation_hold_native_success_and_terminal_recording(self):
        from starter.scripts.evaluate_scripted import RecordedEnv
        from starter.evaluation.evaluate import run_episode
        from starter.interfaces.action import ActionSpec

        q, image = self.initial(), self.image()
        class Env:
            name = "synthetic"
            action_spec = ActionSpec(frequency_hz=60)
            def reset(self, **kwargs):
                self.n = 0
                return Observation(image.copy(), np.r_[q, np.zeros(15)]), {}
            def step(self, action):
                self.n += 1
                terminal_image = image.copy()
                terminal_image[0,0] = self.n
                return Observation(terminal_image, np.r_[action, np.zeros(15)]), 0, self.n == 3, False, {"success": self.n == 3}
        stream = io.StringIO()
        env = RecordedEnv(Env(), stream)
        env.policy = ScriptedLiftCube(self.calibration(), self.kin())
        rec = run_episode(env, env.policy, seed=17, episode_id=0, deterministic=True, checkpoint=None)
        self.assertEqual(rec.length, 3)  # Stop midway through a two-tick hold.
        self.assertTrue(rec.success)  # Only the environment sets success.
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(rows[1]["sent_motor"], rows[2]["sent_motor"])
        self.assertEqual(len(env.frames), 4)
        self.assertEqual(env.frames[-1][0,0,0], 3)

    def test_calibration_preflight_needs_independent_poses(self):
        from starter.scripts.evaluate_scripted import check_kinematics
        c = self.calibration()
        kin = self.kin()
        c["fk_checks"] = []
        for d in (-10,0,10):
            motor = degrees_to_motor(np.array([d,0,20,0,0,0], float))
            c["fk_checks"].append({"motor": motor.tolist(), "tcp_xyz": [d*.001,0,.17],
                                   "tcp_rotation": np.eye(3).tolist()})
        check_kinematics(ScriptedLiftCube(c, kin))
        c["fk_checks"][0]["tcp_xyz"][0] += .02
        with self.assertRaisesRegex(ValueError, "FK check"):
            check_kinematics(ScriptedLiftCube(c, kin))

    def test_preflight_roundtrip_uses_each_reference_rotation(self):
        from starter.scripts.evaluate_scripted import check_kinematics
        kin=self.kin(); original=kin.forward_kinematics
        def fk(q):
            pose=original(q); a=np.deg2rad(q[0])
            pose[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
            return pose
        kin.forward_kinematics=fk
        kin.inverse_kinematics=lambda q,pose,**kw:q.copy()
        c=self.calibration(); c['fk_checks']=[]
        for d in (10,20,30):
            q=np.array([d,0,20,0,0.]); pose=fk(q)
            c['fk_checks'].append(dict(motor=degrees_to_motor(np.r_[q,0]).tolist(),
                tcp_xyz=pose[:3,3].tolist(),tcp_rotation=pose[:3,:3].tolist()))
        check_kinematics(ScriptedLiftCube(c,kin))

    def test_waypoint_preflight_does_not_infer_phase_from_height(self):
        c=self.calibration(); c['hover_z']=c['lift_z']; c['lift_rotation']=np.diag([-1.,-1.,1.])
        p=ScriptedLiftCube(c,self.kin()); q=self.initial()
        p.last=q.copy(); p.phase='localize'; p.stable=2; p.pixel=np.array([32,32])
        rotations=[]
        p._solve=lambda xyz,measured,grip,rotation=None: rotations.append(rotation) or q.copy()
        p.act(self.obs(q))
        np.testing.assert_array_equal(rotations[0],c['grasp_rotation'])
        np.testing.assert_array_equal(rotations[2],c['lift_rotation'])

    def test_geometric_clearance_is_a_height_subgoal_not_success(self):
        if importlib.util.find_spec('scipy') is None:
            self.skipTest('Geometric mode requires SciPy')
        from starter.scripted_geometry import GeometricLiftCube
        c=self.calibration()
        c.update(kinematics_backend='joint_chain',gravity_compensation=True,
            tcp_offset_m=[0,0,0],grasp_asset_sha256='0'*64,
            grasp_presets=[dict(tilt=0,rotation=np.eye(3).tolist(),tcp_offset_m=[0,0,0])])
        kin=self.kin();kin.asset_sha256='0'*64
        p=GeometricLiftCube(c,kin);q=self.initial()
        p.phase='lift';p.clear_lift=True;p.transport_rotation=np.eye(3)
        p.target=np.array([0,0,c['hover_z']]);p.last=q.copy()
        for _ in range(3):
            sent=p.act(self.obs(q))
            self.assertLessEqual(np.max(np.abs(sent-q)),3.00001)
        self.assertFalse(p.clear_lift)
        self.assertEqual(p.phase,'lift')  # No success input or declaration.
        self.assertEqual(p.target[2],c['lift_z'])
        self.assertIsNone(p.failure_reason)
        p.reset()
        self.assertFalse(p.clear_lift)
        self.assertEqual(p.clearance_stable,0)
        c['grasp_asset_sha256']='1'*64
        with self.assertRaisesRegex(ValueError,'same source asset'):GeometricLiftCube(c,kin)
        c['grasp_asset_sha256']='0'*64
        c['grasp_presets'][0]['rotation'][0][0]=2
        with self.assertRaises(ValueError):GeometricLiftCube(c,kin)

    def test_runtime_calibration_uses_camera_base_only_and_records_reset(self):
        if importlib.util.find_spec('scipy') is None:
            self.skipTest('Runtime calibration requires SciPy')
        import torch
        from types import SimpleNamespace
        from scipy.spatial.transform import Rotation
        from starter.camera_calibration import read_camera_to_base
        from starter.scripts.evaluate_scripted import RecordedEnv
        camera=SimpleNamespace(_view=SimpleNamespace(get_world_poses=lambda:(
            torch.tensor([[.4,.6,.8]],dtype=torch.float64),torch.tensor([[1.,0,0,0]],dtype=torch.float64))))
        base_R=Rotation.from_euler('z',90,degrees=True).as_matrix()
        quat=Rotation.from_matrix(base_R).as_quat()[[3,0,1,2]]
        robot=SimpleNamespace(body_names=['base'],data=SimpleNamespace(
            body_pos_w=torch.tensor([[[.1,.2,.3]]],dtype=torch.float64),
            body_quat_w=torch.tensor(quat.reshape(1,1,4),dtype=torch.float64)))
        class Scene:
            def __getitem__(self,key):
                if key=='front':return camera
                if key=='robot':return robot
                raise AssertionError('Non-calibration scene access')
        obs=self.obs(self.initial())
        env=SimpleNamespace(name='synthetic',action_spec=ScriptedLiftCube.action_spec,
            _unwrapped=SimpleNamespace(scene=Scene()),reset=lambda **kwargs:(obs,{}))
        T=read_camera_to_base(env)
        np.testing.assert_allclose(T[:3,:3],base_R.T@np.diag([1.,-1.,-1.]),atol=1e-12)
        np.testing.assert_allclose(T[:3,3],base_R.T@np.array([.3,.4,.5]),atol=1e-12)
        stream=io.StringIO();recorder=RecordedEnv(env,stream,runtime_camera_calibration=True)
        recorder.policy=ScriptedLiftCube(self.calibration(),self.kin())
        recorder.reset(seed=17)
        np.testing.assert_array_equal(recorder.policy.c['camera_to_base'],T)
        self.assertEqual(json.loads(stream.getvalue())['calibration']['source'],'runtime-camera-base-api')

    def test_plan_while_opening_preserves_jaw_motion_and_has_visual_fallback(self):
        from starter.scripted_geometry import GeometricLiftCube
        c=self.calibration()
        c.update(kinematics_backend='joint_chain',gravity_compensation=True,tcp_offset_m=[0,0,0],
            grasp_asset_sha256='0'*64,grasp_presets=[dict(tilt=0,rotation=np.eye(3).tolist(),tcp_offset_m=[0,0,0])])
        def make(enabled):
            config=dict(c,plan_while_opening=enabled);kin=self.kin();kin.asset_sha256='0'*64
            p=GeometricLiftCube(config,kin)
            p.select_pose=lambda point,q:None  # No IK is needed to verify opening.
            return p
        for image,expected in ((self.image(),'hover'),(np.zeros((64,64,3),np.uint8),'localize')):
            old,new=make(False),make(True);q=self.initial();q[5]=0
            for _ in range(120):
                obs=self.obs(q,image);a,b=old.act(obs),new.act(obs)
                np.testing.assert_array_equal(a,b)
                q=a
                if old.phase!='open':break
            self.assertEqual(old.phase,'localize');self.assertEqual(new.phase,expected)
            self.assertIsNone(new.failure_reason)
            if expected=='localize':
                held=new.act(self.obs(q,image))
                self.assertEqual(new.phase,'failed')
                np.testing.assert_array_equal(held,q)
        invalid=dict(c,plan_while_opening='yes')
        with self.assertRaisesRegex(ValueError,'must be boolean'):validate_calibration(invalid)
        invalid=dict(c,plan_while_opening=True,grasp_presets=[])
        with self.assertRaisesRegex(ValueError,'requires geometric'):validate_calibration(invalid)


if __name__ == "__main__":
    unittest.main()
