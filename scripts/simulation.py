import time
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
import mujoco.viewer

from typing import Optional
from robot_descriptions.loaders.mujoco import load_robot_description

from iekf.kinematics import G1ForwardKinematics
from iekf.right_invariant import RIEKF
from iekf.utils.types import RobotState, IMUMeasurement

import os

class Simulator:
    def __init__(
        self,
        npz_path: Optional[str] = None,
        duration: Optional[float] = None,
        show_viewer: Optional[bool] = False,
    ):
        self.model = load_robot_description("g1_mj_description")
        self.data = mujoco.MjData(self.model)
        dt = self.model.opt.timestep
        self.show_viewer = show_viewer

        self.session = None

        self.motion_data = None
        if npz_path:
            self.motion_data = np.load(npz_path)
            if duration is None:
                key = "ctrl" if "ctrl" in self.motion_data else "qpos"
                if key in self.motion_data:
                    duration = len(self.motion_data[key]) * dt

        self.duration = duration or 0.0
        self.current_step = 0

        # Instatiate the forward kinematics
        self.left_foot_frame_id = "left_ankle_roll_joint"
        self.right_foot_frame_id = "right_ankle_roll_joint"
        kin = G1ForwardKinematics("g1")
        # Instatiate the IEKF model
        self.iekf = RIEKF(kin, dt)

        self.measurement_data = {
            "actual_body_pos": [],
            "actual_body_vel": [],
            "actual_body_ort": [],
            "est_body_pos": [],
            "est_body_vel": [],
            "est_body_ort": [],
        }

    def _init_states(self):
        state = RobotState()
        rot = Rotation(self.data.qpos[3:7], scalar_first=True)
        np
        state.X = state.make_state(
            R=rot.as_matrix(),
            v=self.data.qvel[:3],
            p=self.data.qpos[:3],
            dl=np.zeros((3,)),
            dr=np.zeros((3,)),
        )
        state.P = np.eye(15) * 0.01**2
        state.P[3:6, 3:6] = np.eye(3) * 0.1**2

        return state

    def step(self, i: int):
        qpos = np.zeros_like(self.data.qpos)
        qvel = np.zeros_like(self.data.qvel)

        qpos[:3] = self.motion_data["body_pos_w"][i, 0, :]
        qpos[3:7] = self.motion_data["body_quat_w"][i, 0, :]
        qpos[7:] = self.motion_data["joint_pos"][i, :]

        qvel[:3] = self.motion_data["body_lin_vel_w"][i, 0, :]
        qvel[3:6] = self.motion_data["body_ang_vel_w"][i, 0, :]
        qvel[6:] = self.motion_data["joint_vel"][i, :]

        self.data.qpos = qpos
        self.data.qvel = qvel

        mujoco.mj_forward(self.model, self.data)

    def estimate_state(self, state: RobotState):
        gyro_noise = np.random.normal(0.0, 0.01, (3,))
        accel_noise = np.random.normal(0.0, 0.1, (3,))
        encoder_noise = np.random.normal(0.0, np.pi / 180.0, (29,))
        imu_data = IMUMeasurement(
            gyro=self.data.sensor("imu-pelvis-angular-velocity").data + gyro_noise,
            accel=self.data.sensor("imu-pelvis-linear-acceleration").data + accel_noise,
        )
        return self.iekf.predict(state, imu_data, self.data.qpos[7:] + encoder_noise)

    def run(self):
        """Run the full simulation duration."""
        steps = self.motion_data["joint_pos"].shape[0]

        # Initialize the setup
        state = self._init_states()

        if self.show_viewer:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = 1  # Base body of the robot
                viewer.cam.distance = 3.0  # Optional: adjust zoom distance
                viewer.cam.lookat[2] = 1.0  # Optional: look slightly higher

                for i in range(steps):
                    # Measure everything else
                    state = self.estimate_state(state)

                    # Now save estimated state from IEKF and mujoco for comparison
                    self.measurement_data["actual_body_pos"].append(self.data.qpos[:3])
                    self.measurement_data["actual_body_vel"].append(self.data.qvel[:3])
                    self.measurement_data["actual_body_ort"].append(self.data.qpos[3:7])

                    R, v, p, _, _ = state.unpack_state()
                    rot = Rotation.from_matrix(R)
                    self.measurement_data["est_body_pos"].append(p)
                    self.measurement_data["est_body_vel"].append(v)
                    self.measurement_data["est_body_ort"].append(
                        rot.as_quat(scalar_first=True)
                    )

                    self.step(i)

                    if viewer.is_running():
                        viewer.sync()

                    time.sleep(0.02)

                for key in self.measurement_data.keys():
                    self.measurement_data[key] = np.stack(self.measurement_data[key])

                if not os.path.exists("results"):
                    os.makedirs("results")
                np.savez("results/dancing.npz", **self.measurement_data)

    def _should_stop(self) -> bool:
        """Check if simulation should terminate (e.g., end of motion data)."""
        if self.motion_data is not None:
            max_steps = len(
                self.motion_data["ctrl"]
                if "ctrl" in self.motion_data
                else self.motion_data["qpos"]
            )
            return self.current_step >= max_steps
        return False
