import time
import numpy as np
import mujoco
import mujoco.viewer

from typing import Optional

from humanoid.se.iekf.dynamics import IEKFDynamics, IMUMeasurement


class Simulator:
    def __init__(
        self,
        xml_path: str,
        npz_path: Optional[str] = None,
        duration: Optional[float] = None,
        dt: Optional[float] = 0.002,
        show_viewer: Optional[bool] = False,
    ):  
        self.dynamics = IEKFDynamics()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt
        self.dt = dt
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

        self.measurement_data = {
            "actual_body_pos": [],
            "actual_body_vel": [],
            "actual_body_ort": [],
            "est_body_pos": [],
            "est_body_vel": [],
            "est_body_ort": []
        }

    def init_state(self):
        return self.dynamics.make_state(
            R=self.dynamics.quart_to_rot(self.data.qpos[3:7]),
            v=self.data.qvel[:3],
            p=self.data.qpos[:3]
        )
    
    def estimate_state(self, X: np.array, imu_gyro: np.array, imu_acc: np.array, dt: float, gyro_bias: np.array, accel_bias: np.array):
        imu = IMUMeasurement(
            gyro=imu_gyro,
            accel=imu_acc,
        )

        return self.dynamics.propagate_state(
            X=X,
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            imu=imu,
            dt=dt
        )


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

    def run(self):
        """Run the full simulation duration."""
        steps = self.motion_data["joint_pos"].shape[0]

        # Initialize the setup
        X = self.init_state()
        # TODO: will add measurement noise
        gyro_bias = np.zeros(3)
        accel_bias = np.zeros(3)
        dt = 0.05 # The motion data is 50 frames per second I believe this is the equivalent sampling rate
        
        if self.show_viewer:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = 1  # Base body of the robot
                viewer.cam.distance = 3.0  # Optional: adjust zoom distance
                viewer.cam.lookat[2] = 1.0  # Optional: look slightly higher

                for i in range(steps):
                    # Measure everything else
                    X = self.estimate_state(
                        X=X,
                        imu_acc=self.data.sensor("imu_acc").data,
                        imu_gyro=self.data.sensor("imu_gyro").data,
                        gyro_bias=gyro_bias,
                        accel_bias=accel_bias,
                        dt=dt,
                    )
                    self.step(i)
                    R, v, p, dl, dr = self.dynamics.unpack_state(X)

                    # Now save estimated state from IEKF and mujoco for comparison
                    self.measurement_data["actual_body_pos"].append(self.data.qpos[:3])
                    self.measurement_data["actual_body_vel"].append(self.data.qvel[:3])
                    self.measurement_data["actual_body_ort"].append(self.dynamics.quart_to_rot(self.data.qpos[3:7]))

                    self.measurement_data["est_body_pos"].append(p)
                    self.measurement_data["est_body_vel"].append(v)
                    self.measurement_data["est_body_ort"].append(R)

                    if viewer.is_running():
                        viewer.sync()

                    time.sleep(0.02)

                for key in self.measurement_data.keys():
                    self.measurement_data[key] = np.stack(self.measurement_data[key])

                np.savez("dancing.npz", **self.measurement_data)

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
