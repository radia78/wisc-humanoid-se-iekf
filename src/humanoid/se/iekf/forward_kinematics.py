import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description


class G1ForwardKinematics:
    """Pinocchio-based forward kinematics for the Unitree G1."""

    LEFT_FOOT_FRAME = "left_ankle_roll_link"
    RIGHT_FOOT_FRAME = "right_ankle_roll_link"

    def __init__(self):
        robot = load_robot_description("g1_description")
        self.model = robot.model
        self.data = robot.data

        self.left_foot_id = self.model.getFrameId(self.LEFT_FOOT_FRAME)
        self.right_foot_id = self.model.getFrameId(self.RIGHT_FOOT_FRAME)

        if self.left_foot_id >= self.model.nframes:
            raise RuntimeError(f"Frame '{self.LEFT_FOOT_FRAME}' not found")
        if self.right_foot_id >= self.model.nframes:
            raise RuntimeError(f"Frame '{self.RIGHT_FOOT_FRAME}' not found")

        self.nq = self.model.nq  # configuration dimension
        self.nv = self.model.nv  # velocity dimension (num DOF)

    def compute_foot_positions(self, joint_angles):
        """Compute foot positions in body (pelvis) frame.

        Args:
            joint_angles: (nq,) joint configuration vector.

        Returns:
            dict with 'left' and 'right' foot positions as (3,) arrays.
        """
        q = np.asarray(joint_angles, dtype=float).ravel()
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        left_pos = self.data.oMf[self.left_foot_id].translation.copy()
        right_pos = self.data.oMf[self.right_foot_id].translation.copy()

        return {"left": left_pos, "right": right_pos}

    def compute_foot_jacobian(self, joint_angles, foot="left"):
        """Compute position Jacobian (3 x nv) for a foot frame.

        Args:
            joint_angles: (nq,) joint configuration vector.
            foot: 'left' or 'right'.

        Returns:
            (3, nv) position Jacobian in the local-world-aligned frame.
        """
        q = np.asarray(joint_angles, dtype=float).ravel()
        frame_id = self.left_foot_id if foot == "left" else self.right_foot_id

        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Full 6xnv Jacobian — rows 0:3 are linear velocity, 3:6 are angular
        J_full = pin.getFrameJacobian(
            self.model,
            self.data,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return J_full[:3, :]  # (3, nv)
