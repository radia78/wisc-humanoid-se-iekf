import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description


class ForwardKinematics:
    def __init__(self, robot_name):
        robot = load_robot_description(f"{robot_name}_description")
        self.model = robot.model
        self.data = robot.data

        self.left_foot_id = self.model.getFrameId(self.LEFT_FOOT_FRAME)
        self.right_foot_id = self.model.getFrameId(self.LEFT_FOOT_FRAME)

        self.nq = self.model.nq  # configuration dimension
        self.nv = self.model.nv  # velocity dimension (num DOF)

    def compute_foot_positions(self, q):
        # Push the kinematics forward
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        left_pos = self.data.oMf[self.left_foot_id].translation.copy()
        right_pos = self.data.oMf[self.right_foot_id].translation.copy()

        # Get the Jacobians
        pin.computeJointJacobians(self.model, self.data, q)
        J_l = pin.getFrameJacobian(
            self.model,
            self.data,
            self.left_foot_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3, :]

        J_r = pin.getFrameJacobian(
            self.model,
            self.data,
            self.right_foot_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3, :]

        return (left_pos, J_l, right_pos, J_r)


class G1ForwardKinematics(ForwardKinematics):
    LEFT_FOOT_FRAME = "left_ankle_roll_link"
    RIGHT_FOOT_FRAME = "right_ankle_roll_link"
