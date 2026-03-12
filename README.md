# wisc-humanoid-se-iekf
Mujoco implementation of the IEKF

## Libraries
1. (Pinnochio)[https://stack-of-tasks.github.io/pinocchio/]
2. (Mujoco)[https://mujoco.readthedocs.io/en/stable/overview.html]
3. Numpy

## Tips on setting up reproducibility
1. Setup pyenv for your work environment, this way you isolate your python version and packages
2. Install ruff. Messy code is already half of the trouble, use ruff to format everything
3. We need to containerize our application/research. In this case, just have Docker setup but please learn how to write a dockerfile to run your simulation

## Goal
Make a basic simulation with the IEKF and unitree G1 URDF.
This requires several things:
1. Simulation backend: Giving control inputs and inject measurement noise to the "sensors"
2. Forward kinematics via Pinnochio
3. Writing IEKF via Numpy
4. Putting it all together and plotting the groundtruth walking vs IEKF
