from humanoid.se.sim.simulation import Simulator, MODEL_PATH, NPZ_PATH

if __name__ == "__main__":
    sim = Simulator(xml_path=MODEL_PATH, npz_path=NPZ_PATH, show_viewer=True)
    sim.run()
