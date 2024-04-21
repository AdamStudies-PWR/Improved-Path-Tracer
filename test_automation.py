import os
import subprocess

EXECUTABLE = "./tracer"

DEPTHS = [10]
# SAMPLES = [40, 80, 200, 400, 1000, 2000, 5000]
SAMPLES = [40]
SCENES = ["scenes/maze.json", "scenes/mirrors.json", "scenes/spheres.json"]

if not os.path.exists("tracer"):
    print("Executable not found")
    exit(1)


for scene in SCENES:
    for depth in DEPTHS:
        for sample in SAMPLES:
            print("\nStarting: " + scene + " Depth=" + str(depth) + " Samples=" + str(sample))
            subprocess.run([EXECUTABLE + " -d="+str(depth) + " -s="+str(sample) + " " + scene], shell=True, check=True)
