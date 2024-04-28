import os
import resource
import subprocess

from multiprocessing import Process


EXECUTABLE = "./tracer"
BENCHMARK_FILE = "benchmark.txt"

DEPTHS = [10]
SAMPLES = [40, 80, 200, 400, 1000, 2000, 5000]
SCENES = ["scenes/maze.json", "scenes/mirrors.json", "scenes/spheres.json"]


def to_mebibytes(kibibytes) -> str:
    return str(round(kibibytes/1024, 2))


def test(scene, depth, sample):
    print("\nStarting: " + scene + " Depth=" + str(depth) + " Samples=" + str(sample))
    subprocess.check_call([EXECUTABLE + " -d="+str(depth) + " -s="+str(sample) + " " + scene], shell=True)
    max_mem = (resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    with open(BENCHMARK_FILE, 'a') as file:
        file.write(to_mebibytes(max_mem) + "\n")


def main():
    if not os.path.exists("tracer"):
        print("Executable not found")
        exit(1)

    if os.path.exists(BENCHMARK_FILE):
        os.remove(BENCHMARK_FILE)

    for scene in SCENES:
        for depth in DEPTHS:
            for sample in SAMPLES:
                p = Process(target=test, args=(scene, depth, sample))
                p.start()
                p.join()


if __name__ == '__main__':
    main()
