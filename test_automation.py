import argparse
import os
import resource
import subprocess

from multiprocessing import Process


EXECUTABLE = "./tracer"
BENCHMARK_FILE = "benchmark.txt"

DEPTHS = [10]
SAMPLES = [40, 80, 200, 400, 1000, 2000, 5000]
SCENES = ["scenes/maze.json", "scenes/mirrors.json", "scenes/spheres.json"]


parser = argparse.ArgumentParser(
    prog="TraceBench",
    description="Benchamrk tool and memory profiler for tracer program.")
parser.add_argument('-o', '--one', action='store_false', help='Enable execution of single test case.')
parser.add_argument('-s', '--samples', default="40", help="Specifies number of samples per pixel.")
parser.add_argument('-d', '--depth', default="10", help="Specifies max number of reflections per ray.")
parser.add_argument('-p', '--path', default="scenes/spheres.json", help="Specifies path to json file with scene data.")


def to_mebibytes(kibibytes) -> str:
    return str(round(kibibytes/1024, 2))


def test(scene, depth, sample):
    print("\nStarting: " + scene + " Depth=" + str(depth) + " Samples=" + str(sample))
    subprocess.check_call([EXECUTABLE + " -d="+str(depth) + " -s="+str(sample) + " " + scene], shell=True)
    maxrss = (resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    mib_used = to_mebibytes(maxrss)
    print("Memory used: " + mib_used + " MiB")
    with open(BENCHMARK_FILE, 'a') as file:
        file.write(mib_used + "\n")


def test_many():
    for scene in SCENES:
        for depth in DEPTHS:
            for sample in SAMPLES:
                p = Process(target=test, args=(scene, depth, sample))
                p.start()
                p.join()


def main() -> int:
    if not os.path.exists("tracer"):
        print("Executable not found")
        return 1

    if os.path.exists(BENCHMARK_FILE):
        os.remove(BENCHMARK_FILE)

    args = parser.parse_args()
    if not args.one:
        test(args.path, args.depth, args.samples)
    else:
        test_many()

    return 0


if __name__ == '__main__':
    exit(main())
