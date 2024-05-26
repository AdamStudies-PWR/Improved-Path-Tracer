import argparse
import os
import resource
import subprocess

from multiprocessing import Process, Manager


EXECUTABLE = "./tracer"
BENCHMARK_FILE = "benchmark.txt"

TIMEOUT = 86400

DEPTHS = [10]
SAMPLES = [40, 80, 200, 400, 1000, 2000, 5000, 10000]
SCENES = ["scenes/spheres.json", "scenes/maze.json", "scenes/mirrors.json"]


parser = argparse.ArgumentParser(
    prog="TraceBench",
    description="Benchamrk tool and memory profiler for tracer program.")
parser.add_argument('-o', '--one', action='store_false', help='Enable execution of single test case.')
parser.add_argument('-s', '--samples', default="40", help="Specifies number of samples per pixel.")
parser.add_argument('-d', '--depth', default="10", help="Specifies max number of reflections per ray.")
parser.add_argument('-p', '--path', default="scenes/spheres.json", help="Specifies path to json file with scene data.")


def write_timeout(scene, sample, depth):
    id = scene.split("/")[1].split(".")[0] + "D" + str(depth) + "S" + str(sample)
    line = id + ";DNF;DNF;DNF"
    with open(BENCHMARK_FILE, 'a') as file:
        file.write(line + "\n")


def to_mebibytes(kibibytes) -> str:
    return str(round(kibibytes/1024, 2))


def test(scene, depth, sample, result):
    print("Starting: " + scene + " Depth=" + str(depth) + " Samples=" + str(sample))
    result[0] = True

    proc = subprocess.Popen([EXECUTABLE + " -d="+str(depth) + " -s="+str(sample) + " " + scene], shell=True)
    try:
        proc.wait(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        print("\nTimeout! Skipping further execution for scene/depth combination.\n")
        result[0] = False

    maxrss = (resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    result[1] = to_mebibytes(maxrss)


def test_many():
    too_long: bool
    for scene in SCENES:
        for depth in DEPTHS:
            too_long = False
            for sample in SAMPLES:
                if not too_long:
                    manager = Manager()
                    result = manager.dict()
                    p = Process(target=test, args=(scene, depth, sample, result))
                    p.start()
                    p.join()
                    if not result[0]:
                        write_timeout(scene, sample, depth)
                        too_long = True
                        continue
                    mib_used = result[1]
                    print("CPU Memory used: " + mib_used + " MiB\n")
                    with open(BENCHMARK_FILE, 'a') as file:
                        file.write(mib_used + "\n")
                else:
                    write_timeout(scene, sample, depth)


def main() -> int:
    if not os.path.exists("tracer"):
        print("Executable not found")
        return 1

    if os.path.exists(BENCHMARK_FILE):
        os.remove(BENCHMARK_FILE)

    args = parser.parse_args()
    if not args.one:
        manager = Manager()
        result = manager.dict()
        test(args.path, args.depth, args.samples, result)
        if not result[0]:
            write_timeout(args.path, args.samples, args.depth)
            return
        mib_used = result[1]
        print("CPU Memory used: " + mib_used + " MiB")
        with open(BENCHMARK_FILE, 'a') as file:
            file.write(mib_used + "\n")
    else:
        test_many()

    return 0


if __name__ == '__main__':
    exit(main())
