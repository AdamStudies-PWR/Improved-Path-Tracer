# Improved Path-Tracer
Prgram used for my master thesis.

## Usage

```
tracer -d=10 -s=40 scene.json
```

or

```
tracer --depth=10 --samples=40 scene.json
```

* [-d / --depth] - Specifies number of max reflections per ray. This value is optional. Default value is 10.
* [-s / --samples] - Specifies number of samples per pixel. This value is optional. Default value is 40.
* [scene.json] - Path to json file with scene data. This value is mandatory. Example scenes can be found in *scenes* folder.

## Output

On output two files are created:
* sceneNameDXSY.png - rendered image.
* benchmark.txt - Contains render id (sceneNameDXSY) and information on how long render took. If this file exists new result will be appended to it

## Memory profiling

To automate testing and analyze memory usage use test_automation.py

```
python3 test_automation.py
```

Using argument [-o / --one] will enable execution of single test case.

To specify scene parametres (only works in single test mode) use parametres:
* [-d / --depth] - pases argument to -d argument of main program [Default: 10]
* [-s / --samples] - pases argument to -s argument of main program [Default: 40]
* [-p / --path] - specify which scene should be used for single test [Default: scene/spheres.json]

Results will be appended to the benchmark.txt file.

To mesure GPU memmory usage this program uses nvidia-smi which probes memory every 500 ms.

**Important** Runing test_automation.py will remove existing benchmark.txt file!

## Requirements

ImageMagic needs to be installed from source using: https://softcreatr.github.io/imei/

## External libraries used
1. *Lohmann, N. (2022). JSON for Modern C++ (Version 3.11.2) [Computer software]. https://github.com/nlohmann*
2. *https://imagemagick.org/Magick++/ (2023), Magic++ ImageMagic 7 (7.1.1-24) [Computer software]. https://github.com/ImageMagick/ImageMagick/releases/tag/7.1.1-24*
