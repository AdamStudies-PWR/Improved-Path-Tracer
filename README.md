# ImprovedPathTracer
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

## Memory profiling

In seprate terminal window run ```nvidia-smi```:
```
nvidia-smi --query-compute-apps=timestamp,name,used_memory --format=csv -lms 500 > tracer_data.csv
```

* [--query-compute-apps=timestamp,name,used_memory] - This specifies what kind of data is logged. In this example:
    * timestamp - At what time measurment was taken.
    * name - Name of program.
    * used_memory - How much memory was consumed by program.
* [--format=csv] - Specifies format used for logged data.
* [-lms 500] - This specifies the time interval in which measurments are taken. In thiss example measurments are taken 500 miliseconds.
* [> tracer_data.csv] - This saves output of program to file, in this example "tracer_data.csv".

## Requirements

ImageMagic needs to be installed from source using: https://softcreatr.github.io/imei/

## External libraries used
1. *Lohmann, N. (2022). JSON for Modern C++ (Version 3.11.2) [Computer software]. https://github.com/nlohmann*
2. *https://imagemagick.org/Magick++/ (2023), Magic++ ImageMagic 7 (7.1.1-24) [Computer software]. https://github.com/ImageMagick/ImageMagick/releases/tag/7.1.1-24*
