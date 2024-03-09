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

[-d / --depth] - Specifies number of max reflections per ray. This value is optional. Default value is 10.
<br>[-s / --samples] - Specifies number of samples per pixel. This value is optional. Default value is 40.
<br>[scene.json] - Path to json file with scene data. This value is mandatory. Example scenes can be found in *scenes* folder.

## Memory profiling

Use nvidia visual profiler

```
nsys profile --cuda-memory-usage=true ./tracer -d=10 -s=40 scene.json
```

Then open the **report*X*.nsys-rep** in NVIDIA Nsight Compute

Additionally memory usage can be validate by running ```nvidia-smi``` while program is running.

## Requirements

ImageMagic needs to be installed from source using: https://softcreatr.github.io/imei/

## External libraries used
1. *Lohmann, N. (2022). JSON for Modern C++ (Version 3.11.2) [Computer software]. https://github.com/nlohmann*
2. *https://imagemagick.org/Magick++/ (2023), Magic++ ImageMagic 7 (7.1.1-24) [Computer software]. https://github.com/ImageMagick/ImageMagick/releases/tag/7.1.1-24*
