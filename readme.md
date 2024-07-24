# Hailo8L on Raspberry Pi 5 in C++

This is a minimal example of running YOLOv8 inference on a Raspberry Pi 5 with
a Hailo8L TPU, using the C++ API.

The files in here are sourced/adapted from https://github.com/hailo-ai/Hailo-Application-Code-Examples
and https://github.com/raspberrypi/rpicam-apps

The OS I've used here is Raspberry Pi OS. I initially tried Ubuntu 24.04, but `raspi-config` there seems to be
too old, preventing one from configuring the latest Raspberry Pi 5 firmware, which is necessary for support
of the Hailo8L TPU.

## Instructions

1. `sudo apt install hailofw hailort build-essentials`

2. `curl -o yolov8s.hef https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8l/yolov8s.hef`

3. `g++ -o yolohailo yolov8.cpp -lhailort && ./yolohailo`

Expected output:

```
infer_model N inputs: 1
infer_model N outputs: 1
infer_model inputstream[0]: 'yolov8s/input_layer1' (height: 640, width: 640, features: 3) (hailo_format = type: UINT8, order: NHWC, flags: 0), frame_size: 1228800 bytes
infer_model outputstream[0]: 'yolov8s/yolov8_nms_postprocess' (height: 80, width: 100, features: 0) (hailo_format = type: FLOAT32, order: HAILO NMS, flags: 0), frame_size: 160320 bytes
input_name: yolov8s/input_layer1
input_frame_size: 1228800
Output tensor yolov8s/yolov8_nms_postprocess, 160320 bytes, shape (80, 100, 0)
  Quantization scale: 1.000000 offset: 0.000000
Output shape: 80, 100
class: 0, confidence: 0.93, 98,87 - 244,520
class: 0, confidence: 0.90, 248,114 - 403,524
class: 16, confidence: 0.93, 453,346 - 596,543
class: 26, confidence: 0.84, 301,186 - 385,341
SUCCESS
```

### Other YOLOv8 Models

You can change the model filename at the top of yolov8.cpp from `yolov8s.hef` to whatever you like.
And you can find a list of all available models here:

https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8l_object_detection.rst

### Measuring FPS / Batch Size

The example inside [advanced/yolov8-fps.cpp](./advanced/yolov8-fps.cpp) measures the FPS achievable
by serially running the model, waiting for results, and running again (i.e. no model parallelism),
at batch size 8.

In order to compile this example, you'll need to be running version 4.18 or later of the Hailo runtime.

The following forum post shows how to install 4.18 on a Raspberry Pi 5. Hopefully this will soon
be as simple as an "apt upgrade".

https://community.hailo.ai/t/still-unable-to-run-4-18-on-rpi5/1985/14?u=rogojin
