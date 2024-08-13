
# Speed Estimation Tutorial

This repository contains a tutorial and implementation for speed estimation using object detection, tracking, and perspective transformation. The code utilizes the YOLOv8 model, ByteTrack algorithm, and OpenCV for processing video inputs and calculating the speed of detected objects in a real-world scenario.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Details of the Implementation](#details-of-the-implementation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to estimate the speed of moving objects in a video by performing object detection, tracking, and perspective transformation. The YOLOv8 model is used for object detection, while the ByteTrack algorithm helps in tracking these objects across frames. Perspective transformation ensures that the calculated speed is accurate by considering the real-world dimensions of the observed area.

## Requirements

- Python 3.8+
- OpenCV
- Numpy
- Supervision
- Ultralytics (for YOLOv8)

## Installation

First, clone the repository:

```bash
git clone https://github.com/bnutfilloyev/Speed-Estimation-tutorial.git
cd Speed-Estimation-tutorial
```

Next, install the required packages:

```bash
pip install -r requirements.txt
```

Make sure you have the YOLOv8 model file (`yolov8x.pt`) in the project directory or modify the code to load a model of your choice.

## Usage

To run the speed estimation, place your video file in the directory and update the file path in the code. Then, execute the script:

```bash
python speed_estimation.py
```

The processed video with speed annotations will be saved as `speed_estimation.mp4`.

## Details of the Implementation

### View Transformation

The view transformation is handled by the `ViewTransformer` class, which maps points from the source video frame to the target real-world coordinates using a perspective transformation matrix.

### Object Detection and Tracking

- **YOLOv8**: Used for object detection in each frame.
- **ByteTrack**: Employed to track the detected objects across frames, providing each object with a unique tracker ID.

### Speed Calculation

The speed is calculated by tracking the movement of each object over time. The vertical movement in the transformed space is measured, and the speed is calculated based on the distance traveled over the time elapsed, then converted into kilometers per hour (km/h).

### Annotations

The detected objects, their traces, bounding boxes, and speed labels are annotated onto the video frames using the Supervision library.

## Contributing

Contributions are welcome! If you have any ideas or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Make sure to add a `requirements.txt` file and any other necessary files before uploading the `README.md` to your repository.