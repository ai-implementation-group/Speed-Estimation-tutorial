# Speed Estimation Tutorial

This repository provides a comprehensive tutorial and implementation for estimating the speed of moving objects in videos using advanced techniques like object detection, tracking, and perspective transformation. The project leverages the YOLOv8 model, ByteTrack algorithm, and OpenCV to process video inputs and accurately calculate the speed of detected objects in real-world scenarios.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The objective of this project is to estimate the speed of moving objects within a video by integrating object detection, tracking, and perspective transformation. The YOLOv8 model is utilized for object detection, while the ByteTrack algorithm tracks these objects across video frames. Perspective transformation is applied to ensure that speed calculations are grounded in real-world dimensions.

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Supervision
- Ultralytics (for YOLOv8)

## Installation

### 1. Clone the Repository

Begin by cloning the repository to your local machine:

```bash
git clone https://github.com/bnutfilloyev/Speed-Estimation-tutorial.git
cd Speed-Estimation-tutorial
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Prepare Model and Video

Ensure the YOLOv8 model file (`yolov8x.pt`) is present in the project directory. If you prefer a different model, update the code accordingly.

Download the input video from [this link](https://www.youtube.com/watch?v=KBsqQez-O4w) and place it in the project directory.

## Usage

To perform speed estimation, place your video file in the project directory and update the file path in the code if necessary. Then, run the script:

```bash
python speed_estimation.py
```

The script will generate a processed video with speed annotations saved as `output.mp4`.

## Implementation Details

### View Transformation

The `ViewTransformer` class handles the view transformation, mapping points from the source video frame to target real-world coordinates using a perspective transformation matrix. This step is crucial for accurate speed estimation.

### Object Detection and Tracking

- **YOLOv8**: This model is employed for detecting objects in each video frame with high accuracy and efficiency.
- **ByteTrack**: After detection, ByteTrack assigns a unique ID to each object and tracks it across frames, ensuring continuous monitoring of the object’s movement.

### Speed Calculation

Speed is calculated by measuring the vertical movement of each object in the transformed space. The distance traveled by the object over a period is calculated, and the speed is then converted into kilometers per hour (km/h).

### Annotations

Annotations include bounding boxes, traces of object movement, and speed labels. These are added to each video frame using the Supervision library, resulting in a clear and informative output.

## Contributing

Contributions are highly encouraged! If you have any suggestions, ideas, or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. For more details, refer to the [LICENSE](LICENSE) file.
