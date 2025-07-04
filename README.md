# Hand Gesture Volume Control

A real-time volume control application that uses hand tracking to adjust system volume through finger distance gestures.

## Features

- **Gesture-based Volume Control**: Control system volume by adjusting the distance between thumb and index finger
- **Real-time Hand Tracking**: Uses MediaPipe for accurate hand landmark detection
- **Volume Locking**: Prevents accidental volume changes when gestures are held steady
- **Visual Feedback**: Live volume bar display and hand landmark visualization
- **Gesture Reset**: Make a fist to unlock volume controls

## Requirements

```
opencv-python
mediapipe
numpy
pycaw
comtypes
```

## Installation

1. Install required dependencies:
```bash
pip install opencv-python mediapipe numpy pycaw comtypes
```

2. Run the application:
```bash
python main.py
```

## Usage

### Basic Controls
- **Volume Adjustment**: Position your hand in front of the camera and adjust the distance between your thumb and index finger
  - Closer fingers = Lower volume
  - Farther apart = Higher volume
- **Volume Lock**: Hold your gesture steady for 2 seconds to lock the current volume level
- **Unlock**: Make a fist (bring thumb and pinky together) to unlock volume controls
- **Exit**: Press 'q' to quit the application

### Visual Interface
- White circles indicate thumb and index finger positions
- Colored line shows the distance between fingers
- Volume bar on the left displays current volume level
- Volume percentage shown in the top-left corner

## Technical Implementation

### Hand Tracking
- Uses MediaPipe Hands solution for real-time hand detection
- Processes RGB video frames at standard webcam resolution
- Tracks 21 hand landmarks with sub-pixel accuracy

### Volume Control
- Integrates with Windows Core Audio API through pycaw
- Maps finger distance (20-150 pixels) to volume range (0-100%)
- Applies volume changes in real-time with minimal latency

### Gesture Recognition
- **Distance Calculation**: Uses 3D Euclidean distance between thumb and index finger landmarks
- **Volume Locking**: Implements a threshold-based locking mechanism to prevent unintended changes
- **Fist Detection**: Measures distance between thumb and pinky for gesture reset

### Lock Mechanism
- Activates when volume remains within ±10% threshold for 2 seconds
- Also triggers at volume extremes (≤1% or ≥99%)
- Prevents accidental volume adjustments during steady gestures

## System Requirements

- **Platform**: Windows (uses Windows Core Audio API)
- **Hardware**: Webcam with decent lighting conditions
- **Performance**: Runs at standard webcam frame rates (typically 30 FPS)

## Limitations

- Windows-only compatibility due to pycaw dependency
- Requires adequate lighting for reliable hand detection
- Single-hand tracking (processes first detected hand)
- May experience occasional tracking loss in poor lighting conditions

## Configuration

Key parameters can be adjusted in the source code:
- `lock_threshold`: Volume stability threshold (default: 10%)
- `lock_duration`: Time required to trigger lock (default: 2 seconds)
- Distance mapping range: Currently set to 20-150 pixels

## Troubleshooting

- **No volume response**: Ensure your hand is clearly visible and well-lit
- **Erratic behavior**: Check for consistent hand positioning within camera frame
- **Permission errors**: Run with administrator privileges if audio control fails
- **Performance issues**: Close other camera applications and ensure adequate system resources
