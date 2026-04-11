# carla-dvs-sim2real

Generate labeled YOLO training data from a simulated event camera (DVS) in
CARLA, with Sim2Real noise injection so the trained model works on real hardware.

## What this does

Event cameras output pixel-level brightness **changes** instead of full frames.
This makes them fast and power-efficient, but standard RGB-trained models don't
work on their output directly.

This project bridges that gap:

1. **Emulates the sensor output** — CARLA's DVS camera is configured to match
   the GenX320's resolution, contrast thresholds, and EHC (Event Histogram
   Contrast) output format used by OpenMV.
2. **Auto-labels from simulation** — CARLA knows the exact position of every
   actor, so bounding box labels are generated automatically without manual
   annotation.
3. **Injects Sim2Real noise** — Background activity noise and hot pixels are
   added so the model generalizes to real sensor data (based on the v2e model).

## Scripts

| File | Description |
|---|---|
| `live_viewer.py` | Real-time RGB vs DVS side-by-side viewer with a randomly spawned vehicle. Adjust framerate, contrast, brightness, and noise interactively. |
| `dataset_generator.py` | Runs 11 predefined traffic scenarios and saves a complete YOLO-format dataset with labels and `data.yaml`. |

## Requirements

- **CARLA 0.9.x** — server must be running on `localhost:2000`
- Python 3.8+

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install the CARLA Python API manually from your CARLA installation:

```bash
pip install <CARLA_ROOT>/PythonAPI/carla/dist/carla-*.whl
```

## Usage

### Live viewer

```bash
python live_viewer.py
```

Keyboard controls:

| Key | Action |
|---|---|
| `Q` | Quit |
| `+` / `-` | Increase / decrease EHC framerate |
| `C` / `V` | Increase / decrease contrast |
| `B` / `N` | Increase / decrease brightness (neutral value) |
| `X` | Toggle Sim2Real noise |
| `D` | Toggle lens distortion (needs calibration data) |
| `S` | Save screenshot |

### Dataset generation

```bash
python dataset_generator.py
```

Output structure:

```
dvs_dataset/
  images/
    train/   <- clean EHC frames (no overlays)
    val/
  labels/
    train/   <- YOLO .txt label files
    val/
  debug/     <- debug images with colored bounding boxes
  data.yaml  <- ready for YOLO training
```

Train YOLO after collecting data:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='dvs_dataset/data.yaml', epochs=100, imgsz=320)
```

## Scenarios

11 scripted traffic scenarios covering a range of situations:

| # | Scenario | Weather |
|---|---|---|
| 1 | Ego approaches parked car | Clear |
| 2 | Car approaches stationary ego | Clear |
| 3 | Pedestrian crosses road | Clear |
| 4 | Car from the side | Clear |
| 5 | Lead car brakes suddenly | Clear |
| 6 | Night driving, parked car | Night |
| 7 | Rainy conditions | Rain |
| 8 | Multiple obstacles | Clear |
| 9 | Pedestrian at night | Night |
| 10 | Pedestrian walking toward ego | Clear |
| 11 | Row of parked cars | Clear |

## Label quality filters

Each frame passes a filter pipeline before a label is saved:

1. **Distance** — objects farther than 40 m are ignored
2. **Minimum size** — bounding boxes smaller than 10 px on any side are dropped
3. **Truncation** — objects more than 60 % outside the frame are dropped.
   **Exception:** objects closer than 15 m skip this filter. At short range the
   object fills most of the frame and is naturally clipped at the borders — these
   are the critical Time-to-Collision frames and must be kept.
4. **Visibility** — raycasts detect occluding geometry
5. **Event activity** — regions with fewer than 2 % active pixels are skipped.
   A stationary object in DVS produces no events; labeling it would teach the
   model the wrong pattern and cause false positives on real data.

## Sensor parameters

| Parameter | Value |
|---|---|
| Resolution | 320 × 320 (live viewer) / 320 × 240 (dataset generator) |
| Positive threshold | 0.25 |
| Negative threshold | 0.25 |
| Refractory period | 1000 ns |
| EHC framerate | 50 Hz |
| EHC contrast | 16 |
| Neutral value | 128 |

## Classes

| ID | Name |
|---|---|
| 0 | car |
| 1 | pedestrian |

## References

- [CARLA Simulator](https://carla.org/)
- [v2e: From Video Frames to Realistic DVS Events](https://arxiv.org/abs/2006.07722) — noise model
- [Ultralytics YOLO](https://docs.ultralytics.com/)
