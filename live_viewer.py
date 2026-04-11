"""
Live POV Viewer: RGB + DVS Event Camera (GenX320 Emulation) v3.0
=================================================================
Spawns a random vehicle in CARLA with an RGB and a DVS camera attached.
Renders the DVS output in EHC (Event Histogram Contrast) format,
matching the output of the GenX320 sensor running on OpenMV.

EHC output format:
  - Pixel value 128 = no event (neutral background)
  - Pixel value > 128 = ON-event  (brightness increasing)
  - Pixel value < 128 = OFF-event (brightness decreasing)
  - Contrast multiplier: x16 (sensor.set_contrast)

Three main design packages:
  Package 1 (CRITICAL): Thread-safe event buffering
    - threading.Lock() protects the event buffer
    - On render: copy + clear buffer atomically (clean time window)
    - No data loss, no double-processing

  Package 2 (HIGH): Decoupled sim-rate vs. EHC frame rate
    - SIM_HZ = 200 Hz (smooth physics, no teleportation)
    - EHC output = 50 Hz (GenX320 default)
    - Events from 4 physics ticks are accumulated per EHC frame

  Package 3 (MEDIUM): Sim2Real sensor imperfections
    - Background Activity Noise (Poisson process, v2e model)
    - Hot pixels (stuck pixels that always fire)
    - Lens distortion placeholder (cv2.remap, needs calibration data)

Keyboard controls:
  Q           - Quit
  +/-         - EHC frame rate (sensor.set_framerate)
  C/V         - Contrast (sensor.set_contrast)
  B/N         - Brightness / neutral value (sensor.set_brightness)
  X           - Noise ON/OFF
  D           - Distortion ON/OFF (when calibration data is available)
  S           - Screenshot
"""

import carla
import cv2
import numpy as np
import random
import time
import threading
from queue import Queue

# ===========================================================================
# GenX320 Sensor Configuration (CARLA DVS Blueprint)
# ===========================================================================
WIDTH  = 320
HEIGHT = 320

GENX320_DVS_PARAMS = {
    "image_size_x":              str(WIDTH),
    "image_size_y":              str(HEIGHT),
    "fov":                       "70",       # Adjust to match your lens
    "positive_threshold":        "0.25",     # GenX320 nominal: 25%
    "negative_threshold":        "0.25",     # GenX320 nominal: 25%
    "sigma_positive_threshold":  "0.02",     # Pixel-to-pixel manufacturing variation
    "sigma_negative_threshold":  "0.02",
    "refractory_period_ns":      "1000",     # 1 us
    "use_log":                   "true",
    "log_eps":                   "0.001",
}

# ===========================================================================
# GenX320 EHC Output Parameters (OpenMV Defaults)
# ===========================================================================
EHC_FRAMERATE    = 50     # Hz -> 20ms accumulation period
EHC_CONTRAST     = 16     # Event multiplier
EHC_BRIGHTNESS   = 128    # Neutral value (no event)

# ===========================================================================
# Package 2: Simulation Rate
# ===========================================================================
# MODE 1 - LIVE / FAST (default):
#   SIM_HZ = EHC_FRAMERATE (50 Hz). Each tick = 1 EHC frame.
#   No ticks wasted. Runs almost as fast as a plain RGB script.
#   DVS quality: good enough for preview and initial datasets.
#
# MODE 2 - OFFLINE / PRECISE (for final training data):
#   SIM_HZ = 200+. Multiple physics ticks per EHC frame.
#   Smoother event trails, but 4x slower.
#   Change SIM_HZ to 200 when collecting the final dataset.
#
SIM_HZ = 200       # <<< For fast preview: 50, for precise recording: 200
FIXED_DELTA = 1.0 / SIM_HZ

# ===========================================================================
# Package 3: Sim2Real Noise Parameters
# ===========================================================================
NOISE_ENABLED = True

# Background Activity: Poisson rate per pixel per second
# Based on v2e paper (Hu et al., CVPR 2021): default ~1 Hz/pixel
# GenX320 has built-in STC filter, so slightly less than older sensors
BA_NOISE_RATE_HZ = 0.5      # Events per pixel per second

# Hot Pixels: stuck pixels that always fire (manufacturing defects)
# Typically ~0.05-0.1% of all pixels
HOT_PIXEL_FRACTION = 0.001  # 0.1% = ~102 pixels at 320x320

# Lens Distortion (placeholder)
# Activate only after obtaining real camera calibration parameters
DISTORTION_ENABLED = False
# Placeholder values — replace with output from cv2.calibrateCamera()
CAMERA_MATRIX = np.array([
    [250.0,   0.0, 160.0],  # fx, 0, cx
    [  0.0, 250.0, 160.0],  # 0, fy, cy
    [  0.0,   0.0,   1.0],
], dtype=np.float64)
# Distortion Coefficients [k1, k2, p1, p2, k3]
DIST_COEFFS = np.array([-0.15, 0.05, 0.0, 0.0, 0.0], dtype=np.float64)


# ===========================================================================
# Package 1: Thread-safe Event Buffer
# ===========================================================================

class ThreadSafeEventBuffer:
    """
    Thread-safe buffer for DVS events.

    Background (CARLA docs + issues #3653, #7108):
      sensor.listen() callbacks run in CARLA background threads,
      NOT in the main thread. Even in synchronous mode, zero or more
      callbacks may arrive per world.tick().
      CARLA's own sensor_synchronization.py example uses a thread-safe
      Queue for exactly this reason.

    Design:
      - Lock protects all accesses to the internal event list
      - drain() returns ALL collected events AND clears the buffer
        atomically -> each EHC frame gets a clean time window
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._chunks = []

    def append(self, events):
        """Called from the CARLA callback thread."""
        with self._lock:
            self._chunks.append(events)

    def drain(self):
        """
        Returns all collected events and clears the buffer.
        Called from the main thread when an EHC frame is rendered.

        Returns:
            np.ndarray: All events since the last drain(), or empty array
        """
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.dtype([
                    ('x', np.uint16), ('y', np.uint16),
                    ('t', np.int64), ('pol', np.bool_),
                ]))
            combined = np.concatenate(self._chunks)
            self._chunks.clear()
            return combined

    def __len__(self):
        with self._lock:
            return sum(len(c) for c in self._chunks)


# ===========================================================================
# Package 3: Sim2Real Noise Model
# ===========================================================================

def generate_hot_pixel_map(height, width, fraction):
    """
    Creates a fixed map of hot pixels.
    Hot pixels are manufacturing defects that fire in every frame.
    The map is created ONCE and stays constant during the session.
    """
    n_hot = int(height * width * fraction)
    hot_y = np.random.randint(0, height, size=n_hot)
    hot_x = np.random.randint(0, width, size=n_hot)
    # Fixed polarity per hot pixel (mostly ON)
    hot_pol = np.random.choice([-1, 1], size=n_hot, p=[0.3, 0.7])
    return hot_y, hot_x, hot_pol


def apply_background_noise(pixel_bins, ba_rate_hz, frame_duration_s):
    """
    Adds background activity noise (Package 3).

    Model based on v2e (Hu et al., CVPR 2021):
      - Each pixel fires spontaneous events as a Poisson process
      - Rate: ba_rate_hz events per pixel per second
      - In a frame of duration T the probability of at least 1
        noise event: p = 1 - exp(-rate * T)
      - For low rates (rate*T << 1): p ≈ rate * T

    Noise events are uniformly distributed ON/OFF (50/50).
    """
    p_noise = ba_rate_hz * frame_duration_s

    noise_mask = np.random.random((pixel_bins.shape)) < p_noise
    noise_pol = np.where(np.random.random(pixel_bins.shape) < 0.5, 1, -1)
    pixel_bins[noise_mask] += noise_pol[noise_mask]

    return pixel_bins


def apply_hot_pixels(pixel_bins, hot_y, hot_x, hot_pol):
    """Adds hot pixel events. These fire in EVERY frame."""
    counts = np.random.randint(1, 4, size=len(hot_y))
    np.add.at(pixel_bins, (hot_y, hot_x), hot_pol * counts)
    return pixel_bins


def build_distortion_maps(camera_matrix, dist_coeffs, width, height):
    """
    Pre-builds lookup tables for cv2.remap().
    Called ONCE at startup.

    PLACEHOLDER: Replace with values from real camera calibration.

    Calibration steps:
      1. Capture checkerboard images with the real sensor
      2. Run cv2.calibrateCamera()
      3. Replace CAMERA_MATRIX and DIST_COEFFS above
    """
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), alpha=0.0
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_matrix,
        (width, height), cv2.CV_32FC1
    )
    return map_x, map_y


def apply_lens_distortion(frame, map_x, map_y):
    """Applies lens distortion to the frame."""
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=128)


# ===========================================================================
# GenX320 EHC Frame Renderer
# ===========================================================================

def render_genx320_ehc(events, contrast, brightness,
                       noise_enabled, ba_rate_hz, frame_duration_s,
                       hot_y, hot_x, hot_pol,
                       distortion_enabled, dist_map_x, dist_map_y):
    """
    Renders events exactly as the GenX320 does in EHC mode on OpenMV,
    with optional Sim2Real enhancements.

    GenX320 EHC formula (from OpenMV docs):
      1. Start with pixel_bin = 0 for all pixels
      2. ON-event  (pol=True):  pixel_bin += 1
      3. OFF-event (pol=False): pixel_bin -= 1
      4. (Package 3) Add background noise + hot pixels
      5. output = brightness + (pixel_bin * contrast)
      6. Clamp to [0, 255]
      7. (Package 3) Optionally apply lens distortion
    """
    # Step 1: Build event bins
    pixel_bins = np.zeros((HEIGHT, WIDTH), dtype=np.int32)

    if len(events) > 0:
        on_mask = events['pol']
        on_events = events[on_mask]
        if len(on_events) > 0:
            np.add.at(pixel_bins, (on_events['y'], on_events['x']), 1)

        off_events = events[~on_mask]
        if len(off_events) > 0:
            np.add.at(pixel_bins, (off_events['y'], off_events['x']), -1)

    # Step 2 (Package 3): Sim2Real Noise
    if noise_enabled:
        pixel_bins = apply_background_noise(
            pixel_bins, ba_rate_hz, frame_duration_s
        )
        pixel_bins = apply_hot_pixels(pixel_bins, hot_y, hot_x, hot_pol)

    # Step 3: GenX320 EHC formula
    output = brightness + (pixel_bins * contrast)
    gray = np.clip(output, 0, 255).astype(np.uint8)

    # Step 4 (Package 3): Lens distortion
    if distortion_enabled and dist_map_x is not None:
        gray = apply_lens_distortion(gray, dist_map_x, dist_map_y)

    # Grayscale -> 3-channel BGR for OpenCV
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ===========================================================================
# Main
# ===========================================================================

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    world = client.get_world()

    # Synchronous mode + Traffic Manager
    traffic_manager = client.get_trafficmanager(8000)
    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.substepping = True
    settings.max_substep_delta_time = 0.005
    settings.max_substeps = 10
    traffic_manager.set_synchronous_mode(True)
    world.apply_settings(settings)

    actors = []

    # Live parameters (adjustable via keyboard)
    ehc_fps        = EHC_FRAMERATE
    ehc_contrast   = EHC_CONTRAST
    ehc_brightness = EHC_BRIGHTNESS
    noise_on       = NOISE_ENABLED
    distortion_on  = DISTORTION_ENABLED
    frame_count    = 0

    try:
        # --- Spawn vehicle ---
        bp_lib = world.get_blueprint_library()
        model3 = bp_lib.filter('*model3*')
        if len(model3) > 0:
            vehicle_bp = model3[0]
        else:
            vehicle_bp = random.choice(bp_lib.filter('vehicle.*'))
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True, 8000)
        actors.append(vehicle)
        print(f"Vehicle spawned at {spawn_point.location}")

        cam_transform = carla.Transform(carla.Location(x=2.0, z=2.4))

        # --- RGB Camera ---
        rgb_bp = bp_lib.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(WIDTH))
        rgb_bp.set_attribute('image_size_y', str(HEIGHT))
        rgb_bp.set_attribute('fov', GENX320_DVS_PARAMS["fov"])
        rgb_cam = world.spawn_actor(rgb_bp, cam_transform, attach_to=vehicle)
        actors.append(rgb_cam)

        # --- DVS Camera (GenX320 parameters) ---
        dvs_bp = bp_lib.find('sensor.camera.dvs')
        for attr, val in GENX320_DVS_PARAMS.items():
            dvs_bp.set_attribute(attr, val)
        dvs_cam = world.spawn_actor(dvs_bp, cam_transform, attach_to=vehicle)
        actors.append(dvs_cam)

        # Package 1: Thread-safe data structures
        event_buffer = ThreadSafeEventBuffer()
        rgb_lock = threading.Lock()
        rgb_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        def rgb_callback(image):
            nonlocal rgb_frame
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((HEIGHT, WIDTH, 4))
            with rgb_lock:
                rgb_frame = arr[:, :, :3].copy()

        def dvs_callback(data):
            if len(data) > 0:
                events = np.frombuffer(data.raw_data, dtype=np.dtype([
                    ('x', np.uint16),
                    ('y', np.uint16),
                    ('t', np.int64),
                    ('pol', np.bool_),
                ]))
                event_buffer.append(events.copy())

        rgb_cam.listen(rgb_callback)
        dvs_cam.listen(dvs_callback)

        # Package 3: Generate noise maps once at startup
        hot_y, hot_x, hot_pol = generate_hot_pixel_map(
            HEIGHT, WIDTH, HOT_PIXEL_FRACTION
        )

        dist_map_x, dist_map_y = None, None
        if DISTORTION_ENABLED:
            dist_map_x, dist_map_y = build_distortion_maps(
                CAMERA_MATRIX, DIST_COEFFS, WIDTH, HEIGHT
            )

        ticks_per_frame = max(1, SIM_HZ // ehc_fps)

        print("\n" + "=" * 65)
        print("  GenX320 EHC Emulation v3.0 - Live Viewer")
        print("=" * 65)
        print(f"  Resolution:         {WIDTH}x{HEIGHT}")
        print(f"  Threshold:          {GENX320_DVS_PARAMS['positive_threshold']}")
        print(f"  Refractory:         {GENX320_DVS_PARAMS['refractory_period_ns']} ns")
        print(f"  Sim rate:           {SIM_HZ} Hz (physics)")
        print(f"  EHC framerate:      {ehc_fps} Hz ({1000/ehc_fps:.0f}ms)")
        print(f"  Ticks per frame:    {ticks_per_frame}")
        print(f"  EHC contrast:       x{ehc_contrast}")
        print(f"  EHC brightness:     {ehc_brightness}")
        print(f"  Noise:              {'ON' if noise_on else 'OFF'}")
        print(f"    BA rate:          {BA_NOISE_RATE_HZ} Hz/px")
        print(f"    Hot pixels:       {int(HEIGHT*WIDTH*HOT_PIXEL_FRACTION)}")
        print(f"  Lens distortion:    {'ON' if distortion_on else 'OFF (calibration missing)'}")
        print(f"  Thread-safety:      Lock-based buffering")
        print("-" * 65)
        print("  [Q] Quit       [+/-] Framerate  [C/V] Contrast")
        print("  [B/N] Bright.  [X] Noise        [D] Distortion  [S] Save")
        print("=" * 65 + "\n")

        tick = 0

        while True:
            world.tick()
            tick += 1

            # Package 2: Decouple sim rate from EHC rate
            ticks_per_frame = max(1, SIM_HZ // ehc_fps)

            if tick % ticks_per_frame != 0:
                continue

            # Package 1: Atomically drain events
            events = event_buffer.drain()

            frame_duration_s = ticks_per_frame * FIXED_DELTA

            dvs_view = render_genx320_ehc(
                events, ehc_contrast, ehc_brightness,
                noise_on, BA_NOISE_RATE_HZ, frame_duration_s,
                hot_y, hot_x, hot_pol,
                distortion_on, dist_map_x, dist_map_y,
            )

            with rgb_lock:
                rgb_view = rgb_frame.copy()

            n_events = len(events)
            accum_ms = frame_duration_s * 1000
            frame_count += 1

            noise_str = " +NOISE" if noise_on else ""
            dist_str = " +DIST" if distortion_on else ""
            add_overlay(rgb_view, "RGB Camera", "")
            add_overlay(
                dvs_view,
                f"GenX320 EHC {ehc_fps}Hz x{ehc_contrast}{noise_str}{dist_str}",
                f"Events: {n_events:,}  ({accum_ms:.0f}ms)  mid={ehc_brightness}"
            )

            combined = np.hstack([rgb_view, dvs_view])
            cv2.imshow('GenX320 DVS v3.0  |  RGB vs Event Camera', combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('+'), ord('=')):
                ehc_fps = min(ehc_fps + 10, SIM_HZ)
                print(f"  Framerate: {ehc_fps} Hz ({1000/ehc_fps:.0f}ms)")
            elif key == ord('-'):
                ehc_fps = max(ehc_fps - 10, 10)
                print(f"  Framerate: {ehc_fps} Hz ({1000/ehc_fps:.0f}ms)")
            elif key == ord('c'):
                ehc_contrast = min(ehc_contrast + 4, 128)
                print(f"  Contrast: x{ehc_contrast}")
            elif key == ord('v'):
                ehc_contrast = max(ehc_contrast - 4, 1)
                print(f"  Contrast: x{ehc_contrast}")
            elif key == ord('b'):
                ehc_brightness = min(ehc_brightness + 8, 255)
                print(f"  Brightness: {ehc_brightness}")
            elif key == ord('n'):
                ehc_brightness = max(ehc_brightness - 8, 0)
                print(f"  Brightness: {ehc_brightness}")
            elif key == ord('x'):
                noise_on = not noise_on
                print(f"  Noise: {'ON' if noise_on else 'OFF'}")
            elif key == ord('d'):
                if dist_map_x is None:
                    dist_map_x, dist_map_y = build_distortion_maps(
                        CAMERA_MATRIX, DIST_COEFFS, WIDTH, HEIGHT
                    )
                distortion_on = not distortion_on
                print(f"  Distortion: {'ON' if distortion_on else 'OFF'}")
            elif key == ord('s'):
                ts = int(time.time())
                cv2.imwrite(f"genx320_rgb_{ts}.png", rgb_view)
                cv2.imwrite(f"genx320_dvs_{ts}.png", dvs_view)
                print(f"  Screenshots saved: genx320_*_{ts}.png")

    finally:
        print("\nCleaning up...")
        cv2.destroyAllWindows()
        for actor in reversed(actors):
            if actor.is_alive:
                actor.destroy()
        world.apply_settings(original_settings)
        print(f"Done! ({frame_count} frames rendered)")


# ===========================================================================
# Overlay helper
# ===========================================================================

def add_overlay(frame, title, info):
    """Semi-transparent info bar at the top."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, title, (8, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                cv2.LINE_AA)
    if info:
        cv2.putText(frame, info, (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1,
                    cv2.LINE_AA)


# ===========================================================================
if __name__ == "__main__":
    main()
