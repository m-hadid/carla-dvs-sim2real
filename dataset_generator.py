"""
CARLA DVS Multi-Scenario Dataset Generator
===========================================
Generates a labeled YOLO dataset from CARLA simulation using a DVS
(Dynamic Vision Sensor) camera emulating the GenX320 event camera.

The script runs through a set of predefined traffic scenarios and saves:
  - EHC (Event Histogram Contrast) frames as training images
  - YOLO-format bounding box labels (auto-generated from simulator ground truth)
  - Optional debug images showing the full label filter pipeline
  - data.yaml for direct use with ultralytics YOLO

Label filter pipeline (applied to every frame):
  1. Actor still alive?
  2. Distance to camera < MAX_DIST_M?
  3. 3D bounding box projectable and large enough on screen?
  4. Truncation at image border < MAX_TRUNCATION?
     NOTE: skipped for objects closer than NEAR_DIST_M — these are the
     critical TTC frames where the object fills the frame and must still
     be detected even when partially cut off at the image border.
  5. Visibility check via raycast (no occluding objects)?
  6. Event activity check (enough DVS events in the bounding box region)?

Classes: car (0), pedestrian (1)

Output folder structure:
  dvs_dataset/
    images/train/   <- clean EHC frames (no overlays)
    images/val/
    labels/train/   <- YOLO .txt label files
    labels/val/
    debug/          <- debug images with colored bounding boxes (optional)
    data.yaml       <- ready for YOLO training

Usage:
  Make sure a CARLA server is running on localhost:2000, then:
    python dataset_generator.py
"""

import carla
import cv2
import numpy as np
import random
import os
import time
import threading
import yaml

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR           = 'dvs_dataset'
FRAMES_PER_SCENARIO  = 80
DEBUG_IMAGES         = True   # Save separate debug images with bounding boxes

# ─── Quality Filters ──────────────────────────────────────────────────────────
MAX_DIST_M           = 40.0   # Max label distance (DVS 320px -> ~40m is reasonable)
MIN_BBOX_PX          = 10     # Min side length of a bounding box in pixels
MIN_EVENT_RATIO      = 0.02   # Min fraction of active pixels inside a bbox (2%)
MAX_TRUNCATION       = 0.6    # Max fraction of bbox clipped at image border
NEAR_DIST_M          = 15.0   # Below this distance, truncation filter is skipped.
                               # At 3-15 m the object fills most of the frame —
                               # these are the critical TTC frames and must be kept.

# ─── GenX320 Sensor Parameters ────────────────────────────────────────────────
WIDTH  = 320
HEIGHT = 240

GENX320_DVS_PARAMS = {
    "image_size_x":              str(WIDTH),
    "image_size_y":              str(HEIGHT),
    "fov":                       "70",
    "positive_threshold":        "0.25",
    "negative_threshold":        "0.25",
    "sigma_positive_threshold":  "0.02",
    "sigma_negative_threshold":  "0.02",
    "refractory_period_ns":      "1000",
    "use_log":                   "true",
    "log_eps":                   "0.001",
}

# ─── EHC Parameters ───────────────────────────────────────────────────────────
EHC_CONTRAST    = 16
EHC_BRIGHTNESS  = 128
SIM_HZ          = 50
FIXED_DELTA     = 1.0 / SIM_HZ
EHC_HZ          = 50
TICKS_PER_FRAME = max(1, SIM_HZ // EHC_HZ)

# ─── Sim2Real Noise ───────────────────────────────────────────────────────────
NOISE_ENABLED       = True
BA_NOISE_RATE_HZ    = 0.5
HOT_PIXEL_FRACTION  = 0.001

# ─── YOLO Classes ─────────────────────────────────────────────────────────────
CLASS_NAMES = {0: 'car', 1: 'pedestrian'}
CLASS_CAR        = 0
CLASS_PEDESTRIAN = 1


# ══════════════════════════════════════════════════════════════════════════════
# Thread-safe Event Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ThreadSafeEventBuffer:
    def __init__(self):
        self._lock   = threading.Lock()
        self._chunks = []
        self._dtype  = np.dtype([
            ('x', np.uint16), ('y', np.uint16),
            ('t', np.int64),  ('pol', np.bool_),
        ])

    def append(self, events):
        with self._lock:
            self._chunks.append(events)

    def drain(self):
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=self._dtype)
            combined = np.concatenate(self._chunks)
            self._chunks.clear()
            return combined


# ══════════════════════════════════════════════════════════════════════════════
# Sim2Real Noise
# ══════════════════════════════════════════════════════════════════════════════

def generate_hot_pixel_map(height, width, fraction):
    n = int(height * width * fraction)
    hy = np.random.randint(0, height, n)
    hx = np.random.randint(0, width, n)
    hp = np.random.choice([-1, 1], n, p=[0.3, 0.7])
    return hy, hx, hp

def apply_background_noise(bins, rate_hz, dt_s):
    p = rate_hz * dt_s
    mask = np.random.random(bins.shape) < p
    pol = np.where(np.random.random(bins.shape) < 0.5, 1, -1)
    bins[mask] += pol[mask]
    return bins

def apply_hot_pixels(bins, hy, hx, hp):
    counts = np.random.randint(1, 4, len(hy))
    np.add.at(bins, (hy, hx), hp * counts)
    return bins


# ══════════════════════════════════════════════════════════════════════════════
# GenX320 EHC Renderer
# Returns both the rendered frame and the raw event bins (for activity check)
# ══════════════════════════════════════════════════════════════════════════════

def render_ehc(events, hy, hx, hp, dt_s):
    """
    Returns:
        frame_bgr: (H, W, 3) uint8 — EHC image for YOLO training
        pixel_bins: (H, W) int32 — raw event counts (for activity check)
    """
    bins = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    if len(events) > 0:
        on  = events[events['pol']]
        off = events[~events['pol']]
        if len(on):  np.add.at(bins, (on['y'],  on['x']),   1)
        if len(off): np.add.at(bins, (off['y'], off['x']), -1)

    # Save raw bins before noise (for activity check)
    raw_bins = bins.copy()

    if NOISE_ENABLED:
        bins = apply_background_noise(bins, BA_NOISE_RATE_HZ, dt_s)
        bins = apply_hot_pixels(bins, hy, hx, hp)

    gray = np.clip(EHC_BRIGHTNESS + bins * EHC_CONTRAST, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), raw_bins


# ══════════════════════════════════════════════════════════════════════════════
# Camera Projection + BBox
# ══════════════════════════════════════════════════════════════════════════════

FOV   = float(GENX320_DVS_PARAMS['fov'])
FOCAL = WIDTH / (2.0 * np.tan(np.radians(FOV / 2.0)))
CX, CY = WIDTH / 2.0, HEIGHT / 2.0


def world_to_cam(loc, cam_tf):
    """Transforms a world point into camera coordinate system."""
    w2c = np.linalg.inv(np.array(cam_tf.get_matrix()))
    return w2c @ np.array([loc.x, loc.y, loc.z, 1.0])


def proj(cp):
    """Projects a camera-space point onto the 2D image plane."""
    if cp[0] <= 0.5:  # Must be at least 0.5m in front of camera
        return None
    return (FOCAL * cp[1] / cp[0] + CX,
            FOCAL * (-cp[2]) / cp[0] + CY)


def get_bbox_3d(actor, cam_tf):
    """
    Projects the 3D bounding box of an actor onto the camera image.
    Returns the unclipped box (for truncation check).

    Returns:
        raw_bbox: (x1, y1, x2, y2) unclipped, or None
        clipped_bbox: (x1, y1, x2, y2) clipped to image bounds, or None
    """
    bb = actor.bounding_box
    vt = actor.get_transform()
    e  = bb.extent

    pts_2d = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                loc = carla.Location(
                    x=bb.location.x + sx * e.x,
                    y=bb.location.y + sy * e.y,
                    z=bb.location.z + sz * e.z,
                )
                world_pt = vt.transform(loc)
                cam_pt   = world_to_cam(world_pt, cam_tf)
                px       = proj(cam_pt)
                if px:
                    pts_2d.append(px)

    if len(pts_2d) < 2:
        return None, None

    xs = [p[0] for p in pts_2d]
    ys = [p[1] for p in pts_2d]

    raw_x1, raw_y1 = min(xs), min(ys)
    raw_x2, raw_y2 = max(xs), max(ys)
    raw_bbox = (raw_x1, raw_y1, raw_x2, raw_y2)

    x1 = max(0, int(raw_x1))
    y1 = max(0, int(raw_y1))
    x2 = min(WIDTH - 1,  int(raw_x2))
    y2 = min(HEIGHT - 1, int(raw_y2))

    if x2 - x1 < MIN_BBOX_PX or y2 - y1 < MIN_BBOX_PX:
        return raw_bbox, None

    clipped_bbox = (x1, y1, x2, y2)
    return raw_bbox, clipped_bbox


def check_truncation(raw_bbox, clipped_bbox):
    """
    Measures how much of the bbox is clipped at the image border.
    Returns: truncation ratio (0.0 = fully visible, 1.0 = completely gone)
    """
    rx1, ry1, rx2, ry2 = raw_bbox
    cx1, cy1, cx2, cy2 = clipped_bbox

    raw_area     = max((rx2 - rx1) * (ry2 - ry1), 1e-6)
    clipped_area = (cx2 - cx1) * (cy2 - cy1)

    return 1.0 - (clipped_area / raw_area)


def check_visibility_raycast(world, cam_location, target_location):
    """
    Casts a ray from the camera to the target.
    Checks if anything is in between (wall, another car, etc.).

    Returns:
        True if target is visible (ray hits target or nothing)
        False if something is occluding the target
    """
    direction = target_location - cam_location
    dist = direction.length()
    if dist < 0.1:
        return True

    result = world.cast_ray(cam_location, target_location)

    if not result:
        return True  # Nothing hit = clear line of sight

    first_hit = result[0]
    hit_dist = cam_location.distance(first_hit.location)
    target_dist = cam_location.distance(target_location)

    return hit_dist >= (target_dist - 2.5)


def check_event_activity(raw_bins, bbox, min_ratio):
    """
    DVS-specific check: does the bbox region contain enough events?

    A stationary object in DVS produces NO events (pixel stays at 128).
    Labeling a flat-gray region as 'car' would teach YOLO the wrong pattern,
    causing false positives on real data.

    Args:
        raw_bins: (H, W) int32 — raw event counts (before noise)
        bbox: (x1, y1, x2, y2) clipped bbox
        min_ratio: minimum fraction of active pixels (e.g. 0.02 = 2%)

    Returns:
        True if the region has sufficient event activity
    """
    x1, y1, x2, y2 = bbox
    region = raw_bins[y1:y2, x1:x2]

    if region.size == 0:
        return False

    active_pixels = np.count_nonzero(region)
    total_pixels  = region.size

    return (active_pixels / total_pixels) >= min_ratio


def bbox_to_yolo(bbox):
    x1, y1, x2, y2 = bbox
    xc = ((x1 + x2) / 2.0) / WIDTH
    yc = ((y1 + y2) / 2.0) / HEIGHT
    w  = (x2 - x1) / WIDTH
    h  = (y2 - y1) / HEIGHT
    return xc, yc, w, h


# ══════════════════════════════════════════════════════════════════════════════
# Full Label Filter Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def generate_labels(world, cam_sensor, raw_bins, actors_classes):
    """
    Generates filtered YOLO labels for one frame.

    Filter pipeline (in order):
      1. Actor still alive?
      2. Distance to camera < MAX_DIST_M?
      3. 3D bbox projectable and large enough?
      4. Truncation at border < MAX_TRUNCATION?
      5. Visibility (raycast, no occluding object)?
      6. Event activity (enough DVS events in region)?

    Returns:
        labels: list of (class_id, xc, yc, w, h) — YOLO format
        debug_bboxes: list of (bbox, class_id, reason) — for debug image
    """
    cam_tf  = cam_sensor.get_transform()
    cam_loc = cam_tf.location

    labels       = []
    debug_bboxes = []

    for actor, cls_id in actors_classes:
        if not actor.is_alive:
            continue

        # Filter 1: Distance
        actor_loc = actor.get_location()
        dist = cam_loc.distance(actor_loc)
        if dist > MAX_DIST_M:
            continue

        # Filter 2: BBox projection
        raw_bbox, clipped_bbox = get_bbox_3d(actor, cam_tf)
        if clipped_bbox is None:
            continue

        # Filter 3: Truncation — skipped for nearby objects (TTC critical zone).
        # At < NEAR_DIST_M the object is close, fills most of the frame, and
        # is naturally clipped at the borders. Dropping these frames would
        # remove exactly the data the model needs for collision avoidance.
        # For distant objects the filter still prevents labeling tiny slivers.
        if raw_bbox is not None and dist > NEAR_DIST_M:
            trunc = check_truncation(raw_bbox, clipped_bbox)
            if trunc > MAX_TRUNCATION:
                debug_bboxes.append((clipped_bbox, cls_id, 'TRUNC'))
                continue

        # Filter 4: Visibility (raycast)
        target = carla.Location(
            x=actor_loc.x,
            y=actor_loc.y,
            z=actor_loc.z + actor.bounding_box.extent.z
        )
        visible = check_visibility_raycast(world, cam_loc, target)
        if not visible:
            debug_bboxes.append((clipped_bbox, cls_id, 'OCCL'))
            continue

        # Filter 5: Event activity
        has_events = check_event_activity(raw_bins, clipped_bbox, MIN_EVENT_RATIO)
        if not has_events:
            debug_bboxes.append((clipped_bbox, cls_id, 'NO_EVT'))
            continue

        # All filters passed -> create label
        xc, yc, w, h = bbox_to_yolo(clipped_bbox)
        labels.append((cls_id, xc, yc, w, h))
        debug_bboxes.append((clipped_bbox, cls_id, 'OK'))

    return labels, debug_bboxes


def draw_debug_image(frame, debug_bboxes):
    """
    Creates a SEPARATE debug image with colored bounding boxes.
    NOT saved as a training image!

    Green  = label accepted (OK)
    Yellow = rejected: truncation
    Red    = rejected: occlusion
    Blue   = rejected: no events
    """
    debug = frame.copy()
    colors = {
        'OK':     (0, 255, 0),
        'TRUNC':  (0, 255, 255),
        'OCCL':   (0, 0, 255),
        'NO_EVT': (255, 100, 0),
    }
    for bbox, cls_id, reason in debug_bboxes:
        x1, y1, x2, y2 = bbox
        color = colors.get(reason, (128, 128, 128))
        thickness = 2 if reason == 'OK' else 1
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, thickness)
        label = f"{CLASS_NAMES.get(cls_id, '?')} {reason}"
        cv2.putText(debug, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    return debug


# ══════════════════════════════════════════════════════════════════════════════
# YOLO data.yaml Generator
# ══════════════════════════════════════════════════════════════════════════════

def write_data_yaml(output_dir):
    data = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val':   'images/val',
        'names': CLASS_NAMES,
    }
    path = os.path.join(output_dir, 'data.yaml')
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  data.yaml written: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Dataset Generator
# ══════════════════════════════════════════════════════════════════════════════

class DataGenerator:
    def __init__(self):
        print("Connecting to CARLA...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20)

        print("Loading Town01...")
        self.world = self.client.load_world('Town01')
        time.sleep(2)

        self.bp_lib    = self.world.get_blueprint_library()
        self.spawn_pts = self.world.get_map().get_spawn_points()
        self.car_bps   = [b for b in self.bp_lib.filter('*vehicle*')
                          if b.has_attribute('number_of_wheels')
                          and int(b.get_attribute('number_of_wheels')) == 4]
        self.walker_bps = self.bp_lib.filter('*walker.pedestrian*')

        # Sync + substepping
        self.tm = self.client.get_trafficmanager(8000)
        self.orig_settings = self.world.get_settings()
        s = self.world.get_settings()
        s.synchronous_mode    = True
        s.fixed_delta_seconds = FIXED_DELTA
        s.substepping         = True
        s.max_substep_delta_time = 0.005
        s.max_substeps        = 10
        self.tm.set_synchronous_mode(True)
        self.world.apply_settings(s)

        # Create output folders (YOLO structure)
        for split in ('train', 'val'):
            os.makedirs(f'{OUTPUT_DIR}/images/{split}', exist_ok=True)
            os.makedirs(f'{OUTPUT_DIR}/labels/{split}', exist_ok=True)
        if DEBUG_IMAGES:
            os.makedirs(f'{OUTPUT_DIR}/debug', exist_ok=True)

        # Noise maps (created once)
        self.hot_y, self.hot_x, self.hot_p = generate_hot_pixel_map(
            HEIGHT, WIDTH, HOT_PIXEL_FRACTION)

        # Ego vehicle + DVS camera
        self.ego     = None
        self.dvs_cam = None
        self.buffer  = ThreadSafeEventBuffer()
        self._spawn_ego_and_camera()

        self.scene_actors = []
        self.total_saved  = 0
        self.total_empty  = 0

    def _spawn_ego_and_camera(self):
        ego_bp = self.bp_lib.filter('*model3*')[0]
        self.ego = self.world.spawn_actor(ego_bp, self.spawn_pts[0])
        self.ego.apply_control(carla.VehicleControl(hand_brake=True))

        dvs_bp = self.bp_lib.find('sensor.camera.dvs')
        for k, v in GENX320_DVS_PARAMS.items():
            dvs_bp.set_attribute(k, v)
        cam_tf = carla.Transform(carla.Location(x=2.0, z=2.4))
        self.dvs_cam = self.world.spawn_actor(dvs_bp, cam_tf, attach_to=self.ego)

        def _cb(data):
            if len(data) > 0:
                self.buffer.append(
                    np.frombuffer(data.raw_data, dtype=np.dtype([
                        ('x', np.uint16), ('y', np.uint16),
                        ('t', np.int64),  ('pol', np.bool_),
                    ])).copy()
                )
        self.dvs_cam.listen(_cb)

    # ── Utilities ──────────────────────────────────────────────────────────

    def _teleport_ego(self, spawn_idx=0, transform=None):
        tf = transform if transform else self.spawn_pts[spawn_idx]
        self.ego.set_transform(tf)
        self.ego.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.ego.apply_control(carla.VehicleControl(hand_brake=True))

    def _set_weather(self, preset):
        presets = {
            'clear': carla.WeatherParameters(
                cloudiness=0, precipitation=0, wind_intensity=0,
                sun_altitude_angle=60, fog_density=0, wetness=0),
            'sunset': carla.WeatherParameters(
                cloudiness=15, precipitation=0, wind_intensity=5,
                sun_altitude_angle=15, fog_density=0, wetness=0),
            'night': carla.WeatherParameters(
                cloudiness=20, precipitation=0, wind_intensity=0,
                sun_altitude_angle=-90, fog_density=0, wetness=0),
            'rain': carla.WeatherParameters(
                cloudiness=80, precipitation=80, wind_intensity=10,
                sun_altitude_angle=40, precipitation_deposits=60, wetness=80),
            'fog': carla.WeatherParameters(
                cloudiness=60, precipitation=0, wind_intensity=0,
                sun_altitude_angle=30, fog_density=50, wetness=0),
        }
        self.world.set_weather(presets.get(preset, presets['clear']))

    def _spawn_walker(self, location):
        bp = random.choice(self.walker_bps)
        walker = self.world.try_spawn_actor(
            bp, carla.Transform(location, carla.Rotation()))
        if walker is None:
            return None, None
        ctrl_bp = self.bp_lib.find('controller.ai.walker')
        ctrl = self.world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
        self.world.tick()
        ctrl.start()
        return walker, ctrl

    def _cleanup_scene(self):
        for a in reversed(self.scene_actors):
            try:
                if a.is_alive:
                    if hasattr(a, 'stop'):
                        a.stop()
                    a.destroy()
            except Exception:
                pass
        self.scene_actors.clear()
        for _ in range(5):
            self.world.tick()

    def _warm_up(self, ticks=50):
        """Let the scene settle before recording."""
        self.buffer.drain()
        for _ in range(ticks):
            self.world.tick()
        self.buffer.drain()

    # ── Frame saving ───────────────────────────────────────────────────────

    def _save_frame(self, scenario_name, actors_classes):
        events = self.buffer.drain()
        dt_s = TICKS_PER_FRAME * FIXED_DELTA

        frame, raw_bins = render_ehc(
            events, self.hot_y, self.hot_x, self.hot_p, dt_s)

        labels, debug_bboxes = generate_labels(
            self.world, self.dvs_cam, raw_bins, actors_classes)

        split = 'val' if random.random() < 0.2 else 'train'
        name  = f"{scenario_name}_{self.total_saved:05d}"

        # Save clean training image (NO overlays!)
        cv2.imwrite(f"{OUTPUT_DIR}/images/{split}/{name}.png", frame)

        # Save labels
        label_path = f"{OUTPUT_DIR}/labels/{split}/{name}.txt"
        with open(label_path, 'w') as f:
            for cls_id, xc, yc, w, h in labels:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # Save debug image separately
        if DEBUG_IMAGES:
            debug_img = draw_debug_image(frame, debug_bboxes)
            cv2.imwrite(f"{OUTPUT_DIR}/debug/{name}.png", debug_img)

        self.total_saved += 1
        has_labels = len(labels) > 0
        if not has_labels:
            self.total_empty += 1

        return has_labels

    def _run_collection(self, scenario_name, actors_classes,
                        ego_control_fn=None, npc_control_fn=None,
                        stop_fn=None, max_frames=FRAMES_PER_SCENARIO):
        saved = 0
        tick  = 0
        empty_streak = 0

        while saved < max_frames:
            if ego_control_fn:
                self.ego.apply_control(ego_control_fn(tick))
            if npc_control_fn:
                npc_control_fn(tick)
            self.world.tick()
            tick += 1

            if stop_fn and stop_fn():
                break

            if tick % TICKS_PER_FRAME == 0:
                had_label = self._save_frame(scenario_name, actors_classes)
                if had_label:
                    saved += 1
                    empty_streak = 0
                else:
                    empty_streak += 1

                if empty_streak > 30:
                    print(f"    WARNING: {empty_streak} empty frames in a row, aborting scenario")
                    break

                if saved % 20 == 0 and saved > 0:
                    print(f"    [{saved}/{max_frames}] frames saved")

    # ══════════════════════════════════════════════════════════════════════
    # Scenarios
    # ══════════════════════════════════════════════════════════════════════

    def scenario_ego_to_parked_car(self):
        print("  Scenario 1: Ego approaches parked car")
        self._set_weather('clear')
        self._teleport_ego(0)

        wp = self.world.get_map().get_waypoint(self.spawn_pts[0].location)
        for _ in range(17):
            wp = (wp.next(2.0) or [wp])[0]
        npc = self.world.try_spawn_actor(random.choice(self.car_bps), wp.transform)
        if not npc:
            return
        npc.apply_control(carla.VehicleControl(hand_brake=True))
        self.scene_actors.append(npc)

        self.ego.apply_control(carla.VehicleControl(throttle=0.35))
        ego_fwd = self.ego.get_transform().get_forward_vector()
        self.ego.set_target_velocity(ego_fwd * 8.0)
        self._warm_up(50)

        dist = lambda: self.ego.get_location().distance(npc.get_location())
        self._run_collection(
            'ego_to_parked', [(npc, CLASS_CAR)],
            ego_control_fn=lambda t: carla.VehicleControl(throttle=0.35),
            stop_fn=lambda: dist() < 1.5,  # collect through the full TTC zone (3-5m)
        )

    def scenario_car_to_ego(self):
        print("  Scenario 2: Car approaches stationary ego")
        self._set_weather('clear')
        self._teleport_ego(0)
        self.ego.apply_control(carla.VehicleControl(hand_brake=True))

        wp = self.world.get_map().get_waypoint(self.spawn_pts[0].location)
        for _ in range(18):
            wp = (wp.next(2.0) or [wp])[0]
        rot = wp.transform.rotation
        rot.yaw += 180
        npc = self.world.try_spawn_actor(
            random.choice(self.car_bps),
            carla.Transform(wp.transform.location, rot))
        if not npc:
            return
        self.scene_actors.append(npc)

        npc.apply_control(carla.VehicleControl(throttle=0.4))
        npc_fwd = npc.get_transform().get_forward_vector()
        npc.set_target_velocity(npc_fwd * 8.0)
        self._warm_up(50)

        dist = lambda: self.ego.get_location().distance(npc.get_location())
        self._run_collection(
            'car_to_ego', [(npc, CLASS_CAR)],
            ego_control_fn=lambda t: carla.VehicleControl(hand_brake=True),
            npc_control_fn=lambda t: npc.apply_control(
                carla.VehicleControl(throttle=0.4)),
            stop_fn=lambda: dist() < 1.5,  # collect through the full TTC zone
        )

    def scenario_pedestrian_crossing(self):
        print("  Scenario 3: Pedestrian crosses the road")
        self._set_weather('clear')
        self._teleport_ego(0)

        ego_loc = self.ego.get_location()
        ego_fwd = self.ego.get_transform().get_forward_vector()
        start = carla.Location(
            x=ego_loc.x + ego_fwd.x * 15 + ego_fwd.y * 4,
            y=ego_loc.y + ego_fwd.y * 15 - ego_fwd.x * 4, z=0.5)
        dest = carla.Location(
            x=ego_loc.x + ego_fwd.x * 15 - ego_fwd.y * 6,
            y=ego_loc.y + ego_fwd.y * 15 + ego_fwd.x * 6, z=0.0)

        walker, ctrl = self._spawn_walker(start)
        if not walker:
            return
        ctrl.go_to_location(dest)
        ctrl.set_max_speed(1.5)
        self.scene_actors.extend([walker, ctrl])

        self.ego.apply_control(carla.VehicleControl(throttle=0.25))
        self.ego.set_target_velocity(ego_fwd * 4.0)
        self._warm_up(50)

        self._run_collection(
            'ped_cross', [(walker, CLASS_PEDESTRIAN)],
            ego_control_fn=lambda t: carla.VehicleControl(throttle=0.25),
        )

    def scenario_car_from_side(self):
        print("  Scenario 4: Car approaching from the side")
        self._set_weather('clear')
        self._teleport_ego(0)

        ego_loc = self.ego.get_location()
        ego_fwd = self.ego.get_transform().get_forward_vector()
        ego_yaw = self.ego.get_transform().rotation.yaw

        side_loc = carla.Location(
            x=ego_loc.x + ego_fwd.x * 20 + ego_fwd.y * 10,
            y=ego_loc.y + ego_fwd.y * 20 - ego_fwd.x * 10, z=0.5)
        npc = self.world.try_spawn_actor(
            random.choice(self.car_bps),
            carla.Transform(side_loc, carla.Rotation(yaw=ego_yaw + 90)))
        if not npc:
            return
        self.scene_actors.append(npc)

        self.ego.apply_control(carla.VehicleControl(throttle=0.3))
        self.ego.set_target_velocity(ego_fwd * 6.0)
        npc.apply_control(carla.VehicleControl(throttle=0.4))
        npc.set_target_velocity(npc.get_transform().get_forward_vector() * 8.0)
        self._warm_up(50)

        dist = lambda: self.ego.get_location().distance(npc.get_location())
        self._run_collection(
            'car_side', [(npc, CLASS_CAR)],
            ego_control_fn=lambda t: carla.VehicleControl(throttle=0.3),
            npc_control_fn=lambda t: npc.apply_control(
                carla.VehicleControl(throttle=0.4)),
            stop_fn=lambda: dist() < 3.0,
        )

    def scenario_slow_leader_brake(self):
        print("  Scenario 5: Lead car brakes suddenly")
        self._set_weather('clear')
        self._teleport_ego(0)

        wp = self.world.get_map().get_waypoint(self.spawn_pts[0].location)
        for _ in range(8):
            wp = (wp.next(2.0) or [wp])[0]
        npc = self.world.try_spawn_actor(random.choice(self.car_bps), wp.transform)
        if not npc:
            return
        self.scene_actors.append(npc)

        fwd = self.ego.get_transform().get_forward_vector()
        self.ego.apply_control(carla.VehicleControl(throttle=0.3))
        self.ego.set_target_velocity(fwd * 8.0)
        npc.apply_control(carla.VehicleControl(throttle=0.3))
        npc.set_target_velocity(npc.get_transform().get_forward_vector() * 8.0)
        self._warm_up(50)

        def npc_ctrl(tick):
            if tick < 40:
                npc.apply_control(carla.VehicleControl(throttle=0.3))
            else:
                npc.apply_control(carla.VehicleControl(brake=1.0))

        dist = lambda: self.ego.get_location().distance(npc.get_location())
        self._run_collection(
            'leader_brake', [(npc, CLASS_CAR)],
            ego_control_fn=lambda t: carla.VehicleControl(throttle=0.32),
            npc_control_fn=npc_ctrl,
            stop_fn=lambda: dist() < 2.0,
        )

    def scenario_night_approach(self):
        print("  Scenario 6: Night driving approaching parked car")
        self._set_weather('night')
        self._teleport_ego(0)
        self.ego.set_light_state(carla.VehicleLightState.LowBeam)

        wp = self.world.get_map().get_waypoint(self.spawn_pts[0].location)
        for _ in range(17):
            wp = (wp.next(2.0) or [wp])[0]
        npc = self.world.try_spawn_actor(random.choice(self.car_bps), wp.transform)
        if not npc:
            return
        npc.apply_control(carla.VehicleControl(hand_brake=True))
        self.scene_actors.append(npc)

        self.ego.apply_control(carla.VehicleControl(throttle=0.35))
        self.ego.set_target_velocity(
            self.ego.get_transform().get_forward_vector() * 8.0)
        self._warm_up(50)

        dist = lambda: self.ego.get_location().distance(npc.get_location())
        self._run_collection(
            'night_approach', [(npc, CLASS_CAR)],
            ego_control_fn=lambda t: carla.VehicleControl(throttle=0.35),
            stop_fn=lambda: dist() < 1.5,  # collect through the full TTC zone
        )
        self.ego.set_light_state(carla.VehicleLightState.NONE)

    def scenario_rain_approach(self):
        print("  Scenario 7: Rainy conditions")
        self._set_weather('rain')
        self._teleport_ego(0)

        wp = self.world.get_map().get_waypoint(self.spawn_pts[0].location)
        for _ in range(17):
            wp = (wp.next(2.0) or [wp])[0]
        npc = self.world.try_spawn_actor(random.choice(self.car_bps), wp.transform)
        if not npc:
            return
        npc.apply_control(carla.VehicleControl(hand_brake=True))
        self.scene_actors.append(npc)

        self.ego.apply_control(carla.VehicleControl(throttle=0.35))
        self.ego.set_target_velocity(
            self.ego.get_transform().get_forward_vector() * 8.0)
        self._warm_up(50)

        dist = lambda: self.ego.get_location().distance(npc.get_location())
        self._run_collection(
            'rain_approach', [(npc, CLASS_CAR)],
            ego_control_fn=lambda t: carla.VehicleControl(throttle=0.35),
            stop_fn=lambda: dist() < 1.5,  # collect through the full TTC zone
        )

    def scenario_multi_obstacle(self):
        print("  Scenario 8: Multiple obstacles")
        self._set_weather('clear')
        self._teleport_ego(0)

        ego_loc = self.ego.get_location()
        ego_fwd = self.ego.get_transform().get_forward_vector()
        npcs = []
        for fwd_d, side_d in [(20, 0), (35, 2), (30, -2)]:
            loc = carla.Location(
                x=ego_loc.x + ego_fwd.x * fwd_d + ego_fwd.y * side_d,
                y=ego_loc.y + ego_fwd.y * fwd_d - ego_fwd.x * side_d, z=0.5)
            wp = self.world.get_map().get_waypoint(loc)
            npc = self.world.try_spawn_actor(
                random.choice(self.car_bps), wp.transform)
            if npc:
                npc.apply_control(carla.VehicleControl(hand_brake=True))
                self.scene_actors.append(npc)
                npcs.append(npc)
        if not npcs:
            return

        self.ego.apply_control(carla.VehicleControl(throttle=0.3))
        self.ego.set_target_velocity(ego_fwd * 7.0)
        self._warm_up(50)

        actors_classes = [(n, CLASS_CAR) for n in npcs]
        dist = lambda: min(
            self.ego.get_location().distance(n.get_location())
            for n in npcs if n.is_alive)
        self._run_collection(
            'multi_obs', actors_classes,
            ego_control_fn=lambda t: carla.VehicleControl(throttle=0.3),
            stop_fn=lambda: dist() < 1.5,  # collect through the full TTC zone
        )

    def scenario_pedestrian_night(self):
        print("  Scenario 9: Pedestrian at night")  # was 9, unchanged
        self._set_weather('night')
        self._teleport_ego(0)
        self.ego.set_light_state(carla.VehicleLightState.LowBeam)

        ego_loc = self.ego.get_location()
        ego_fwd = self.ego.get_transform().get_forward_vector()
        start = carla.Location(
            x=ego_loc.x + ego_fwd.x * 18 + ego_fwd.y * 3,
            y=ego_loc.y + ego_fwd.y * 18 - ego_fwd.x * 3, z=0.5)
        dest = carla.Location(
            x=ego_loc.x + ego_fwd.x * 18 - ego_fwd.y * 5,
            y=ego_loc.y + ego_fwd.y * 18 + ego_fwd.x * 5, z=0.0)

        walker, ctrl = self._spawn_walker(start)
        if not walker:
            return
        ctrl.go_to_location(dest)
        ctrl.set_max_speed(1.5)
        self.scene_actors.extend([walker, ctrl])

        self.ego.apply_control(carla.VehicleControl(throttle=0.25))
        self.ego.set_target_velocity(ego_fwd * 5.0)
        self._warm_up(50)

        self._run_collection(
            'ped_night', [(walker, CLASS_PEDESTRIAN)],
            ego_control_fn=lambda t: carla.VehicleControl(throttle=0.25),
        )
        self.ego.set_light_state(carla.VehicleLightState.NONE)

    def scenario_pedestrian_toward_ego(self):
        print("  Scenario 10: Pedestrian walking toward ego")
        self._set_weather('clear')
        self._teleport_ego(0)
        self.ego.apply_control(carla.VehicleControl(hand_brake=True))

        ego_loc = self.ego.get_location()
        ego_fwd = self.ego.get_transform().get_forward_vector()
        start = carla.Location(
            x=ego_loc.x + ego_fwd.x * 20,
            y=ego_loc.y + ego_fwd.y * 20, z=0.5)

        walker, ctrl = self._spawn_walker(start)
        if not walker:
            return
        ctrl.go_to_location(ego_loc)
        ctrl.set_max_speed(1.4)
        self.scene_actors.extend([walker, ctrl])
        self._warm_up(50)

        dist = lambda: self.ego.get_location().distance(walker.get_location())
        self._run_collection(
            'ped_toward', [(walker, CLASS_PEDESTRIAN)],
            ego_control_fn=lambda t: carla.VehicleControl(hand_brake=True),
            stop_fn=lambda: dist() < 2.0,
        )

    def scenario_parked_row(self):
        print("  Scenario 11: Row of parked cars")
        self._set_weather('clear')
        self._teleport_ego(0)

        ego_loc = self.ego.get_location()
        ego_fwd = self.ego.get_transform().get_forward_vector()
        npcs = []
        for i in range(4):
            fwd_d, side_d = 10 + i * 8, 3
            loc = carla.Location(
                x=ego_loc.x + ego_fwd.x * fwd_d + ego_fwd.y * side_d,
                y=ego_loc.y + ego_fwd.y * fwd_d - ego_fwd.x * side_d, z=0.5)
            wp = self.world.get_map().get_waypoint(loc)
            npc = self.world.try_spawn_actor(
                random.choice(self.car_bps), wp.transform)
            if npc:
                npc.apply_control(carla.VehicleControl(hand_brake=True))
                self.scene_actors.append(npc)
                npcs.append(npc)
        if not npcs:
            return

        self.ego.apply_control(carla.VehicleControl(throttle=0.35))
        self.ego.set_target_velocity(ego_fwd * 8.0)
        self._warm_up(50)

        self._run_collection(
            'parked_row', [(n, CLASS_CAR) for n in npcs],
            ego_control_fn=lambda t: carla.VehicleControl(throttle=0.35),
        )

    # ── Run all scenarios ──────────────────────────────────────────────────

    def run_all(self):
        scenarios = [
            self.scenario_ego_to_parked_car,
            self.scenario_car_to_ego,
            self.scenario_pedestrian_crossing,
            self.scenario_car_from_side,
            self.scenario_slow_leader_brake,
            self.scenario_night_approach,
            self.scenario_rain_approach,
            self.scenario_multi_obstacle,
            self.scenario_pedestrian_night,
            self.scenario_pedestrian_toward_ego,
            self.scenario_parked_row,
        ]
        print(f"\n{'='*60}")
        print(f"  DVS Dataset Generator — {len(scenarios)} scenarios")  # 11 scenarios
        print(f"  Output: {OUTPUT_DIR}/")
        print(f"  Filters: dist<{MAX_DIST_M}m  bbox>{MIN_BBOX_PX}px"
              f"  events>{MIN_EVENT_RATIO*100:.0f}%  trunc<{MAX_TRUNCATION*100:.0f}%")
        print(f"{'='*60}\n")

        for i, fn in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}]", end=' ')
            try:
                fn()
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._cleanup_scene()

        write_data_yaml(OUTPUT_DIR)

        print(f"\n{'='*60}")
        print(f"  Done!")
        print(f"  Saved:           {self.total_saved} frames")
        print(f"  Without labels:  {self.total_empty}"
              f" ({100*self.total_empty/max(self.total_saved,1):.1f}%)")
        print(f"  Output:          {OUTPUT_DIR}/")
        print(f"{'='*60}")

    def shutdown(self):
        self._cleanup_scene()
        try:
            if self.dvs_cam and self.dvs_cam.is_alive:
                self.dvs_cam.stop()
                self.dvs_cam.destroy()
            if self.ego and self.ego.is_alive:
                self.ego.destroy()
        except Exception:
            pass
        self.orig_settings.synchronous_mode = False
        self.world.apply_settings(self.orig_settings)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    gen = None
    try:
        gen = DataGenerator()
        gen.run_all()
    except KeyboardInterrupt:
        print("\nAborted by user.")
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        if gen:
            gen.shutdown()
        print("Cleanup complete.")
