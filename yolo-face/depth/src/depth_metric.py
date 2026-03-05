"""
Depth Metric microservice — Depth Anything V2 metric monocular depth.

Subscribes to ``frames:raw``, runs the **metric** variant of Depth Anything V2
on each frame, back-projects the depth map into a point cloud using a pinhole
camera model, renders the cloud with PyVista (off-screen, Eye-Dome Lighting,
pixel/square points coloured by metric depth), then publishes the rendered BGR
frame to ``frames:depth`` — exactly like the regular depth service but with true
metric scale and a 3-D perspective render.

Camera intrinsics — pinhole back-projection
-------------------------------------------
For each pixel (u, v) with measured depth z (metres):

    x = (u - c_x) / f_x · z
    y = (v - c_y) / f_y · z
    z = z

where (c_x, c_y) is the principal point (default: frame centre) and
(f_x, f_y) are the focal lengths in pixels.

Because the pixel grid is fixed per-resolution it is computed once and reused
every frame.  The PyVista mesh's point array is updated **in-place** each frame
(``cloud.points[:] = new_pts``), so VTK never reallocates the geometry buffer —
critical for keeping throughput high on the GB10's 273 GB/s memory bandwidth.

Render aesthetic
----------------
* Eye-Dome Lighting (EDL) for deep silhouettes / depth cues.
* Points rendered as filled squares (``render_points_as_spheres=False``).
* ``point_size`` is configurable (default 6 — chunky/pixelated look).
* Scalar colouring by metric Z depth with the ``magma`` colormap.
* Black background.

Environment variables
---------------------
# GPU vs CPU
# The renderer is configured to use the EGL/OpenGL backend so all drawing
# work is submitted to the device.  VTK will still spawn a handful of worker
# threads to prepare commands — seeing many ``vtkRenderThreads`` on ``top``
# is normal and does **not** mean the rendering is occurring on the CPU.
# If you want to reduce CPU-side threading you can limit the number of
# OMP/MKL threads (``OMP_NUM_THREADS=1`` etc.) or set ``VTK_MAX_THREADS``.

  DEPTH_MODEL       Encoder variant.  Options: vits vitb vitl vitg  Default: vits
  DEPTH_DATASET     Metric checkpoint set.  Options: hypersim vkitti  Default: vkitti
  DEPTH_MAX_DEPTH   Maximum metric depth in metres.  Default: 20.0
  FOCAL_LENGTH_X    f_x in pixels.  Default: 470.4
  FOCAL_LENGTH_Y    f_y in pixels.  Default: 470.4
  PRINCIPAL_X       c_x in pixels.  Default: frame_width  / 2
  PRINCIPAL_Y       c_y in pixels.  Default: frame_height / 2
  POINT_SIZE        PyVista point size.  Default: 6
  DEPTH_CMAP        PyVista colormap name.  Default: magma
  RENDER_WIDTH      Output frame width  (0 = match input).  Default: 0
  RENDER_HEIGHT     Output frame height (0 = match input).  Default: 0
  SHOW_FPS          Overlay FPS counter (1/0).  Default: 1
  REDIS_URL         Redis connection string.

Performance tips
----------------
* Lower ``RENDER_WIDTH`` / ``RENDER_HEIGHT`` to reduce the GPU/VTK work; halving
  either dimension roughly quadruples framerate.  These variables are the
  easiest way to trade quality for speed.
* ``EDL_PASSES`` controls the number of eye‑dome lighting passes (default 8).
  Reducing it to 2‑4 shrinks render time at the cost of slightly flatter
  shading.
* The model itself runs on CUDA and is already reasonably fast, but you can
  choose smaller encoders (``vits`` vs ``vitb``) or switch to the ``DEPTH_DATASET``
  checkpoint with smaller resolution.
* The renderer allocates and copies a tensor of depth values every frame;
  this has been optimised in the code below but avoiding high resolutions is
  still the most effective lever for increasing FPS.
* If you only need a coarse point cloud, consider subsampling the input frame
  before publishing or set ``RENDER_WIDTH``/``RENDER_HEIGHT`` accordingly.
"""

import os
import time

import cv2
import numpy as np
import torch

# Force VTK to use the EGL (GPU) headless backend before pyvista is imported.
# Without this the container has no X display and VTK falls back to nothing,
# causing a segfault on the first render() call.
os.environ.setdefault("VTK_DEFAULT_EGL_DEVICE_INDEX", "0")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pyvista as pv

from frame_bus import FrameBus
from logging_config import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEPTH_MODEL     = os.environ.get("DEPTH_MODEL",     "vits")
DEPTH_DATASET   = os.environ.get("DEPTH_DATASET",   "vkitti")
DEPTH_MAX_DEPTH = float(os.environ.get("DEPTH_MAX_DEPTH", "20.0"))
FOCAL_LENGTH_X  = float(os.environ.get("FOCAL_LENGTH_X",  "470.4"))
FOCAL_LENGTH_Y  = float(os.environ.get("FOCAL_LENGTH_Y",  "470.4"))
PRINCIPAL_X     = os.environ.get("PRINCIPAL_X")   # None → frame_width  / 2
PRINCIPAL_Y     = os.environ.get("PRINCIPAL_Y")   # None → frame_height / 2
POINT_SIZE      = int(os.environ.get("POINT_SIZE",     "6"))
DEPTH_CMAP      = os.environ.get("DEPTH_CMAP",      "magma")
RENDER_WIDTH    = int(os.environ.get("RENDER_WIDTH",   "0"))
RENDER_HEIGHT   = int(os.environ.get("RENDER_HEIGHT",  "0"))
SHOW_FPS        = os.environ.get("SHOW_FPS",         "1") == "1"

_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

_HF_REPOS = {
    "vits": {
        "hypersim": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
        "vkitti":   "depth-anything/Depth-Anything-V2-Metric-VKITTI-Small",
    },
    "vitb": {
        "hypersim": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Base",
        "vkitti":   "depth-anything/Depth-Anything-V2-Metric-VKITTI-Base",
    },
    "vitl": {
        "hypersim": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
        "vkitti":   "depth-anything/Depth-Anything-V2-Metric-VKITTI-Large",
    },
    "vitg": {
        "hypersim": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Giant",
        "vkitti":   "depth-anything/Depth-Anything-V2-Metric-VKITTI-Giant",
    },
}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class MetricDepthWrapper:
    """Depth Anything V2 (metric) inference wrapper."""

    def __init__(self, encoder: str, dataset: str, max_depth: float):
        from depth_anything_v2.dpt import DepthAnythingV2
        from huggingface_hub import hf_hub_download

        if encoder not in _MODEL_CONFIGS:
            raise ValueError(f"Unknown encoder '{encoder}'. Choose from: {list(_MODEL_CONFIGS)}")
        if dataset not in ("hypersim", "vkitti"):
            raise ValueError(f"Unknown dataset '{dataset}'. Choose from: hypersim, vkitti")

        log.info(
            f"[DepthMetric] Loading Depth Anything V2 metric: "
            f"encoder={encoder}  dataset={dataset}  max_depth={max_depth} m"
        )
        model = DepthAnythingV2(**_MODEL_CONFIGS[encoder])
        # max_depth is not a constructor arg in the base class — set it as an
        # attribute so that infer_image() uses it for metric scaling.
        model.max_depth = max_depth

        repo_id   = _HF_REPOS[encoder][dataset]
        ckpt_name = f"depth_anything_v2_metric_{dataset}_{encoder}.pth"
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)

        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        self._model = model.to("cuda").eval()
        total_norm = 0.0
        for p in self._model.parameters():
            total_norm += p.data.float().norm().item()
        log.info(f"[DepthMetric] Model ready (param norm {total_norm:.3f})")

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a metric depth map (H×W float32, metres)."""
        log.debug(f"[DepthMetricWrapper] infer called input min/max {frame_bgr.min()}/{frame_bgr.max()} shape {frame_bgr.shape}")
        depth = self._model.infer_image(frame_bgr)  # H×W float32
        log.debug(f"[DepthMetricWrapper] infer output min/max {depth.min()}/{depth.max()} shape {depth.shape} dtype {depth.dtype}")
        return depth


# ---------------------------------------------------------------------------
# PyVista point-cloud renderer
# ---------------------------------------------------------------------------

class PointCloudRenderer:
    """Off-screen PyVista renderer for metric depth point clouds.

    Initialised once the input resolution is known (first frame).  Subsequent
    frames reuse the same plotter and update the point buffer **in-place** —
    no VTK mesh recreation, no Python heap allocation in the hot path.

    Render pipeline
    ---------------
    1.  Pinhole back-projection  →  (H*W, 3) float32 XYZ
    2.  ``cloud.points[:] = new_pts``  — in-place buffer update
    3.  ``cloud["z_depth"] = z``       — scalar update for colourmap
    4.  ``plotter.render()``
    5.  ``plotter.screenshot(return_img=True)``  →  H×W×3 RGB uint8
    6.  ``cv2.cvtColor(rgb, BGR)``  →  ready for Redis publish
    """

    def __init__(
        self,
        h: int,
        w: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        max_depth: float,
        point_size: int,
        cmap: str,
        render_w: int,
        render_h: int,
    ):
        self._h, self._w = h, w
        out_w = render_w if render_w > 0 else w
        out_h = render_h if render_h > 0 else h

        # ------------------------------------------------------------------
        # Pre-compute normalised pixel-direction grid (once per resolution)
        #
        #   x = (u - c_x) / f_x · z      →  right  (+X)
        #   y = (v - c_y) / f_y · z      →  down   (negated → +Y up in 3-D)
        #   z = z                         →  depth  (+Z away from camera)
        #
        u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        self._dx: np.ndarray = ((u - cx) / fx).ravel()   # (H*W,)
        self._dy: np.ndarray = ((v - cy) / fy).ravel()   # (H*W,)

        # Pre-allocated XYZ buffer — overwritten in-place every frame
        # we'll keep a host copy for VTK, but the heavy math is pushed
        # onto the GPU with PyTorch so the CPU stays mostly idle.
        self._pts = np.zeros((h * w, 3), dtype=np.float32)

        # also keep CUDA versions of the direction vectors and a GPU
        # buffer; these live on the device and are reused each frame.
        # converting here avoids repeated allocations in the hot path.
        self._dx_t = torch.from_numpy(self._dx).to("cuda")
        self._dy_t = torch.from_numpy(self._dy).to("cuda")
        self._pts_t = torch.zeros((h * w, 3), dtype=torch.float32, device="cuda")

        # allocate a persistent scalar buffer for depth values on the GPU
        # so we don't call ``torch.from_numpy(...).to('cuda')`` each frame.
        self._z_t = torch.zeros(h * w, dtype=torch.float32, device="cuda")
        # keep a host array reference too so we can ravel without copying
        self._z_cpu = np.zeros(h * w, dtype=np.float32)

        # ------------------------------------------------------------------
        # PyVista plotter setup
        log.info(
            f"[DepthMetric] Building PyVista plotter  {out_w}×{out_h}  "
            f"point_size={point_size}  cmap={cmap}  EDL=on"
        )

        self._cloud = pv.PolyData(self._pts.copy())
        self._cloud["z_depth"] = np.zeros(h * w, dtype=np.float32)

        self._plotter = pv.Plotter(off_screen=True, window_size=[out_w, out_h])
        self._plotter.set_background("black")

        try:
            self._plotter.disable_anti_aliasing()
        except Exception:
            log.warning("[DepthMetric] could not disable anti-aliasing on plotter")

        # Eye-Dome Lighting — silhouettes + enhanced depth perception
        # allow the number of passes to be tuned via environment variable
        edl_passes = int(os.environ.get("EDL_PASSES", "2"))
        try:
            self._plotter.enable_eye_dome_lighting(n_passes=edl_passes)
        except Exception:
            # some pyvista versions take no argument
            self._plotter.enable_eye_dome_lighting()

        # Points actor — square pixels, scalar colouring by Z depth.
        # store the returned actor so we can tweak its scalar range each frame
        self._actor = self._plotter.add_points(
            self._cloud,
            scalars="z_depth",
            cmap=cmap,
            clim=[0.0, max_depth],
            point_size=point_size,
            render_points_as_spheres=False,
            lighting=False,        # EDL handles all shading
            show_scalar_bar=False,
        )

        # Camera: positioned slightly in front of the origin, looking along +Z
        # (standard pinhole geometry — scene extends away from camera)
        self._plotter.camera.position    = (0.0,  0.0, -0.5)
        self._plotter.camera.focal_point = (0.0,  0.0,  5.0)
        self._plotter.camera.up          = (0.0, -1.0,  0.0)  # -Y = up (image coords)
        self._plotter.camera.clipping_range = (0.01, max_depth * 2)

        # Match the renderer's perspective to the camera intrinsics so the
        # rendered image occupies the same field-of-view as the incoming frame.
        # PyVista uses the *vertical* view angle measured in degrees.
        # Formula: fov_y = 2 * arctan((h/2) / fy).
        try:
            import numpy as _np
            fov_y = 2.0 * _np.degrees(_np.arctan2((self._h / 2.0), fy))
            log.info(f"[DepthMetric] setting camera view_angle to {fov_y:.2f}°")
            self._plotter.camera.view_angle = float(fov_y)
        except Exception:
            # if numpy isn't available for some reason just keep the default
            log.warning("[DepthMetric] failed to compute camera view_angle, leaving default")

        # force an initial render so the window size and camera are settled
        self._plotter.render()
        log.info("[DepthMetric] PyVista plotter ready")

    # ------------------------------------------------------------------

    def render(self, depth: np.ndarray) -> np.ndarray:
        """Back-project *depth*, update the cloud in-place, return BGR frame.

        Args:
            depth: H×W float32 metric depth map (metres).

        Returns:
            BGR uint8 ndarray of shape (render_h, render_w, 3).
        """
        if depth.shape != (self._h, self._w):
            depth = cv2.resize(depth, (self._w, self._h), interpolation=cv2.INTER_LINEAR)

        # move the back‑projection math to the GPU.  converting the
        # depth map to a tensor and doing the multiplies in CUDA keeps
        # the CPU threads mostly idle; only the final copy back to host
        # is performed here when the points are handed to VTK.
        # ensure we have a float32 contiguous array; avoid copying if not
        depth_cpu = depth.astype(np.float32, copy=False)
        # keep a host flat copy for scalar operations (used by VTK)
        # use the pre-allocated buffer to avoid a second allocation
        z = depth_cpu.ravel()

        # copy scalars into the persistent GPU tensor rather than creating a
        # new tensor each frame. non_blocking is safe here because we don't
        # use the tensor until the copy completes but it allows overlap.
        self._z_t.copy_(torch.from_numpy(z), non_blocking=True)

        # GPU compute (use the prepopulated self._z_t)
        self._pts_t[:, 0] = self._dx_t * self._z_t    # X = (u - cx)/fx · z
        self._pts_t[:, 1] = self._dy_t * self._z_t    # Y = (v - cy)/fy · z
        self._pts_t[:, 2] = self._z_t                 # Z = metric depth (m)

        # pull the result back to host buffer for VTK
        # (VTK/PyVista only accepts numpy arrays, so this copy is unavoidable)
        self._pts[:] = self._pts_t.cpu().numpy()

        # Update VTK geometry:
        #   - points: must be *assigned* (not sliced in-place) so PyVista
        #     advances the VTK pipeline MTime and re-uploads geometry.
        #   - scalars: must be updated *in-place* ([:]=) so PyVista keeps
        #     its reference to the original array; reassignment loses it.
        log.debug(f"[DepthMetric] render: pts min/max {self._pts.min():.3f}/{self._pts.max():.3f}")
        self._cloud.points = self._pts
        log.debug("[DepthMetric] render: points assigned, invoking Modified() on points")
        self._cloud.GetPoints().Modified()
        log.debug(f"[DepthMetric] render: z scalar min/max {z.min():.3f}/{z.max():.3f}")
        self._cloud["z_depth"][:] = z

        # update colour-mapping range to the current frame's depth min/max;
        # this boosts contrast so subtle variations inside the person show up
        try:
            zmin, zmax = float(z.min()), float(z.max())
            if zmax > zmin:
                self._actor.mapper.SetScalarRange(zmin, zmax)
        except Exception:
            # mapper may not be accessible in some compile-time builds, ignore
            log.debug("[DepthMetric] failed to update scalar range")

        self._plotter.render()
        log.debug("[DepthMetric] render: plotter.render() completed")

        # Returns H×W×3 RGB uint8
        rgb = self._plotter.screenshot(return_img=True)
        log.debug(f"[DepthMetric] render: screenshot shape {rgb.shape}, mean {rgb.mean():.1f}")

        # when output dimensions differ from input, apply nearest-neighbour
        # interpolation so large points remain blocky rather than blurred.
        if rgb.shape[0] != self._h or rgb.shape[1] != self._w:
            rgb = cv2.resize(rgb, (self._w, self._h), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# FPS overlay  (matches depth.py style)
# ---------------------------------------------------------------------------

def _fps_overlay(im: np.ndarray, fps: int) -> None:
    text         = f"FPS: {fps}"
    font         = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.0, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    h, w         = im.shape[:2]
    fx           = w - tw - 15
    cv2.rectangle(im, (fx - 5, 25 - th - 5), (fx + tw + 5, 25 + bl), (255, 255, 255), -1)
    cv2.putText(im, text, (fx, 25), font, scale, (104, 31, 17), thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    log.info(
        f"[DepthMetric] Starting  model={DEPTH_MODEL}  dataset={DEPTH_DATASET}  "
        f"max_depth={DEPTH_MAX_DEPTH} m"
    )
    model = MetricDepthWrapper(
        encoder=DEPTH_MODEL,
        dataset=DEPTH_DATASET,
        max_depth=DEPTH_MAX_DEPTH,
    )

    bus = FrameBus()
    log.info("[DepthMetric] Subscribing to frames:raw")

    renderer: PointCloudRenderer | None = None
    fps_counter, fps_timer, fps_display = 0, time.time(), 0

    frame_idx = 0
    for raw_frame, _ in bus.subscribe("raw"):
        frame_idx += 1
        log.debug(f"[DepthMetric] got frame {frame_idx} shape {raw_frame.shape} min/max {raw_frame.min()}/{raw_frame.max()}")
        # Lazy renderer init — deferred until first frame so resolution is
        # known and the PyVista window is sized correctly
        if renderer is None:
            h, w = raw_frame.shape[:2]
            cx   = float(PRINCIPAL_X) if PRINCIPAL_X else w / 2.0
            cy   = float(PRINCIPAL_Y) if PRINCIPAL_Y else h / 2.0
            renderer = PointCloudRenderer(
                h=h, w=w,
                fx=FOCAL_LENGTH_X, fy=FOCAL_LENGTH_Y,
                cx=cx, cy=cy,
                max_depth=DEPTH_MAX_DEPTH,
                point_size=POINT_SIZE,
                cmap=DEPTH_CMAP,
                render_w=RENDER_WIDTH,
                render_h=RENDER_HEIGHT,
            )

        depth       = model.infer(raw_frame)       # H×W float32 metres
        log.debug(f"[DepthMetric] depth map stats min/max {depth.min():.3f}/{depth.max():.3f}")
        depth_frame = renderer.render(depth)        # H×W×3 BGR uint8
        log.debug(f"[DepthMetric] depth_frame stats mean {depth_frame.mean():.1f} shape {depth_frame.shape}")

        if SHOW_FPS:
            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer   = now
            _fps_overlay(depth_frame, fps_display)

        bus.publish("depth", depth_frame)


if __name__ == "__main__":
    main()
