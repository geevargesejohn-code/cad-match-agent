"""Pure-CPU mesh renderer: no OpenGL, no GPU, works the same on Windows, macOS and Linux.

Triangles are projected with NumPy and painted back-to-front with OpenCV
(painter's algorithm). That is plenty for silhouettes and shaded previews of
mechanical parts, and it keeps installation trivial.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import trimesh

MAX_RENDER_FACES = 40000


def fibonacci_sphere(n: int, rotation: np.ndarray | None = None) -> np.ndarray:
    """n evenly spread unit directions covering the whole sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    pts = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    if rotation is not None:
        pts = pts @ rotation.T
    return pts


def refine_directions(n: int) -> np.ndarray:
    """A second, denser set of directions offset from the default set."""
    rot = trimesh.transformations.rotation_matrix(0.61, [0.3, 0.8, 0.52])[:3, :3]
    return fibonacci_sphere(n, rotation=rot)


def _camera_axes(direction: np.ndarray, roll_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right/up/forward axes for a camera at `direction`, looking at the origin."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    forward = -d
    hint = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    if roll_deg:
        a = math.radians(roll_deg)
        right, up = right * math.cos(a) + up * math.sin(a), -right * math.sin(a) + up * math.cos(a)
    return right, up, forward


def prepare_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Simplify very heavy meshes so rendering stays fast (shape is unaffected at image resolution)."""
    if len(mesh.faces) <= MAX_RENDER_FACES:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=MAX_RENDER_FACES)
    except Exception:  # fast-simplification not installed: render the full mesh, just slower
        return mesh


def render(
    mesh: trimesh.Trimesh,
    direction,
    size: int | tuple[int, int] = 224,
    roll_deg: float = 0.0,
    fov_deg: float | None = None,
    color: tuple[int, int, int] = (205, 205, 205),
    background: np.ndarray | tuple[int, int, int] = (255, 255, 255),
    fill: float = 0.9,
    supersample: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a mesh that is centered at the origin and fits in a unit sphere.

    Returns (rgb uint8 HxWx3, mask bool HxW). Orthographic by default; pass fov_deg
    for a perspective camera (used to make realistic synthetic photos).
    """
    w, h = (size, size) if isinstance(size, int) else size
    W, H = w * supersample, h * supersample
    right, up, forward = _camera_axes(np.asarray(direction, float), roll_deg)

    v = mesh.vertices
    tris = mesh.faces
    if fov_deg is None:
        scale = fill * min(W, H) / 2.0
        xs, ys = v @ right * scale, v @ up * scale
        depth = v @ forward
    else:
        dist = 1.0 / math.sin(math.radians(fov_deg) / 2) / fill
        cam = -forward * dist
        rel = v - cam
        z = rel @ forward
        f_px = (min(W, H) / 2.0) / math.tan(math.radians(fov_deg) / 2)
        xs, ys = (rel @ right) * f_px / z, (rel @ up) * f_px / z
        depth = z
    px = np.stack([xs + W / 2.0, H / 2.0 - ys], axis=1)

    # Two-sided Lambert shading with a key light and a fill light, both fixed to the camera.
    normals = mesh.face_normals
    key = 0.45 * right + 0.65 * up - 0.6 * forward
    key /= np.linalg.norm(key)
    fill_light = -0.6 * right - 0.2 * up - 0.75 * forward
    fill_light /= np.linalg.norm(fill_light)
    shade = 0.28 + 0.62 * np.abs(normals @ key) + 0.2 * np.abs(normals @ fill_light)
    shade = np.clip(shade, 0, 1.05)
    face_colors = np.clip(np.outer(shade, color), 0, 255).astype(np.uint8)

    order = np.argsort(-depth[tris].mean(axis=1))  # farthest first
    SHIFT = 4
    pts_all = np.round(px[tris] * (1 << SHIFT)).astype(np.int32)

    if isinstance(background, np.ndarray):
        img = cv2.resize(background, (W, H), interpolation=cv2.INTER_LINEAR).copy()
    else:
        img = np.empty((H, W, 3), np.uint8)
        img[:] = background
    mask = np.zeros((H, W), np.uint8)

    for idx in order:
        pts = pts_all[idx]
        c = face_colors[idx]
        cv2.fillConvexPoly(img, pts, (int(c[0]), int(c[1]), int(c[2])), lineType=cv2.LINE_8, shift=SHIFT)
        cv2.fillConvexPoly(mask, pts, 255, lineType=cv2.LINE_8, shift=SHIFT)

    if supersample > 1:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
    return img, mask > 127


THUMBNAIL_DIRECTION = np.array([0.55, -0.7, 0.45])


def thumbnail(mesh: trimesh.Trimesh, size: int = 256) -> np.ndarray:
    img, _ = render(prepare_mesh(mesh), THUMBNAIL_DIRECTION, size=size, color=(120, 170, 225),
                    background=(246, 247, 249))
    return img
