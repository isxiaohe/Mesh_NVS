#!/usr/bin/env python3
"""
Render Depth and Normals from Mesh using COLMAP Cameras

This script renders depth maps and camera-space normal vectors from a triangle mesh
using Open3D ray tracing, with camera poses from a COLMAP reconstruction.

Output is organized by train/test splits from a JSON file.

Usage:
    python render_depth_and_normals.py \
        --input_dir /path/to/colmap \
        --mesh_path /path/to/mesh.ply \
        --output_dir /path/to/output

Input Requirements:
    --input_dir: Must contain:
        - images/: RGB images (not directly used, but referenced by COLMAP)
        - sparse/0/: COLMAP reconstruction with images.bin and cameras.bin
        - train_test_lists.json: JSON file with 'train' and 'test' keys,
          each containing a list of image filenames (e.g., ["00001.jpg", ...])

    --mesh_path: Path to a triangle mesh in .ply format

Output Structure:
    output_dir/
    ├── train/
    │   ├── 00000_depth.npy     # Raw depth (H, W) float32
    │   ├── 00000_depth.png     # Depth visualization
    │   ├── 00000_normal.npy    # Raw normals (H, W, 3) float32 in [-1, 1]
    │   ├── 00000_normal.png    # Normal visualization (RGB mapped to [0, 1])
    │   ├── 00001_depth.npy
    │   └── ...
    └── test/
        └── ... (same structure)

Details:
    - Depth: Z-depth in camera coordinate system (same units as mesh)
    - Normals: Camera coordinate system, interpolated from vertex normals
    - Background pixels (no ray hit) have depth=0 and normals=[0, 0, 0]
    - Visualization maps normals from [-1, 1] to [0, 1] for RGB display
    - Depth visualization uses min-max normalization with plasma colormap

Dependencies:
    - open3d
    - numpy
    - opencv-python
    - matplotlib
    - scipy
    - tqdm
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Import COLMAP loader utilities
import sys
sys.path.append(str(Path(__file__).parent))
from colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text


def render_depth_and_normals(mesh, intrinsic, extrinsic, width, height):
    """
    Render depth and camera-space normals using Open3D ray tracing.

    Args:
        mesh: o3d.geometry.TriangleMesh (with vertex normals computed)
        intrinsic: (3, 3) numpy array - camera intrinsic matrix
        extrinsic: (4, 4) numpy array - world-to-camera transformation
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        depth: (H, W) float32 - Z-depth in camera space
        normals: (H, W, 3) float32 - Camera-space normals, range [-1, 1]
        mask: (H, W) bool - True where ray hit the mesh
    """
    # Ensure mesh has vertex normals for smooth interpolation
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Get numpy data
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_normals = np.asarray(mesh.vertex_normals)

    # Create raycasting scene
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    # Generate camera rays
    intrinsic_t = o3d.core.Tensor(intrinsic, dtype=o3d.core.float32)
    extrinsic_t = o3d.core.Tensor(extrinsic, dtype=o3d.core.float32)
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=intrinsic_t,
        extrinsic_matrix=extrinsic_t,
        width_px=width,
        height_px=height
    )

    # Cast rays
    ans = scene.cast_rays(rays)

    # Extract hit information
    t_hit = ans['t_hit'].numpy()                      # Hit distance
    primitive_ids = ans['primitive_ids'].numpy()      # Triangle ID
    primitive_uvs = ans['primitive_uvs'].numpy()      # Barycentric coords

    # Build valid hit mask
    hit_mask = np.isfinite(t_hit)

    # ==================== Compute Smooth Normals ====================

    # Safe indices for array indexing (replace invalid with 0)
    safe_ids = np.where(hit_mask, primitive_ids, 0)

    # Get vertex indices for each hit triangle
    v_indices = triangles[safe_ids]  # (H, W, 3)

    # Get normals at triangle vertices
    n0 = vertex_normals[v_indices[:, :, 0]]
    n1 = vertex_normals[v_indices[:, :, 1]]
    n2 = vertex_normals[v_indices[:, :, 2]]

    # Barycentric interpolation
    u = primitive_uvs[:, :, 0:1]
    v = primitive_uvs[:, :, 1:2]
    w = 1.0 - u - v

    # Interpolate world-space normals
    world_normals = w * n0 + u * n1 + v * n2

    # Normalize
    norm = np.linalg.norm(world_normals, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0  # Avoid division by zero
    world_normals = world_normals / norm

    # Transform to camera space: N_cam = N_world @ R.T
    R_cam = extrinsic[:3, :3]
    camera_normals = world_normals @ R_cam.T

    # Set background normals to zero
    camera_normals[~hit_mask] = [0.0, 0.0, 0.0]

    # ==================== Compute Z-Depth ====================

    # Ray direction in world coordinates
    rays_numpy = rays.numpy()
    dir_world = rays_numpy[:, :, 3:6]

    # Transform ray direction to camera space
    dir_cam = dir_world @ R_cam.T

    # True Z-depth = Euclidean distance * Z-component of ray direction
    z_depth = t_hit * dir_cam[:, :, 2]

    # Set background depth to 0
    z_depth[~hit_mask] = 0.0

    return z_depth.astype(np.float32), camera_normals.astype(np.float32), hit_mask


def visualize_normals(normals, mask):
    """
    Convert normals to RGB image for visualization.

    Maps normals from [-1, 1] to [0, 1].
    Background pixels (mask=False) set to black.

    Args:
        normals: (H, W, 3) float32, range [-1, 1]
        mask: (H, W) bool, True for valid pixels

    Returns:
        vis: (H, W, 3) uint8, RGB image
    """
    vis = (normals + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
    vis[~mask] = [0.0, 0.0, 0.0]  # Background to black
    return (vis * 255).astype(np.uint8)


def visualize_depth(depth, mask):
    """
    Convert depth to RGB image with colormap for visualization.

    Uses min-max normalization on valid pixels only.
    Background pixels set to black.

    Args:
        depth: (H, W) float32
        mask: (H, W) bool, True for valid pixels

    Returns:
        vis: (H, W, 3) uint8, RGB image with colormap
    """
    vis_normalized = np.zeros(depth.shape, dtype=np.float32)

    if mask.any():
        # Min-max normalization on valid pixels
        valid_depth = depth[mask]
        dmin = valid_depth.min()
        dmax = valid_depth.max()

        if dmax > dmin:
            vis_normalized[mask] = (depth[mask] - dmin) / (dmax - dmin)
        else:
            vis_normalized[mask] = 0.0

    # Apply colormap (automatically handles the 3 channels)
    vis_colored = plt.cm.plasma(vis_normalized)[..., :3]  # (H, W, 3) in [0, 1]
    return (vis_colored * 255).astype(np.uint8)


def colmap_intrinsics_to_matrix(cam):
    """
    Convert COLMAP camera intrinsics to 3x3 intrinsic matrix.

    Args:
        cam: Camera namedtuple from colmap_loader

    Returns:
        intrinsic: (3, 3) numpy array
    """
    params = np.array(cam.params, dtype=np.float64).reshape(-1)

    if params.size == 3:  # SIMPLE_PINHOLE: [f, cx, cy]
        f, cx, cy = params.tolist()
        fx, fy = f, f
    elif params.size >= 4:  # PINHOLE: [fx, fy, cx, cy, ...]
        fx, fy, cx, cy = params[:4].tolist()
    else:
        raise ValueError(f"Unsupported COLMAP camera params size={params.size}")

    intrinsic = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return intrinsic


def colmap_extrinsics_to_matrix(qvec, tvec):
    """
    Convert COLMAP extrinsics to 4x4 world-to-camera matrix.

    Args:
        qvec: (4,) quaternion [w, x, y, z]
        tvec: (3,) translation vector

    Returns:
        extrinsic: (4, 4) numpy array, world-to-camera transformation
    """
    qvec = np.asarray(qvec, dtype=np.float64).reshape(4)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

    # Convert quaternion to rotation matrix
    # COLMAP uses (w, x, y, z), scipy uses (x, y, z, w)
    rot = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
    Rwc = rot.as_matrix()  # World-to-camera rotation

    # Build 4x4 matrix
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = Rwc
    extrinsic[:3, 3] = tvec

    return extrinsic


def render_split(
    mesh,
    image_list,
    split_name,
    image_name_to_extrinsics,
    cam_intrinsics,
    output_dir
):
    """
    Render depth and normals for a list of images in a split.

    Args:
        mesh: o3d.geometry.TriangleMesh
        image_list: List of image filenames (e.g., ["00001.jpg", ...])
        split_name: "train" or "test"
        image_name_to_extrinsics: Dict mapping image name to extrinsics
        cam_intrinsics: Dict mapping camera_id to intrinsics
        output_dir: Path object for this split's output directory

    Returns:
        Tuple of (processed_count, skipped_count)
    """
    split_output_dir = Path(output_dir) / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for idx, image_name in enumerate(tqdm(image_list, desc=f"Rendering {split_name}")):
        # Get camera data
        extr = image_name_to_extrinsics.get(image_name)
        if extr is None:
            print(f"[WARN] Image {image_name} not in COLMAP extrinsics, skipping")
            skipped += 1
            continue

        intr = cam_intrinsics.get(extr.camera_id)
        if intr is None:
            print(f"[WARN] Camera {extr.camera_id} not in intrinsics, skipping {image_name}")
            skipped += 1
            continue

        # Build camera matrices
        intrinsic_matrix = colmap_intrinsics_to_matrix(intr)
        extrinsic_matrix = colmap_extrinsics_to_matrix(extr.qvec, extr.tvec)

        width, height = int(intr.width), int(intr.height)

        # Render
        try:
            depth, normals, mask = render_depth_and_normals(
                mesh, intrinsic_matrix, extrinsic_matrix, width, height
            )
        except Exception as e:
            print(f"[ERROR] Failed to render {image_name}: {e}")
            skipped += 1
            continue

        # Generate output filenames
        base_name = f"{idx:05d}"
        depth_npy_path = split_output_dir / f"{base_name}_depth.npy"
        depth_png_path = split_output_dir / f"{base_name}_depth.png"
        normal_npy_path = split_output_dir / f"{base_name}_normal.npy"
        normal_png_path = split_output_dir / f"{base_name}_normal.png"

        # Save NPY files
        np.save(depth_npy_path, depth)
        np.save(normal_npy_path, normals)

        # Save visualizations
        depth_vis = visualize_depth(depth, mask)
        normal_vis = visualize_normals(normals, mask)

        cv2.imwrite(str(depth_png_path), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(normal_png_path), cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))

        processed += 1

    return processed, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Render depth and normals from mesh using COLMAP cameras"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="COLMAP format directory with images/, sparse/0/, and train_test_lists.json"
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        required=True,
        help="Path to mesh file (.ply)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory for saving results"
    )

    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip rendering train split"
    )
    
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip rendering test split"
    )
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    mesh_path = Path(args.mesh_path)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not input_dir.exists():
        raise RuntimeError(f"Input directory not found: {input_dir}")

    images_dir = input_dir / "images"
    sparse_dir = input_dir / "sparse" / "0"
    split_file = input_dir / "train_test_lists.json"

    if not images_dir.exists():
        raise RuntimeError(f"Images directory not found: {images_dir}")
    if not sparse_dir.exists():
        raise RuntimeError(f"Sparse directory not found: {sparse_dir}")
    if not split_file.exists():
        raise RuntimeError(f"Train/test split file not found: {split_file}")

    # ========================================================================
    # Load and validate mesh
    # ========================================================================

    print(f"[INFO] Loading mesh from {mesh_path}")
    try:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load mesh from {mesh_path}: {e}")

    if mesh.is_empty():
        raise RuntimeError(f"Empty mesh: {mesh_path}")

    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Mesh has no vertices: {mesh_path}")

    if len(mesh.triangles) == 0:
        raise RuntimeError(f"Mesh has no triangles: {mesh_path}")

    if not mesh.has_vertex_normals():
        print("[INFO] Computing vertex normals")
        mesh.compute_vertex_normals()

    print(f"[INFO] Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    # ========================================================================
    # Load COLMAP camera data
    # ========================================================================

    print(f"[INFO] Loading COLMAP data from {sparse_dir}")

    try:
        cam_extrinsics = read_extrinsics_binary(sparse_dir / "images.bin")
    except:
        cam_extrinsics = read_extrinsics_text(sparse_dir / "images.txt")

    try:
        cam_intrinsics = read_intrinsics_binary(sparse_dir / "cameras.bin")
    except Exception as e:
        cam_intrinsics = read_intrinsics_text(sparse_dir / "cameras.txt")

    if len(cam_extrinsics) == 0:
        raise RuntimeError("No cameras found in COLMAP data")

    print(f"[INFO] Loaded {len(cam_extrinsics)} cameras")

    # Build mapping from image name to camera data
    image_name_to_extrinsics = {img.name: img for img in cam_extrinsics.values()}

    # ========================================================================
    # Load train/test splits
    # ========================================================================

    print(f"[INFO] Loading train/test splits from {split_file}")

    try:
        with open(split_file, 'r') as f:
            splits = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load split file {split_file}: {e}")

    if 'train' not in splits or 'test' not in splits:
        raise RuntimeError("Split file must contain 'train' and 'test' keys")

    train_images = splits['train']
    test_images = splits['test']

    if not isinstance(train_images, list) or not isinstance(test_images, list):
        raise RuntimeError("Train and test must be lists in split file")

    if len(train_images) == 0 and len(test_images) == 0:
        raise RuntimeError("Both train and test splits are empty")

    print(f"[INFO] Train: {len(train_images)} images, Test: {len(test_images)} images")

    # Create output directories
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Render all images
    # ========================================================================

    # Render train split
    if not args.skip_train:
        print("\n[INFO] Rendering train split")
        train_processed, train_skipped = render_split(
            mesh=mesh,
            image_list=train_images,
            split_name="train",
            image_name_to_extrinsics=image_name_to_extrinsics,
            cam_intrinsics=cam_intrinsics,
            output_dir=output_dir
        )

    # Render test split
    if not args.skip_test:
        print("\n[INFO] Rendering test split")
        if os.path.exists(output_dir / "test") and len(list((output_dir / "test").glob("*"))) > 0:
            print("[INFO] Reusing existing test images")
            args.skip_test = True
        else:
            test_processed, test_skipped = render_split(
                mesh=mesh,
                image_list=test_images,
                split_name="test",
                image_name_to_extrinsics=image_name_to_extrinsics,
                cam_intrinsics=cam_intrinsics,
                output_dir=output_dir
            )

    # Print summary
    print("\n" + "="*50)
    print("RENDERING COMPLETE")
    print("="*50)
    if not args.skip_train:
        print(f"Train: {train_processed}/{len(train_images)} processed, {train_skipped} skipped")
    if not args.skip_test:
        print(f"Test:  {test_processed}/{len(test_images)} processed, {test_skipped} skipped")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - {output_dir / 'train'}")
    print(f"  - {output_dir / 'test'}")


if __name__ == "__main__":
    main()
