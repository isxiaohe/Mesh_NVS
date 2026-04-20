import os
import cv2
import numpy as np
from pathlib import Path

from glob import glob 
import open3d as o3d
import open3d.visualization.rendering as rendering
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, readColmapCameras



def colmap_intrinsics_to_o3d_pinhole(cam):
    w, h = int(cam.width), int(cam.height)
    p = np.array(cam.params, dtype=np.float64).reshape(-1)

    if p.size == 3:  # SIMPLE_PINHOLE: [f, cx, cy]
        f, cx, cy = p.tolist()
        fx, fy = f, f
    elif p.size >= 4:  # PINHOLE / OPENCV-like: [fx, fy, cx, cy, ...]
        fx, fy, cx, cy = p[:4].tolist()
    else:
        raise ValueError(f"Unsupported COLMAP camera params size={p.size}")

    intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    return intr


def colmap_qvec_tvec_to_world_to_cam(qvec, tvec):
    qvec = np.asarray(qvec, dtype=np.float64).reshape(4)  # (w,x,y,z)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

    rot = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])  # (x,y,z,w)
    Rwc = rot.as_matrix()  # world -> camera
    twc = tvec

    extr = np.eye(4, dtype=np.float64)
    extr[:3, :3] = Rwc
    extr[:3, 3] = twc
    return extr


def create_offscreen_renderer(width, height):
    try:
        return rendering.OffscreenRenderer(width, height, headless=True)
    except TypeError:
        return rendering.OffscreenRenderer(width, height)


def add_mesh_to_scene(scene, mesh: o3d.geometry.TriangleMesh, name="mesh"):
    # 用 unlit：更稳定，不依赖光照（尤其你只是想对齐原图/看深度）
    mtl = rendering.MaterialRecord()
    mtl.shader = "defaultUnlit"
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]

    if scene.has_geometry(name):
        scene.remove_geometry(name)
    scene.add_geometry(name, mesh, mtl)


def read_image_any_ext(images_dir: Path, image_name: str):
    """
    COLMAP image.name 可能是 .jpg/.png；这里按原名读。
    如果原名不存在，再尝试常见扩展名替换。
    """
    p = images_dir / image_name
    if p.exists():
        return cv2.imread(str(p), cv2.IMREAD_COLOR), p

    stem = Path(image_name).stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p2 = images_dir / f"{stem}{ext}"
        if p2.exists():
            return cv2.imread(str(p2), cv2.IMREAD_COLOR), p2

    return None, p


def depth_to_vis(depth: np.ndarray, invalid_val=0.0):
    """
    depth: float32 (H,W), view-space z (距离，单位同 mesh)
    返回：uint8 (H,W) 可视化图（越近越亮），以及用于显示的有效范围
    """
    d = depth.copy()
    # 通常 0 表示背景/无命中
    mask = np.isfinite(d) & (d > invalid_val)

    if mask.sum() < 10:
        return np.zeros_like(d, dtype=np.uint8), (0.0, 0.0)

    valid = d[mask]
    # 用分位数避免极端值导致全黑/全白
    dmin = float(np.percentile(valid, 2.0))
    dmax = float(np.percentile(valid, 98.0))
    if dmax <= dmin + 1e-6:
        dmax = dmin + 1e-3

    d_clip = np.clip(d, dmin, dmax)
    # 近处亮：做反转
    vis = 1.0 - (d_clip - dmin) / (dmax - dmin)
    vis[~mask] = 0.0
    vis_u8 = (vis * 255.0).astype(np.uint8)
    return vis_u8, (dmin, dmax)

def depth_to_valid_mask(depth: np.ndarray, invalid_val: float = 0.0) -> np.ndarray:
    """
    depth: float32 (H, W), view-space z
    Returns uint8 (H, W): 1 where depth is valid (ray hit mesh), 0 otherwise.
    """
    d = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(d) & (d > invalid_val)
    return valid.astype(np.uint8)

def save_intrinsic_txt(cam_intrinsic, out_path):
    fx, fy, cx, cy = cam_intrinsic.params
    w = cam_intrinsic.width
    h = cam_intrinsic.height
    with open(out_path, "w") as f:
        f.write(f"{w} {h} {fx} {fy} {cx} {cy}\n")

def interactive_show_three_windows(
    input_root: Path,
    mesh_path: Path,
    cam_extrinsics_data: dict,
    cam_intrinsics_data: dict,
    save_dir: Path = None,
):
    input_root = Path(input_root)
    images_dir = input_root / "images"
    assert images_dir.exists(), f"Not found: {images_dir}"

    mesh_path = Path(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Empty mesh: {mesh_path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    if save_dir is not None:
        save_dir = Path(save_dir)
        # (save_dir / "vis").mkdir(parents=True, exist_ok=True)
        rendered_depth_dir = save_dir / "rendered_depth"
        rendered_depth_dir.mkdir(parents=True, exist_ok=True)
        # rendered_rgb_dir = save_dir / "rendered_rgb"
        # rendered_rgb_dir.mkdir(parents=True, exist_ok=True)
        # extrinsic_dir = save_dir / "extrinsic"
        # extrinsic_dir.mkdir(parents=True, exist_ok=True)
        # intrinsic_dir = save_dir / "intrinsic"
        # intrinsic_dir.mkdir(parents=True, exist_ok=True)
        rendered_mask_dir = save_dir / "rendered_mask"
        rendered_mask_dir.mkdir(parents=True, exist_ok=True)
        

    # 读取cam_extrinsics_data 并存成 key是image_name的dict
    cam_extrinsics_data_dict = {}
    renderer_cache = {}  # (w,h) -> renderer
    for img_id, img in cam_extrinsics_data.items():
        cam_extrinsics_data_dict[img.name] = img
    image_list = glob(os.path.join(images_dir, '*.png'))
    image_list = sorted(image_list, key=lambda x: int(Path(x).stem))
    for idx, image_id in enumerate(image_list):
        image_name = Path(image_id).name
        img = cam_extrinsics_data_dict.get(image_name, None)
        if image_name not in cam_extrinsics_data_dict:
            print(f"[Warn] image {image_name} not in cam_extrinsics_data, skip.")
            continue
        orig_bgr, orig_path = read_image_any_ext(images_dir, image_name)
        if orig_bgr is None:
            print(f"[Warn] Original not found for {image_name} under {images_dir}")
            # 继续也可以渲染，但窗口1会用黑图占位
            orig_bgr = np.zeros((cam_intrinsics_data[img.camera_id].height,
                                 cam_intrinsics_data[img.camera_id].width, 3), dtype=np.uint8)

        # 相机内外参
        # save extrinsic and intrinsic
        cam_extrinsic = cam_extrinsics_data_dict[image_name]
        cam_intrinsic = cam_intrinsics_data[img.camera_id]
        qvec, tvec = cam_extrinsic.qvec, cam_extrinsic.tvec
        rot = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])  # colmap format (w, x, y, z)
        Rwc = rot.as_matrix()   # world -> camera
        twc = tvec.reshape(3,)
        extri_mat = np.eye(4)
        extri_mat[:3, :3] = Rwc
        extri_mat[:3, 3] = twc
        # np.savetxt( extrinsic_dir / image_name.replace('.png', '.txt'),extri_mat)
        cam = cam_intrinsics_data[img.camera_id]
        # save_intrinsic_txt(cam_intrinsic, intrinsic_dir / image_name.replace('.png', '.txt'))
        # ===
    
        intr = colmap_intrinsics_to_o3d_pinhole(cam)
        w, h = int(intr.width), int(intr.height)
        extr_w2c = colmap_qvec_tvec_to_world_to_cam(img.qvec, img.tvec)

        # renderer 复用
        key = (w, h)
        if key not in renderer_cache:
            r = create_offscreen_renderer(w, h)
            r.scene.set_background([0.0, 0.0, 0.0, 1.0])
            add_mesh_to_scene(r.scene, mesh, name="mesh")
            renderer_cache[key] = r
        renderer = renderer_cache[key]

        # setup camera
        
        renderer.setup_camera(intr, extr_w2c)

        # render color (Open3D -> RGBA uint8)
        color_o3d = renderer.render_to_image()
        color_rgba = np.asarray(color_o3d)
        if color_rgba.shape[-1] == 4:
            color_rgb = color_rgba[..., :3]
        else:
            color_rgb = color_rgba
        # 转 BGR 给 OpenCV
        render_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

        # render depth (float32)
        depth_o3d = renderer.render_to_depth_image(z_in_view_space=True)
        depth = np.asarray(depth_o3d).astype(np.float32)
        depth_vis_u8, (dmin, dmax) = depth_to_vis(depth)

        # 为了更好看，可以上伪彩（可选）；这里默认灰度
        depth_vis_bgr = cv2.cvtColor(depth_vis_u8, cv2.COLOR_GRAY2BGR)
        # save depth 
        plt.imsave(rendered_depth_dir/image_name, depth, cmap='Spectral')
        np.save(rendered_depth_dir / (image_name.split('.')[0] + ".npy"), depth)   
        # cv2.imwrite(rendered_rgb_dir / image_name, render_bgr)

        valid_mask = depth_to_valid_mask(depth)  # (H,W) uint8 0/1
        np.save(rendered_mask_dir / (image_name.split('.')[0] + ".npy"), valid_mask)
        cv2.imwrite(rendered_mask_dir / image_name, valid_mask * 255)



# -------------------------
# 你实际调用示例（把读取接口接上）
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default='/mnt/fillipo/junfeng/BIGAI-demo/youtube-tidy-data/0NoI5tLKApU',
                        help="包含 images/ 和 colmap/ 的根目录")
    parser.add_argument("--mesh_path", type=str, default='/mnt/fillipo/junfeng/BIGAI-demo/svpp-results/test-params/0NoI5tLKApU-small-color-loss/tetra_meshes/tetra_mesh_binary_search_7_iter_7000.ply',
                        help="Path to the mesh file (.ply)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Path to save the rendered depth and rgb")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    mesh_path = Path(args.mesh_path)
    colmap_dir = input_root / 'sparse/0'
    if args.save_dir is None:
        save_dir = mesh_path.parent / "rendering"
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # 你已有的读取函数
    cam_extrinsics_data = read_extrinsics_binary(colmap_dir / "images.bin")
    cam_intrinsics_data = read_intrinsics_binary(colmap_dir / "cameras.bin")
    interactive_show_three_windows(
        input_root=input_root,
        mesh_path=mesh_path,
        cam_extrinsics_data=cam_extrinsics_data,
        cam_intrinsics_data=cam_intrinsics_data,
        save_dir=save_dir,
    )
