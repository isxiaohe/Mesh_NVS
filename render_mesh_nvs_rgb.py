import json
import os
from os import makedirs
import torch
import torchvision
from random import randint
from tqdm import tqdm
import numpy as np
import cv2

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from mesh_nvs_utils import RenderWithColorField, RenderWithMeshColors, get_mesh_from_ply


from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

from scene.mesh import MeshRasterizer
from scene.mesh import ScalableMeshRenderer as MeshRenderer
from scene.dataset_readers import sceneLoadTypeCallbacks

from utils.general_utils import safe_state
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

faces_dict = {}

def train_color_field(
    ply_file: str,
    train_cameras: list,
    camera_extent: float,
    color_field_type: str = "ColorFieldVM",
    iterations: int = 5000,
    TV_loss_weight: float = 1.0
) -> RenderWithColorField:
    """
    Trains a color field model using a 3D mesh and a set of camera viewpoints.

    Args:
        dataset (ModelParams): Parameters related to the dataset.
        pipeline (PipelineParams): Parameters related to the training pipeline.
        color_field_type (str, optional): The type of color field to use.

    Returns:
        RenderWithColorField: The trained render with color field object.
    """
    # Load the mesh from the PLY file
    
    mesh = get_mesh_from_ply(ply_file)

    # Set up the mesh renderer
    mesh_rasterizer = MeshRasterizer(cameras=train_cameras)
    mesh_renderer = MeshRenderer(mesh_rasterizer)

    # Initialize the render with color field
    render_with_color_field = RenderWithColorField(mesh, color_field_type, camera_extent)
    lr_factor = 0.1 ** (1 / iterations)
    viewpoint_stack = None
    print(f"[INFO] Training color field for {iterations} iterations")
    progress_bar = tqdm(range(iterations + 1), desc="Training color field", leave=True)
    for iteration in progress_bar:
        render_with_color_field.optimizer.zero_grad()

        # Get a random viewpoint
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
            viewpoint_idx_stack = list(range(len(viewpoint_stack)))

        random_view_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_idx = viewpoint_idx_stack.pop(random_view_idx)
        viewpoint_cam = viewpoint_stack.pop(random_view_idx)
        
        # Render the mesh
        mesh_color = render_with_color_field(viewpoint_idx, viewpoint_cam, mesh_renderer, faces_dict=faces_dict)
        if mesh_color is None:
            print(f"[WARNING] Mesh is empty after culling. Skipping iteration {iteration}. Image Name: {viewpoint_cam.image_name}")
            continue
        # Compute the loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = ((mesh_color - gt_image) ** 2).mean()

        # Compute the TV loss
        TV_loss_weight *= lr_factor
        tv_loss = render_with_color_field.color_field.TV_loss() * TV_loss_weight
        loss += tv_loss
        
        loss.backward()
        render_with_color_field.optimizer.step()

        # Update learning rate
        for param_group in render_with_color_field.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

               
    return render_with_color_field
    
def render_set(output_path, name, views, render_with_color_field):
    render_path = os.path.join(output_path, name, "renders")
    gts_path = os.path.join(output_path, name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # Get the mesh
    if not views:
        return
    
    skipped_image_name = ()
    
    # Set up the mesh renderer
    mesh_renderer = MeshRenderer(MeshRasterizer(cameras=views))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering, depth, normal = render_with_color_field(idx, view, mesh_renderer, training=False)
        if rendering is None:
            print(f"[WARNING] Mesh is empty after culling. Skipping rendering for image {view.image_name}")
            skipped_image_name += (view.image_name,)
            continue
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # # save original depth and normal if available
        # if depth is not None:
        #     # save as np
        #     np.save(os.path.join(render_path, '{0:05d}'.format(idx) + "_depth.npy"), depth.cpu().numpy())
        # if normal is not None:
        #     # save as np
        #     np.save(os.path.join(render_path, '{0:05d}'.format(idx) + "_normal.npy"), normal.cpu().numpy())

        # # visualize depth and normal if available
        # # use plasma
        # # depth H * W
        # # normal 3 * H * W
        # if depth is not None:
        #     depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        #     depth_vis = cv2.applyColorMap((depth_vis.cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        #     cv2.imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + "_depth.png"), depth_vis)
        # if normal is not None:
        #     # normalize normal to [0, 1]
        #     normal_vis = (normal + 1) / 2
        #     # 转换维度：(3, H, W) -> (H, W, 3) 并转换为 uint8
        #     normal_vis = (normal_vis.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        #     normal_vis = cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + "_normal.png"), normal_vis)
        
    return skipped_image_name

def render_sets(train_cameras, test_cameras, output_path, skip_train : bool, skip_test : bool, render_with_color_field : RenderWithColorField, prefix_dir : str = "mesh"):
    with torch.no_grad():

        if not skip_train:
             render_set(output_path, f"{prefix_dir}/train", train_cameras, render_with_color_field)

        if not skip_test:
            return render_set(output_path, f"{prefix_dir}/test", test_cameras, render_with_color_field)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Color Field Training andTesting script parameters")
    
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--ply_file", required=True, type=str)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--resolution", default=-1, type=float)
    parser.add_argument("--data_device", default="cuda", type=str)
    parser.add_argument("--use_mesh_colors", action="store_true",
                        help="Use mesh vertex colors directly instead of training neural color field")
    args = parser.parse_args()

    output_path = args.output_path if args.output_path else os.path.join(os.path.dirname(args.ply_file), "mesh_nvs_fixed")
    print(f"[INFO] Rendering " + args.ply_file)
    print(f"[INFO] Output path: {output_path}")
    # breakpoint()
    # Instantiate scene
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, "images", True)
    else:
        raise ValueError("Only Colmap data sets are supported for now")


    print("Loading Training Cameras")
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)
    print("Loading Test Cameras")
    test_cameras = cameraList_from_camInfos(scene_info.test_cameras, 1.0, args)
    cameras_extent = scene_info.nerf_normalization["radius"]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Choose rendering method based on command line argument
    if args.use_mesh_colors:
        print("[INFO] Using mesh vertex colors directly (no neural color field training)")
        # Load mesh with vertex colors
        mesh = get_mesh_from_ply(args.ply_file)
        render_with_color_field = RenderWithMeshColors(mesh=mesh)
    else:
        print("[INFO] Training neural color field...")
        render_with_color_field = train_color_field(ply_file=args.ply_file,
                                                    train_cameras=train_cameras,
                                                    camera_extent=cameras_extent)


    import json
    os.makedirs(os.path.join(output_path, "mesh"), exist_ok=True)
    with open(os.path.join(output_path, "mesh", "faces_dict.json"), "w") as f:
        json.dump(faces_dict, f)

    skipped_image_names = render_sets(train_cameras=train_cameras,
                test_cameras=test_cameras,
                output_path=output_path,
                skip_train=args.skip_train,
                skip_test=args.skip_test,
                render_with_color_field=render_with_color_field)

    if skipped_image_names:
        print(f"[WARNING] The following images were skipped during rendering due to empty mesh after culling: {skipped_image_names}")
        # save
        with open(os.path.join(output_path, "mesh", "skipped_images.json"), "w") as f:
            json.dump(skipped_image_names, f)
