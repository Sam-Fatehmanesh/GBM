#!/usr/bin/env python3
"""
Generate a rotating 3D video of zebrafish brain regions using masks.

This script loads brain region masks, combines them with different colors,
and creates a rotating 3D visualization saved as a video using PyVista for GPU acceleration.
"""

import os
import numpy as np
import torch
from masks import load_zebrafish_masks

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista not available. Install with: pip install pyvista")


def create_voxel_cubes(mask_loader, brain_regions, colors, cube_size=0.8):
    """
    Create 3D cube meshes for each voxel in brain regions.
    
    Args:
        mask_loader: ZebrafishMaskLoader instance
        brain_regions: List of brain region names to include
        colors: List of color names for each region
        cube_size: Size of each voxel cube (0.0-1.0)
        
    Returns:
        region_info: List of tuples (mesh, color, name) for each region
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for this visualization. Install with: pip install pyvista")
    
    region_info = []
    
    print(f"Creating voxel cubes for {len(brain_regions)} regions...")
    
    for i, region_name in enumerate(brain_regions):
        if region_name in mask_loader:
            # Get the mask and convert to numpy
            mask = mask_loader[region_name].cpu().numpy().astype(bool)
            
            if np.any(mask):
                # Get coordinates of active voxels
                coords = np.where(mask)
                if len(coords[0]) > 0:
                    # Create a combined mesh for all voxels in this region
                    region_mesh = pv.MultiBlock()
                    
                    # Create individual cubes for each voxel
                    voxel_positions = list(zip(coords[0], coords[1], coords[2]))  # z, y, x
                    
                    # Limit number of voxels for performance (sample if too many)
                    max_voxels = 5000  # Adjust based on performance needs
                    if len(voxel_positions) > max_voxels:
                        # Sample random voxels to stay within limit
                        import random
                        random.seed(42)  # For reproducibility
                        voxel_positions = random.sample(voxel_positions, max_voxels)
                        print(f"  {region_name}: Sampling {max_voxels} of {len(coords[0])} voxels")
                    
                    # Create all cubes at once using a more efficient method
                    points = []
                    cubes = []
                    
                    for idx, (z, y, x) in enumerate(voxel_positions):
                        # Create cube vertices around this position
                        # Cube goes from (x-0.5, y-0.5, z-0.5) to (x+0.5, y+0.5, z+0.5)
                        s = cube_size / 2  # Half size
                        
                        # 8 vertices of a cube
                        cube_vertices = np.array([
                            [x-s, y-s, z-s],  # 0
                            [x+s, y-s, z-s],  # 1
                            [x+s, y+s, z-s],  # 2
                            [x-s, y+s, z-s],  # 3
                            [x-s, y-s, z+s],  # 4
                            [x+s, y-s, z+s],  # 5
                            [x+s, y+s, z+s],  # 6
                            [x-s, y+s, z+s],  # 7
                        ])
                        
                        base_idx = len(points)
                        points.extend(cube_vertices)
                        
                        # Define the 6 faces of the cube (each face has 4 vertices)
                        cube_faces = np.array([
                            [4, base_idx+0, base_idx+1, base_idx+2, base_idx+3],  # bottom
                            [4, base_idx+4, base_idx+7, base_idx+6, base_idx+5],  # top
                            [4, base_idx+0, base_idx+4, base_idx+5, base_idx+1],  # front
                            [4, base_idx+2, base_idx+6, base_idx+7, base_idx+3],  # back
                            [4, base_idx+0, base_idx+3, base_idx+7, base_idx+4],  # left
                            [4, base_idx+1, base_idx+5, base_idx+6, base_idx+2],  # right
                        ])
                        
                        cubes.extend(cube_faces)
                    
                    # Create the mesh from all points and faces
                    if points:
                        points = np.array(points)
                        faces = np.array(cubes).ravel()
                        
                        region_mesh = pv.PolyData(points, faces)
                        
                        region_info.append((region_mesh, colors[i], region_name))
                        print(f"  {region_name}: {len(voxel_positions)} voxel cubes created")
                    else:
                        print(f"  {region_name}: No cubes created")
                else:
                    print(f"  {region_name}: No active voxels")
            else:
                print(f"  {region_name}: Empty mask")
        else:
            print(f"  Warning: Region '{region_name}' not found")
    
    return region_info


def create_brain_video_pyvista(
    output_file='zebrafish_brain_rotation.mp4',
    target_shape=(20, 64, 32),  # Reasonable size for smooth surfaces
    fps=30,
    duration_seconds=8,
    quality=8  # 1-10, higher is better quality
):
    """
    Create a rotating video of zebrafish brain regions using PyVista.
    
    Args:
        output_file: Output video filename
        target_shape: Target shape for brain masks
        fps: Frames per second for video
        duration_seconds: Duration of video in seconds
        quality: Video quality (1-10)
    """
    if not PYVISTA_AVAILABLE:
        print("Error: PyVista is not available. Install with: pip install pyvista")
        return
    
    print("Creating zebrafish brain rotation video with PyVista...")
    print(f"Target shape: {target_shape}")
    print(f"Output file: {output_file}")
    
    # Load masks
    print("\nLoading brain masks...")
    mask_loader = load_zebrafish_masks(target_shape=target_shape)
    
    # Define brain regions to visualize with colors
    brain_regions = [
        'whole_brain',
    ]
    
    # Color names for PyVista
    colors = ['red']
    region_names = [
        'whole_brain',
    ]
    
    # Filter to only available regions
    available_regions = []
    available_colors = []
    available_names = []
    
    for region, color, name in zip(brain_regions, colors, region_names):
        if region in mask_loader:
            available_regions.append(region)
            available_colors.append(color)
            available_names.append(name)
            print(f"  Including: {name}")
        else:
            print(f"  Skipping: {name} (not found)")
    
    if not available_regions:
        print("Error: No brain regions found!")
        return
    
    # Create voxel cubes for brain regions
    print(f"\nCreating voxel cubes...")
    region_info = create_voxel_cubes(mask_loader, available_regions, available_colors, cube_size=0.9)
    
    if not region_info:
        print("Error: No voxel cubes created!")
        return
    
    print(f"Successfully created {len(region_info)} brain regions with voxel cubes")
    
    # Set up PyVista plotter
    print("\nSetting up PyVista visualization...")
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    
    # Add each brain region to the plotter
    for mesh, color, region_name in region_info:
        plotter.add_mesh(
            mesh, 
            color=color, 
            opacity=0.2,
            smooth_shading=True,
            name=region_name
        )
    
    # Set up lighting and camera
    plotter.add_light(pv.Light(position=(10, 10, 10), focal_point=(0, 0, 0)))
    plotter.set_background('black')
    
    # Position camera for good view
    # Calculate center point of the brain
    center_x, center_y, center_z = target_shape[2]/2, target_shape[1]/2, target_shape[0]/2
    plotter.camera.focal_point = (center_x, center_y, center_z)
    
    # Set initial camera position (further back to see the whole brain)
    plotter.camera.position = (center_x + 80, center_y + 80, center_z + 60)
    plotter.camera.up = (0, 0, 1)
    
    # Calculate number of frames
    total_frames = int(fps * duration_seconds)
    print(f"Creating animation with {total_frames} frames at {fps} FPS...")
    
    # Set up video recording
    plotter.open_movie(output_file, framerate=fps, quality=quality)
    
    # Create rotation animation
    print("Generating frames...")
    center_x, center_y, center_z = target_shape[2]/2, target_shape[1]/2, target_shape[0]/2
    
    for frame in range(total_frames):
        # Calculate rotation angle (full 360 degree rotation)
        angle = frame * 360.0 / total_frames
        
        # Update camera position (orbit around the brain)
        radius = max(target_shape) * 2.5  # Scale radius based on brain size
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        z = target_shape[0] * 0.8  # Keep some elevation relative to brain height
        
        plotter.camera.position = (
            x + center_x, 
            y + center_y, 
            z + center_z
        )
        plotter.camera.focal_point = (center_x, center_y, center_z)
        
        # Write frame
        plotter.write_frame()
        
        # Progress update
        if frame % (total_frames // 10) == 0:
            progress = (frame / total_frames) * 100
            print(f"  Progress: {progress:.1f}%")
    
    # Finalize video
    plotter.close()
    
    print(f"✓ Video saved successfully!")
    print(f"  File: {output_file}")
    print(f"  Duration: {duration_seconds}s")
    print(f"  FPS: {fps}")
    print(f"  Frames: {total_frames}")
    print(f"  Quality: {quality}/10")


def create_simple_point_cloud_video(
    output_file='zebrafish_brain_points.mp4',
    target_shape=(15, 64, 32),
    fps=24,
    duration_seconds=8,
    point_size=3
):
    """
    Create a simpler point cloud visualization if mesh generation fails.
    
    Args:
        output_file: Output video filename
        target_shape: Target shape for brain masks
        fps: Frames per second
        duration_seconds: Duration in seconds
        point_size: Size of points
    """
    if not PYVISTA_AVAILABLE:
        print("Error: PyVista is not available. Install with: pip install pyvista")
        return
    
    print("Creating point cloud brain video...")
    
    # Load masks
    mask_loader = load_zebrafish_masks(target_shape=target_shape)
    
    # Define brain regions and colors
    brain_regions = [
        'prosencephalon_(forebrain)',
        'mesencephalon_(midbrain)', 
        'rhombencephalon_(hindbrain)',
        'cerebellum'
    ]
    
    # RGB color values
    colors = [
        [255, 100, 100],  # Red
        [100, 255, 100],  # Green  
        [100, 100, 255],  # Blue
        [255, 255, 100],  # Yellow
    ]
    
    # Set up plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.set_background('black')
    
    # Create point clouds for each region
    for i, region_name in enumerate(brain_regions):
        if region_name in mask_loader:
            mask = mask_loader[region_name].cpu().numpy().astype(bool)
            
            # Get coordinates of active voxels
            coords = np.where(mask)
            if len(coords[0]) > 0:
                # Create points
                points = np.column_stack([coords[2], coords[1], coords[0]])  # x, y, z
                
                # Create point cloud
                point_cloud = pv.PolyData(points)
                
                # Add to plotter
                plotter.add_mesh(
                    point_cloud,
                    color=colors[i],
                    point_size=point_size,
                    render_points_as_spheres=True,
                    opacity=0.8,
                    name=region_name
                )
                
                print(f"  Added {len(points)} points for {region_name}")
    
    # Set camera
    plotter.camera.position = (80, 80, 80)
    plotter.camera.focal_point = (target_shape[2]/2, target_shape[1]/2, target_shape[0]/2)
    
    # Create animation
    total_frames = int(fps * duration_seconds)
    plotter.open_movie(output_file, framerate=fps, quality=8)
    
    for frame in range(total_frames):
        angle = frame * 360.0 / total_frames
        radius = 120
        
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        z = 60
        
        plotter.camera.position = (
            x + target_shape[2]/2, 
            y + target_shape[1]/2, 
            z + target_shape[0]/2
        )
        
        plotter.write_frame()
        
        if frame % (total_frames // 10) == 0:
            print(f"  Progress: {(frame / total_frames) * 100:.1f}%")
    
    plotter.close()
    print(f"✓ Point cloud video saved: {output_file}")


def main():
    """Main function to create the brain video."""
    print("=== Zebrafish Brain 3D Video Generator (PyVista) ===\n")
    
    if not PYVISTA_AVAILABLE:
        print("PyVista is not installed. Please install it with:")
        print("  pip install pyvista")
        print("  # or")
        print("  conda install -c conda-forge pyvista")
        return
    
    # Create output directory
    output_dir = "videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the mesh-based video
    output_file = os.path.join(output_dir, "zebrafish_brain_meshes.mp4")
    
    try:
        create_brain_video_pyvista(
            output_file=output_file,
            target_shape=(20, 64, 32),  # Good balance of detail and performance
            fps=30,
            duration_seconds=8,
            quality=8
        )
    except Exception as e:
        print(f"Mesh-based video failed: {e}")
        print("Trying point cloud version...")
        
        # Fallback to point cloud version
        output_file_points = os.path.join(output_dir, "zebrafish_brain_points.mp4")
        create_simple_point_cloud_video(
            output_file=output_file_points,
            target_shape=(15, 64, 32),
            fps=24,
            duration_seconds=8,
            point_size=4
        )
    
    print(f"\n=== Video creation complete! ===")


if __name__ == "__main__":
    main() 