#!/usr/bin/env python3
"""
Simple Three.js-based web viewer for brain region masks and neuron positions.

This script creates a web interface that displays:
1. 3D brain region masks as semi-transparent point clouds
2. Static neuron position point cloud  
3. Interactive controls for region toggling and visualization parameters

Usage:
  python web_simple_brain_viewer.py \
    --subject-h5 /home/user/gbm3/GBM3/processed_spike_voxels_2018/subject_14.h5 \
    --mask-h5 /home/user/gbm3/GBM3/processed_spike_voxels_2018_masks/subject_14_mask.h5 \
    --host 0.0.0.0 --port 8053

Then open http://<server_ip>:8053 in your browser.
"""

import os
import argparse
import json
import numpy as np
import h5py
from flask import Flask, render_template_string, jsonify, make_response

app = Flask(__name__)

# Global data storage
NEURON_DATA = None
MASK_DATA = None


def load_neuron_positions(h5_path, max_neurons=100000):
    """Load neuron positions with intelligent subsampling."""
    print(f"Loading neuron positions from {h5_path}...")

    with h5py.File(h5_path, "r") as f:
        cell_positions = f["cell_positions"][:]  # (N, 3)
        num_neurons = int(f["num_neurons"][()])

        print(f"  Loaded {num_neurons} neurons")

        # Intelligent subsampling for better brain structure visibility
        if num_neurons > max_neurons:
            print(
                f"  Subsampling {num_neurons} neurons to {max_neurons} for visualization"
            )

            # Use random sampling but ensure we get good coverage
            np.random.seed(42)  # For reproducible results
            indices = np.random.choice(num_neurons, size=max_neurons, replace=False)
            indices = np.sort(indices)  # Keep some spatial coherence

            cell_positions = cell_positions[indices]
            display_neurons = max_neurons
        else:
            display_neurons = num_neurons

        print(f"  Final neuron count: {display_neurons}")
        print(f"  Position shape: {cell_positions.shape}")
        print(
            f"  Position ranges (normalized): X=[{cell_positions[:, 0].min():.3f}, {cell_positions[:, 0].max():.3f}], "
            f"Y=[{cell_positions[:, 1].min():.3f}, {cell_positions[:, 1].max():.3f}], "
            f"Z=[{cell_positions[:, 2].min():.3f}, {cell_positions[:, 2].max():.3f}]"
        )

        # Scale positions from [0,1] to [0,100] for better visualization with reasonable point sizes
        cell_positions_scaled = cell_positions * 100.0
        print(
            f"  Position ranges (scaled): X=[{cell_positions_scaled[:, 0].min():.1f}, {cell_positions_scaled[:, 0].max():.1f}], "
            f"Y=[{cell_positions_scaled[:, 1].min():.1f}, {cell_positions_scaled[:, 1].max():.1f}], "
            f"Z=[{cell_positions_scaled[:, 2].min():.1f}, {cell_positions_scaled[:, 2].max():.1f}]"
        )

        return {
            "positions": cell_positions_scaled.astype(np.float32),
            "num_neurons": display_neurons,
            "total_neurons": num_neurons,
        }


def load_mask_data(h5_path):
    """Load brain region masks and metadata from mask H5 file."""
    print(f"Loading mask data from {h5_path}...")

    with h5py.File(h5_path, "r") as f:
        label_volume = f["label_volume"][:]  # (X, Y, Z) voxel grid
        region_names = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in f["region_names"][:]
        ]

        # Get grid metadata
        grid_shape = f.attrs.get("grid_shape_xyz", label_volume.shape)
        if isinstance(grid_shape, np.ndarray):
            grid_shape = grid_shape.tolist()

        print(f"  Loaded label volume: {label_volume.shape}")
        print(f"  Number of regions: {len(region_names)}")
        print(f"  Grid shape XYZ: {grid_shape}")

        return {
            "label_volume": label_volume.astype(np.int16),
            "region_names": region_names,
            "grid_shape": list(grid_shape),
            "num_regions": len(region_names),
        }


def extract_region_surfaces(label_volume, region_id, subsample=4):
    """Extract surface vertices for a specific region."""
    try:
        from scipy import ndimage

        # Simple surface extraction: find voxels that are on the boundary
        region_mask = label_volume == region_id

        if not np.any(region_mask):
            return np.array([]).reshape(0, 3)

        # Subsample for performance
        if subsample > 1:
            region_mask = region_mask[::subsample, ::subsample, ::subsample]

        # Find surface voxels (those adjacent to non-region voxels)
        # Erode the mask to find interior voxels
        interior = ndimage.binary_erosion(region_mask)

        # Surface voxels are in the region but not in the interior
        surface = region_mask & (~interior)

        # Get coordinates of surface voxels
        coords = np.array(np.where(surface)).T

        if len(coords) == 0:
            return np.array([]).reshape(0, 3)

        if subsample > 1:
            coords = coords * subsample

        # Convert to normalized coordinates [0, 1] then scale to [0, 100] like neurons
        coords = coords.astype(np.float32)
        coords[:, 0] /= label_volume.shape[0]  # X
        coords[:, 1] /= label_volume.shape[1]  # Y
        coords[:, 2] /= label_volume.shape[2]  # Z

        # Scale to [0, 100] coordinate space to match neurons
        coords *= 100.0

        return coords

    except Exception as e:
        print(f"Error in extract_region_surfaces for region {region_id}: {e}")
        return np.array([]).reshape(0, 3)


@app.route("/")
def index():
    """Serve the main HTML page with Three.js visualization."""
    import time

    timestamp = int(time.time())

    html_template = (
        """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Brain Viewer - v"""
        + str(timestamp)
        + """</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { margin: 0; padding: 0; overflow: hidden; font-family: Arial, sans-serif; }
        #container { width: 100vw; height: 100vh; display: flex; }
        #controls { 
            width: 320px; 
            background: #2a2a2a; 
            color: white; 
            padding: 15px; 
            overflow-y: auto;
            box-sizing: border-box;
        }
        #viewport { flex: 1; position: relative; }
        #canvas { width: 100%; height: 100%; display: block; }
        .control-group { margin-bottom: 20px; }
        .control-group h3 { margin: 0 0 10px 0; color: #4a9eff; }
        .slider-container { margin: 10px 0; }
        .slider-container label { display: block; margin-bottom: 5px; font-size: 12px; }
        input[type="range"] { width: 100%; }
        .checkbox-container { margin: 5px 0; }
        .checkbox-container input { margin-right: 8px; }
        .region-list { max-height: 250px; overflow-y: auto; border: 1px solid #555; padding: 10px; }
        .region-item { margin: 3px 0; font-size: 11px; }
        .region-item input { margin-right: 5px; }
        
        /* Dual range slider styles */
        .dual-range-slider {
            position: relative;
            height: 20px;
            margin: 10px 0;
        }
        .dual-range-slider input[type="range"] {
            position: absolute;
            width: 100%;
            height: 20px;
            top: 0;
            left: 0;
            -webkit-appearance: none;
            background: transparent;
            pointer-events: none;
        }
        .dual-range-slider input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            height: 20px;
            width: 20px;
            border-radius: 50%;
            background: #4a9eff;
            cursor: pointer;
            pointer-events: all;
            border: 2px solid #fff;
        }
        .dual-range-slider input[type="range"]::-moz-range-thumb {
            height: 20px;
            width: 20px;
            border-radius: 50%;
            background: #4a9eff;
            cursor: pointer;
            pointer-events: all;
            border: 2px solid #fff;
        }
        .dual-range-slider::before {
            content: '';
            position: absolute;
            top: 9px;
            left: 0;
            right: 0;
            height: 2px;
            background: #555;
        }
        .range-fill {
            position: absolute;
            top: 9px;
            height: 2px;
            background: #4a9eff;
            pointer-events: none;
        }
        .range-values {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #ccc;
            margin-top: 5px;
        }
        #loading { 
            position: absolute; 
            top: 50%; 
            left: 50%; 
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 20px;
            border-radius: 10px;
            z-index: 1000;
        }
        button { 
            background: #4a9eff; 
            color: white; 
            border: none; 
            padding: 8px 12px; 
            border-radius: 4px; 
            cursor: pointer; 
            margin: 2px;
        }
        button:hover { background: #3a8eef; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="container">
        <div id="controls">
            <h2 style="margin-top: 0;">Simple Brain Viewer</h2>
            
            <div class="control-group">
                <h3>Neuron Points</h3>
                <div class="checkbox-container">
                    <input type="checkbox" id="show-neurons" checked> <label for="show-neurons">Show Neurons</label>
                </div>
                <div class="slider-container">
                    <label for="point-size">Point Size:</label>
                    <input type="range" id="point-size" min="0.5" max="3.0" value="1.0" step="0.1">
                    <span id="point-size-display">1.0</span>
                </div>
                <div class="slider-container">
                    <label for="neuron-opacity">Opacity:</label>
                    <input type="range" id="neuron-opacity" min="0.1" max="1.0" value="0.6" step="0.05">
                    <span id="neuron-opacity-display">0.6</span>
                </div>
            </div>
            
            <div class="control-group">
                <h3>Spatial Filter</h3>
                <div class="slider-container">
                    <label for="x-filter">X-Axis Range:</label>
                    <div class="dual-range-slider">
                        <input type="range" id="x-min-slider" min="0" max="100" value="0" step="1">
                        <input type="range" id="x-max-slider" min="0" max="100" value="100" step="1">
                        <div class="range-fill" id="x-range-fill"></div>
                    </div>
                    <div class="range-values">
                        <span id="x-min-value">0.0</span> - <span id="x-max-value">1.0</span>
                    </div>
                </div>
                <div class="slider-container">
                    <label for="y-filter">Y-Axis Range:</label>
                    <div class="dual-range-slider">
                        <input type="range" id="y-min-slider" min="0" max="100" value="0" step="1">
                        <input type="range" id="y-max-slider" min="0" max="100" value="100" step="1">
                        <div class="range-fill" id="y-range-fill"></div>
                    </div>
                    <div class="range-values">
                        <span id="y-min-value">0.0</span> - <span id="y-max-value">1.0</span>
                    </div>
                </div>
                <button onclick="resetSpatialFilter()" style="width: 100%; padding: 6px; margin: 5px 0;">Reset Filter</button>
            </div>
            
            <div class="control-group">
                <h3>Region Masks</h3>
                <div class="slider-container">
                    <label for="mask-opacity">Mask Opacity:</label>
                    <input type="range" id="mask-opacity" min="0.1" max="0.8" value="0.4" step="0.05">
                    <span id="mask-opacity-display">0.4</span>
                </div>
                <div class="slider-container">
                    <label for="mask-size">Mask Point Size:</label>
                    <input type="range" id="mask-size" min="0.5" max="3.0" value="1.0" step="0.1">
                    <span id="mask-size-display">1.0</span>
                </div>
                <div style="margin: 10px 0;">
                    <button onclick="selectAllRegions()">Select All</button>
                    <button onclick="clearAllRegions()">Clear All</button>
                    <button onclick="selectTopRegions()">Top 10</button>
                </div>
                <div class="region-list" id="region-list">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
            
            <div class="control-group">
                <h3>View Controls</h3>
                <button onclick="resetCamera()" style="width: 100%; padding: 8px; margin: 5px 0;">Reset Camera</button>
                <div class="checkbox-container">
                    <input type="checkbox" id="auto-rotate"> <label for="auto-rotate">Auto Rotate</label>
                </div>
            </div>
        </div>
        
        <div id="viewport">
            <div id="loading">Loading data...</div>
            <canvas id="canvas"></canvas>
        </div>
    </div>

    <script>
        // Three.js scene setup
        let scene, camera, renderer, controls;
        let neuronPoints, regionMeshes = {};
        
        // Data storage
        let neuronData = null;
        let maskData = null;
        let cachedRegionGeometry = {};
        
        // Initialize the 3D scene
        function initThreeJS() {
            const canvas = document.getElementById('canvas');
            const viewport = document.getElementById('viewport');
            
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x111111);
            
            // Camera - updated for [0,100] coordinate space
            camera = new THREE.PerspectiveCamera(75, viewport.clientWidth / viewport.clientHeight, 0.1, 10000);
            camera.position.set(150, 150, 150);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
            renderer.setSize(viewport.clientWidth, viewport.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            
            // Controls - updated for [0,100] coordinate space
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(50, 50, 50);
            
            // Lights
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // Handle window resize
            window.addEventListener('resize', onWindowResize);
            
            animate();
        }
        
        function onWindowResize() {
            const viewport = document.getElementById('viewport');
            camera.aspect = viewport.clientWidth / viewport.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(viewport.clientWidth, viewport.clientHeight);
        }
        
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            
            // Auto-rotate if enabled
            if (document.getElementById('auto-rotate').checked) {
                scene.rotation.y += 0.005;
            }
            
            renderer.render(scene, camera);
        }
        
        // Load data from server
        async function loadData() {
            try {
                console.log('Loading neuron data...');
                const neuronResponse = await fetch('/api/neuron_data');
                neuronData = await neuronResponse.json();
                
                console.log('Loading mask data...');
                const maskResponse = await fetch('/api/mask_data');
                maskData = await maskResponse.json();
                
                console.log('Data loaded:', {
                    neurons: neuronData.num_neurons,
                    total_neurons: neuronData.total_neurons,
                    regions: maskData.num_regions
                });
                
                initializeUI();
                createNeuronVisualization();
                document.getElementById('loading').style.display = 'none';
                
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('loading').innerHTML = 'Error loading data: ' + error.message;
            }
        }
        
        function initializeUI() {
            // Setup region list
            const regionList = document.getElementById('region-list');
            regionList.innerHTML = '';
            
            // Show all regions
            for (let i = 0; i < maskData.region_names.length; i++) {
                const regionDiv = document.createElement('div');
                regionDiv.className = 'region-item';
                regionDiv.innerHTML = `
                    <input type="checkbox" id="region-${i+1}" value="${i+1}">
                    <label for="region-${i+1}">${maskData.region_names[i]}</label>
                `;
                regionList.appendChild(regionDiv);
            }
            
            // Add event listeners
            setupEventListeners();
        }
        
        function setupEventListeners() {
            // Neuron point controls
            document.getElementById('show-neurons').addEventListener('change', (e) => {
                if (neuronPoints) {
                    neuronPoints.visible = e.target.checked;
                }
            });
            
            document.getElementById('point-size').addEventListener('input', (e) => {
                document.getElementById('point-size-display').textContent = e.target.value;
                if (neuronPoints && neuronPoints.material) {
                    neuronPoints.material.size = parseFloat(e.target.value);
                }
            });
            
            document.getElementById('neuron-opacity').addEventListener('input', (e) => {
                document.getElementById('neuron-opacity-display').textContent = e.target.value;
                if (neuronPoints && neuronPoints.material) {
                    neuronPoints.material.opacity = parseFloat(e.target.value);
                }
            });
            
            // Mask controls
            // Individual region checkboxes control visibility directly
            document.getElementById('mask-opacity').addEventListener('input', (e) => {
                document.getElementById('mask-opacity-display').textContent = e.target.value;
                updateMaskOpacity(parseFloat(e.target.value));
            });
            
            document.getElementById('mask-size').addEventListener('input', (e) => {
                document.getElementById('mask-size-display').textContent = e.target.value;
                updateMaskSize(parseFloat(e.target.value));
            });
            
            // Region checkboxes - handle individual region loading/unloading
            document.getElementById('region-list').addEventListener('change', async (e) => {
                if (e.target.type === 'checkbox') {
                    const regionId = parseInt(e.target.value);
                    if (e.target.checked) {
                        // Load and show this region
                        if (!regionMeshes[regionId]) {
                            await createRegionMesh(regionId);
                        }
                        if (regionMeshes[regionId]) {
                            regionMeshes[regionId].visible = true;
                        }
                    } else {
                        // Hide this region
                        if (regionMeshes[regionId]) {
                            regionMeshes[regionId].visible = false;
                        }
                    }
                }
            });
            
            // Spatial filter event listeners
            document.getElementById('x-min-slider').addEventListener('input', updateSpatialFilterControls);
            document.getElementById('x-max-slider').addEventListener('input', updateSpatialFilterControls);
            document.getElementById('y-min-slider').addEventListener('input', updateSpatialFilterControls);
            document.getElementById('y-max-slider').addEventListener('input', updateSpatialFilterControls);
            
            // Initialize dual range sliders
            initializeDualRangeSliders();
        }
        
        // Global variables for data bounds and filtering
        let dataBounds = { minX: 0, maxX: 1, minY: 0, maxY: 1, minZ: 0, maxZ: 1 };
        let spatialFilter = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
        
        function calculateDataBounds() {
            if (!neuronData || !neuronData.positions || neuronData.positions.length === 0) return;
            
            let minX = 1, maxX = 0, minY = 1, maxY = 0, minZ = 1, maxZ = 0;
            
            for (const pos of neuronData.positions) {
                minX = Math.min(minX, pos[0]);
                maxX = Math.max(maxX, pos[0]);
                minY = Math.min(minY, pos[1]);
                maxY = Math.max(maxY, pos[1]);
                minZ = Math.min(minZ, pos[2]);
                maxZ = Math.max(maxZ, pos[2]);
            }
            
            dataBounds = { minX, maxX, minY, maxY, minZ, maxZ };
            
            // Initialize spatial filter to full range
            spatialFilter = { xMin: minX, xMax: maxX, yMin: minY, yMax: maxY };
            
            console.log('Data bounds:', dataBounds);
        }
        
        function applySpatialFilter(positions) {
            const filtered = [];
            
            for (let i = 0; i < positions.length; i++) {
                const pos = positions[i];
                const x = pos[0], y = pos[1];
                
                if (x >= spatialFilter.xMin && x <= spatialFilter.xMax &&
                    y >= spatialFilter.yMin && y <= spatialFilter.yMax) {
                    filtered.push(pos);
                }
            }
            
            return filtered;
        }
        
        function createNeuronVisualization() {
            if (!neuronData) return;
            
            calculateDataBounds();
            updateNeuronVisualization();
        }
        
        function updateNeuronVisualization() {
            // Remove existing neuron points
            if (neuronPoints) {
                scene.remove(neuronPoints);
                if (neuronPoints.geometry) neuronPoints.geometry.dispose();
                if (neuronPoints.material) neuronPoints.material.dispose();
                neuronPoints = null;
            }
            
            const pointSize = parseFloat(document.getElementById('point-size').value);
            const opacity = parseFloat(document.getElementById('neuron-opacity').value);
            
            // Apply spatial filtering
            const filteredPositions = applySpatialFilter(neuronData.positions);
            
            if (filteredPositions.length === 0) {
                console.log('No neurons pass the current filters');
                return;
            }
            
            console.log(`Rendering ${filteredPositions.length} neurons after filtering`);
            
            // Create geometry for points
            const geometry = new THREE.BufferGeometry();
            
            // Convert positions to flat array
            const positions = new Float32Array(filteredPositions.length * 3);
            for (let i = 0; i < filteredPositions.length; i++) {
                positions[i * 3] = filteredPositions[i][0];
                positions[i * 3 + 1] = filteredPositions[i][1];
                positions[i * 3 + 2] = filteredPositions[i][2];
            }
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            // Create material for points
            const material = new THREE.PointsMaterial({
                size: pointSize,
                color: 0x00aaff,  // Bright blue
                transparent: true,
                opacity: opacity,
                sizeAttenuation: true,
                alphaTest: 0.05  // Helps with transparency rendering
            });
            
            // Create points mesh
            neuronPoints = new THREE.Points(geometry, material);
            scene.add(neuronPoints);
            
            console.log('Neuron visualization updated');
        }
        
        // Function removed - individual checkboxes control regions directly
        
        async function loadRegionMasks() {
            // Load masks for checked regions
            const checkboxes = document.querySelectorAll('#region-list input:checked');
            
            for (const checkbox of checkboxes) {
                const regionId = parseInt(checkbox.value);
                if (!regionMeshes[regionId]) {
                    await createRegionMesh(regionId);
                }
                if (regionMeshes[regionId]) {
                    regionMeshes[regionId].visible = true;
                }
            }
        }
        
        async function createRegionMesh(regionId) {
            try {
                console.log(`Loading region ${regionId}...`);
                const response = await fetch(`/api/region_surface/${regionId}`);
                const data = await response.json();
                
                if (data.vertices.length === 0) {
                    console.log(`No surface found for region ${regionId}`);
                    return;
                }
                
                // Create geometry from vertices - simple points approach
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(data.vertices.length * 3);
                
                // Fill positions array
                for (let i = 0; i < data.vertices.length; i++) {
                    positions[i * 3] = data.vertices[i][0];
                    positions[i * 3 + 1] = data.vertices[i][1];
                    positions[i * 3 + 2] = data.vertices[i][2];
                }
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                
                // Create material with random color
                const color = new THREE.Color().setHSL(Math.random(), 0.8, 0.6);
                const material = new THREE.PointsMaterial({
                    color: color,
                    size: 1.5,
                    transparent: true,
                    opacity: 0.8,
                    sizeAttenuation: true
                });
                
                // Create points mesh
                const mesh = new THREE.Points(geometry, material);
                regionMeshes[regionId] = mesh;
                scene.add(mesh);
                
                console.log(`Region ${regionId} loaded with ${data.vertices.length} surface points`);
                
            } catch (error) {
                console.error(`Error loading region ${regionId}:`, error);
            }
        }
        
        function hideAllMasks() {
            Object.values(regionMeshes).forEach(mesh => {
                mesh.visible = false;
            });
        }
        
        function updateMaskOpacity(opacity) {
            Object.values(regionMeshes).forEach(mesh => {
                mesh.material.opacity = opacity;
            });
        }
        
        function updateMaskSize(size) {
            Object.values(regionMeshes).forEach(mesh => {
                mesh.material.size = size;
            });
        }
        
        function updateRegionVisibility() {
            const checkboxes = document.querySelectorAll('#region-list input');
            
            checkboxes.forEach(checkbox => {
                const regionId = parseInt(checkbox.value);
                if (regionMeshes[regionId]) {
                    // Region is visible only if its individual checkbox is checked
                    regionMeshes[regionId].visible = checkbox.checked;
                }
            });
        }
        
        async function selectAllRegions() {
            document.querySelectorAll('#region-list input[type="checkbox"]').forEach(cb => {
                cb.checked = true;
            });
            // Load all checked regions
            await loadRegionMasks();
        }
        
        function clearAllRegions() {
            document.querySelectorAll('#region-list input[type="checkbox"]').forEach(cb => {
                cb.checked = false;
            });
            hideAllMasks();
        }
        
        async function selectTopRegions() {
            clearAllRegions();
            // Select first 10 regions
            for (let i = 1; i <= Math.min(10, maskData.num_regions); i++) {
                const checkbox = document.getElementById(`region-${i}`);
                if (checkbox) {
                    checkbox.checked = true;
                }
            }
            // Load the selected regions
            await loadRegionMasks();
        }
        
        function initializeDualRangeSliders() {
            // Initialize range fills
            updateRangeFill('x', 0, 100);
            updateRangeFill('y', 0, 100);
        }
        
        function updateRangeFill(axis, minValue, maxValue) {
            const fill = document.getElementById(axis + '-range-fill');
            fill.style.left = minValue + '%';
            fill.style.width = (maxValue - minValue) + '%';
        }
        
        function updateSpatialFilterControls() {
            // Get slider values
            const xMinSlider = parseFloat(document.getElementById('x-min-slider').value);
            const xMaxSlider = parseFloat(document.getElementById('x-max-slider').value);
            const yMinSlider = parseFloat(document.getElementById('y-min-slider').value);
            const yMaxSlider = parseFloat(document.getElementById('y-max-slider').value);
            
            // Ensure min <= max
            const xMin = Math.min(xMinSlider, xMaxSlider);
            const xMax = Math.max(xMinSlider, xMaxSlider);
            const yMin = Math.min(yMinSlider, yMaxSlider);
            const yMax = Math.max(yMinSlider, yMaxSlider);
            
            // Update range fills
            updateRangeFill('x', xMin, xMax);
            updateRangeFill('y', yMin, yMax);
            
            // Convert to actual coordinate values
            spatialFilter.xMin = dataBounds.minX + (xMin / 100) * (dataBounds.maxX - dataBounds.minX);
            spatialFilter.xMax = dataBounds.minX + (xMax / 100) * (dataBounds.maxX - dataBounds.minX);
            spatialFilter.yMin = dataBounds.minY + (yMin / 100) * (dataBounds.maxY - dataBounds.minY);
            spatialFilter.yMax = dataBounds.minY + (yMax / 100) * (dataBounds.maxY - dataBounds.minY);
            
            // Update display values
            document.getElementById('x-min-value').textContent = spatialFilter.xMin.toFixed(2);
            document.getElementById('x-max-value').textContent = spatialFilter.xMax.toFixed(2);
            document.getElementById('y-min-value').textContent = spatialFilter.yMin.toFixed(2);
            document.getElementById('y-max-value').textContent = spatialFilter.yMax.toFixed(2);
            
            // Update visualization
            updateNeuronVisualization();
            updateRegionVisibility(); // Also update regions with spatial filter
        }
        
        function resetSpatialFilter() {
            // Reset sliders to full range
            document.getElementById('x-min-slider').value = 0;
            document.getElementById('x-max-slider').value = 100;
            document.getElementById('y-min-slider').value = 0;
            document.getElementById('y-max-slider').value = 100;
            
            // Update everything
            updateSpatialFilterControls();
        }
        
        function resetCamera() {
            camera.position.set(150, 150, 150);
            controls.target.set(50, 50, 50);
            controls.reset();
        }
        
        // Initialize everything
        document.addEventListener('DOMContentLoaded', () => {
            initThreeJS();
            loadData();
        });
    </script>
</body>
</html>
"""
    )
    response = make_response(render_template_string(html_template))
    # Add cache-busting headers to force browser refresh
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/favicon.ico")
def favicon():
    """Return empty favicon to avoid 404 errors."""
    from flask import Response

    return Response("", 204)


@app.route("/api/neuron_data")
def api_neuron_data():
    """Return neuron position data."""
    try:
        if NEURON_DATA is None:
            return jsonify({"error": "Neuron data not loaded"}), 500

        return jsonify(
            {
                "num_neurons": int(NEURON_DATA["num_neurons"]),
                "total_neurons": int(NEURON_DATA["total_neurons"]),
                "positions": NEURON_DATA["positions"].tolist(),
            }
        )
    except Exception as e:
        print(f"Error in api_neuron_data: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/mask_data")
def api_mask_data():
    """Return mask dataset metadata."""
    try:
        if MASK_DATA is None:
            return jsonify({"error": "Mask data not loaded"}), 500

        return jsonify(
            {
                "num_regions": int(MASK_DATA["num_regions"]),
                "region_names": MASK_DATA["region_names"],
                "grid_shape": MASK_DATA["grid_shape"],
            }
        )
    except Exception as e:
        print(f"Error in api_mask_data: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/region_surface/<int:region_id>")
def api_region_surface(region_id):
    """Return surface vertices for a brain region."""
    try:
        if MASK_DATA is None:
            return jsonify({"error": "Mask data not loaded"}), 500

        if region_id < 1 or region_id > MASK_DATA["num_regions"]:
            return jsonify(
                {
                    "error": f"Invalid region ID {region_id}, valid range: 1-{MASK_DATA['num_regions']}"
                }
            ), 400

        # Extract surface for the region
        vertices = extract_region_surfaces(
            MASK_DATA["label_volume"], region_id, subsample=4
        )
        return jsonify({"region_id": region_id, "vertices": vertices.tolist()})
    except Exception as e:
        print(f"Error in api_region_surface for region {region_id}: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"region_id": region_id, "vertices": []})


def main():
    global NEURON_DATA, MASK_DATA

    parser = argparse.ArgumentParser(
        description="Simple web viewer for brain neuron positions and masks"
    )
    parser.add_argument(
        "--subject-h5",
        required=True,
        help="Path to subject HDF5 file with neuron positions",
    )
    parser.add_argument(
        "--mask-h5", required=True, help="Path to mask HDF5 file with region data"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8053, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Validate files exist
    if not os.path.exists(args.subject_h5):
        raise FileNotFoundError(f"Subject HDF5 file not found: {args.subject_h5}")
    if not os.path.exists(args.mask_h5):
        raise FileNotFoundError(f"Mask HDF5 file not found: {args.mask_h5}")

    # Load data
    print("Loading data...")
    NEURON_DATA = load_neuron_positions(
        args.subject_h5, max_neurons=100000
    )  # Show all neurons - no subsampling
    MASK_DATA = load_mask_data(args.mask_h5)
    print("Data loaded successfully!")

    print(f"\nStarting server at http://{args.host}:{args.port}")
    print("Open this URL in your browser to view the visualization.")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
