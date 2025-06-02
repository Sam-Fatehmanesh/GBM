// Global variables for animation control
let currentScene = null;
let currentRenderer = null;
let currentCamera = null;
let currentControls = null;
let baselineMesh = null;
let predictionMesh = null;
let baselineData = null;
let predictionData = null;
let animationFrameId = null;

// Global animation state
let baselineAnimationState = null;
let predictionAnimationState = null;

// Time control variables
let isPaused = false;
let manualTimeControl = false;
let totalTimeVolumes = 0;

// Spatial filtering variables
let spatialFilter = {
  xMin: null,
  xMax: null,
  yMin: null,
  yMax: null
};

// Data bounds for slider initialization
let dataBounds = {
  minX: 0,
  maxX: 100,
  minY: 0,
  maxY: 100
};

// Region overlay system
let regionOverlays = {};
let regionColors = {};

// Generate unique colors for regions
function generateRegionColor(regionName) {
  if (regionColors[regionName]) {
    return regionColors[regionName];
  }
  
  // Generate a unique color based on region name hash
  let hash = 0;
  for (let i = 0; i < regionName.length; i++) {
    hash = regionName.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  // Convert hash to HSL color with high saturation and brightness
  const hue = Math.abs(hash) % 360;
  const saturation = 70 + (Math.abs(hash) % 30); // 70-100%
  const lightness = 50 + (Math.abs(hash) % 20);  // 50-70%
  
  // Convert HSL to RGB
  const c = (1 - Math.abs(2 * lightness / 100 - 1)) * saturation / 100;
  const x = c * (1 - Math.abs((hue / 60) % 2 - 1));
  const m = lightness / 100 - c / 2;
  
  let r, g, b;
  if (hue < 60) { r = c; g = x; b = 0; }
  else if (hue < 120) { r = x; g = c; b = 0; }
  else if (hue < 180) { r = 0; g = c; b = x; }
  else if (hue < 240) { r = 0; g = x; b = c; }
  else if (hue < 300) { r = x; g = 0; b = c; }
  else { r = c; g = 0; b = x; }
  
  r = Math.round((r + m) * 255);
  g = Math.round((g + m) * 255);
  b = Math.round((b + m) * 255);
  
  const color = (r << 16) | (g << 8) | b;
  regionColors[regionName] = color;
  return color;
}

// Spatial filtering helper functions
function applySpatialFilter(coords) {
  if (!coords || coords.length === 0) return coords;
  
  // If spatial filter is not properly initialized, return all coords
  if (spatialFilter.xMin === null || spatialFilter.xMax === null || 
      spatialFilter.yMin === null || spatialFilter.yMax === null) {
    return coords;
  }
  
  return coords.filter(c => {
    const x = c[2]; // x coordinate
    const y = c[1]; // y coordinate
    
    // Apply X filter
    if (x < spatialFilter.xMin || x > spatialFilter.xMax) return false;
    
    // Apply Y filter
    if (y < spatialFilter.yMin || y > spatialFilter.yMax) return false;
    
    return true;
  });
}

function updateSpatialFilter() {
  // Get X-axis slider values and convert to actual coordinate values
  const xMinSlider = parseFloat(document.getElementById('x-min-slider').value);
  const xMaxSlider = parseFloat(document.getElementById('x-max-slider').value);
  
  // Ensure min <= max by swapping if necessary for X-axis
  const xMin = Math.min(xMinSlider, xMaxSlider);
  const xMax = Math.max(xMinSlider, xMaxSlider);
  
  // Convert X-axis slider percentages to actual coordinate values
  spatialFilter.xMin = dataBounds.minX + (xMin / 100) * (dataBounds.maxX - dataBounds.minX);
  spatialFilter.xMax = dataBounds.minX + (xMax / 100) * (dataBounds.maxX - dataBounds.minX);
  
  // Update X-axis display values
  document.getElementById('x-min-value').textContent = Math.round(spatialFilter.xMin);
  document.getElementById('x-max-value').textContent = Math.round(spatialFilter.xMax);
  
  // Update X-axis range fill visuals
  updateRangeFill('x', xMin, xMax);
  
  // Y-axis is handled by the custom slider, so we don't process it here
  
  // Recreate all visualizations with new filter
  updateAnimation(); // Updates baseline and prediction activities
  updateRegionOverlays(); // Updates region overlays
}

function updateRangeFill(axis, minValue, maxValue) {
  if (axis === 'x') {
    const fill = document.getElementById('x-range-fill');
    const minPercent = minValue;
    const maxPercent = maxValue;
    fill.style.left = minPercent + '%';
    fill.style.width = (maxPercent - minPercent) + '%';
  } else if (axis === 'y') {
    const fill = document.getElementById('y-range-fill');
    // For vertical sliders, we need to calculate from bottom
    // Higher values should be at the top, lower values at the bottom
    const minPercent = minValue;
    const maxPercent = maxValue;
    fill.style.bottom = minPercent + '%';
    fill.style.height = (maxPercent - minPercent) + '%';
  }
}

function initializeDualRangeSliders() {
  // Initialize X-axis range fill
  updateRangeFill('x', 0, 100);
  
  // Initialize Y-axis range fill  
  updateRangeFill('y', 0, 100);
  
  // Set up event listeners for X-axis sliders
  const xMinSlider = document.getElementById('x-min-slider');
  const xMaxSlider = document.getElementById('x-max-slider');
  
  // X-axis slider event listeners
  xMinSlider.addEventListener('input', () => {
    updateRangeFill('x', Math.min(xMinSlider.value, xMaxSlider.value), Math.max(xMinSlider.value, xMaxSlider.value));
  });
  
  xMaxSlider.addEventListener('input', () => {
    updateRangeFill('x', Math.min(xMinSlider.value, xMaxSlider.value), Math.max(xMinSlider.value, xMaxSlider.value));
  });
  
  // Initialize custom Y-axis slider
  initializeYAxisSlider();
}

function initializeYAxisSlider() {
  const container = document.querySelector('.y-dual-range-container');
  const minHandle = document.getElementById('y-min-handle');
  const maxHandle = document.getElementById('y-max-handle');
  
  let isDragging = false;
  let dragHandle = null;
  let containerRect = null;
  
  // Y-axis slider values (0-100)
  let yMinValue = 0;
  let yMaxValue = 100;
  
  function updateYAxisVisuals(updateFilter = true) {
    // Update handle positions
    minHandle.style.bottom = yMinValue + '%';
    maxHandle.style.bottom = yMaxValue + '%';
    
    // Update range fill
    const fill = document.getElementById('y-range-fill');
    fill.style.bottom = yMinValue + '%';
    fill.style.height = (yMaxValue - yMinValue) + '%';
    
    // Only update spatial filter if requested and data bounds are available
    if (updateFilter && dataBounds.minY !== undefined) {
      updateSpatialFilterFromYAxis(yMinValue, yMaxValue);
    }
  }
  
  function updateSpatialFilterFromYAxis(minVal, maxVal) {
    // Convert to actual coordinate values and update spatial filter
    spatialFilter.yMin = dataBounds.minY + (minVal / 100) * (dataBounds.maxY - dataBounds.minY);
    spatialFilter.yMax = dataBounds.minY + (maxVal / 100) * (dataBounds.maxY - dataBounds.minY);
    
    // Update display values
    document.getElementById('y-min-value').textContent = Math.round(spatialFilter.yMin);
    document.getElementById('y-max-value').textContent = Math.round(spatialFilter.yMax);
    
    // Update all visualizations with new filter
    updateAnimation(); // Updates baseline and prediction activities
    updateRegionOverlays(); // Updates region overlays
  }
  
  function startDrag(e, handle) {
    isDragging = true;
    dragHandle = handle;
    containerRect = container.getBoundingClientRect();
    
    handle.classList.add('dragging');
    document.addEventListener('mousemove', onDrag);
    document.addEventListener('mouseup', stopDrag);
    e.preventDefault();
  }
  
  function onDrag(e) {
    if (!isDragging || !dragHandle || !containerRect) return;
    
    const y = e.clientY - containerRect.top;
    const containerHeight = containerRect.height;
    
    // Convert to percentage (inverted because we're measuring from top but want bottom-up values)
    let percentage = ((containerHeight - y) / containerHeight) * 100;
    percentage = Math.max(0, Math.min(100, percentage));
    
    if (dragHandle.dataset.type === 'min') {
      yMinValue = percentage;
      // Ensure min doesn't go above max
      if (yMinValue > yMaxValue) {
        yMaxValue = yMinValue;
      }
    } else if (dragHandle.dataset.type === 'max') {
      yMaxValue = percentage;
      // Ensure max doesn't go below min
      if (yMaxValue < yMinValue) {
        yMinValue = yMaxValue;
      }
    }
    
    updateYAxisVisuals();
  }
  
  function stopDrag() {
    if (dragHandle) {
      dragHandle.classList.remove('dragging');
    }
    isDragging = false;
    dragHandle = null;
    containerRect = null;
    document.removeEventListener('mousemove', onDrag);
    document.removeEventListener('mouseup', stopDrag);
  }
  
  // Add event listeners
  minHandle.addEventListener('mousedown', (e) => startDrag(e, minHandle));
  maxHandle.addEventListener('mousedown', (e) => startDrag(e, maxHandle));
  
  // Initialize visuals without updating filter (will be updated later when data bounds are set)
  updateYAxisVisuals(false);
  
  // Store function reference for later initialization
  window.finalizeYAxisSlider = () => {
    updateYAxisVisuals(true);
  };
}

function calculateDataBounds(data) {
  if (!data || !data.coords || data.coords.length === 0) return null;
  
  const coords = data.coords;
  let minX = coords[0][2], maxX = coords[0][2];
  let minY = coords[0][1], maxY = coords[0][1];
  
  for (let i = 1; i < coords.length; i++) {
    const x = coords[i][2];
    const y = coords[i][1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  
  return { minX, maxX, minY, maxY };
}

function initializeSpatialControls() {
  // Set default bounds based on baseline data
  if (baselineData) {
    const bounds = calculateDataBounds(baselineData);
    if (bounds) {
      dataBounds = bounds;
      
      // Initialize sliders to full range
      document.getElementById('x-min-slider').value = 0;
      document.getElementById('x-max-slider').value = 100;
      
      // Update initial display values
      document.getElementById('x-min-value').textContent = bounds.minX;
      document.getElementById('x-max-value').textContent = bounds.maxX;
      document.getElementById('y-min-value').textContent = bounds.minY;
      document.getElementById('y-max-value').textContent = bounds.maxY;
      
      // Set initial filter to full range (no filtering)
      spatialFilter.xMin = bounds.minX;
      spatialFilter.xMax = bounds.maxX;
      spatialFilter.yMin = bounds.minY;
      spatialFilter.yMax = bounds.maxY;
      
      // Finalize Y-axis slider now that bounds are set
      if (window.finalizeYAxisSlider) {
        window.finalizeYAxisSlider();
      }
    }
  }
}

// Time control functions
function togglePause() {
  isPaused = !isPaused;
  const btn = document.getElementById('pause-btn');
  btn.textContent = isPaused ? 'Play' : 'Pause';
}

function setManualTime(timeIndex) {
  manualTimeControl = true;
  
  // Update baseline animation to specific time
  if (baselineAnimationState && document.getElementById('show-baseline').checked) {
    baselineAnimationState.currentVolIdx = timeIndex % baselineAnimationState.volumeIndices.length;
    updateBaselineToCurrentTime();
  }
  
  // Update prediction animation to specific time
  if (predictionAnimationState && document.getElementById('show-predictions').checked) {
    predictionAnimationState.currentVolIdx = timeIndex % predictionAnimationState.volumeIndices.length;
    updatePredictionToCurrentTime();
  }
  
  // Update time display
  document.getElementById('current-time').textContent = timeIndex;
  
  // Reset manual control after a brief delay
  setTimeout(() => {
    manualTimeControl = false;
  }, 100);
}

function updateTimeControls() {
  // Calculate total time volumes from available data
  let maxVolumes = 0;
  if (baselineAnimationState) {
    maxVolumes = Math.max(maxVolumes, baselineAnimationState.volumeIndices.length);
  }
  if (predictionAnimationState) {
    maxVolumes = Math.max(maxVolumes, predictionAnimationState.volumeIndices.length);
  }
  
  totalTimeVolumes = maxVolumes;
  
  if (totalTimeVolumes > 0) {
    document.getElementById('time-slider').max = totalTimeVolumes - 1;
    document.getElementById('total-time').textContent = totalTimeVolumes - 1;
  }
}

async function loadRegionMask(regionName) {
  try {
    const res = await fetch(`/regions/${encodeURIComponent(regionName)}/mask_json`);
    const data = await res.json();
    return {
      name: regionName,
      coords: data.coords,
      color: generateRegionColor(regionName)
    };
  } catch (err) {
    console.error(`Failed to load region ${regionName}:`, err);
    return null;
  }
}

function createRegionOverlay(regionData) {
  if (!regionData || !regionData.coords || regionData.coords.length === 0) return null;
  
  // Apply spatial filtering to region coordinates
  const filteredCoords = applySpatialFilter(regionData.coords);
  if (filteredCoords.length === 0) return null;
  
  const cubeGeo = new THREE.BoxGeometry(1, 1, 1);
  const cubeMat = new THREE.MeshBasicMaterial({ 
    color: regionData.color,
    transparent: true, 
    opacity: 0.4, // Increased opacity for wireframe visibility
    wireframe: true, // Use wireframe to show region boundaries without blocking activity
    depthTest: true,
    depthWrite: false // Don't write to depth buffer so activity can render on top
  });
  
  const mesh = new THREE.InstancedMesh(cubeGeo, cubeMat, filteredCoords.length);
  const dummy = new THREE.Object3D();
  
  filteredCoords.forEach((c, i) => {
    // c is [z, y, x] format - use original coordinates without reflection
    dummy.position.set(c[2], c[1], c[0]);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  });
  
  // Set rendering order to render regions before activity
  mesh.renderOrder = 1;
  
  return mesh;
}

async function updateRegionOverlays() {
  // Remove existing overlays
  Object.values(regionOverlays).forEach(mesh => {
    if (mesh) {
      currentScene.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
  });
  regionOverlays = {};
  
  // Get selected regions
  const selectedRegions = getSelectedRegions();
  
  // Load and create overlays for selected regions
  for (const regionName of selectedRegions) {
    const regionData = await loadRegionMask(regionName);
    if (regionData) {
      const overlay = createRegionOverlay(regionData);
      if (overlay) {
        regionOverlays[regionName] = overlay;
        currentScene.add(overlay);
      }
    }
  }
}

// Update fetchRegions to add event listeners for region selection
async function fetchRegions() {
  // Show loading message and hide container
  document.getElementById('loading-masks').style.display = 'block';
  document.getElementById('regions-container').style.display = 'none';
  
  const res = await fetch('/regions');
  const data = await res.json();
  const container = document.getElementById('regions-container');
  
  // Clear existing content
  container.innerHTML = '';
  
  data.regions.forEach(region => {
    const checkboxDiv = document.createElement('div');
    checkboxDiv.className = 'region-checkbox';
    
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = `region-${region}`;
    checkbox.value = region;
    
    // Add event listener for region overlay updates
    checkbox.addEventListener('change', updateRegionOverlays);
    
    const label = document.createElement('label');
    label.htmlFor = `region-${region}`;
    label.textContent = region;
    
    checkboxDiv.appendChild(checkbox);
    checkboxDiv.appendChild(label);
    container.appendChild(checkboxDiv);
  });
  
  // Hide loading message and show container
  document.getElementById('loading-masks').style.display = 'none';
  container.style.display = 'block';
}

function getSelectedRegions() {
  const checkboxes = document.querySelectorAll('#regions-container input[type="checkbox"]:checked');
  return Array.from(checkboxes).map(cb => cb.value);
}

async function simulate() {
  const regions = getSelectedRegions();
  const fraction = parseFloat(document.getElementById('fraction').value);
  const steps = parseInt(document.getElementById('steps').value, 10);
  const statusDiv = document.getElementById('status');
  
  if (regions.length === 0) {
    statusDiv.innerHTML = '⚠ Please select at least one brain region';
    return;
  }
  
  statusDiv.innerHTML = '<span class="loading-indicator"></span>Submitting simulation...';

  const res = await fetch('/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ selected_regions: regions, activation_fraction: fraction, prediction_steps: steps })
  });
  const result = await res.json();
  const jobId = result.job_id;
  statusDiv.innerHTML = `<span class="loading-indicator"></span>Job: ${jobId}`;
  pollStatus(jobId);
}

async function pollStatus(jobId) {
  const statusDiv = document.getElementById('status');
  const interval = setInterval(async () => {
    const res = await fetch(`/status/${jobId}`);
    const data = await res.json();
    // Unwrap nested status
    const jobStatus = data.status.status || data.status;
    statusDiv.innerHTML = `<span class="loading-indicator"></span>Status: ${jobStatus}`;
    if (jobStatus === 'done' || jobStatus === 'error') {
      clearInterval(interval);
      if (jobStatus === 'done') {
        statusDiv.innerHTML = '✓ Simulation complete';
        loadResults(jobId);
      } else {
        statusDiv.innerHTML = '✗ Simulation failed';
      }
    }
  }, 2000);
}

async function loadResults(jobId) {
  const statusDiv = document.getElementById('status');
  statusDiv.innerHTML = '<span class="loading-indicator"></span>Loading results...';

  const res = await fetch(`/result/${jobId}`);
  const data = await res.json();
  if (data.status.status !== 'done') {
    statusDiv.innerHTML = '✗ Job not complete';
    return;
  }

  // Load both heatmaps
  const heatmapAll = document.getElementById('heatmap-all');
  const heatmapTop12 = document.getElementById('heatmap-top12');
  heatmapAll.src = `/results/${jobId}/heatmap.png`;
  heatmapTop12.src = `/results/${jobId}/heatmap_top12.png`;

  document.getElementById('download-summary-json').href = `/results/${jobId}/summary.json`;
  document.getElementById('download-summary-csv').href = `/results/${jobId}/summary.csv`;
  document.getElementById('download-probabilities').href = `/results/${jobId}/predicted_probabilities.npy`;

  // Show results panel
  const resultsPanel = document.getElementById('results');
  resultsPanel.classList.add('visible');
  
  statusDiv.innerHTML = '✓ Results ready';
  
  // Load prediction data and enable predictions checkbox
  await loadPredictionData(jobId);
  document.getElementById('show-predictions').disabled = false;
}

async function loadPredictionData(jobId) {
  try {
    const res = await fetch(`/results/${jobId}/predicted_sequence_json`);
    const data = await res.json();
    predictionData = {
      coords: data.coords,
      Z: data.Z,
      zStart: data.zStart,
      color: 0xff00ff, // Neon violet
      opacity: 0.8
    };
  } catch (err) {
    console.error('Failed to load prediction data:', err);
  }
}

function initViewer() {
  const container = document.getElementById('viewer');
  
  // Clear any existing scene
  if (currentRenderer) {
    container.removeChild(currentRenderer.domElement);
    currentRenderer.dispose();
  }
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }
  
  // Clear region overlays
  Object.values(regionOverlays).forEach(mesh => {
    if (mesh) {
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
  });
  regionOverlays = {};
  
  // Three.js setup for fullscreen background
  currentScene = new THREE.Scene();
  currentCamera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  currentRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  currentRenderer.setSize(window.innerWidth, window.innerHeight);
  currentRenderer.setClearColor(0x000000, 1); // Pure black background
  container.appendChild(currentRenderer.domElement);
  
  currentControls = new THREE.OrbitControls(currentCamera, currentRenderer.domElement);
  currentControls.enableDamping = true;
  currentControls.dampingFactor = 0.05;
  
  // Handle window resize
  function onWindowResize() {
    currentCamera.aspect = window.innerWidth / window.innerHeight;
    currentCamera.updateProjectionMatrix();
    currentRenderer.setSize(window.innerWidth, window.innerHeight);
  }
  window.addEventListener('resize', onWindowResize);
  
  // Start animation loop
  animate();
  
  // Load region overlays for any pre-selected regions
  setTimeout(() => {
    updateRegionOverlays();
  }, 100);
}

async function initBaselineData() {
  try {
    const res = await fetch('/baseline_mask_json');
    const data = await res.json();
    baselineData = {
      coords: data.coords,
      Z: data.Z,
      zStart: data.zStart,
      color: 0x04ff00, // Neon green
      opacity: 0.8
    };
    
    // Set up camera position based on baseline data
    if (baselineData.coords.length > 0) {
      setupCamera();
    }
    
    // Initialize spatial controls with data bounds
    initializeSpatialControls();
    
    updateAnimation();
  } catch(e) {
    console.error('Baseline data error:', e);
  }
}

function setupCamera() {
  if (!baselineData || !baselineData.coords.length) return;
  
  // Calculate center and bounds from first volume
  const firstVolCoords = baselineData.coords.filter(c => Math.floor(c[0] / baselineData.Z) === 0);
  if (firstVolCoords.length === 0) return;
  
  const positions = firstVolCoords.map(c => [c[2], c[1], (c[0] + baselineData.zStart) % baselineData.Z]);
  const xs = positions.map(p => p[0]);
  const ys = positions.map(p => p[1]);
  const zs = positions.map(p => p[2]);
  
  const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
  const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
  const centerZ = (Math.min(...zs) + Math.max(...zs)) / 2;
  
  const sizeX = Math.max(...xs) - Math.min(...xs);
  const sizeY = Math.max(...ys) - Math.min(...ys);
  const sizeZ = Math.max(...zs) - Math.min(...zs);
  const maxDim = Math.max(sizeX, sizeY, sizeZ);
  
  currentCamera.position.set(centerX + maxDim * 1.5, centerY + maxDim * 1.5, centerZ + maxDim * 1.5);
  currentControls.target.set(centerX, centerY, centerZ);
  currentControls.update();
}

function updateAnimation() {
  if (!currentScene) return;
  
  // Clear existing activity meshes (but preserve region overlays)
  if (baselineMesh) {
    currentScene.remove(baselineMesh);
    baselineMesh.geometry.dispose();
    baselineMesh.material.dispose();
    baselineMesh = null;
  }
  if (predictionMesh) {
    currentScene.remove(predictionMesh);
    predictionMesh.geometry.dispose();
    predictionMesh.material.dispose();
    predictionMesh = null;
  }
  
  const showBaseline = document.getElementById('show-baseline').checked;
  const showPredictions = document.getElementById('show-predictions').checked;
  
  // Create baseline mesh if enabled and data available
  if (showBaseline && baselineData) {
    createBaselineAnimation();
  }
  
  // Create prediction mesh if enabled and data available
  if (showPredictions && predictionData) {
    createPredictionVisualization();
  }
  
  // Region overlays are preserved and don't need to be recreated
}

function createBaselineAnimation() {
  if (!baselineData) return;
  
  const cubeGeo = new THREE.BoxGeometry(1, 1, 1);
  const cubeMat = new THREE.MeshBasicMaterial({ 
    color: baselineData.color, 
    transparent: true, 
    opacity: baselineData.opacity,
    wireframe: false,
    depthTest: true,
    depthWrite: true
  });
  
  // Initialize persistent animation state
  baselineAnimationState = {
    volumes: {},
    volumeIndices: [],
    currentVolIdx: 0,
    lastUpdateTime: performance.now(),
    updateInterval: 500,
    cubeGeo: cubeGeo,
    cubeMat: cubeMat
  };
  
  // Group coords by volume and apply spatial filtering
  baselineData.coords.forEach(c => {
    const sliceIdx = c[0];
    const volIdx = Math.floor(sliceIdx / baselineData.Z);
    baselineAnimationState.volumes[volIdx] = baselineAnimationState.volumes[volIdx] || [];
    baselineAnimationState.volumes[volIdx].push(c);
  });
  
  // Apply spatial filtering to each volume
  Object.keys(baselineAnimationState.volumes).forEach(volIdx => {
    baselineAnimationState.volumes[volIdx] = applySpatialFilter(baselineAnimationState.volumes[volIdx]);
  });
  
  baselineAnimationState.volumeIndices = Object.keys(baselineAnimationState.volumes).map(n => parseInt(n)).sort((a, b) => a - b);
  
  if (baselineAnimationState.volumeIndices.length === 0) return;
  
  // Create initial mesh
  baselineMesh = createMeshForVolume(baselineAnimationState.currentVolIdx, baselineAnimationState);
  if (baselineMesh) {
    currentScene.add(baselineMesh);
  }
  
  // Update time controls
  updateTimeControls();
}

function createMeshForVolume(volIdx, animationState) {
  const coords = animationState.volumes[animationState.volumeIndices[volIdx]];
  if (!coords || coords.length === 0) return null;
  
  const mesh = new THREE.InstancedMesh(animationState.cubeGeo, animationState.cubeMat, coords.length);
  const dummy = new THREE.Object3D();
  
  coords.forEach((c, i) => {
    const zStart = animationState === baselineAnimationState ? baselineData.zStart : predictionData.zStart;
    const Z = animationState === baselineAnimationState ? baselineData.Z : predictionData.Z;
    dummy.position.set(c[2], c[1], (c[0] + zStart) % Z);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  });
  
  // Set higher render order so activity renders after regions
  mesh.renderOrder = 2;
  
  return mesh;
}

function updateBaselineToCurrentTime() {
  if (!baselineAnimationState || !baselineData) return;
  
  // Remove current mesh
  if (baselineMesh) {
    currentScene.remove(baselineMesh);
    baselineMesh.geometry.dispose();
    baselineMesh.material.dispose();
  }
  
  // Create new mesh for current volume
  baselineMesh = createMeshForVolume(baselineAnimationState.currentVolIdx, baselineAnimationState);
  if (baselineMesh) {
    currentScene.add(baselineMesh);
  }
}

function createPredictionVisualization() {
  if (!predictionData) return;
  
  const cubeGeo = new THREE.BoxGeometry(1, 1, 1);
  const cubeMat = new THREE.MeshBasicMaterial({ 
    color: predictionData.color, 
    transparent: true, 
    opacity: predictionData.opacity,
    wireframe: false,
    depthTest: true,
    depthWrite: true
  });
  
  // Initialize persistent animation state
  predictionAnimationState = {
    volumes: {},
    volumeIndices: [],
    currentVolIdx: 0,
    lastUpdateTime: performance.now(),
    updateInterval: 500,
    cubeGeo: cubeGeo,
    cubeMat: cubeMat
  };
  
  // Group coords by volume index (time volume) like baseline and apply spatial filtering
  predictionData.coords.forEach(c => {
    const sliceIdx = c[0];
    const volIdx = Math.floor(sliceIdx / predictionData.Z);
    predictionAnimationState.volumes[volIdx] = predictionAnimationState.volumes[volIdx] || [];
    predictionAnimationState.volumes[volIdx].push(c);
  });
  
  // Apply spatial filtering to each volume
  Object.keys(predictionAnimationState.volumes).forEach(volIdx => {
    predictionAnimationState.volumes[volIdx] = applySpatialFilter(predictionAnimationState.volumes[volIdx]);
  });
  
  predictionAnimationState.volumeIndices = Object.keys(predictionAnimationState.volumes).map(n => parseInt(n)).sort((a, b) => a - b);
  
  if (predictionAnimationState.volumeIndices.length === 0) return;
  
  // Create initial mesh
  predictionMesh = createMeshForVolume(predictionAnimationState.currentVolIdx, predictionAnimationState);
  if (predictionMesh) {
    currentScene.add(predictionMesh);
  }
  
  // Update time controls
  updateTimeControls();
}

function updatePredictionToCurrentTime() {
  if (!predictionAnimationState || !predictionData) return;
  
  // Remove current mesh
  if (predictionMesh) {
    currentScene.remove(predictionMesh);
    predictionMesh.geometry.dispose();
    predictionMesh.material.dispose();
  }
  
  // Create new mesh for current volume
  predictionMesh = createMeshForVolume(predictionAnimationState.currentVolIdx, predictionAnimationState);
  if (predictionMesh) {
    currentScene.add(predictionMesh);
  }
}

function updateBaselineAnimation(currentTime) {
  if (!baselineAnimationState || !baselineData) return;
  if (isPaused || manualTimeControl) return;
  
  if (currentTime - baselineAnimationState.lastUpdateTime >= baselineAnimationState.updateInterval) {
    // Advance to next volume
    baselineAnimationState.currentVolIdx = (baselineAnimationState.currentVolIdx + 1) % baselineAnimationState.volumeIndices.length;
    baselineAnimationState.lastUpdateTime = currentTime;
    
    updateBaselineToCurrentTime();
    
    // Update time slider
    document.getElementById('time-slider').value = baselineAnimationState.currentVolIdx;
    document.getElementById('current-time').textContent = baselineAnimationState.currentVolIdx;
  }
}

function updatePredictionAnimation(currentTime) {
  if (!predictionAnimationState || !predictionData) return;
  if (isPaused || manualTimeControl) return;
  
  if (currentTime - predictionAnimationState.lastUpdateTime >= predictionAnimationState.updateInterval) {
    // Advance to next volume
    predictionAnimationState.currentVolIdx = (predictionAnimationState.currentVolIdx + 1) % predictionAnimationState.volumeIndices.length;
    predictionAnimationState.lastUpdateTime = currentTime;
    
    updatePredictionToCurrentTime();
    
    // Update time slider (use whichever animation is visible)
    if (document.getElementById('show-predictions').checked) {
      document.getElementById('time-slider').value = predictionAnimationState.currentVolIdx;
      document.getElementById('current-time').textContent = predictionAnimationState.currentVolIdx;
    }
  }
}

function animate() {
  animationFrameId = requestAnimationFrame(animate);
  
  const currentTime = performance.now();
  
  // Update baseline animation
  if (document.getElementById('show-baseline').checked && baselineData) {
    updateBaselineAnimation(currentTime);
  }
  
  // Update prediction animation
  if (document.getElementById('show-predictions').checked && predictionData) {
    updatePredictionAnimation(currentTime);
  }
  
  if (currentControls) currentControls.update();
  if (currentRenderer && currentScene && currentCamera) {
    currentRenderer.render(currentScene, currentCamera);
  }
}

// Event listeners
document.getElementById('simulateBtn').addEventListener('click', simulate);
document.getElementById('show-baseline').addEventListener('change', updateAnimation);
document.getElementById('show-predictions').addEventListener('change', updateAnimation);

// Time control event listeners
document.getElementById('pause-btn').addEventListener('click', togglePause);
document.getElementById('time-slider').addEventListener('input', (e) => {
  setManualTime(parseInt(e.target.value));
});

// Spatial filtering event listeners (both min and max sliders trigger the same update)
document.getElementById('x-min-slider').addEventListener('change', updateSpatialFilter);
document.getElementById('x-max-slider').addEventListener('change', updateSpatialFilter);

// Initialize on page load
window.addEventListener('load', () => {
  fetchRegions();
  initViewer();
  initBaselineData();
  initializeDualRangeSliders();
  
  // Disable predictions checkbox initially
  document.getElementById('show-predictions').disabled = true;
});

// Add these new functions for heatmap toggle
function showAllRegions() {
  document.getElementById('heatmap-all').style.display = 'block';
  document.getElementById('heatmap-top12').style.display = 'none';
  document.getElementById('heatmap-all-btn').classList.add('active');
  document.getElementById('heatmap-top12-btn').classList.remove('active');
}

function showTop12() {
  document.getElementById('heatmap-all').style.display = 'none';
  document.getElementById('heatmap-top12').style.display = 'block';
  document.getElementById('heatmap-all-btn').classList.remove('active');
  document.getElementById('heatmap-top12-btn').classList.add('active');
}

// Fullscreen functionality
function openFullscreen() {
  // Get the currently visible heatmap
  const allVisible = document.getElementById('heatmap-all').style.display !== 'none';
  const currentHeatmap = allVisible ? 
    document.getElementById('heatmap-all') : 
    document.getElementById('heatmap-top12');
  
  if (currentHeatmap.requestFullscreen) {
    currentHeatmap.requestFullscreen();
  } else if (currentHeatmap.webkitRequestFullscreen) { // Safari
    currentHeatmap.webkitRequestFullscreen();
  } else if (currentHeatmap.msRequestFullscreen) { // IE11
    currentHeatmap.msRequestFullscreen();
  }
}

// Listen for fullscreen change events to handle exit
document.addEventListener('fullscreenchange', handleFullscreenChange);
document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
document.addEventListener('msfullscreenchange', handleFullscreenChange);

function handleFullscreenChange() {
  // This function can be used to handle any cleanup when exiting fullscreen
  // Currently no specific action needed
}

// Update the expandHeatmap function to use the currently visible heatmap
function expandHeatmap() {
  const modal = document.getElementById('heatmap-modal');
  const heatmapExpanded = document.getElementById('heatmap-expanded');
  
  // Check which heatmap is currently visible
  const allVisible = document.getElementById('heatmap-all').style.display !== 'none';
  const heatmapSrc = allVisible ? 
    document.getElementById('heatmap-all').src : 
    document.getElementById('heatmap-top12').src;
  
  heatmapExpanded.src = heatmapSrc;
  modal.style.display = 'flex';
} 