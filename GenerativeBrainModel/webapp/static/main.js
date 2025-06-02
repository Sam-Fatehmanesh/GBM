async function fetchRegions() {
  // Show loading message and hide select
  document.getElementById('loading-masks').style.display = 'block';
  document.getElementById('regions').style.display = 'none';
  const res = await fetch('/regions');
  const data = await res.json();
  const sel = document.getElementById('regions');
  data.regions.forEach(region => {
    const opt = document.createElement('option');
    opt.value = region;
    opt.text = region;
    sel.add(opt);
  });
  // Hide loading message and show select
  document.getElementById('loading-masks').style.display = 'none';
  sel.style.display = 'block';
}

async function simulate() {
  const regions = Array.from(document.getElementById('regions').selectedOptions).map(o => o.value);
  const fraction = parseFloat(document.getElementById('fraction').value);
  const steps = parseInt(document.getElementById('steps').value, 10);
  const statusDiv = document.getElementById('status');
  statusDiv.innerText = 'Submitting simulation...';

  const res = await fetch('/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ selected_regions: regions, activation_fraction: fraction, prediction_steps: steps })
  });
  const result = await res.json();
  const jobId = result.job_id;
  statusDiv.innerText = `Job submitted: ${jobId}`;
  pollStatus(jobId);
}

async function pollStatus(jobId) {
  const statusDiv = document.getElementById('status');
  const interval = setInterval(async () => {
    const res = await fetch(`/status/${jobId}`);
    const data = await res.json();
    // Unwrap nested status
    const jobStatus = data.status.status || data.status;
    statusDiv.innerText = `Status: ${jobStatus}`;
    if (jobStatus === 'done' || jobStatus === 'error') {
      clearInterval(interval);
      if (jobStatus === 'done') {
        loadResults(jobId);
      } else {
        statusDiv.innerText += ' (Error)';
      }
    }
  }, 2000);
}

async function loadResults(jobId) {
  const statusDiv = document.getElementById('status');
  statusDiv.innerText = 'Loading results...';

  const res = await fetch(`/result/${jobId}`);
  const data = await res.json();
  if (data.status.status !== 'done') {
    statusDiv.innerText = 'Job not complete.';
    return;
  }

  const heatmap = document.getElementById('heatmap');
  heatmap.src = `/results/${jobId}/heatmap.png`;

  document.getElementById('download-summary-json').href = `/results/${jobId}/summary.json`;
  document.getElementById('download-summary-csv').href = `/results/${jobId}/summary.csv`;
  document.getElementById('download-probabilities').href = `/results/${jobId}/predicted_probabilities.npy`;

  document.getElementById('results').style.display = 'block';
  statusDiv.innerText = 'Results ready.';
  // Initialize 3D voxel viewer
  initVoxelViewer(jobId);
}

async function initVoxelViewer(jobId) {
  try {
    const res = await fetch(`/results/${jobId}/activation_mask_json`);
    const data = await res.json();
    const coords = data.coords; // [[z,y,x], ...]
    const container = document.getElementById('viewer');
    // Clear any existing canvas
    container.innerHTML = '';
    // Three.js setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    // Create points geometry
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(coords.length * 3);
    coords.forEach((c, i) => {
      // Map x,y,z to three.js axes: x=c[2], y=c[1], z=c[0]
      positions[i*3] = c[2];
      positions[i*3+1] = c[1];
      positions[i*3+2] = c[0];
    });
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    // Compute bounding box after setting positions
    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox;
    const center = bbox.getCenter(new THREE.Vector3());
    const material = new THREE.PointsMaterial({ color: 0x00ff00, size: 0.5 });
    const points = new THREE.Points(geometry, material);
    scene.add(points);
    // Position camera
    camera.position.set(center.x + 100, center.y + 100, center.z + 100);
    controls.target.copy(center);
    controls.update();
    // Animation loop - rotate the points for a looped animation
    function animate() {
      requestAnimationFrame(animate);
      points.rotation.y += 0.005; // rotate for looped animation
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  } catch (err) {
    console.error('Voxel viewer error:', err);
  }
}

async function initBaselineViewer() {
  try {
    const res = await fetch('/baseline_mask_json');
    const data = await res.json();
    const coordsAll = data.coords; // [[sliceIdx, y, x], ...]
    const Z = data.Z; // slices per 3D volume
    const zStart = data.zStart; // starting z-plane offset
    const container = document.getElementById('viewer');
    container.innerHTML = '';
    // Three.js setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    // Group coords by volume index (time volume)
    const volumes = {};
    coordsAll.forEach(c => {
      const sliceIdx = c[0];
      const volIdx = Math.floor(sliceIdx / Z);
      volumes[volIdx] = volumes[volIdx] || [];
      volumes[volIdx].push(c);
    });
    const volumeIndices = Object.keys(volumes).map(n => parseInt(n)).sort((a, b) => a - b);
    const numVolumes = volumeIndices.length;
    // Prepare cube geometry and material for voxels
    const cubeGeo = new THREE.BoxGeometry(1, 1, 1);
    const cubeMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    let mesh;
    // Compute center for camera target from first volume
    const firstVolCoords = volumes[volumeIndices[0]];
    const pos0 = new Float32Array(firstVolCoords.length * 3);
    firstVolCoords.forEach((c, i) => {
      pos0[i*3] = c[2];
      pos0[i*3+1] = c[1];
      pos0[i*3+2] = (c[0] + zStart) % Z;
    });
    const geom0 = new THREE.BufferGeometry();
    geom0.setAttribute('position', new THREE.BufferAttribute(pos0, 3));
    geom0.computeBoundingBox();
    const center = geom0.boundingBox.getCenter(new THREE.Vector3());
    // Initialize camera
    camera.position.set(center.x + 100, center.y + 100, center.z + 100);
    controls.target.copy(center);
    controls.update();
    // Animation variables
    let volIdx = 0;
    // Draw initial volume
    {
      const coords = volumes[volumeIndices[volIdx]];
      mesh = new THREE.InstancedMesh(cubeGeo, cubeMat, coords.length);
      const dummy = new THREE.Object3D();
      coords.forEach((c, i) => {
        dummy.position.set(c[2], c[1], (c[0] + zStart) % Z);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      });
      scene.add(mesh);
    }
    // Initial render
    controls.update();
    renderer.render(scene, camera);
    // Timing for volume updates (0.5s per volume)
    let lastUpdateTime = performance.now();
    const updateInterval = 500; // milliseconds
    // Animation loop with timed stepping
    function animate(time) {
      requestAnimationFrame(animate);
      if (time - lastUpdateTime >= updateInterval) {
        // Advance to next volume
        volIdx = (volIdx + 1) % numVolumes;
        // Remove previous mesh
        scene.remove(mesh);
        mesh.geometry.dispose();
        mesh.material.dispose();
        // Build and render new volume cubes
        const coords = volumes[volumeIndices[volIdx]];
        mesh = new THREE.InstancedMesh(cubeGeo, cubeMat, coords.length);
        const dummy2 = new THREE.Object3D();
        coords.forEach((c, i) => {
          dummy2.position.set(c[2], c[1], (c[0] + zStart) % Z);
          dummy2.updateMatrix();
          mesh.setMatrixAt(i, dummy2.matrix);
        });
        scene.add(mesh);
        lastUpdateTime = time;
      }
      controls.update();
      renderer.render(scene, camera);
    }
    requestAnimationFrame(animate);
  } catch(e) {
    console.error('Baseline viewer error:', e);
  }
}

document.getElementById('simulateBtn').addEventListener('click', simulate);
window.addEventListener('load', () => {
  fetchRegions();
  initBaselineViewer();
}); 