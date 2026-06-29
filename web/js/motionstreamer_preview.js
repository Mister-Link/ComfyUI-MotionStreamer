import { app } from "../../../../scripts/app.js";

const VIEWER_HTML = `<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #16182a; color: #ccc; font: 12px monospace; overflow: hidden; display: flex; flex-direction: column; height: 100vh; user-select: none; }
  canvas { flex: 1; display: block; cursor: grab; min-height: 0; }
  canvas.dragging { cursor: grabbing; }
  #controls { padding: 6px 10px; background: #1e2030; display: flex; align-items: center; gap: 8px; flex-shrink: 0; border-top: 1px solid #2a2d42; position: relative; }
  button { background: #2a2d42; color: #ccc; border: 1px solid #444; padding: 3px 10px; cursor: pointer; border-radius: 3px; font: 12px monospace; }
  button:hover { background: #3a3d52; }
  button.active { background: #2a52a0; color: #fff; border-color: #5080dd; }
  input[type=range] { flex: 1; accent-color: #5080dd; }
  #status { position: absolute; top: 8px; left: 10px; color: #778; pointer-events: none; max-width: 80%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #bonesPanel { display: none; position: absolute; bottom: calc(100% + 4px); left: 10px; background: #1e2030; border: 1px solid #2a2d42; border-radius: 4px; padding: 8px 12px; flex-direction: column; gap: 6px; z-index: 10; white-space: nowrap; }
  #bonesPanel.open { display: flex; }
  #bonesPanel label { display: flex; align-items: center; gap: 7px; cursor: pointer; color: #ccc; }
  #bonesPanel input[type=checkbox] { accent-color: #5080dd; cursor: pointer; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="status">Waiting for motion data…</div>
<div id="controls">
  <div id="bonesPanel">
    <label><input type="checkbox" id="chkFace" checked> Face</label>
    <label><input type="checkbox" id="chkHead" checked> Head</label>
  </div>
  <button id="btnPlay">▶ Play</button>
  <button id="btnFollow">⊙ Follow</button>
  <button id="btnBones">Bones…</button>
  <input type="range" id="scrubber" min="0" max="100" value="0">
  <span id="frameLabel" style="white-space:nowrap;min-width:60px;text-align:right">0 / 0</span>
</div>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const scrubber = document.getElementById('scrubber');
const frameLabel = document.getElementById('frameLabel');
const btnPlay = document.getElementById('btnPlay');
const btnFollow = document.getElementById('btnFollow');
const btnBones = document.getElementById('btnBones');
const bonesPanel = document.getElementById('bonesPanel');
const chkFace = document.getElementById('chkFace');
const chkHead = document.getElementById('chkHead');
const statusEl = document.getElementById('status');

let showFace = true, showHead = true;

let xyz = null, bones = [], numFrames = 0, numJoints = 22, fps = 30;
let currentFrame = 0, playing = false, animId = null, lastTime = 0;
let rotY = 0.4, rotX = -0.15, zoom = 1.0;
let dragging = false, lastMX = 0, lastMY = 0, dragIsPan = false;
let panX = 0, panY = 0;
let norm = null;
let followMode = false;

function resize() {
  canvas.width = canvas.clientWidth || 512;
  canvas.height = canvas.clientHeight || 400;
  render();
}

// Scale is derived from the character body size at frame 0, not total travel,
// so zoom stays sensible even for long-distance motions.
function computeNorm() {
  if (!xyz) return null;
  let minX=Infinity, maxX=-Infinity, minY=Infinity, maxY=-Infinity, minZ=Infinity, maxZ=-Infinity;
  for (let j = 0; j < numJoints; j++) {
    const [x, y, z] = getJoint(0, j);
    if (x < minX) minX=x; if (x > maxX) maxX=x;
    if (y < minY) minY=y; if (y > maxY) maxY=y;
    if (z < minZ) minZ=z; if (z > maxZ) maxZ=z;
  }
  const range = Math.max(maxX-minX, maxY-minY, maxZ-minZ, 0.01);
  const [rx0, , rz0] = getJoint(0, 0);
  return { cx: rx0, cy: (minY+maxY)/2, cz: rz0, scale: 1/range, floorY: minY };
}

function drawGrid() {
  if (!norm) return;
  const { floorY, scale, cx: baseCX, cz: baseCZ } = norm;
  let gcx = baseCX, gcz = baseCZ;
  if (followMode && numFrames > 0) {
    const root = getJoint(Math.floor(currentFrame) % numFrames, 0);
    gcx = root[0]; gcz = root[2];
  }
  const step = 0.3 / scale;
  const n = 7;
  const half = n * step;
  ctx.save();
  ctx.strokeStyle = '#252840';
  ctx.lineWidth = 0.8;
  for (let i = -n; i <= n; i++) {
    const x = gcx + i * step;
    const z = gcz + i * step;
    const [ax0, ay0] = project(x, floorY, gcz - half);
    const [ax1, ay1] = project(x, floorY, gcz + half);
    ctx.beginPath(); ctx.moveTo(ax0, ay0); ctx.lineTo(ax1, ay1); ctx.stroke();
    const [bx0, by0] = project(gcx - half, floorY, z);
    const [bx1, by1] = project(gcx + half, floorY, z);
    ctx.beginPath(); ctx.moveTo(bx0, by0); ctx.lineTo(bx1, by1); ctx.stroke();
  }
  ctx.restore();
}

// Set zoom so the character fills ~75% of the viewport at the current rotation.
function autoFrame() {
  if (!norm) return;
  const savedZoom = zoom;
  zoom = 1.0;
  let minSX=Infinity, maxSX=-Infinity, minSY=Infinity, maxSY=-Infinity;
  for (let j = 0; j < numJoints; j++) {
    const [sx, sy] = project(...getJoint(0, j));
    if (sx < minSX) minSX=sx; if (sx > maxSX) maxSX=sx;
    if (sy < minSY) minSY=sy; if (sy > maxSY) maxSY=sy;
  }
  const cW = maxSX - minSX || 1, cH = maxSY - minSY || 1;
  zoom = Math.min((canvas.width * 0.75) / cW, (canvas.height * 0.75) / cH);
  if (!isFinite(zoom) || zoom <= 0) zoom = savedZoom;
}

function getJoint(frame, joint) {
  const i = (frame * numJoints + joint) * 3;
  return [xyz[i], xyz[i+1], xyz[i+2]];
}

function project(x, y, z) {
  const { scale, cx: baseCX, cy, cz: baseCZ } = norm;
  let cx = baseCX, cz = baseCZ;
  if (followMode) {
    const f = Math.floor(currentFrame) % numFrames;
    const root = getJoint(f, 0); // Pelvis
    cx = root[0]; cz = root[2];
  }
  let px = (x-cx)*scale, py = (y-cy)*scale, pz = (z-cz)*scale;
  const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
  const rx = px*cosY - pz*sinY;
  const rz = px*sinY + pz*cosY;
  const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
  const ry = py*cosX - rz*sinX;
  const rz2 = py*sinX + rz*cosX;
  const fov = Math.min(canvas.width, canvas.height) * 0.55 * zoom;
  const camZ = rz2 + 2.5;
  return [rx/camZ*fov + canvas.width/2 + panX, -ry/camZ*fov + canvas.height/2 + panY];
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!xyz || !norm) return;
  drawGrid();
  const f = Math.floor(currentFrame) % numFrames;
  const pts = [];
  for (let j = 0; j < numJoints; j++) pts.push(project(...getJoint(f, j)));

  ctx.lineWidth = 2.5;
  ctx.strokeStyle = '#4a70cc';
  for (const [a, b] of bones) {
    if (a >= pts.length || b >= pts.length) continue;
    // Head bone: Neck(12) → Head(15)
    if (!showHead && ((a === 12 && b === 15) || (a === 15 && b === 12))) continue;
    ctx.beginPath();
    ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]);
    ctx.stroke();
  }

  for (let j = 0; j < pts.length; j++) {
    if (!showFace && j === 15) continue; // Head joint dot
    const [sx, sy] = pts[j];
    ctx.beginPath();
    ctx.arc(sx, sy, 3.5, 0, Math.PI*2);
    ctx.fillStyle = '#ccd';
    ctx.fill();
  }
}

function step(ts) {
  if (!playing) return;
  if (!lastTime) lastTime = ts;
  currentFrame = (currentFrame + (ts - lastTime) / 1000 * fps) % numFrames;
  lastTime = ts;
  const f = Math.floor(currentFrame);
  scrubber.value = f;
  frameLabel.textContent = f + ' / ' + numFrames;
  render();
  animId = requestAnimationFrame(step);
}

function setPlaying(v) {
  playing = v;
  btnPlay.textContent = v ? '⏸ Pause' : '▶ Play';
  if (v) { lastTime = 0; animId = requestAnimationFrame(step); }
  else if (animId) { cancelAnimationFrame(animId); animId = null; }
}

function setFollow(v) {
  if (!v && followMode && norm && numFrames > 0) {
    const root = getJoint(Math.floor(currentFrame) % numFrames, 0);
    norm.cx = root[0];
    norm.cz = root[2];
  }
  followMode = v;
  btnFollow.classList.toggle('active', v);
  render();
}

btnPlay.addEventListener('click', () => setPlaying(!playing));
btnFollow.addEventListener('click', () => setFollow(!followMode));
btnBones.addEventListener('click', () => {
  const open = bonesPanel.classList.toggle('open');
  btnBones.classList.toggle('active', open);
});
chkFace.addEventListener('change', () => { showFace = chkFace.checked; render(); });
chkHead.addEventListener('change', () => { showHead = chkHead.checked; render(); });
scrubber.addEventListener('input', () => {
  currentFrame = +scrubber.value;
  frameLabel.textContent = Math.floor(currentFrame) + ' / ' + numFrames;
  render();
});

canvas.addEventListener('mousedown', e => {
  dragging = true; dragIsPan = e.shiftKey;
  lastMX = e.clientX; lastMY = e.clientY;
  canvas.classList.add('dragging');
});
window.addEventListener('mousemove', e => {
  if (!dragging) return;
  const dx = e.clientX - lastMX, dy = e.clientY - lastMY;
  if (dragIsPan) {
    panX += dx; panY += dy;
  } else {
    rotY += dx * 0.012;
    rotX = Math.max(-1.4, Math.min(1.4, rotX - dy * 0.012));
  }
  lastMX = e.clientX; lastMY = e.clientY;
  render();
});
window.addEventListener('mouseup', () => { dragging = false; canvas.classList.remove('dragging'); });
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  zoom = Math.max(0.2, Math.min(8.0, zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
  render();
}, { passive: false });

window.addEventListener('message', e => {
  if (e.data?.type === 'LOAD_MOTION') {
    const d = e.data.motionData;
    xyz = new Float32Array(d.xyz);
    numFrames = d.num_frames;
    numJoints = d.num_joints || 22;
    fps = d.fps || 30;
    bones = d.bones || [];
    currentFrame = 0;
    panX = 0; panY = 0;
    norm = computeNorm();
    autoFrame();
    setFollow(false);
    scrubber.max = numFrames - 1;
    scrubber.value = 0;
    frameLabel.textContent = '0 / ' + numFrames;
    statusEl.textContent = d.text || '';
    render();
    setPlaying(true);
  } else if (e.data?.type === 'RESIZE') {
    resize();
  }
});

new ResizeObserver(resize).observe(canvas);
resize();
if (window.parent) window.parent.postMessage({ type: 'VIEWER_READY' }, '*');
</script>
</body>
</html>`;

app.registerExtension({
    name: "motionstreamer.previewanimation",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MotionStreamerPreviewAnimation") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const r = onNodeCreated?.apply(this, arguments);

            const iframe = document.createElement("iframe");
            iframe.style.cssText = "width:100%;height:100%;border:none;background:#16182a;display:block;";

            const blob = new Blob([VIEWER_HTML], { type: "text/html" });
            iframe.src = URL.createObjectURL(blob);
            iframe.addEventListener("load", () => { iframe._blobUrl = iframe.src; });

            const widget = this.addDOMWidget("preview", "MS_MOTION_PREVIEW", iframe, {
                getValue() { return ""; },
                setValue() {}
            });
            widget.computeSize = (w) => [w || 512, (w || 512) * 1.05];
            widget.element = iframe;
            this.motionViewerIframe = iframe;
            this.motionViewerReady = false;

            window.addEventListener("message", e => {
                if (e.data?.type === "VIEWER_READY") this.motionViewerReady = true;
            });

            const notifyResize = () => iframe.contentWindow?.postMessage({ type: "RESIZE" }, "*");
            this.onResize = () => requestAnimationFrame(notifyResize);

            const ro = new ResizeObserver(() => requestAnimationFrame(notifyResize));
            ro.observe(iframe);

            const onRemoved = this.onRemoved;
            this.onRemoved = function() {
                ro.disconnect();
                if (iframe._blobUrl) URL.revokeObjectURL(iframe._blobUrl);
                onRemoved?.apply(this, arguments);
            };

            this.setSize([512, 570]);

            const onExecuted = this.onExecuted;
            this.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);
                if (!message?.motion_data?.[0]) return;
                try {
                    const motionData = JSON.parse(message.motion_data[0]);
                    const send = () => iframe.contentWindow?.postMessage({ type: "LOAD_MOTION", motionData }, "*");
                    if (this.motionViewerReady) {
                        send();
                    } else {
                        const t = setInterval(() => { if (this.motionViewerReady) { clearInterval(t); send(); } }, 50);
                        setTimeout(() => { clearInterval(t); send(); }, 2000);
                    }
                } catch(e) {
                    console.error("[MotionStreamer] Failed to parse motion data:", e);
                }
            };

            return r;
        };
    }
});
