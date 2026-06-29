import { app } from "../../../../scripts/app.js";

const VIEWER_HTML = `<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #16182a; color: #ccc; font: 12px monospace; overflow: hidden; display: flex; flex-direction: column; height: 100vh; user-select: none; }
  canvas { flex: 1; display: block; cursor: grab; min-height: 0; }
  canvas.dragging { cursor: grabbing; }
  #controls { padding: 6px 10px; background: #1e2030; display: flex; align-items: center; gap: 8px; flex-shrink: 0; border-top: 1px solid #2a2d42; }
  button { background: #2a2d42; color: #ccc; border: 1px solid #444; padding: 3px 10px; cursor: pointer; border-radius: 3px; font: 12px monospace; }
  button:hover { background: #3a3d52; }
  button.active { background: #2a52a0; color: #fff; border-color: #5080dd; }
  input[type=range] { flex: 1; accent-color: #5080dd; }
  #status { position: absolute; top: 8px; left: 10px; color: #778; pointer-events: none; max-width: 80%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="status">Waiting for motion data…</div>
<div id="controls">
  <button id="btnPlay">▶ Play</button>
  <button id="btnFollow">⊙ Follow</button>
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
const statusEl = document.getElementById('status');

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
    const la = projectLine(x, floorY, gcz - half, x, floorY, gcz + half);
    if (la) { ctx.beginPath(); ctx.moveTo(la[0][0], la[0][1]); ctx.lineTo(la[1][0], la[1][1]); ctx.stroke(); }
    const lb = projectLine(gcx - half, floorY, z, gcx + half, floorY, z);
    if (lb) { ctx.beginPath(); ctx.moveTo(lb[0][0], lb[0][1]); ctx.lineTo(lb[1][0], lb[1][1]); ctx.stroke(); }
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
    const p = project(...getJoint(0, j));
    if (!p) continue;
    const [sx, sy] = p;
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

const NEAR = 0.1;

function toViewSpace(x, y, z) {
  const { scale, cx: baseCX, cy, cz: baseCZ } = norm;
  let cx = baseCX, cz = baseCZ;
  if (followMode) {
    const f = Math.floor(currentFrame) % numFrames;
    const root = getJoint(f, 0);
    cx = root[0]; cz = root[2];
  }
  let px = (x-cx)*scale, py = (y-cy)*scale, pz = (z-cz)*scale;
  const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
  const rx = px*cosY - pz*sinY;
  const rz = px*sinY + pz*cosY;
  const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
  const ry = py*cosX - rz*sinX;
  const rz2 = py*sinX + rz*cosX;
  return { rx, ry, camZ: rz2 + 2.5 };
}

function screenFromView(v) {
  const fov = Math.min(canvas.width, canvas.height) * 0.55 * zoom;
  return [v.rx/v.camZ*fov + canvas.width/2 + panX, -v.ry/v.camZ*fov + canvas.height/2 + panY];
}

// Returns screen [x,y] or null if behind near plane.
function project(x, y, z) {
  const v = toViewSpace(x, y, z);
  if (v.camZ < NEAR) return null;
  return screenFromView(v);
}

// Returns [[x0,y0],[x1,y1]] with near-plane clipping, or null if fully behind.
function projectLine(x1, y1, z1, x2, y2, z2) {
  let v1 = toViewSpace(x1, y1, z1);
  let v2 = toViewSpace(x2, y2, z2);
  if (v1.camZ < NEAR && v2.camZ < NEAR) return null;
  if (v1.camZ < NEAR) {
    const t = (NEAR - v1.camZ) / (v2.camZ - v1.camZ);
    v1 = { rx: v1.rx + t*(v2.rx-v1.rx), ry: v1.ry + t*(v2.ry-v1.ry), camZ: NEAR };
  } else if (v2.camZ < NEAR) {
    const t = (NEAR - v2.camZ) / (v1.camZ - v2.camZ);
    v2 = { rx: v2.rx + t*(v1.rx-v2.rx), ry: v2.ry + t*(v1.ry-v2.ry), camZ: NEAR };
  }
  return [screenFromView(v1), screenFromView(v2)];
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
    const pa = pts[a], pb = pts[b];
    if (pa && pb) {
      ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
    } else if (!pa || !pb) {
      // One endpoint behind camera — clip to near plane
      const [ja, jb] = [getJoint(f, a), getJoint(f, b)];
      const line = projectLine(...ja, ...jb);
      if (line) { ctx.beginPath(); ctx.moveTo(line[0][0], line[0][1]); ctx.lineTo(line[1][0], line[1][1]); ctx.stroke(); }
    }
  }

  for (let j = 0; j < pts.length; j++) {
    if (!pts[j]) continue;
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
