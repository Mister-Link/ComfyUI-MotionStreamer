# link-comfy-motionstreamer

ComfyUI proof-of-concept nodes for the MotionStreamer text-to-motion model.

## Nodes

- `MotionStreamer Load Model`
- `MotionStreamer Generate`
- `MotionStreamer Preview`
- `MotionStreamer Save Outputs`

## Expected local setup

Default paths assume:

- MotionStreamer repo at `/home/linkray/MotionStreamer`
- 272-dim representation repo at `/home/linkray/272-dim-Motion-Representation`
- Blender installed on `PATH`

The node package will also try to reuse Python packages from `/home/linkray/venv`
if ComfyUI's own venv does not have `sentence_transformers`.

## Outputs

The save node can write:

- normalized motion `.npy`
- preview `.mp4`
- `.bvh`
- `.fbx`
