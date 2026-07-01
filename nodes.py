import json
import os
import re
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import importlib
from pathlib import Path
import types

import numpy as np
import torch

COMFY_ROOT = Path("/data/comfy/ComfyUI")
if str(COMFY_ROOT) not in sys.path and COMFY_ROOT.exists():
    sys.path.insert(0, str(COMFY_ROOT))

import comfy.model_management as model_management

CURRENT_DIR = Path(__file__).resolve().parent
LIB_DIR = CURRENT_DIR / "lib"
DATA_DIR = CURRENT_DIR / "data"

if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

try:
    import folder_paths
    COMFY_OUTPUT_DIR = Path(folder_paths.get_output_directory())
    _MODELS_DIR = Path(folder_paths.models_dir) / "motionstreamer"
except Exception:
    COMFY_OUTPUT_DIR = CURRENT_DIR.parent.parent / "output"
    _MODELS_DIR = COMFY_ROOT / "models" / "motionstreamer"


def _add_external_site_packages() -> None:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for candidate in [Path("/home/linkray/venv/lib") / version / "site-packages"]:
        if candidate.exists():
            s = str(candidate)
            if s not in sys.path:
                sys.path.append(s)


_add_external_site_packages()


def _sanitize_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return text or "motion"


def _timestamp() -> str:
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"{ms:03d}"


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@contextmanager
def _mounted_models_package():
    package_root = LIB_DIR / "models"
    package_name = "models"
    previous_package = sys.modules.get(package_name)
    previous_submodules = {
        k: v for k, v in list(sys.modules.items())
        if k == package_name or k.startswith("models.")
    }
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_root)]
    package.__file__ = str(package_root / "__init__.py")
    sys.modules[package_name] = package
    try:
        yield
    finally:
        for k in [k for k in sys.modules if k == package_name or k.startswith("models.")]:
            if k not in previous_submodules:
                del sys.modules[k]
        for k, v in previous_submodules.items():
            sys.modules[k] = v
        if previous_package is None and package_name not in previous_submodules and package_name in sys.modules:
            del sys.modules[package_name]


def _ensure_models(mdir: Path) -> None:
    import huggingface_hub
    from huggingface_hub import snapshot_download

    mdir.mkdir(parents=True, exist_ok=True)
    hf_logging = getattr(huggingface_hub, "logging", None)
    if hf_logging is not None:
        enable_progress_bars = getattr(hf_logging, "enable_progress_bars", None)
        if callable(enable_progress_bars):
            enable_progress_bars()

    tae_ok = (mdir / "Causal_TAE" / "net_last.pth").exists()
    trans_ok = (mdir / "t2m_model" / "latest.pth").exists() or \
               (mdir / "Experiments" / "t2m_model" / "latest.pth").exists()
    enc_ok = (mdir / "sentencet5-xxl").exists()

    if not tae_ok or not trans_ok:
        patterns = []
        sizes = []
        if not tae_ok:
            patterns.append("Causal_TAE/*")
            sizes.append("TAE ~291 MB")
        if not trans_ok:
            patterns.append("Experiments/t2m_model/*")
            sizes.append("transformer ~925 MB")
        print(f"[MotionStreamer] Downloading from HuggingFace (lxxiao/MotionStreamer): {', '.join(sizes)}")
        print(f"[MotionStreamer]   saving to: {mdir}")
        snapshot_download(
            repo_id="lxxiao/MotionStreamer",
            local_dir=str(mdir),
            allow_patterns=patterns,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[MotionStreamer] MotionStreamer weights download complete.")

    if not enc_ok:
        print(f"[MotionStreamer] Downloading sentence-t5-xxl text encoder (~10 GB) — this will take a while.")
        print(f"[MotionStreamer]   saving to: {mdir / 'sentencet5-xxl'}")
        snapshot_download(
            repo_id="sentence-transformers/sentence-t5-xxl",
            local_dir=str(mdir / "sentencet5-xxl"),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[MotionStreamer] Text encoder download complete.")


@dataclass
class MotionStreamerModelWrapper:
    device: torch.device
    tae_model: torch.nn.Module
    transformer: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray
    reference_end_latent: torch.Tensor


@dataclass
class MotionStreamerData:
    text: str
    seed: int
    motion_272: np.ndarray
    motion_272_denorm: np.ndarray
    xyz: np.ndarray


class MotionStreamerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "offload_to_cpu": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MOTIONSTREAMER_MODEL", "MOTIONSTREAMER_TEXT_ENCODER")
    RETURN_NAMES = ("model", "text_encoder")
    FUNCTION = "load"
    CATEGORY = "MotionStreamer"

    def load(self, offload_to_cpu: bool = False):
        from sentence_transformers import SentenceTransformer
        from comfy.utils import ProgressBar

        mdir = _MODELS_DIR

        # Download missing weights (prints progress to terminal)
        _ensure_models(mdir)

        device = torch.device("cpu") if offload_to_cpu else (
            model_management.get_torch_device() if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"[MotionStreamer] Loading models from {mdir} on {device}")

        # 3 tracked steps: TAE, transformer, text encoder
        pbar = ProgressBar(3)

        with _mounted_models_package():
            tae = importlib.import_module("models.tae")
            llama_model = importlib.import_module("models.llama_model")

            net = tae.Causal_HumanTAE(
                hidden_size=1024, down_t=2, stride_t=2, depth=3,
                dilation_growth_rate=3, activation="relu",
                latent_dim=16, clip_range=[-30, 20],
            )
            config = llama_model.LLaMAHFConfig.from_name("Normal_size")
            config.block_size = 78
            trans_encoder = llama_model.LLaMAHF(config, 9, 16, device)

            print("[MotionStreamer] Loading TAE weights ...")
            tae_ckpt = torch.load(mdir / "Causal_TAE" / "net_last.pth", map_location="cpu")
            net.load_state_dict(tae_ckpt["net"], strict=True)
            net.eval().to(device)
            pbar.update(1)

            # support both original Experiments/ path and flattened t2m_model/ path
            trans_path = mdir / "t2m_model" / "latest.pth"
            if not trans_path.exists():
                trans_path = mdir / "Experiments" / "t2m_model" / "latest.pth"
            print("[MotionStreamer] Loading transformer weights ...")
            trans_ckpt = torch.load(trans_path, map_location="cpu")
            new_state = {}
            for k, v in trans_ckpt["trans"].items():
                new_state[".".join(k.split(".")[1:]) if k.split(".")[0] == "module" else k] = v
            trans_encoder.load_state_dict(new_state, strict=True)
            trans_encoder.eval().to(device)
            pbar.update(1)

        mean = np.load(DATA_DIR / "Mean.npy")
        std = np.load(DATA_DIR / "Std.npy")
        reference_end_latent = torch.from_numpy(
            np.load(DATA_DIR / "reference_end_latent_t2m_272.npy")
        ).to(device)

        model = MotionStreamerModelWrapper(
            device=device,
            tae_model=net,
            transformer=trans_encoder,
            mean=mean,
            std=std,
            reference_end_latent=reference_end_latent,
        )

        print("[MotionStreamer] Loading text encoder ...")
        enc = SentenceTransformer(str(mdir / "sentencet5-xxl"))
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        pbar.update(1)

        print("[MotionStreamer] All models loaded.")
        return (model, enc)


class MotionStreamerGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MOTIONSTREAMER_MODEL",),
                "text_encoder": ("MOTIONSTREAMER_TEXT_ENCODER",),
                "text": ("STRING", {"default": "A person is walking forward.", "multiline": True}),
                "negative_text": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 123, "min": 0, "max": 0x7FFFFFFF}),
            },
            "optional": {
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Length of the generated animation in seconds."}),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Prompt strength. Higher follows the prompt more closely (but can be more unstable)."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "Motion randomness. Higher is more random (but can be more unstable)."}),
            },
        }

    RETURN_TYPES = ("MOTIONSTREAMER_DATA",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "generate"
    CATEGORY = "MotionStreamer"

    def generate(self, model, text_encoder, text, negative_text, seed,
                 duration=10.0, cfg_scale=4.0, temperature=1.0):
        torch.manual_seed(seed)
        if model.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        _FPS = 30
        _UNIT = 4  # frames per latent token
        length = max(_UNIT, (round(duration * _FPS) // _UNIT) * _UNIT)

        with torch.no_grad():
            motion_latents = model.transformer.sample_for_eval_CFG_inference(
                text=text,
                negative_text=negative_text,
                length=length,
                tokenizer=text_encoder,
                device=model.device,
                cfg=cfg_scale,
                temperature=temperature,
            )
            motion_seqs = model.tae_model.forward_decoder(motion_latents)

        motion = motion_seqs.squeeze(0).detach().cpu().numpy().astype(np.float32)
        motion_denorm = (motion * model.std + model.mean).astype(np.float32)
        xyz = _recover_xyz_from_motion(motion, model.mean, model.std).astype(np.float32)
        print(f"[MotionStreamer] Generated motion for: {text}")
        return (MotionStreamerData(text, seed, motion, motion_denorm, xyz),)


def _accumulate_rotations(relative_rotations: np.ndarray) -> np.ndarray:
    total = [relative_rotations[0]]
    for rel in relative_rotations[1:]:
        total.append(np.matmul(rel, total[-1]))
    return np.array(total)


def _recover_xyz_from_motion(motion_272: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    face_utils = _load_module_from_path("ms_face_z", LIB_DIR / "face_z_align_util.py")
    final_x = motion_272 * std + mean
    njoint = 22
    nfrm, _ = final_x.shape
    positions_no_heading = final_x[:, 8 : 8 + 3 * njoint].reshape(nfrm, -1, 3)
    velocities_root_xy = final_x[:, :2]
    global_heading_diff_rot = final_x[:, 2:8]

    global_heading_rot = _accumulate_rotations(
        face_utils.rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy()
    )
    inv_rot = np.transpose(global_heading_rot, (0, 2, 1))
    positions = np.matmul(
        np.repeat(inv_rot[:, None], njoint, axis=1),
        positions_no_heading[..., None],
    ).squeeze(-1)

    vel_xyz = np.zeros((velocities_root_xy.shape[0], 3))
    vel_xyz[:, 0] = velocities_root_xy[:, 0]
    vel_xyz[:, 2] = velocities_root_xy[:, 1]
    vel_xyz[1:] = np.matmul(inv_rot[:-1], vel_xyz[1:, :, None]).squeeze(-1)
    root_trans = np.cumsum(vel_xyz, axis=0)
    positions[:, :, 0] += root_trans[:, 0:1]
    positions[:, :, 2] += root_trans[:, 2:]
    return positions



_BONES_22 = [
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 14), (14, 17), (17, 19), (19, 21),
    (9, 13), (13, 16), (16, 18), (18, 20),
]


class MotionStreamerPreviewAnimation:
    BONES = _BONES_22

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"motion": ("MOTIONSTREAMER_DATA",)}}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "preview"
    CATEGORY = "MotionStreamer"
    OUTPUT_NODE = True

    def preview(self, motion: MotionStreamerData):
        num_frames, num_joints, _ = motion.xyz.shape
        data = {
            "xyz": motion.xyz.flatten().tolist(),
            "num_frames": num_frames,
            "num_joints": num_joints,
            "fps": 30,
            "text": motion.text,
            "bones": self.BONES,
        }
        motion_json = json.dumps(data)
        return {"ui": {"motion_data": [motion_json]}, "result": ()}


ASSETS_DIR = CURRENT_DIR / "assets"
_TEMPLATE_FBX = ASSETS_DIR / "boy_Rigging_smplx_tex.fbx"


def _get_rot_matrices_from_motion(motion_272: np.ndarray) -> tuple:
    """Return (rot_matrices (T, 22, 3, 3), root_trans (T, 3)) from motion_272."""
    face_utils = _load_module_from_path("ms_face_z", LIB_DIR / "face_z_align_util.py")
    njoint = 22
    nfrm = motion_272.shape[0]

    rot_mat = face_utils.rotation_6d_to_matrix(
        torch.from_numpy(motion_272[:, 8 + 6 * njoint: 8 + 12 * njoint]).reshape(nfrm, -1, 6)
    ).numpy()

    global_heading_diff = motion_272[:, 2:8]
    vel_xy = motion_272[:, :2]
    height = motion_272[:, 8: 8 + 3 * njoint].reshape(nfrm, -1, 3)[:, 0, 1]

    global_rot = _accumulate_rotations(
        face_utils.rotation_6d_to_matrix(torch.from_numpy(global_heading_diff)).numpy()
    )
    inv_rot = np.transpose(global_rot, (0, 2, 1))
    rot_mat[:, 0] = np.matmul(inv_rot, rot_mat[:, 0])

    vel_xyz = np.zeros((nfrm, 3))
    vel_xyz[:, 0] = vel_xy[:, 0]
    vel_xyz[:, 2] = vel_xy[:, 1]
    vel_xyz[1:] = np.matmul(inv_rot[:-1], vel_xyz[1:, :, None]).squeeze(-1)
    root_trans = np.cumsum(vel_xyz, axis=0)
    root_trans[:, 1] = height

    return rot_mat, root_trans


class MotionStreamerExportFBX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("MOTIONSTREAMER_DATA",),
                "output_dir": ("STRING", {"default": "motionstreamer"}),
                "filename_prefix": ("STRING", {"default": "motion"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_path",)
    FUNCTION = "export"
    CATEGORY = "MotionStreamer"
    OUTPUT_NODE = True

    def export(self, motion, output_dir, filename_prefix):
        from ms_fbx_export import write_fbx_with_character

        output_root = COMFY_OUTPUT_DIR / output_dir
        output_root.mkdir(parents=True, exist_ok=True)

        stem = f"{_sanitize_filename(filename_prefix)}_{_timestamp()}"
        fbx_p = output_root / f"{stem}.fbx"

        motion_for_export = getattr(motion, "motion_272_denorm", None)
        if motion_for_export is None:
            raise RuntimeError(
                "Motion data is missing denormalized joint rotations required for FBX export. "
                "Regenerate the motion with the updated MotionStreamer node."
            )

        rot_matrices, root_trans = _get_rot_matrices_from_motion(motion_for_export)

        ok = write_fbx_with_character(
            template_fbx_path=str(_TEMPLATE_FBX),
            rot_matrices=rot_matrices,
            translations=root_trans,
            save_path=str(fbx_p),
            fps=30.0,
            scale=100.0,
        )
        if not ok:
            raise RuntimeError(f"FBX export failed — output not found at {fbx_p}")

        # Safety-net: remove any sidecar files that may appear next to the output FBX
        for _p in (Path(str(fbx_p) + ".png"), fbx_p.parent / (fbx_p.stem + ".fbm")):
            try:
                if _p.is_dir():
                    shutil.rmtree(_p, ignore_errors=True)
                elif _p.exists():
                    _p.unlink()
            except OSError:
                pass

        fbx_rel = str(fbx_p.relative_to(COMFY_OUTPUT_DIR)).replace("\\", "/")
        download_url = f"/view?filename={fbx_rel}&type=output"
        fbx_filename = fbx_p.name
        print(f"[MotionStreamer] Exported FBX: {fbx_p}")
        return {
            "ui": {
                "text": [
                    f'<a href="{download_url}" download="{fbx_filename}">Download: {fbx_filename}</a>'
                ]
            },
            "result": (str(fbx_p),),
        }


NODE_CLASS_MAPPINGS = {
    "MotionStreamerLoader": MotionStreamerLoader,
    "MotionStreamerGenerate": MotionStreamerGenerate,
    "MotionStreamerPreviewAnimation": MotionStreamerPreviewAnimation,
    "MotionStreamerExportFBX": MotionStreamerExportFBX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MotionStreamerLoader": "MotionStreamer Loader",
    "MotionStreamerGenerate": "MotionStreamer Generate",
    "MotionStreamerPreviewAnimation": "MotionStreamer Preview Animation (3D)",
    "MotionStreamerExportFBX": "MotionStreamer Export FBX",
}
