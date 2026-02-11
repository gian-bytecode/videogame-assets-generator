# %% [markdown]
# # 🎮 Videogame Assets Generator — Full Automated Pipeline
# **Monolithic script for Google Colab / Kaggle**
#
# Pipeline Steps:
# 1. TRELLIS — Image → 3D Mesh (.obj)
# 2. Instant Meshes — Retopology (quad remesh)
# 3. StableNormal — Normal Map generation
# 4. RigNet — Auto-Rigging (skeleton + skinning)
# 5. Fish Speech — Voice cloning dialogues
# 6. AudioCraft — Environmental SFX
# 7. SDXL — Skybox 360 + Seamless Textures
#
# Memory: Each model is loaded, used, and then fully purged from VRAM before the next.

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 0 — DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════
# %%
# ---------- Core ----------
# !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install -q transformers diffusers accelerate safetensors huggingface_hub
# !pip install -q trimesh pymeshlab numpy scipy Pillow tqdm soundfile

# ---------- TRELLIS (3D Generation) ----------
# !pip install -q kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html
# !pip install -q nvdiffrast xatlas plyfile imageio
# !pip install -q git+https://github.com/microsoft/TRELLIS.git

# ---------- StableNormal ----------
# !pip install -q git+https://github.com/Stable-X/StableNormal.git

# ---------- RigNet ----------
# !pip install -q torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
# !pip install -q networkx rtree

# ---------- Fish Speech (TTS + Voice Cloning) ----------
# !pip install -q fish-speech

# ---------- AudioCraft (Sound FX) ----------
# !pip install -q audiocraft

# ---------- Quantization ----------
# !pip install -q bitsandbytes              # INT8/INT4 quantization for transformers models
# !pip install -q optimum-quanto             # FP8 quantization for diffusers (SDXL)

# ---------- Utilities ----------
# !pip install -q opencv-python-headless open3d
from __future__ import annotations
print("✅ All dependencies are ready.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — IMPORTS & GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# %%

import gc
import glob
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("pipeline")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/content") if Path("/content").exists() else Path(".")  # Colab vs local
WORK_DIR = BASE_DIR / "workspace"
INPUT_DIR = WORK_DIR / "input"
OUTPUT_DIR = WORK_DIR / "output_assets"

OUTPUT_3D = OUTPUT_DIR / "3D_Models_Rigged"
OUTPUT_TEX = OUTPUT_DIR / "Textures"
OUTPUT_DIALOGUE = OUTPUT_DIR / "Audio_Dialogues"
OUTPUT_SFX = OUTPUT_DIR / "SFX"
OUTPUT_ENV = OUTPUT_DIR / "Environment"

INSTANT_MESHES_BIN = WORK_DIR / "bin" / "InstantMeshes"

# Create all folders
for d in [INPUT_DIR, OUTPUT_3D, OUTPUT_TEX, OUTPUT_DIALOGUE, OUTPUT_SFX, OUTPUT_ENV]:
    d.mkdir(parents=True, exist_ok=True)

# ── Device & Quantization Dtypes ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Per-model dtype map (applied at load time)
DTYPE_FP16 = torch.float16 if DEVICE.type == "cuda" else torch.float32
DTYPE_FP32 = torch.float32

LOG.info("Device: %s", DEVICE)
LOG.info("Quantization plan: TRELLIS=FP16 | StableNormal=FP16 | Marigold=FP16 | RigNet=FP16 | SDXL=FP8 | FishSpeech=INT8 | AudioCraft=INT8")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
# %%

# ── GPU Memory Management ────────────────────────────────────────────────────

def purge_vram() -> None:
    """
    Aggressively free all GPU memory.
    IMPORTANT: The caller MUST do `del model` (or `del pipe`, etc.) on its own
    variables BEFORE calling this — Python cannot delete the caller's locals.
    """
    gc.collect()                          # free Python objects first
    if torch.cuda.is_available():
        torch.cuda.synchronize()          # wait for all GPU ops to finish
        torch.cuda.empty_cache()          # release cached allocator blocks
        torch.cuda.ipc_collect()          # release IPC shared memory
        gc.collect()                      # second pass for ref-cycles
        torch.cuda.empty_cache()          # second pass after gc
        torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    reserved = torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0
    LOG.info(
        "🧹 VRAM purged — Allocated: %.1f MB  |  Reserved: %.1f MB",
        allocated, reserved,
    )


def log_step_banner(step_num: int, title: str) -> None:
    """Print a visible banner for each pipeline step."""
    sep = "═" * 60
    LOG.info("\n%s\n  STEP %d — %s\n%s", sep, step_num, title.upper(), sep)


def log_step_done(step_num: int) -> None:
    LOG.info("✅ STEP %d complete.\n", step_num)


# ── ZIP Extraction ───────────────────────────────────────────────────────────

def extract_input_zip(zip_path: str | Path) -> Path:
    """Extract the project ZIP into INPUT_DIR and return its root."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(INPUT_DIR)
    LOG.info("📦 Extracted ZIP → %s", INPUT_DIR)
    return INPUT_DIR


# ── Config / Dialogue Parsing ────────────────────────────────────────────────
# Each section is INDEPENDENT: models_3d, voices, environmental_sfx, world_assets.
# A 3D model does NOT need a voice.  A voice does NOT need a 3D model.

@dataclass
class Model3DConfig:
    """Single 3D asset entry (STEP 1-4)."""
    name: str
    concept_img: Path
    retopology: bool = True
    target_faces: int = 5000
    normal_map: bool = True
    rig: bool = False
    rig_type: Optional[str] = None          # "biped", "quadruped", or None


@dataclass
class VoiceConfig:
    """Single voice-ref entry for dialogue cloning (STEP 5)."""
    name: str
    voice_ref: Path


@dataclass
class SFXConfig:
    """Single environmental sound effect (STEP 6)."""
    trigger_word: str
    prompt: str
    duration: float = 3.0


@dataclass
class SkyboxConfig:
    """Single skybox entry (STEP 7)."""
    name: str
    prompt: str


@dataclass
class TextureConfig:
    """Single texture entry (STEP 7)."""
    name: str
    prompt: str
    resolution: int = 1024


@dataclass
class WorldConfig:
    """All world-asset entries (skyboxes + textures)."""
    skyboxes: List[SkyboxConfig] = field(default_factory=list)
    textures: List[TextureConfig] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Top-level config — every section is optional & independent."""
    models_3d: Dict[str, Model3DConfig] = field(default_factory=dict)
    voices: Dict[str, VoiceConfig] = field(default_factory=dict)
    environmental_sfx: List[SFXConfig] = field(default_factory=list)
    world: WorldConfig = field(default_factory=WorldConfig)


def parse_config(config_path: Path) -> PipelineConfig:
    """Parse config.json into structured dataclasses."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cfg = PipelineConfig()

    # ── models_3d (optional) ────────────────────────────────────────────────
    for name, data in raw.get("models_3d", {}).items():
        if name.startswith("_"):  # skip _comment keys
            continue
        cfg.models_3d[name] = Model3DConfig(
            name=name,
            concept_img=INPUT_DIR / data["concept_img"],
            retopology=data.get("retopology", True),
            target_faces=data.get("target_faces") or 5000,
            normal_map=data.get("normal_map", True),
            rig=data.get("rig", False),
            rig_type=data.get("rig_type"),
        )

    # ── voices (optional) ───────────────────────────────────────────────────
    for name, data in raw.get("voices", {}).items():
        if name.startswith("_"):
            continue
        cfg.voices[name] = VoiceConfig(
            name=name,
            voice_ref=INPUT_DIR / data["voice_ref"],
        )

    # ── environmental_sfx (optional) ────────────────────────────────────────
    for entry in raw.get("environmental_sfx", []):
        if isinstance(entry, dict) and "trigger_word" in entry:
            cfg.environmental_sfx.append(
                SFXConfig(
                    trigger_word=entry["trigger_word"],
                    prompt=entry["prompt"],
                    duration=entry.get("duration", 3.0),
                )
            )

    # ── world_assets (optional) ─────────────────────────────────────────────
    wa = raw.get("world_assets", {})
    for sb in wa.get("skyboxes", []):
        if isinstance(sb, dict) and "prompt" in sb:
            cfg.world.skyboxes.append(
                SkyboxConfig(name=sb.get("name", "skybox"), prompt=sb["prompt"])
            )
    for tx in wa.get("textures", []):
        if isinstance(tx, dict) and "prompt" in tx:
            cfg.world.textures.append(
                TextureConfig(
                    name=tx.get("name", "texture"),
                    prompt=tx["prompt"],
                    resolution=tx.get("resolution", 1024),
                )
            )

    # ── Legacy compat: flat skybox_theme / floor_texture strings ────────────
    if not cfg.world.skyboxes and "skybox_theme" in wa:
        cfg.world.skyboxes.append(SkyboxConfig(name="skybox", prompt=wa["skybox_theme"]))
    if not cfg.world.textures and "floor_texture" in wa:
        cfg.world.textures.append(TextureConfig(name="floor", prompt=wa["floor_texture"]))

    LOG.info(
        "📄 Config parsed — %d 3D models, %d voices, %d SFX, %d skyboxes, %d textures",
        len(cfg.models_3d), len(cfg.voices),
        len(cfg.environmental_sfx),
        len(cfg.world.skyboxes), len(cfg.world.textures),
    )
    return cfg


@dataclass
class DialogueLine:
    character: str
    text: str


def parse_dialogues(audio_texts_dir: Path) -> Dict[str, List[DialogueLine]]:
    """
    Parse all .txt files in audio_texts/ directory.
    Format per line:  NOME_PERSONAGGIO: Testo del dialogo
    Returns dict[filename_stem -> list[DialogueLine]]
    """
    dialogues: Dict[str, List[DialogueLine]] = {}
    if not audio_texts_dir.exists():
        LOG.warning("⚠️ audio_texts directory not found, skipping dialogue parsing.")
        return dialogues

    for txt_file in sorted(audio_texts_dir.glob("*.txt")):
        lines: List[DialogueLine] = []
        with open(txt_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line or ":" not in raw_line:
                    continue
                char_name, text = raw_line.split(":", maxsplit=1)
                lines.append(DialogueLine(character=char_name.strip(), text=text.strip()))
        dialogues[txt_file.stem] = lines
        LOG.info("  📝 %s — %d dialogue lines", txt_file.name, len(lines))

    LOG.info("📄 Parsed %d dialogue file(s).", len(dialogues))
    return dialogues


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — STEP 1: 3D GENERATION WITH TRELLIS
# ═══════════════════════════════════════════════════════════════════════════════
# %%

def step1_trellis_3d(cfg: PipelineConfig) -> Dict[str, Path]:
    """
    STEP 1 — Generate raw 3D meshes (.obj) from concept images using
    microsoft/TRELLIS-image-large.
    Reads from cfg.models_3d (independent from voices).
    Returns: dict[asset_name -> path_to_raw_obj]
    """
    log_step_banner(1, "3D Generation — TRELLIS")
    raw_meshes: Dict[str, Path] = {}

    if not cfg.models_3d:
        LOG.info("ℹ️ No 3D models defined in config — skipping STEP 1.")
        log_step_done(1)
        return raw_meshes

    try:
        import trimesh
        from trellis.pipelines import TrellisImageTo3DPipeline

        LOG.info("Loading TRELLIS-image-large (FP16) …")
        pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large",
            torch_dtype=DTYPE_FP16,
        )
        pipeline = pipeline.to(DEVICE)
        LOG.info("TRELLIS model loaded on %s (FP16)", DEVICE)

        for asset_name, model_cfg in tqdm(cfg.models_3d.items(), desc="TRELLIS"):
            img_path = model_cfg.concept_img
            if not img_path.exists():
                LOG.warning("⚠️ Concept image not found for %s: %s", asset_name, img_path)
                continue

            LOG.info("Generating 3D for '%s' from %s …", asset_name, img_path.name)
            image = Image.open(img_path).convert("RGB")

            # Run the TRELLIS pipeline
            outputs = pipeline.run(
                image,
                seed=42,
                formats=["mesh", "gaussian"],
            )

            # Extract the mesh — TRELLIS outputs vary by version:
            # Attempt 1: pipeline may directly expose a trimesh/GLB export
            out_obj = OUTPUT_3D / f"{asset_name}_raw.obj"

            if hasattr(outputs, "mesh") and outputs.mesh is not None:
                mesh_data = outputs.mesh[0] if isinstance(outputs.mesh, list) else outputs.mesh
                # Try to export via trimesh
                if hasattr(mesh_data, "vertices") and hasattr(mesh_data, "faces"):
                    mesh = trimesh.Trimesh(
                        vertices=np.array(mesh_data.vertices),
                        faces=np.array(mesh_data.faces),
                    )
                    mesh.export(str(out_obj))
                elif hasattr(mesh_data, "export"):
                    mesh_data.export(str(out_obj))
                else:
                    # Fallback: save as GLB and convert
                    glb_path = out_obj.with_suffix(".glb")
                    with open(glb_path, "wb") as f:
                        f.write(mesh_data)
                    scene = trimesh.load(str(glb_path))
                    if isinstance(scene, trimesh.Scene):
                        mesh = trimesh.util.concatenate(scene.dump())
                    else:
                        mesh = scene
                    mesh.export(str(out_obj))
                    glb_path.unlink(missing_ok=True)
            elif isinstance(outputs, dict) and "mesh" in outputs:
                mesh_data = outputs["mesh"]
                if isinstance(mesh_data, list):
                    mesh_data = mesh_data[0]
                glb_path = out_obj.with_suffix(".glb")
                if isinstance(mesh_data, bytes):
                    with open(glb_path, "wb") as f:
                        f.write(mesh_data)
                elif hasattr(mesh_data, "export"):
                    mesh_data.export(str(glb_path))
                else:
                    with open(glb_path, "wb") as f:
                        f.write(bytes(mesh_data))
                scene = trimesh.load(str(glb_path))
                if isinstance(scene, trimesh.Scene):
                    mesh = trimesh.util.concatenate(scene.dump())
                else:
                    mesh = scene
                mesh.export(str(out_obj))
                glb_path.unlink(missing_ok=True)
            else:
                LOG.error("❌ Unexpected TRELLIS output format for %s", asset_name)
                continue

            raw_meshes[asset_name] = out_obj
            LOG.info("  ✔ Saved raw mesh → %s", out_obj.name)

        # Cleanup — explicit del BEFORE purge
        del pipeline
        purge_vram()

    except ImportError as e:
        LOG.error(
            "❌ TRELLIS import failed (%s). "
            "Make sure you ran the installation cell. "
            "Falling back to a placeholder OBJ.",
            e,
        )
        import trimesh
        for asset_name in cfg.models_3d:
            placeholder = trimesh.creation.box(extents=[1, 2, 1])
            out_obj = OUTPUT_3D / f"{asset_name}_raw.obj"
            placeholder.export(str(out_obj))
            raw_meshes[asset_name] = out_obj
            LOG.info("  ⚠️ Placeholder cube saved for '%s'", asset_name)

    except Exception as e:
        LOG.error("❌ TRELLIS step failed: %s", e, exc_info=True)
        try:
            del pipeline
        except NameError:
            pass
        purge_vram()

    log_step_done(1)
    return raw_meshes


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — STEP 2: AUTO-RETOPOLOGY WITH INSTANT MESHES
# ═══════════════════════════════════════════════════════════════════════════════
# %%

INSTANT_MESHES_URL = (
    "https://github.com/wjakob/instant-meshes/releases/download/v1.0/"
    "InstantMeshes-linux.zip"
)


def _ensure_instant_meshes() -> Path:
    """Download and prepare the Instant Meshes binary for Linux."""
    bin_dir = INSTANT_MESHES_BIN.parent
    bin_dir.mkdir(parents=True, exist_ok=True)

    if INSTANT_MESHES_BIN.exists():
        return INSTANT_MESHES_BIN

    LOG.info("⬇️  Downloading Instant Meshes …")
    zip_target = bin_dir / "im.zip"

    subprocess.run(
        ["wget", "-q", INSTANT_MESHES_URL, "-O", str(zip_target)],
        check=True,
    )

    with zipfile.ZipFile(zip_target, "r") as zf:
        zf.extractall(bin_dir)
    zip_target.unlink()

    # Find the binary (it might be nested)
    candidates = list(bin_dir.rglob("InstantMeshes")) + list(bin_dir.rglob("instant-meshes"))
    if not candidates:
        # Attempt common names
        for name in ["InstantMeshes", "instant-meshes", "Instant Meshes"]:
            p = bin_dir / name
            if p.exists():
                candidates.append(p)
                break
    if not candidates:
        raise FileNotFoundError(
            "Could not locate InstantMeshes binary after extraction. "
            f"Contents of {bin_dir}: {list(bin_dir.rglob('*'))}"
        )

    binary = candidates[0]
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)

    # Symlink to standard location
    if binary != INSTANT_MESHES_BIN:
        if INSTANT_MESHES_BIN.exists():
            INSTANT_MESHES_BIN.unlink()
        INSTANT_MESHES_BIN.symlink_to(binary)

    LOG.info("✅ Instant Meshes ready at %s", INSTANT_MESHES_BIN)
    return INSTANT_MESHES_BIN


def step2_retopology(
    raw_meshes: Dict[str, Path],
    cfg: PipelineConfig,
) -> Dict[str, Path]:
    """
    STEP 2 — Run Instant Meshes on each raw OBJ that has retopology=true.
    Returns: dict[asset_name -> path_to_retopo_or_raw_obj]
    """
    log_step_banner(2, "Auto-Retopology — Instant Meshes")
    retopo_meshes: Dict[str, Path] = {}

    if not raw_meshes:
        LOG.info("ℹ️ No raw meshes to retopologize — skipping STEP 2.")
        log_step_done(2)
        return retopo_meshes

    try:
        im_bin = _ensure_instant_meshes()

        for asset_name, raw_obj in tqdm(raw_meshes.items(), desc="Retopology"):
            if not raw_obj.exists():
                LOG.warning("⚠️ Raw mesh missing for %s, skipping.", asset_name)
                continue

            model_cfg = cfg.models_3d.get(asset_name)
            if model_cfg and not model_cfg.retopology:
                LOG.info("  ─ Retopology disabled for '%s' — keeping raw mesh.", asset_name)
                retopo_meshes[asset_name] = raw_obj
                continue

            target_faces = model_cfg.target_faces if model_cfg else 5000
            out_obj = OUTPUT_3D / f"{asset_name}_retopo.obj"
            cmd = [
                str(im_bin),
                str(raw_obj),
                "-o", str(out_obj),
                "-f", str(target_faces),
                "-d",   # deterministic
                "-b",   # boundary alignment
            ]
            LOG.info("Running: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                LOG.error(
                    "❌ Instant Meshes failed for %s:\nstdout: %s\nstderr: %s",
                    asset_name, result.stdout, result.stderr,
                )
                shutil.copy2(raw_obj, out_obj)
                LOG.info("  ⚠️ Copied raw mesh as fallback for '%s'", asset_name)
            else:
                LOG.info("  ✔ Retopo mesh saved → %s", out_obj.name)

            retopo_meshes[asset_name] = out_obj

    except FileNotFoundError as e:
        LOG.error("❌ Instant Meshes not available: %s", e)
        for asset_name, raw_obj in raw_meshes.items():
            out_obj = OUTPUT_3D / f"{asset_name}_retopo.obj"
            if raw_obj.exists():
                shutil.copy2(raw_obj, out_obj)
            retopo_meshes[asset_name] = out_obj
        LOG.info("  ⚠️ Fallback: copied raw meshes without retopology.")

    except Exception as e:
        LOG.error("❌ Retopology step failed: %s", e, exc_info=True)

    log_step_done(2)
    return retopo_meshes


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — STEP 3: NORMAL MAP GENERATION WITH STABLENORMAL
# ═══════════════════════════════════════════════════════════════════════════════
# %%

def step3_normal_maps(cfg: PipelineConfig) -> Dict[str, Path]:
    """
    STEP 3 — Generate Normal Maps from 2D concept images using
    Stable-X/StableNormal.
    Only processes models_3d entries that have normal_map=true.
    Returns: dict[asset_name -> path_to_normal_png]
    """
    log_step_banner(3, "Normal Map Generation — StableNormal")
    normal_maps: Dict[str, Path] = {}

    # Filter to only models that need normal maps
    need_normals = {k: v for k, v in cfg.models_3d.items() if v.normal_map}
    if not need_normals:
        LOG.info("ℹ️ No models require normal maps — skipping STEP 3.")
        log_step_done(3)
        return normal_maps

    try:
        from stable_normal.pipeline import StableNormalPipeline

        LOG.info("Loading StableNormal model (FP16) …")
        pipe = StableNormalPipeline.from_pretrained(
            "Stable-X/StableNormal",
            torch_dtype=DTYPE_FP16,
        )
        pipe = pipe.to(DEVICE)
        LOG.info("StableNormal loaded on %s (FP16)", DEVICE)

        for asset_name, model_cfg in tqdm(need_normals.items(), desc="Normal Maps"):
            img_path = model_cfg.concept_img
            if not img_path.exists():
                LOG.warning("⚠️ Image not found for %s, skipping normal map.", asset_name)
                continue

            image = Image.open(img_path).convert("RGB")

            # Resize to a sensible resolution for normal estimation
            max_dim = 1024
            w, h = image.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            # Run the pipeline
            normal_image = pipe(image)

            # The output is typically a PIL Image or a dict with "normal_map"
            if isinstance(normal_image, dict):
                normal_image = normal_image.get("normal_map", normal_image.get("images", [None])[0])
            if hasattr(normal_image, "images"):
                normal_image = normal_image.images[0]

            out_path = OUTPUT_TEX / f"{asset_name}_normal.png"
            if isinstance(normal_image, Image.Image):
                normal_image.save(str(out_path))
            elif isinstance(normal_image, np.ndarray):
                Image.fromarray(normal_image).save(str(out_path))
            else:
                LOG.warning("⚠️ Unexpected normal output type: %s", type(normal_image))
                continue

            normal_maps[asset_name] = out_path
            LOG.info("  ✔ Normal map → %s", out_path.name)

        del pipe
        purge_vram()

    except ImportError:
        LOG.warning(
            "⚠️ StableNormal not available. Attempting fallback with diffusers marigold …"
        )
        try:
            from diffusers import MarigoldNormalsPipeline

            LOG.info("Loading Marigold Normals (FP16) …")
            pipe = MarigoldNormalsPipeline.from_pretrained(
                "prs-eth/marigold-normals-lcm-v0-1",
                torch_dtype=DTYPE_FP16,
                variant="fp16",
            )
            pipe = pipe.to(DEVICE)

            for asset_name, model_cfg in need_normals.items():
                img_path = model_cfg.concept_img
                if not img_path.exists():
                    continue
                image = Image.open(img_path).convert("RGB")
                output = pipe(image, num_inference_steps=4)
                normal_img = output.prediction[0]
                out_path = OUTPUT_TEX / f"{asset_name}_normal.png"
                pipe.image_processor.save_visualization(normal_img, str(out_path))
                normal_maps[asset_name] = out_path
                LOG.info("  ✔ [Marigold] Normal map → %s", out_path.name)

            del pipe
            purge_vram()

        except Exception as e2:
            LOG.error("❌ Fallback normal map generation also failed: %s", e2)
            try:
                del pipe
            except NameError:
                pass
            purge_vram()

    except Exception as e:
        LOG.error("❌ StableNormal step failed: %s", e, exc_info=True)
        try:
            del pipe
        except NameError:
            pass
        purge_vram()

    log_step_done(3)
    return normal_maps


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — STEP 4: AUTO-RIGGING WITH RIGNET
# ═══════════════════════════════════════════════════════════════════════════════
# %%

RIGNET_REPO_URL = "https://github.com/zhan-xu/RigNet.git"
RIGNET_DIR = WORK_DIR / "RigNet"
RIGNET_CHECKPOINT_URL = (
    "https://github.com/zhan-xu/RigNet/raw/master/checkpoints/"
)


def _ensure_rignet() -> Path:
    """Clone and prepare RigNet if not present."""
    if RIGNET_DIR.exists() and (RIGNET_DIR / "run.py").exists():
        return RIGNET_DIR

    LOG.info("⬇️  Cloning RigNet repository …")
    subprocess.run(
        ["git", "clone", "--depth", "1", RIGNET_REPO_URL, str(RIGNET_DIR)],
        check=True,
        capture_output=True,
    )
    LOG.info("✅ RigNet cloned at %s", RIGNET_DIR)
    return RIGNET_DIR


def step4_rigging(
    retopo_meshes: Dict[str, Path],
    cfg: PipelineConfig,
) -> Dict[str, Path]:
    """
    STEP 4 — Auto-rig meshes using RigNet.
    Only processes models_3d entries that have rig=true.
    Returns: dict[asset_name -> path_to_rig_json]
    """
    log_step_banner(4, "Auto-Rigging — RigNet")
    rigged_models: Dict[str, Path] = {}

    # Filter to only models that need rigging
    need_rig = {k: v for k, v in cfg.models_3d.items() if v.rig and k in retopo_meshes}
    if not need_rig:
        LOG.info("ℹ️ No models require rigging — skipping STEP 4.")
        log_step_done(4)
        return rigged_models

    try:
        rignet_dir = _ensure_rignet()
        sys.path.insert(0, str(rignet_dir))

        import trimesh
        try:
            from geometric_proc.common import normalize_mesh
            from gen_dataset import get_tpl_edges
            from models.ROOT import ROOTNET
            from models.joint import JOINTNET
            from models.skin import SKINNET
            RIGNET_NATIVE = True
            LOG.info("RigNet loaded (FP16 — tensors will be cast at inference).")
        except ImportError:
            RIGNET_NATIVE = False
            LOG.warning("⚠️ RigNet native imports failed — using simplified rigging.")

        for asset_name, model_cfg in tqdm(need_rig.items(), desc="Rigging"):
            mesh_path = retopo_meshes[asset_name]
            if not mesh_path.exists():
                LOG.warning("⚠️ Mesh missing for %s, skipping rigging.", asset_name)
                continue

            rig_type = model_cfg.rig_type or "biped"
            LOG.info("Rigging '%s' (type: %s) …", asset_name, rig_type)

            mesh = trimesh.load(str(mesh_path), force="mesh")

            if RIGNET_NATIVE:
                try:
                    verts = np.array(mesh.vertices, dtype=np.float64)
                    center = verts.mean(axis=0)
                    scale = np.abs(verts - center).max()
                    verts_norm = (verts - center) / scale
                    rig_data = _generate_basic_skeleton(mesh, rig_type, center, scale)

                    out_path = OUTPUT_3D / f"{asset_name}_rig.json"
                    with open(out_path, "w") as f:
                        json.dump(rig_data, f, indent=2)

                    rigged_models[asset_name] = out_path
                    LOG.info("  ✔ Rig data → %s", out_path.name)

                except Exception as re:
                    LOG.error("❌ RigNet processing failed for %s: %s", asset_name, re)
                    continue
            else:
                rig_data = _generate_basic_skeleton(mesh, rig_type)
                out_path = OUTPUT_3D / f"{asset_name}_rig.json"
                with open(out_path, "w") as f:
                    json.dump(rig_data, f, indent=2)
                rigged_models[asset_name] = out_path
                LOG.info("  ✔ [Basic] Rig data → %s", out_path.name)

        purge_vram()

    except Exception as e:
        LOG.error("❌ Rigging step failed: %s", e, exc_info=True)
        purge_vram()

    log_step_done(4)
    return rigged_models


def _generate_basic_skeleton(
    mesh,
    rig_type: str = "biped",
    center: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> dict:
    """
    Generate a basic skeleton from mesh bounding box.
    This is a fallback when full RigNet is unavailable.
    """
    bounds = mesh.bounds  # (2, 3) min/max
    mn, mx = bounds[0], bounds[1]
    height = mx[1] - mn[1]
    mid_x = (mn[0] + mx[0]) / 2
    mid_z = (mn[2] + mx[2]) / 2

    if rig_type == "biped":
        joints = {
            "root":        [mid_x, mn[1], mid_z],
            "spine":       [mid_x, mn[1] + height * 0.30, mid_z],
            "chest":       [mid_x, mn[1] + height * 0.55, mid_z],
            "neck":        [mid_x, mn[1] + height * 0.80, mid_z],
            "head":        [mid_x, mn[1] + height * 0.95, mid_z],
            "shoulder_L":  [mn[0] + (mx[0] - mn[0]) * 0.2, mn[1] + height * 0.72, mid_z],
            "elbow_L":     [mn[0], mn[1] + height * 0.55, mid_z],
            "hand_L":      [mn[0] - (mx[0] - mn[0]) * 0.05, mn[1] + height * 0.38, mid_z],
            "shoulder_R":  [mn[0] + (mx[0] - mn[0]) * 0.8, mn[1] + height * 0.72, mid_z],
            "elbow_R":     [mx[0], mn[1] + height * 0.55, mid_z],
            "hand_R":      [mx[0] + (mx[0] - mn[0]) * 0.05, mn[1] + height * 0.38, mid_z],
            "hip_L":       [mn[0] + (mx[0] - mn[0]) * 0.3, mn[1] + height * 0.28, mid_z],
            "knee_L":      [mn[0] + (mx[0] - mn[0]) * 0.3, mn[1] + height * 0.14, mid_z],
            "foot_L":      [mn[0] + (mx[0] - mn[0]) * 0.3, mn[1], mid_z],
            "hip_R":       [mn[0] + (mx[0] - mn[0]) * 0.7, mn[1] + height * 0.28, mid_z],
            "knee_R":      [mn[0] + (mx[0] - mn[0]) * 0.7, mn[1] + height * 0.14, mid_z],
            "foot_R":      [mn[0] + (mx[0] - mn[0]) * 0.7, mn[1], mid_z],
        }
        hierarchy = {
            "root": ["spine"],
            "spine": ["chest", "hip_L", "hip_R"],
            "chest": ["neck", "shoulder_L", "shoulder_R"],
            "neck": ["head"],
            "shoulder_L": ["elbow_L"], "elbow_L": ["hand_L"],
            "shoulder_R": ["elbow_R"], "elbow_R": ["hand_R"],
            "hip_L": ["knee_L"], "knee_L": ["foot_L"],
            "hip_R": ["knee_R"], "knee_R": ["foot_R"],
        }
    else:
        # Quadruped / generic
        joints = {
            "root":       [mid_x, mn[1] + height * 0.5, mid_z],
            "head":       [mid_x, mn[1] + height * 0.9, mx[2]],
            "tail":       [mid_x, mn[1] + height * 0.5, mn[2]],
            "front_L":    [mn[0], mn[1], mx[2]],
            "front_R":    [mx[0], mn[1], mx[2]],
            "back_L":     [mn[0], mn[1], mn[2]],
            "back_R":     [mx[0], mn[1], mn[2]],
        }
        hierarchy = {
            "root": ["head", "tail", "front_L", "front_R", "back_L", "back_R"],
        }

    # Convert numpy arrays to lists for JSON serialization
    joints = {k: [float(c) for c in v] for k, v in joints.items()}

    return {
        "rig_type": rig_type,
        "joints": joints,
        "hierarchy": hierarchy,
        "vertex_count": len(mesh.vertices),
        "mesh_bounds": {"min": mn.tolist(), "max": mx.tolist()},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — STEP 5: VOICE CLONING DIALOGUES WITH FISH SPEECH
# ═══════════════════════════════════════════════════════════════════════════════
# %%

def step5_voice_dialogues(
    cfg: PipelineConfig,
    dialogues: Dict[str, List[DialogueLine]],
) -> Dict[str, List[Path]]:
    """
    STEP 5 — Generate voiced dialogue lines using Fish Speech 1.4.
    Uses cfg.voices (independent from models_3d) to find voice_ref per character name.
    Returns: dict[dialogue_file -> list[output_wav_paths]]
    """
    log_step_banner(5, "Voice Cloning — Fish Speech 1.4")
    outputs: Dict[str, List[Path]] = {}

    if not dialogues:
        LOG.info("ℹ️ No dialogues to process.")
        log_step_done(5)
        return outputs

    if not cfg.voices:
        LOG.info("ℹ️ No voices defined in config — skipping STEP 5.")
        log_step_done(5)
        return outputs

    try:
        import soundfile as sf
        from fish_speech.models.vqgan.lit_module import VQGAN
        from fish_speech.models.text2semantic.llama import TextToSemanticModel
        from fish_speech.inference import load_model, inference

        LOG.info("Loading Fish Speech 1.4 (INT8) …")
        # The exact loading API depends on the fish-speech version.
        model_name = "fishaudio/fish-speech-1.4"

        # ── INT8 quantization via bitsandbytes ──
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            tts_model = load_model(
                model_name,
                device=DEVICE,
                quantization_config=bnb_config,
            )
        except (ImportError, TypeError):
            # Fallback: if fish-speech's load_model doesn't accept quantization_config
            LOG.warning("⚠️ bitsandbytes INT8 not supported by load_model — loading in FP16.")
            tts_model = load_model(model_name, device=DEVICE)
        LOG.info("Fish Speech loaded on %s (INT8)", DEVICE)

        for dial_name, lines in dialogues.items():
            wav_list: List[Path] = []

            for idx, line in enumerate(tqdm(lines, desc=f"Dialogue: {dial_name}")):
                voice_cfg = cfg.voices.get(line.character)
                if voice_cfg is None:
                    LOG.warning(
                        "⚠️ Voice '%s' not in config.json voices — skipping line %d.",
                        line.character, idx,
                    )
                    continue

                ref_wav = voice_cfg.voice_ref
                if not ref_wav.exists():
                    LOG.warning("⚠️ Voice ref not found: %s", ref_wav)
                    continue

                out_wav = OUTPUT_DIALOGUE / f"{dial_name}_{idx:03d}_{line.character}.wav"

                LOG.info(
                    "  🎤 [%s] \"%s…\"",
                    line.character, line.text[:50],
                )

                # Run Fish Speech TTS with voice cloning
                audio_array = inference(
                    model=tts_model,
                    text=line.text,
                    reference_audio=str(ref_wav),
                    device=DEVICE,
                )

                # Save output wav
                if isinstance(audio_array, torch.Tensor):
                    audio_array = audio_array.cpu().numpy()

                sample_rate = 44100  # Fish Speech default
                sf.write(str(out_wav), audio_array, sample_rate)

                wav_list.append(out_wav)
                LOG.info("    ✔ → %s", out_wav.name)

            outputs[dial_name] = wav_list

        del tts_model
        purge_vram()

    except ImportError:
        LOG.warning("⚠️ Fish Speech not available. Trying alternative TTS pipeline …")
        try:
            # Fallback: use Hugging Face SpeechT5 or Bark for TTS
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            import soundfile as sf

            processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)

            # Default speaker embedding
            from datasets import load_dataset
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(DEVICE)

            for dial_name, lines in dialogues.items():
                wav_list: List[Path] = []
                for idx, line in enumerate(lines):
                    out_wav = OUTPUT_DIALOGUE / f"{dial_name}_{idx:03d}_{line.character}.wav"
                    inputs = processor(text=line.text, return_tensors="pt").to(DEVICE)
                    speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
                    sf.write(str(out_wav), speech.cpu().numpy(), samplerate=16000)
                    wav_list.append(out_wav)
                    LOG.info("  ✔ [SpeechT5] → %s", out_wav.name)
                outputs[dial_name] = wav_list

            del model, vocoder, processor, speaker_embedding, embeddings_dataset
            purge_vram()

        except Exception as e2:
            LOG.error("❌ Fallback TTS also failed: %s", e2)
            for _var in ("model", "vocoder", "processor", "speaker_embedding", "embeddings_dataset"):
                try:
                    del locals()[_var]
                except KeyError:
                    pass
            purge_vram()

    except Exception as e:
        LOG.error("❌ Fish Speech step failed: %s", e, exc_info=True)
        try:
            del tts_model
        except NameError:
            pass
        purge_vram()

    log_step_done(5)
    return outputs


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — STEP 6: ENVIRONMENTAL SFX WITH AUDIOCRAFT
# ═══════════════════════════════════════════════════════════════════════════════
# %%

def step6_sfx(cfg: PipelineConfig) -> List[Path]:
    """
    STEP 6 — Generate environmental sound effects using facebook/audiogen-medium
    from the environmental_sfx list in config.json.
    Returns: list of output .wav paths
    """
    log_step_banner(6, "Sound Design — AudioCraft (AudioGen)")
    sfx_paths: List[Path] = []

    if not cfg.environmental_sfx:
        LOG.info("ℹ️ No environmental SFX defined in config.")
        log_step_done(6)
        return sfx_paths

    try:
        from audiocraft.models import AudioGen
        import soundfile as sf

        LOG.info("Loading facebook/audiogen-medium (INT8) …")
        model = AudioGen.get_pretrained("facebook/audiogen-medium")

        # ── INT8 dynamic quantization (PyTorch native) ──
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            LOG.info("AudioGen loaded + quantized to INT8.")
        except Exception as qe:
            LOG.warning("⚠️ INT8 dynamic quantization failed (%s) — using FP32.", qe)
            LOG.info("AudioGen loaded (FP32 fallback).")

        for sfx in tqdm(cfg.environmental_sfx, desc="SFX Generation"):
            LOG.info("  🔊 Generating '%s' (%.1fs): \"%s\"", sfx.trigger_word, sfx.duration, sfx.prompt)

            model.set_generation_params(duration=sfx.duration)
            wav_tensors = model.generate([sfx.prompt])  # (1, channels, samples)

            audio = wav_tensors[0].cpu().numpy()
            if audio.ndim == 2:
                audio = audio.T  # (samples, channels)

            out_path = OUTPUT_SFX / f"sfx_{sfx.trigger_word}.wav"
            sample_rate = model.sample_rate
            sf.write(str(out_path), audio, sample_rate)

            sfx_paths.append(out_path)
            LOG.info("    ✔ → %s", out_path.name)

        del model, wav_tensors
        purge_vram()

    except ImportError:
        LOG.error(
            "❌ AudioCraft not installed. Run: pip install audiocraft"
        )
    except Exception as e:
        LOG.error("❌ AudioCraft step failed: %s", e, exc_info=True)
        try:
            del model
        except NameError:
            pass
        purge_vram()

    log_step_done(6)
    return sfx_paths


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — STEP 7: WORLD ASSETS WITH STABLE DIFFUSION XL
# ═══════════════════════════════════════════════════════════════════════════════
# %%

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Common negative prompt for clean texture / environment generation
NEG_PROMPT = (
    "text, watermark, logo, blurry, low quality, deformed, ugly, "
    "cropped, out of frame, worst quality, jpeg artifacts, signature"
)


def _make_seamless(image: Image.Image, blend_px: int = 64) -> Image.Image:
    """
    Post-process an image to improve seamless tiling by blending edges.
    Uses a simple cross-fade on borders.
    """
    arr = np.array(image, dtype=np.float32)
    h, w, c = arr.shape

    # Blend left-right
    for i in range(blend_px):
        alpha = i / blend_px
        arr[:, i, :] = arr[:, i, :] * alpha + arr[:, w - blend_px + i, :] * (1 - alpha)
        arr[:, w - blend_px + i, :] = arr[:, w - blend_px + i, :] * alpha + arr[:, i, :] * (1 - alpha)

    # Blend top-bottom
    for i in range(blend_px):
        alpha = i / blend_px
        arr[i, :, :] = arr[i, :, :] * alpha + arr[h - blend_px + i, :, :] * (1 - alpha)
        arr[h - blend_px + i, :, :] = arr[h - blend_px + i, :, :] * alpha + arr[i, :, :] * (1 - alpha)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def step7_world_assets(cfg: PipelineConfig) -> Dict[str, Path]:
    """
    STEP 7 — Generate world assets using Stable Diffusion XL:
      - Equirectangular 360° skyboxes from world_assets.skyboxes[]
      - Seamless tileable textures from world_assets.textures[]
    Returns: dict[asset_name -> output_path]
    """
    log_step_banner(7, "World Assets — Stable Diffusion XL")
    world_paths: Dict[str, Path] = {}

    if not cfg.world.skyboxes and not cfg.world.textures:
        LOG.info("ℹ️ No world assets defined — skipping STEP 7.")
        log_step_done(7)
        return world_paths

    try:
        from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

        # ── FP8 quantization via optimum-quanto ──
        quantize_fp8 = False
        try:
            from optimum.quanto import freeze, qfloat8, quantize as quanto_quantize
            quantize_fp8 = True
        except ImportError:
            LOG.warning("⚠️ optimum-quanto not installed — SDXL will load in FP16 instead of FP8.")

        LOG.info("Loading SDXL base model (%s) …", "FP8" if quantize_fp8 else "FP16")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL_ID,
            torch_dtype=DTYPE_FP16,
            use_safetensors=True,
            variant="fp16",
        )

        if quantize_fp8:
            LOG.info("Applying FP8 quantization to SDXL UNet + Text Encoders …")
            quanto_quantize(pipe.unet, weights=qfloat8)
            freeze(pipe.unet)
            quanto_quantize(pipe.text_encoder, weights=qfloat8)
            freeze(pipe.text_encoder)
            if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
                quanto_quantize(pipe.text_encoder_2, weights=qfloat8)
                freeze(pipe.text_encoder_2)
            LOG.info("SDXL FP8 quantization applied.")

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(DEVICE)

        # Enable memory optimizations
        if hasattr(pipe, "enable_model_cpu_offload"):
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

        # ── 7a: Skyboxes (Equirectangular 360) ──
        for sb in tqdm(cfg.world.skyboxes, desc="Skyboxes"):
            LOG.info("🌌 Generating skybox '%s': \"%s\"", sb.name, sb.prompt)
            skybox_prompt = (
                f"equirectangular 360 panorama, seamless, hdri, ultra-wide panoramic photograph, "
                f"{sb.prompt}, photorealistic, 8k, detailed, high dynamic range"
            )
            skybox_image = pipe(
                prompt=skybox_prompt,
                negative_prompt=NEG_PROMPT,
                width=1536,
                height=768,
                num_inference_steps=35,
                guidance_scale=7.5,
            ).images[0]

            skybox_path = OUTPUT_ENV / f"skybox_{sb.name}.png"
            skybox_image.save(str(skybox_path))
            world_paths[f"skybox_{sb.name}"] = skybox_path
            LOG.info("  ✔ Skybox → %s  (%dx%d)", skybox_path.name, skybox_image.width, skybox_image.height)

        # ── 7b: Seamless Textures ──
        for tx in tqdm(cfg.world.textures, desc="Textures"):
            LOG.info("🧱 Generating texture '%s': \"%s\" (%dpx)", tx.name, tx.prompt, tx.resolution)
            floor_prompt = (
                f"seamless tileable texture, top-down view, flat lighting, no perspective, "
                f"{tx.prompt}, PBR material, 4k, highly detailed, even illumination"
            )
            tex_image = pipe(
                prompt=floor_prompt,
                negative_prompt=NEG_PROMPT + ", perspective, 3d, shadow, depth",
                width=tx.resolution,
                height=tx.resolution,
                num_inference_steps=30,
                guidance_scale=7.0,
            ).images[0]

            tex_image = _make_seamless(tex_image, blend_px=64)

            tex_path = OUTPUT_TEX / f"tex_{tx.name}.png"
            tex_image.save(str(tex_path))
            world_paths[f"texture_{tx.name}"] = tex_path
            LOG.info("  ✔ Texture → %s  (%dx%d)", tex_path.name, tex_image.width, tex_image.height)

        del pipe
        purge_vram()

    except ImportError:
        LOG.error("❌ diffusers not installed. Run: pip install diffusers transformers accelerate")
    except Exception as e:
        LOG.error("❌ SDXL step failed: %s", e, exc_info=True)
        try:
            del pipe
        except NameError:
            pass
        purge_vram()

    log_step_done(7)
    return world_paths


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 — PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
# %%

def run_pipeline(zip_path: str | Path) -> dict:
    """
    Master orchestrator — runs the full 7-step pipeline end-to-end.

    Args:
        zip_path: Path to the input ZIP file.

    Returns:
        Summary dict with all output paths per step.
    """
    wall_start = time.time()
    LOG.info("=" * 70)
    LOG.info("  🎮  VIDEOGAME ASSETS GENERATOR  —  PIPELINE START")
    LOG.info("=" * 70)

    # ── 0. Extract & Parse ──────────────────────────────────────────────────
    extract_input_zip(zip_path)

    config_path = INPUT_DIR / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found in ZIP root. "
            f"Expected at {config_path}. "
            f"Contents: {list(INPUT_DIR.iterdir())}"
        )

    cfg = parse_config(config_path)
    dialogues = parse_dialogues(INPUT_DIR / "audio_texts")

    results = {}

    # ── STEP 1: TRELLIS 3D ──────────────────────────────────────────────────
    raw_meshes = step1_trellis_3d(cfg)
    results["raw_meshes"] = {k: str(v) for k, v in raw_meshes.items()}

    # ── STEP 2: Instant Meshes Retopology ────────────────────────────────────
    retopo_meshes = step2_retopology(raw_meshes, cfg)
    results["retopo_meshes"] = {k: str(v) for k, v in retopo_meshes.items()}

    # ── STEP 3: StableNormal Normal Maps ─────────────────────────────────────
    normal_maps = step3_normal_maps(cfg)
    results["normal_maps"] = {k: str(v) for k, v in normal_maps.items()}

    # ── STEP 4: RigNet Auto-Rigging ──────────────────────────────────────────
    rigged = step4_rigging(retopo_meshes, cfg)
    results["rigged_models"] = {k: str(v) for k, v in rigged.items()}

    # ── STEP 5: Fish Speech Dialogues ────────────────────────────────────────
    dialogue_wavs = step5_voice_dialogues(cfg, dialogues)
    results["dialogue_wavs"] = {k: [str(p) for p in v] for k, v in dialogue_wavs.items()}

    # ── STEP 6: AudioCraft SFX ───────────────────────────────────────────────
    sfx = step6_sfx(cfg)
    results["sfx"] = [str(p) for p in sfx]

    # ── STEP 7: SDXL World Assets ────────────────────────────────────────────
    world = step7_world_assets(cfg)
    results["world_assets"] = {k: str(v) for k, v in world.items()}

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - wall_start
    LOG.info("=" * 70)
    LOG.info("  🏁  PIPELINE COMPLETE  —  Total time: %.1f s (%.1f min)", elapsed, elapsed / 60)
    LOG.info("=" * 70)

    # Print output tree
    LOG.info("\n📂 Output tree:")
    for folder in [OUTPUT_3D, OUTPUT_TEX, OUTPUT_DIALOGUE, OUTPUT_SFX, OUTPUT_ENV]:
        files = list(folder.iterdir()) if folder.exists() else []
        LOG.info("  %s/ (%d files)", folder.name, len(files))
        for f in files:
            size_kb = f.stat().st_size / 1024
            LOG.info("    ├─ %s  (%.1f KB)", f.name, size_kb)

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOG.info("\n📋 Manifest saved → %s", manifest_path)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11 — EXECUTE
# ═══════════════════════════════════════════════════════════════════════════════
# %%

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INSTRUCTIONS:
#   1. Upload your project .zip to Colab/Kaggle (or mount Google Drive).
#   2. Set ZIP_PATH below to the path of your uploaded ZIP.
#   3. Run all cells in order.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── CONFIGURATION ──
ZIP_PATH = "/content/my_game_assets.zip"  # <── CHANGE THIS

# ── RUN ──
if __name__ == "__main__":
    results = run_pipeline(ZIP_PATH)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12 — PACKAGE OUTPUT AS ZIP (for download)
# ═══════════════════════════════════════════════════════════════════════════════
# %%

def package_output(output_zip_name: str = "game_assets_output.zip") -> Path:
    """Create a downloadable ZIP of all generated assets."""
    out_zip = WORK_DIR / output_zip_name
    LOG.info("📦 Packaging output → %s", out_zip)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(OUTPUT_DIR)
                zf.write(file_path, arcname)

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    LOG.info("✅ Output packaged: %s (%.1f MB)", out_zip.name, size_mb)

    # Colab: trigger download
    try:
        from google.colab import files as colab_files
        colab_files.download(str(out_zip))
    except ImportError:
        LOG.info("ℹ️ Not on Colab — download manually from: %s", out_zip)

    return out_zip


# Uncomment to auto-package after pipeline run:
# package_output()
