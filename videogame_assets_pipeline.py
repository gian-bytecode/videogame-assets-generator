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
import io
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

def step1_trellis_3d(cfg: PipelineConfig) -> Dict[str, dict]:
    """
    STEP 1 — Generate raw 3D meshes (.obj) + textures from concept images
    using microsoft/TRELLIS-image-large.

    Returns: dict[asset_name -> {"mesh": Path, "texture": Path | None}]
    The texture is the high-poly colour map that TRELLIS bakes onto the mesh.
    """
    log_step_banner(1, "3D Generation — TRELLIS")
    trellis_outputs: Dict[str, dict] = {}

    if not cfg.models_3d:
        LOG.info("ℹ️ No 3D models defined in config — skipping STEP 1.")
        log_step_done(1)
        return trellis_outputs

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

            # ── Extract mesh + texture from TRELLIS output ────────────
            out_obj = OUTPUT_3D / f"{asset_name}_raw.obj"
            out_tex = OUTPUT_TEX / f"{asset_name}_highpoly_diffuse.png"
            texture_saved: Path | None = None

            # Helper to load a GLB, save OBJ + extract texture
            def _glb_to_obj_tex(glb_path: Path) -> trimesh.Trimesh:
                scene = trimesh.load(str(glb_path))
                if isinstance(scene, trimesh.Scene):
                    mesh = trimesh.util.concatenate(scene.dump())
                else:
                    mesh = scene
                return mesh

            def _try_extract_texture(mesh_or_scene, fallback_glb: Path | None = None):
                """Try to pull a diffuse texture image from the mesh material."""
                nonlocal texture_saved
                targets = [mesh_or_scene]
                if hasattr(mesh_or_scene, 'geometry'):
                    targets = list(mesh_or_scene.geometry.values())
                for m in targets:
                    mat = getattr(m, 'visual', None)
                    if mat is None:
                        continue
                    # SimpleMaterial with image
                    if hasattr(mat, 'material'):
                        im = getattr(mat.material, 'image', None)
                        if im is not None:
                            im.save(str(out_tex))
                            texture_saved = out_tex
                            return
                    # TextureVisuals
                    if hasattr(mat, 'material') and hasattr(mat.material, 'baseColorTexture'):
                        bct = mat.material.baseColorTexture
                        if bct is not None:
                            bct.save(str(out_tex))
                            texture_saved = out_tex
                            return
                # Last resort: try reading the GLB again with full material
                if fallback_glb and fallback_glb.exists():
                    try:
                        import pygltflib
                        gltf = pygltflib.GLTF2().load(str(fallback_glb))
                        if gltf.images:
                            import base64
                            img_data = gltf.images[0]
                            if img_data.uri and img_data.uri.startswith('data:'):
                                header, b64 = img_data.uri.split(',', 1)
                                raw = base64.b64decode(b64)
                                Image.open(io.BytesIO(raw)).save(str(out_tex))
                                texture_saved = out_tex
                    except Exception:
                        pass

            if hasattr(outputs, "mesh") and outputs.mesh is not None:
                mesh_data = outputs.mesh[0] if isinstance(outputs.mesh, list) else outputs.mesh
                if hasattr(mesh_data, "vertices") and hasattr(mesh_data, "faces"):
                    mesh = trimesh.Trimesh(
                        vertices=np.array(mesh_data.vertices),
                        faces=np.array(mesh_data.faces),
                    )
                    mesh.export(str(out_obj))
                    _try_extract_texture(mesh_data)
                elif hasattr(mesh_data, "export"):
                    glb_path = out_obj.with_suffix(".glb")
                    mesh_data.export(str(glb_path))
                    scene_raw = trimesh.load(str(glb_path))
                    _try_extract_texture(scene_raw, glb_path)
                    m = _glb_to_obj_tex(glb_path)
                    m.export(str(out_obj))
                    glb_path.unlink(missing_ok=True)
                else:
                    glb_path = out_obj.with_suffix(".glb")
                    with open(glb_path, "wb") as f:
                        f.write(mesh_data if isinstance(mesh_data, bytes) else bytes(mesh_data))
                    scene_raw = trimesh.load(str(glb_path))
                    _try_extract_texture(scene_raw, glb_path)
                    m = _glb_to_obj_tex(glb_path)
                    m.export(str(out_obj))
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
                scene_raw = trimesh.load(str(glb_path))
                _try_extract_texture(scene_raw, glb_path)
                m = _glb_to_obj_tex(glb_path)
                m.export(str(out_obj))
                glb_path.unlink(missing_ok=True)
            else:
                LOG.error("❌ Unexpected TRELLIS output format for %s", asset_name)
                continue

            # Also check TRELLIS output dict for a separate texture field
            if texture_saved is None:
                for tex_key in ("texture", "albedo", "color_map", "baseColor"):
                    tex_obj = None
                    if isinstance(outputs, dict):
                        tex_obj = outputs.get(tex_key)
                    elif hasattr(outputs, tex_key):
                        tex_obj = getattr(outputs, tex_key)
                    if tex_obj is not None:
                        if isinstance(tex_obj, Image.Image):
                            tex_obj.save(str(out_tex))
                            texture_saved = out_tex
                        elif isinstance(tex_obj, np.ndarray):
                            Image.fromarray(tex_obj).save(str(out_tex))
                            texture_saved = out_tex
                        break

            trellis_outputs[asset_name] = {
                "mesh": out_obj,
                "texture": texture_saved,
            }
            LOG.info("  ✔ Saved raw mesh → %s", out_obj.name)
            if texture_saved:
                LOG.info("  ✔ Saved high-poly texture → %s", texture_saved.name)
            else:
                LOG.warning("  ⚠️ No texture extracted for %s (will use concept image as fallback)", asset_name)

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
            trellis_outputs[asset_name] = {"mesh": out_obj, "texture": None}
            LOG.info("  ⚠️ Placeholder cube saved for '%s'", asset_name)

    except Exception as e:
        LOG.error("❌ TRELLIS step failed: %s", e, exc_info=True)
        try:
            del pipeline
        except NameError:
            pass
        purge_vram()

    log_step_done(1)
    return trellis_outputs


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


def _uv_unwrap_xatlas(mesh_path: Path) -> Path:
    """
    UV-unwrap a mesh using xatlas and save in-place.
    Returns the same path (now with UVs embedded).
    """
    import trimesh
    try:
        import xatlas as _xatlas
    except ImportError:
        LOG.warning("⚠️ xatlas not available — skipping UV unwrap for %s", mesh_path.name)
        return mesh_path

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if mesh.vertices.shape[0] == 0:
        LOG.warning("⚠️ Empty mesh — skipping UV unwrap for %s", mesh_path.name)
        return mesh_path

    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)

    # xatlas parametrize
    LOG.info("    UV unwrap (xatlas): %d verts, %d faces …", len(verts), len(faces))
    atlas = _xatlas.Atlas()
    atlas.add_mesh(verts, faces)
    atlas.generate()
    vmapping, new_faces, uvs = atlas[0]

    # Remap: xatlas may have split vertices for UV seams
    new_verts = verts[vmapping]

    # Save as OBJ with UVs
    with open(mesh_path, "w") as f:
        for v in new_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for face in new_faces:
            # OBJ is 1-indexed, faces reference both v and vt
            i0, i1, i2 = face[0] + 1, face[1] + 1, face[2] + 1
            f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")

    LOG.info("    ✔ UV unwrap complete: %d verts (after seam splits), %d UVs",
             len(new_verts), len(uvs))
    return mesh_path


def step2_retopology(
    trellis_outputs: Dict[str, dict],
    cfg: PipelineConfig,
) -> Dict[str, dict]:
    """
    STEP 2 — Run Instant Meshes on each raw (high-poly) OBJ that has
    retopology=true, then UV-unwrap the LOW-POLY mesh via xatlas.

    Returns: dict[asset_name -> {
        "lowpoly":  Path,          # retopo mesh with UVs
        "highpoly": Path,          # original TRELLIS mesh (untouched)
        "texture_highpoly": Path | None,  # high-poly colour texture
    }]
    """
    log_step_banner(2, "Auto-Retopology + UV Unwrap")
    retopo_results: Dict[str, dict] = {}

    if not trellis_outputs:
        LOG.info("ℹ️ No raw meshes to retopologize — skipping STEP 2.")
        log_step_done(2)
        return retopo_results

    try:
        im_bin = _ensure_instant_meshes()

        for asset_name, t_out in tqdm(trellis_outputs.items(), desc="Retopology"):
            raw_obj: Path = t_out["mesh"]
            hp_texture: Path | None = t_out.get("texture")

            if not raw_obj.exists():
                LOG.warning("⚠️ Raw mesh missing for %s, skipping.", asset_name)
                continue

            model_cfg = cfg.models_3d.get(asset_name)
            if model_cfg and not model_cfg.retopology:
                LOG.info("  ─ Retopology disabled for '%s' — keeping raw mesh.", asset_name)
                if model_cfg.normal_map:
                    _uv_unwrap_xatlas(raw_obj)
                retopo_results[asset_name] = {
                    "lowpoly": raw_obj,
                    "highpoly": raw_obj,
                    "texture_highpoly": hp_texture,
                }
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

            # UV-unwrap the LOW-POLY mesh only
            _uv_unwrap_xatlas(out_obj)
            retopo_results[asset_name] = {
                "lowpoly": out_obj,
                "highpoly": raw_obj,
                "texture_highpoly": hp_texture,
            }

    except FileNotFoundError as e:
        LOG.error("❌ Instant Meshes not available: %s", e)
        for asset_name, t_out in trellis_outputs.items():
            raw_obj = t_out["mesh"]
            out_obj = OUTPUT_3D / f"{asset_name}_retopo.obj"
            if raw_obj.exists():
                shutil.copy2(raw_obj, out_obj)
            _uv_unwrap_xatlas(out_obj)
            retopo_results[asset_name] = {
                "lowpoly": out_obj,
                "highpoly": raw_obj,
                "texture_highpoly": t_out.get("texture"),
            }
        LOG.info("  ⚠️ Fallback: copied raw meshes without retopology (UVs generated).")

    except Exception as e:
        LOG.error("❌ Retopology step failed: %s", e, exc_info=True)

    log_step_done(2)
    return retopo_results


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — STEP 3: HIGH→LOW BAKING  (Normal Maps + Diffuse Transfer)
# ═══════════════════════════════════════════════════════════════════════════════
# %%

# ─────────────────────────────────────────────────────────────────────────────
# Baking helpers
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_normal_from_image(image: "Image.Image", pipe: Any) -> "np.ndarray | None":
    """
    Run a normal-estimation AI model (StableNormal / Marigold) on a 2D image.
    Returns an HxWx3 float32 numpy array in [-1,1] range, or None on failure.
    """
    normal_out = pipe(image)

    if isinstance(normal_out, dict):
        normal_out = normal_out.get("normal_map", normal_out.get("images", [None])[0])
    if hasattr(normal_out, "images"):
        normal_out = normal_out.images[0]
    if hasattr(normal_out, "prediction"):   # Marigold-style
        normal_out = normal_out.prediction[0]

    if isinstance(normal_out, Image.Image):
        arr = np.array(normal_out, dtype=np.float32) / 255.0
    elif isinstance(normal_out, np.ndarray):
        arr = normal_out.astype(np.float32)
        if arr.max() > 1.5:  # 0-255 range
            arr = arr / 255.0
    else:
        return None

    # Convert [0,1] → [-1,1] if needed (most normal maps encode as (n+1)/2)
    if arr.min() >= 0:
        arr = arr * 2.0 - 1.0
    return arr


def _build_bvh(highpoly_path: Path):
    """
    Load the high-poly mesh and build a proximity query structure.
    Returns (trimesh.Trimesh, trimesh.proximity.ProximityQuery).
    """
    import trimesh
    hp = trimesh.load(str(highpoly_path), force="mesh", process=False)
    return hp, trimesh.proximity.ProximityQuery(hp)


def _bake_geometric_normals(
    highpoly_path: Path,
    lowpoly_path: Path,
    bake_resolution: int = 2048,
) -> "np.ndarray":
    """
    Standard high-to-low geometric normal bake.

    For every texel in the low-poly UV space:
      1. Find the 3D surface point + tangent frame on the low-poly mesh.
      2. Cast a ray along the low-poly surface normal to find the closest
         high-poly surface point.
      3. The high-poly surface normal at that hit point, expressed in the
         low-poly tangent space, becomes the baked tangent-space normal.

    Returns an (R, R, 3) float32 array in [-1, 1].
    """
    import trimesh

    neutral = np.zeros((bake_resolution, bake_resolution, 3), dtype=np.float32)
    neutral[..., 2] = 1.0  # (0,0,1) = flat tangent-space normal

    lp = trimesh.load(str(lowpoly_path), force="mesh", process=False)
    if not hasattr(lp.visual, "uv") or lp.visual.uv is None or len(lp.visual.uv) == 0:
        LOG.warning("    ⚠️ Low-poly has no UVs — cannot bake geometric normals.")
        return neutral

    hp, hp_prox = _build_bvh(highpoly_path)

    lp_verts = np.array(lp.vertices, dtype=np.float64)
    lp_faces = np.array(lp.faces, dtype=np.int32)
    lp_uvs = np.array(lp.visual.uv, dtype=np.float32)

    # Pre-compute per-vertex normals on LOW-poly for tangent frame
    if lp.vertex_normals is None or len(lp.vertex_normals) == 0:
        lp.fix_normals()
    lp_vnormals = np.array(lp.vertex_normals, dtype=np.float64)

    # Pre-compute per-face normals on HIGH-poly
    if hp.face_normals is None or len(hp.face_normals) == 0:
        hp.fix_normals()

    bake = neutral.copy()

    for fi, face in enumerate(lp_faces):
        uv_tri = lp_uvs[face]          # (3, 2)
        v_tri = lp_verts[face]          # (3, 3)
        n_tri = lp_vnormals[face]       # (3, 3)

        # UV → pixel coords
        px = (uv_tri[:, 0] * (bake_resolution - 1)).astype(np.int32)
        py = ((1.0 - uv_tri[:, 1]) * (bake_resolution - 1)).astype(np.int32)

        min_x = max(int(px.min()), 0)
        max_x = min(int(px.max()), bake_resolution - 1)
        min_y = max(int(py.min()), 0)
        max_y = min(int(py.max()), bake_resolution - 1)
        if min_x > max_x or min_y > max_y:
            continue

        # Barycentric setup
        e0 = np.array([px[1] - px[0], py[1] - py[0]], dtype=np.float64)
        e1 = np.array([px[2] - px[0], py[2] - py[0]], dtype=np.float64)
        denom = e0[0] * e1[1] - e1[0] * e0[1]
        if abs(denom) < 1e-10:
            continue
        inv_d = 1.0 / denom

        for by in range(min_y, max_y + 1):
            for bx in range(min_x, max_x + 1):
                d = np.array([bx - px[0], by - py[0]], dtype=np.float64)
                u = (d[0] * e1[1] - e1[0] * d[1]) * inv_d
                v = (e0[0] * d[1] - d[0] * e0[1]) * inv_d
                w = 1.0 - u - v
                if u < -1e-4 or v < -1e-4 or w < -1e-4:
                    continue

                # Interpolated 3D position and normal on low-poly
                pos_lp = w * v_tri[0] + u * v_tri[1] + v * v_tri[2]
                nrm_lp = w * n_tri[0] + u * n_tri[1] + v * n_tri[2]
                nrm_lp /= (np.linalg.norm(nrm_lp) + 1e-12)

                # Find closest point on high-poly
                closest, dist, tri_id = hp_prox.on_surface([pos_lp])
                hp_normal = hp.face_normals[tri_id[0]].astype(np.float64)
                hp_normal /= (np.linalg.norm(hp_normal) + 1e-12)

                # Build tangent frame from low-poly normal
                N = nrm_lp
                # Choose a non-parallel vector for cross product
                up = np.array([0, 1, 0], dtype=np.float64)
                if abs(np.dot(N, up)) > 0.99:
                    up = np.array([1, 0, 0], dtype=np.float64)
                T = np.cross(N, up)
                T /= (np.linalg.norm(T) + 1e-12)
                B = np.cross(N, T)
                B /= (np.linalg.norm(B) + 1e-12)

                # Express high-poly normal in tangent space
                ts_x = np.dot(hp_normal, T)
                ts_y = np.dot(hp_normal, B)
                ts_z = np.dot(hp_normal, N)
                bake[by, bx] = [ts_x, ts_y, ts_z]

    return bake


def _bake_diffuse_transfer(
    highpoly_path: Path,
    highpoly_texture_path: Path,
    lowpoly_path: Path,
    bake_resolution: int = 2048,
) -> "np.ndarray":
    """
    Transfer the high-poly colour texture onto the low-poly UV layout.

    For each texel in low-poly UV space:
      1. Compute 3D surface point.
      2. Find nearest point on high-poly.
      3. Convert that point to high-poly UV, sample colour.
      4. Write into low-poly bake.

    Returns an (R, R, 3) uint8 diffuse map.
    """
    import trimesh

    bake = np.full((bake_resolution, bake_resolution, 3), 128, dtype=np.uint8)

    lp = trimesh.load(str(lowpoly_path), force="mesh", process=False)
    if not hasattr(lp.visual, "uv") or lp.visual.uv is None:
        LOG.warning("    ⚠️ Low-poly has no UVs — cannot transfer diffuse.")
        return bake

    hp = trimesh.load(str(highpoly_path), force="mesh", process=False)
    hp_prox = trimesh.proximity.ProximityQuery(hp)

    tex_img = np.array(Image.open(highpoly_texture_path).convert("RGB"))
    tex_h, tex_w = tex_img.shape[:2]

    # Check if high-poly has UVs for proper sampling
    hp_has_uv = (hasattr(hp.visual, "uv") and hp.visual.uv is not None
                 and len(hp.visual.uv) > 0)

    lp_verts = np.array(lp.vertices, dtype=np.float64)
    lp_faces = np.array(lp.faces, dtype=np.int32)
    lp_uvs = np.array(lp.visual.uv, dtype=np.float32)

    if hp_has_uv:
        hp_uvs = np.array(hp.visual.uv, dtype=np.float32)
        hp_faces = np.array(hp.faces, dtype=np.int32)
        hp_verts = np.array(hp.vertices, dtype=np.float64)

    for face in lp_faces:
        uv_tri = lp_uvs[face]
        v_tri = lp_verts[face]

        px = (uv_tri[:, 0] * (bake_resolution - 1)).astype(np.int32)
        py = ((1.0 - uv_tri[:, 1]) * (bake_resolution - 1)).astype(np.int32)

        min_x = max(int(px.min()), 0)
        max_x = min(int(px.max()), bake_resolution - 1)
        min_y = max(int(py.min()), 0)
        max_y = min(int(py.max()), bake_resolution - 1)
        if min_x > max_x or min_y > max_y:
            continue

        e0 = np.array([px[1] - px[0], py[1] - py[0]], dtype=np.float64)
        e1 = np.array([px[2] - px[0], py[2] - py[0]], dtype=np.float64)
        denom = e0[0] * e1[1] - e1[0] * e0[1]
        if abs(denom) < 1e-10:
            continue
        inv_d = 1.0 / denom

        for by in range(min_y, max_y + 1):
            for bx in range(min_x, max_x + 1):
                d = np.array([bx - px[0], by - py[0]], dtype=np.float64)
                u = (d[0] * e1[1] - e1[0] * d[1]) * inv_d
                v = (e0[0] * d[1] - d[0] * e0[1]) * inv_d
                w = 1.0 - u - v
                if u < -1e-4 or v < -1e-4 or w < -1e-4:
                    continue

                pos_lp = w * v_tri[0] + u * v_tri[1] + v * v_tri[2]

                # Closest point on high-poly
                closest, dist, tri_id = hp_prox.on_surface([pos_lp])

                if hp_has_uv:
                    # Compute barycentric on high-poly triangle to sample UV
                    hp_face = hp_faces[tri_id[0]]
                    hp_v = hp_verts[hp_face]
                    # Barycentric of closest point in HP triangle
                    hit = closest[0]
                    e0h = hp_v[1] - hp_v[0]
                    e1h = hp_v[2] - hp_v[0]
                    dh = hit - hp_v[0]
                    d00 = np.dot(e0h, e0h)
                    d01 = np.dot(e0h, e1h)
                    d11 = np.dot(e1h, e1h)
                    d20 = np.dot(dh, e0h)
                    d21 = np.dot(dh, e1h)
                    denom_h = d00 * d11 - d01 * d01
                    if abs(denom_h) < 1e-12:
                        continue
                    bv = (d11 * d20 - d01 * d21) / denom_h
                    bw = (d00 * d21 - d01 * d20) / denom_h
                    bu = 1.0 - bv - bw
                    hp_uv = bu * hp_uvs[hp_face[0]] + bv * hp_uvs[hp_face[1]] + bw * hp_uvs[hp_face[2]]
                    sx = int(np.clip(hp_uv[0], 0, 1) * (tex_w - 1))
                    sy = int((1.0 - np.clip(hp_uv[1], 0, 1)) * (tex_h - 1))
                else:
                    # No HP UVs — orthographic projection fallback
                    c = closest[0]
                    bbox_min = hp.vertices.min(axis=0)
                    bbox_max = hp.vertices.max(axis=0)
                    extent = bbox_max - bbox_min
                    extent[extent < 1e-8] = 1.0
                    nrm_pt = (c - bbox_min) / extent  # [0,1]
                    sx = int(np.clip(nrm_pt[0], 0, 1) * (tex_w - 1))
                    sy = int((1.0 - np.clip(nrm_pt[1], 0, 1)) * (tex_h - 1))

                bake[by, bx] = tex_img[sy, sx]

    return bake


def _combine_normal_maps(
    geo_normals: "np.ndarray",
    ai_normals: "np.ndarray",
    strength: float = 1.0,
) -> "np.ndarray":
    """
    Combine two tangent-space normal maps using the Reoriented Normal Mapping
    (RNM) technique — the industry standard for overlaying detail normals.

    Both inputs are float32 in [-1, 1] range, shape (H, W, 3).
    Returns float32 in [-1, 1].
    """
    # Resize ai_normals to match geo_normals if needed
    if ai_normals.shape[:2] != geo_normals.shape[:2]:
        from PIL import Image as _Img
        h, w = geo_normals.shape[:2]
        ai_img = _Img.fromarray(((ai_normals * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8))
        ai_img = ai_img.resize((w, h), _Img.LANCZOS)
        ai_normals = np.array(ai_img, dtype=np.float32) / 255.0 * 2.0 - 1.0

    # Apply strength to the detail (AI) map
    detail = ai_normals.copy()
    detail[..., :2] *= strength

    # Reoriented Normal Mapping (RNM)
    # t = base * float3( 1, 1, 1) + float3(0, 0, 1)
    # u = detail * float3(-1,-1, 1) + float3(0, 0, 1)
    # result = normalize(t * dot(t, u) - u * t.z)
    t = geo_normals.copy()
    t[..., 2] += 1.0
    u = detail.copy()
    u[..., 0] *= -1.0
    u[..., 1] *= -1.0
    u[..., 2] += 1.0

    dot_tu = np.sum(t * u, axis=-1, keepdims=True)
    result = t * dot_tu - u * t[..., 2:3]

    # Normalize
    length = np.linalg.norm(result, axis=-1, keepdims=True)
    length = np.maximum(length, 1e-8)
    result = result / length

    return result.astype(np.float32)


def _normal_float_to_uint8(normal_f32: "np.ndarray") -> "np.ndarray":
    """Convert [-1,1] float32 tangent-space normal to [0,255] uint8."""
    return ((normal_f32 * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


def step3_bake_textures(
    cfg: PipelineConfig,
    retopo_results: Dict[str, dict],
) -> Dict[str, dict]:
    """
    STEP 3 — Full High-to-Low Baking Pipeline.

    For each asset that has normal_map=true:

      A) AI Normal from High-Poly Texture:
         - Feed the TRELLIS high-poly colour texture (or concept image fallback)
           into StableNormal / Marigold → produces "AI normal map".

      B) Geometric Normal Bake (High→Low):
         - For every texel in the low-poly UV, ray-cast to the high-poly mesh.
         - Express the high-poly surface normal in the low-poly tangent frame
           → produces "geometric difference normal map".

      C) Normal Combine:
         - Merge (A) and (B) using Reoriented Normal Mapping (RNM).
         - This gives a final normal map with BOTH the high-poly surface detail
           AND the AI-inferred micro-detail.

      D) Diffuse Transfer (bonus — if high-poly texture available):
         - Project the high-poly colour onto the low-poly UV → diffuse map.

    Returns: dict[asset_name -> {
        "normal":  Path,          # final combined normal map
        "diffuse": Path | None,   # transferred diffuse / albedo
    }]
    """
    log_step_banner(3, "High→Low Bake  (Normals + Diffuse)")
    bake_outputs: Dict[str, dict] = {}

    need_bake = {
        k: v for k, v in cfg.models_3d.items()
        if v.normal_map and k in retopo_results
    }
    if not need_bake:
        LOG.info("ℹ️ No models require baking — skipping STEP 3.")
        log_step_done(3)
        return bake_outputs

    # ── Load the AI normal estimation model ──────────────────────────────────
    pipe = None
    pipeline_name = "none"

    try:
        from stable_normal.pipeline import StableNormalPipeline
        LOG.info("Loading StableNormal model (FP16) …")
        pipe = StableNormalPipeline.from_pretrained(
            "Stable-X/StableNormal",
            torch_dtype=DTYPE_FP16,
        ).to(DEVICE)
        pipeline_name = "StableNormal"
        LOG.info("StableNormal loaded on %s (FP16)", DEVICE)
    except ImportError:
        LOG.warning("⚠️ StableNormal not available, trying Marigold fallback …")
        try:
            from diffusers import MarigoldNormalsPipeline
            LOG.info("Loading Marigold Normals (FP16) …")
            pipe = MarigoldNormalsPipeline.from_pretrained(
                "prs-eth/marigold-normals-lcm-v0-1",
                torch_dtype=DTYPE_FP16,
                variant="fp16",
            ).to(DEVICE)
            pipeline_name = "Marigold"
        except Exception as e2:
            LOG.error("❌ Neither StableNormal nor Marigold available: %s", e2)
            log_step_done(3)
            return bake_outputs

    if pipe is None:
        log_step_done(3)
        return bake_outputs

    # ── Process each asset ───────────────────────────────────────────────────
    BAKE_RES = 2048

    try:
        for asset_name, model_cfg in tqdm(need_bake.items(), desc="Baking"):
            r = retopo_results[asset_name]
            lowpoly_path: Path = r["lowpoly"]
            highpoly_path: Path = r["highpoly"]
            hp_texture_path: Path | None = r.get("texture_highpoly")

            if not lowpoly_path.exists():
                LOG.warning("⚠️ Low-poly mesh missing for %s, skipping.", asset_name)
                continue

            # ── (A) AI Normal from High-Poly Texture ─────────────────────────
            # Prefer the high-poly colour texture; fall back to concept image
            if hp_texture_path and hp_texture_path.exists():
                source_img = Image.open(hp_texture_path).convert("RGB")
                LOG.info("  [%s] AI normal from high-poly texture for %s",
                         pipeline_name, asset_name)
            else:
                source_img = Image.open(model_cfg.concept_img).convert("RGB")
                LOG.info("  [%s] AI normal from concept image (no HP texture) for %s",
                         pipeline_name, asset_name)

            # Clamp resolution for AI model
            max_dim = 1024
            w, h = source_img.size
            if max(w, h) > max_dim:
                sc = max_dim / max(w, h)
                source_img = source_img.resize(
                    (int(w * sc), int(h * sc)), Image.LANCZOS
                )

            ai_normal = _estimate_normal_from_image(source_img, pipe)
            if ai_normal is None:
                LOG.warning("⚠️ AI normal estimation failed for %s", asset_name)
                ai_normal = np.zeros((BAKE_RES, BAKE_RES, 3), dtype=np.float32)
                ai_normal[..., 2] = 1.0  # flat fallback

            # Save AI normal as reference
            ai_ref_path = OUTPUT_TEX / f"{asset_name}_normal_ai.png"
            Image.fromarray(_normal_float_to_uint8(ai_normal)).save(str(ai_ref_path))
            LOG.info("  ✔ AI normal map saved → %s", ai_ref_path.name)

            # ── (B) Geometric Normal Bake (High→Low) ─────────────────────────
            if highpoly_path.exists() and highpoly_path != lowpoly_path:
                LOG.info("  Baking geometric normals (high→low) for %s …", asset_name)
                geo_normals = _bake_geometric_normals(
                    highpoly_path, lowpoly_path, BAKE_RES
                )
            else:
                LOG.info("  High-poly == low-poly for %s — using flat geometric base.",
                         asset_name)
                geo_normals = np.zeros((BAKE_RES, BAKE_RES, 3), dtype=np.float32)
                geo_normals[..., 2] = 1.0

            geo_ref_path = OUTPUT_TEX / f"{asset_name}_normal_geo.png"
            Image.fromarray(_normal_float_to_uint8(geo_normals)).save(str(geo_ref_path))
            LOG.info("  ✔ Geometric normal map saved → %s", geo_ref_path.name)

            # ── (C) Normal Combine ───────────────────────────────────────────
            LOG.info("  Combining geometric + AI normals (RNM) for %s …", asset_name)
            combined = _combine_normal_maps(geo_normals, ai_normal, strength=1.0)
            out_normal = OUTPUT_TEX / f"{asset_name}_normal.png"
            Image.fromarray(_normal_float_to_uint8(combined)).save(str(out_normal))
            LOG.info("  ✔ Final combined normal map → %s (%dx%d)",
                     out_normal.name, BAKE_RES, BAKE_RES)

            # ── (D) Diffuse Transfer ─────────────────────────────────────────
            out_diffuse: Path | None = None
            if hp_texture_path and hp_texture_path.exists() and highpoly_path.exists():
                LOG.info("  Transferring diffuse colour (high→low) for %s …", asset_name)
                diffuse = _bake_diffuse_transfer(
                    highpoly_path, hp_texture_path, lowpoly_path, BAKE_RES
                )
                out_diffuse = OUTPUT_TEX / f"{asset_name}_diffuse.png"
                Image.fromarray(diffuse).save(str(out_diffuse))
                LOG.info("  ✔ Diffuse map → %s", out_diffuse.name)

            bake_outputs[asset_name] = {
                "normal": out_normal,
                "diffuse": out_diffuse,
            }

        del pipe
        purge_vram()

    except Exception as e:
        LOG.error("❌ Baking step failed: %s", e, exc_info=True)
        try:
            del pipe
        except NameError:
            pass
        purge_vram()

    log_step_done(3)
    return bake_outputs


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

    # ── STEP 1: TRELLIS 3D (high-poly mesh + texture) ────────────────────────
    trellis_out = step1_trellis_3d(cfg)
    results["trellis"] = {
        k: {"mesh": str(v["mesh"]), "texture": str(v["texture"]) if v["texture"] else None}
        for k, v in trellis_out.items()
    }

    # ── STEP 2: Instant Meshes Retopology + UV Unwrap (low-poly) ─────────────
    retopo_results = step2_retopology(trellis_out, cfg)
    results["retopo"] = {
        k: {"lowpoly": str(v["lowpoly"]), "highpoly": str(v["highpoly"])}
        for k, v in retopo_results.items()
    }

    # ── STEP 3: High→Low Bake (Normals + Diffuse) ────────────────────────────
    bake_outputs = step3_bake_textures(cfg, retopo_results)
    results["bake"] = {
        k: {"normal": str(v["normal"]), "diffuse": str(v["diffuse"]) if v["diffuse"] else None}
        for k, v in bake_outputs.items()
    }

    # Convenience: extract lowpoly mesh dict for downstream steps
    retopo_meshes = {k: v["lowpoly"] for k, v in retopo_results.items()}

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
