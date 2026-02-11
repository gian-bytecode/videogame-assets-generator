# 🎮 Videogame Assets Generator — Pipeline Automatica

Script monolitico Python per **Google Colab / Kaggle** che automatizza una pipeline completa di generazione asset per videogiochi.

## Pipeline (7 Step)

| Step | Modello | Input → Output |
|------|---------|----------------|
| 1 | **TRELLIS** (`microsoft/TRELLIS-image-large`) | Immagine concept → Mesh 3D grezza (.obj) |
| 2 | **Instant Meshes** (binary Linux) | Mesh grezza → Mesh pulita a quadrilateri |
| 3 | **StableNormal** (`Stable-X/StableNormal`) | Immagine concept → Normal Map (.png) |
| 4 | **RigNet** (PyTorch Geometric) | Mesh pulita → Skeleton + Skinning (.json) |
| 5 | **Fish Speech 1.4** (`fishaudio/fish-speech-1.4`) | Testo + Voice ref → Dialoghi (.wav) |
| 6 | **AudioCraft** (`facebook/audiogen-medium`) | Prompt testuale → SFX ambientali (.wav) |
| 7 | **SDXL** (`stabilityai/stable-diffusion-xl-base-1.0`) | Prompt → Skybox 360° + Texture seamless |

## Struttura ZIP di Input

```
my_game_assets.zip
├── config.json
├── TRELLIS/
│   ├── knight_concept.png
│   └── dragon_concept.png
├── audio_refs/
│   ├── knight_voice.wav
│   └── dragon_voice.wav
└── audio_texts/
    └── scene_01_confrontation.txt
```

## Formato config.json

```json
{
    "characters": {
        "Knight": {
            "concept_img": "TRELLIS/knight_concept.png",
            "voice_ref": "audio_refs/knight_voice.wav",
            "rig_type": "biped"
        }
    },
    "environmental_sfx": [
        { "trigger_word": "campfire", "prompt": "crackling fire...", "duration": 5.0 }
    ],
    "world_assets": {
        "skybox_theme": "fantasy sunset sky with dramatic clouds",
        "floor_texture": "cobblestone road, PBR material"
    }
}
```

## Formato Dialoghi (.txt)

```
Knight: Halt! Who dares enter the forbidden keep?
Dragon: You are brave, little human.
```

Ogni riga segue il formato `NOME_PERSONAGGIO: Testo`. Il nome viene mappato alla chiave corrispondente in `config.json`.

## Struttura Output

```
output_assets/
├── 3D_Models_Rigged/
│   ├── Knight_raw.obj
│   ├── Knight_retopo.obj
│   ├── Knight_rig.json
│   └── ...
├── Textures/
│   ├── Knight_normal.png
│   └── floor_seamless.png
├── Audio_Dialogues/
│   ├── scene_01_000_Knight.wav
│   └── scene_01_001_Dragon.wav
├── SFX/
│   ├── sfx_campfire.wav
│   └── sfx_sword_clash.wav
├── Environment/
│   └── skybox_360.png
└── manifest.json
```

## Come Usare

### Google Colab
1. Apri un nuovo notebook Colab con GPU (T4 / A100)
2. Copia il contenuto di `videogame_assets_pipeline.py` nelle celle
3. Esegui la cella 0 (installazione dipendenze) — decommenta le righe `!pip install`
4. Carica il tuo ZIP su Colab
5. Imposta `ZIP_PATH` nella cella 11
6. Esegui tutte le celle

### Kaggle
1. Crea un nuovo notebook con GPU P100/T4
2. Carica lo ZIP come dataset
3. Segui gli stessi passi di Colab

## Gestione Memoria

Ogni modello viene:
- Caricato → usato → **cancellato dalla VRAM**
- `del model` + `torch.cuda.empty_cache()` + `gc.collect()`

Questo permette di eseguire tutti e 7 i modelli su una singola GPU (anche T4 16GB).

## Fallback

Lo script include fallback automatici:
- **StableNormal** → Marigold Normals se StableNormal non è installato
- **Fish Speech** → SpeechT5 se Fish Speech non è disponibile
- **Instant Meshes** → copia della mesh originale se il binary non funziona
- **RigNet** → skeleton basico calcolato dal bounding box
