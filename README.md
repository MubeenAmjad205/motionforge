# MotionForge

A modular, Colab-ready image-to-motion video pipeline.

Converts structured JSON scene definitions into individual MP4 clips, packages everything into a downloadable ZIP.

---

## Project Structure

```
motionforge/
├── configs/
│   ├── models.json           # Model capabilities registry
│   └── default_settings.json # Global pipeline defaults
├── data/
│   └── scenes.json           # Your scene definitions
├── inputs/
│   └── images/               # Input scene images (PNG/JPG)
├── outputs/                  # Generated clips and logs (auto-created)
├── logs/                     # Global pipeline logs (auto-created)
├── zip/                      # Output ZIP files (auto-created)
├── src/
│   ├── __init__.py
│   ├── main.py               # CLI entry point
│   ├── scene_loader.py       # JSON loader + defaults merger
│   ├── validators.py         # Scene schema + video output validation
│   ├── model_registry.py     # Model capabilities + adapter registry
│   ├── fallback_manager.py   # Primary → retry → fallback chain
│   ├── queue_manager.py      # Scene queue + checkpoint persistence
│   ├── video_generator.py    # Single-scene generation wrapper
│   ├── postprocess.py        # MoviePy utilities
│   ├── report_generator.py   # JSON manifest + Markdown report
│   ├── zip_exporter.py       # ZIP packaging
│   ├── logger.py             # Structured logging
│   └── model_adapters/
│       ├── __init__.py
│       ├── base_adapter.py   # Abstract base class
│       ├── mock_adapter.py   # ✅ Working — MoviePy image-to-MP4
│       ├── svd_adapter.py    # 🔧 Stub — Stable Video Diffusion XT
│       ├── wan_adapter.py    # 🔧 Stub — Wan2.1 I2V GGUF Q4
│       └── framepack_adapter.py # 🔧 Stub — FramePack low-VRAM
├── notebook.ipynb            # Colab notebook (9 cells)
├── requirements.txt
└── README.md
```

---

## Quick Start (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your scene images

Place images at the paths referenced in `data/scenes.json`:

```
inputs/images/scene_001.png
inputs/images/scene_002.png
inputs/images/scene_003.png
```

### 3. Run the pipeline

```bash
python src/main.py
```

With a custom scenes file:

```bash
python src/main.py --scenes data/scenes.json
```

Resume an interrupted run:

```bash
python src/main.py --resume
```

### 4. Find outputs

```
outputs/
  scene_001_awake_at_2am/
    scene_001.mp4
    scene_001.png          (input image copy)
    scene_001_config.json  (scene config snapshot)
    scene_001_log.txt      (per-scene log)
  project_manifest.json
  generation_report.md
zip/
  stickman_motivation_part_1_output.zip
```

---

## Running in Google Colab

1. Open `notebook.ipynb` in Google Colab.
2. Run **Cell 1** to install dependencies.
3. Run **Cell 3** to set up project paths.
4. Upload your images in **Cell 3b** (or use generated placeholders for testing).
5. Run **Cells 4–9** sequentially.
6. The ZIP is automatically downloaded at the end.

---

## Scene Configuration

Edit `data/scenes.json` to define your scenes:

```json
{
  "project": { "name": "my_project" },
  "global_settings": {
    "default_model": "mock_adapter",
    "fallback_models": ["stable_video_diffusion_xt", "framepack_low_vram"],
    "resolution": { "width": 854, "height": 480 },
    "fps": 16,
    "duration_seconds": 4
  },
  "queue_settings": {
    "max_retries_per_scene": 2,
    "continue_on_error": true
  },
  "scenes": [
    {
      "id": 1,
      "name": "my_scene",
      "input_image": "inputs/images/scene_001.png",
      "motion_prompt": "Character walking forward, smooth animation.",
      "duration_seconds": 5,
      "fps": 16,
      "seed": 42,
      "enabled": true
    }
  ]
}
```

### Scene Fields

| Field | Required | Description |
|---|---|---|
| `id` | ✅ | Unique scene number |
| `name` | ✅ | Human-readable scene name |
| `input_image` | ✅ | Relative path to source image |
| `motion_prompt` | ✅ | Animation instruction for the model |
| `enabled` | ✅ | Set to `false` to skip a scene |
| `duration_seconds` | Optional | Clip length (overrides global) |
| `fps` | Optional | Frame rate (overrides global) |
| `resolution` | Optional | `{"width": N, "height": N}` |
| `seed` | Optional | For reproducibility |
| `negative_prompt` | Optional | What to avoid in generation |
| `model_override` | Optional | Use a specific model for this scene |
| `motion_strength` | Optional | Animation intensity (0.0–1.0) |

---

## Model Backends

| Model Key | Status | VRAM | Notes |
|---|---|---|---|
| `mock_adapter` | ✅ Ready | 0 | MoviePy — no GPU needed |
| `wan2_1_i2v_gguf_q4` | 🔧 Stub | ~12 GB | Best quality — Phase 2 |
| `stable_video_diffusion_xt` | 🔧 Stub | ~8 GB | Fallback 1 — Phase 2 |
| `framepack_low_vram` | 🔧 Stub | ~6 GB | Fallback 2 — Phase 2 |

To add a new model:
1. Create `src/model_adapters/my_model_adapter.py` extending `BaseAdapter`
2. Add its capabilities to `configs/models.json`
3. Register it in `src/main.py` (and the notebook Cell 5)

---

## Fallback Strategy

```
Primary model attempt
  ├── Success → save output ✅
  └── Failure → retry (up to max_retries)
       ├── Success → save output ✅
       └── All retries exhausted → try fallback model 1
            ├── Success → save output ✅
            └── Failure → try fallback model 2
                 ├── Success → save output ✅
                 └── All failed → mark scene failed, continue pipeline
```

Failed scenes are saved with error reports. The pipeline continues processing remaining scenes unless `continue_on_error: false`.

---

## Output Files

### Per-scene folder
- `scene_NNN.mp4` — generated video clip
- `scene_NNN.png` — copy of input image
- `scene_NNN_config.json` — full resolved scene config
- `scene_NNN_log.txt` — detailed per-scene log

### Project level
- `project_manifest.json` — full project summary with video metadata
- `generation_report.md` — human-readable Markdown report
- `failed_scenes.json` — details of any failures
- `project_queue_state.json` — checkpoint for resume

---

## Development Phases

### Phase 1 (Current — MVP) ✅
- MockAdapter producing real MP4s
- Full pipeline: load → validate → queue → generate → report → ZIP
- Resume from checkpoint

### Phase 2 (Next)
- Real Wan2.1 I2V adapter
- Real SVD adapter
- Adaptive resolution downgrade on retry

### Phase 3 (Future)
- Gradio web UI
- Automatic stitching module
- Cloud storage integration
