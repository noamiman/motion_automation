# Smart Home Motion Automation Project

The next smart home won’t be a pile of brittle “if-this-then-that” rules—it will understand routines, explain its choices, and keep you in control. I’m building this project as a step toward that future: an end-to-end engine that watches a camera/video feed, summarizes activity, infers user intentions, and proposes auditable, Home-Assistant–ready YAML automations.

I’m enthusiastic about the space and improving fast. This repo is designed to be practical, private (local by default), and modular, so it can evolve with real households and real feedback—turning raw motion into helpful, transparent automations you actually trust.
> End-to-end: **YOLOv8-Pose** (OpenCV) → event compression → **LLM intent prediction** (few-shot & zero-shot) → **cosine-similarity** filtering → **YAML** generation.

---

## Table of Contents

- [Highlights](#highlights)
- [Pipeline](#pipeline)
- [Repository Structure](#repository-structure)
- [Tech & Techniques](#tech--techniques)
- [Pipeline Diagram](#pipeline-diagram)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Quick Start](#quick-start)
  - [Artifacts](#artifacts)
- [Future Work](#future-work)
- [License](#license)
- [Conclusion](#conclusion)
---

## Highlights

- **Computer Vision:** YOLOv8-Pose model (`yolov8n-pose.pt`) with **OpenCV** (`cv2`) for real-time skeletal/motion analysis.
- **LLMs (Local):** **Ollama** runtime (default `llama3.2:3b`) for **few-shot** intent prediction and **zero-shot** YAML synthesis.
- **Representation Learning:** **Cosine similarity** (configurable `SIM_THRESH`) to select representative events and reduce noise.
- **Controllability:** Probability cutoff (`ACTIONS_MIN_PROB`) and detection limits (`LIMIT_DETECTIONS`) to tune precision/recall.
- **Traceable Artifacts:** Intermediate **JSON/TXT** outputs at each stage plus final **YAML** automations.

---

## Pipeline

All five stages are orchestrated by [`main.py`](./main.py):

1. **Capture & Detect**  
   `MotionAnalyzer` runs YOLOv8-Pose on a camera or video source and writes **motion events JSON** (e.g., `detection/data/motion_analysis_room1.json`).

2. **Compress**  
   Convert verbose events JSON into compact **event lines** (`detection/data/events.txt`) for LLM consumption.

3. **Intent Prediction (Few-shot)**  
   An LLM infers **user intentions** from event lines, producing `output/outputs/representatives_result.json`.  
   Representative events are chosen using **cosine similarity** (`SIM_THRESH`) and an optional **limit**.

4. **Action Recommendations**  
   Generate **automation ideas** with probabilities → `output/outputs/automation_rec.json`  
   Filtered by `ACTIONS_MIN_PROB`.

5. **Zero-shot YAML Generation**  
   Normalize and emit **automation YAML** → `output/outputs/yaml_file_result.yaml`.

---
# Repository Structure

```text
Motion_Analysis/
├─ detection/
│  ├─ data/
│  ├─ video_output.py
│  └─ yolov8n-pose.pt
├─ output/
│  ├─ outputs/
│  │  ├─ automation_rec.json
│  │  ├─ representatives_result.json
│  │  └─ yaml_file_result.yaml
│  ├─ automation_prompt_generator.py
│  ├─ generate_yaml.py
│  └─ intension_predictior.py
├─ prompts/
│  ├─ intension_prompt.txt
│  └─ prompt_for_auto.txt
└─ main.py
```
---

## Tech & Techniques

**Computer Vision**
- **YOLOv8-Pose** (`yolov8n-pose.pt`) for human keypoints & activity cues.
- **OpenCV (cv2)** for frame capture, preprocessing, and I/O.

**NLP / LLM**
- **Ollama** local model: `llama3.2:3b` (configurable).
- **Few-shot prompting** (`prompts/intension_prompt.txt`) to steer intent extraction.
- **Zero-shot prompting** (`prompts/prompt_for_zeroShot.txt`) to synthesize standardized **YAML**.

**Scoring & Filtering**
- **Cosine similarity** (`SIM_THRESH`) for representative event selection.
- **Probability thresholds** (`ACTIONS_MIN_PROB`) to keep strong automation candidates.

**Outputs**
- **JSON/TXT** artifacts for observability.
- **Home-Assistant-style YAML** automations.

---

## Pipeline Diagram

<img width="821" height="786" alt="pipeline-diagram" src="https://github.com/user-attachments/assets/53ee6477-094a-42ea-93a7-9ab355715bec" />

---

## Getting Started

### Prerequisites

- Python **3.10+**
- [Ollama](https://ollama.com) with the model you want (default: `llama3.2:3b`)
    ```bash
    ollama pull llama3.2:3b
- (Optional) GPU for faster YOLOv8-Pose inference

    ```bash
    git clone https://github.com/noamiman/motion_automation.git
    cd motion_automation
    
    python -m venv .venv
    # macOS/Linux
    source .venv/bin/activate
    # Windows
    # .venv\Scripts\activate
    
    pip install -r requirements.txt

### Quick Start

Edit the top of main.py
 if needed:

    MODEL_PATH = "yolov8n-pose.pt"
    VIDEO_SOURCE = 0  # 0 = default camera, or path to a video file
    
    MODEL_NAME = "llama3.2:3b"
    LIMIT_DETECTIONS = 100
    SIM_THRESH = 0.86
    ACTIONS_MIN_PROB = 0.10

Run:

    python main.py

### Artifacts:

- Motion JSON: detection/data/motion_analysis_room1.json

- Events TXT: detection/data/events.txt

- Intent JSON: output/outputs/representatives_result.json

- Actions JSON: output/outputs/automation_rec.json

- Final YAML: output/outputs/yaml_file_result.yaml

### Future Work

- VLM-powered perception: Replace/augment YOLOv8-Pose with vision-language models (VLMs) for richer scene understanding, human-object interaction (HOI), and temporal reasoning.

- Temporal action recognition: Integrate transformers (e.g., TimeSformer/VideoMAE-style) to better capture multi-step activities.

- Graph reasoning: Use scene graphs or GNNs over keypoints/objects to infer relationships (who, what, where).

- Active learning loop: Allow users to accept/reject automations and fine-tune prompts or weights automatically.

- RAG for devices/entities: Ground YAML generation with a retrieval layer over local device/entity catalogs and Home Assistant docs.

- Privacy-first on-device: Quantized models and accelerated backends for fully local inference.

- Evaluation harness: Precision/recall metrics for intent detection; YAML validation and dry-run executor.

- Multi-camera fusion: Room zoning and cross-camera identity tracking to reduce false positives.

- Deployment: Docker compose with GPU support; optional web UI for reviewing events and proposed automations.

# License
## llama3.2:3b
https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/LICENSE.txt


# Conclusion

This project was an end-to-end exploration across CV (YOLOv8-Pose), representation learning (cosine similarity), and local LLMs (Ollama) to turn raw motion into actionable YAML automations. I intentionally tried many technologies; while some choices aren’t the most efficient yet, the exercise delivered a working pipeline and clear insight into what to improve. I’m enthusiastic about the problem space and committed to iterating quickly, measuring results, and raising engineering rigor with each revision.


### 🤝 Let's Connect!
Thanks for checking out my project!
I'm always happy to connect with other data & travel tech enthusiasts.
Find me on [LinkedIn](https://www.linkedin.com/in/noamiman/)
