# CodeGarage_maharashtra_round
# TrustLayer — AI-Powered Digital Authenticity and Trust

**CodeGarage Maharashtra Round** | Multimodal Digital Content Authenticity & Trust

TrustLayer is an AI-powered digital authenticity investigation platform. It analyzes multiple forms of digital content **together** and determines whether the evidence is **Authentic**, **Manipulated**, or **Coordinated Synthetic**, or whether the evidence is too weak to say (**Uncertain / Requires Verification**).

Unlike systems that inspect images, videos, audio, or text independently, TrustLayer focuses on **cross-modal reasoning**. It compares relationships between different evidence sources, detects inconsistencies, combines multiple signals, and provides an explainable trust assessment with confidence and supporting evidence.

---

**Table of Contents**

1. [Problem Statement](#problem-statement)
2. [Core Idea](#core-idea)
3. [Classification Outcomes](#classification-outcomes)
4. [System Architecture](#system-architecture)
5. [Technology Stack](#technology-stack)
6. [AI Model Strategy](#ai-model-strategy)
7. [Dataset Strategy](#dataset-strategy)
8. [Project Structure](#project-structure)
9. [API Overview](#api-overview)
10. [MVP Scope](#mvp-scope)
11. [Development Roadmap](#development-roadmap)
12. [Evaluation](#evaluation)
13. [Responsible AI](#responsible-ai)
14. [Team and Development Principles](#team-and-development-principles)
15. [Current Status](#current-status)
16. [License](#license)

**Problem Statement**

Generative AI has made digital content increasingly difficult to authenticate. Images, videos, audio, messages, and documents can each appear convincing on their own, while inconsistencies **between** them may reveal manipulation or a coordinated synthetic narrative. Existing detection approaches often analyze each artifact independently and may fail when faced with manipulation techniques they have not seen before.

TrustLayer aims to:
- Analyze multiple forms of digital content within a single investigation.
- Detect manipulation or synthetic content across supported modalities.
- Identify relationships and inconsistencies between different inputs.
- Combine individual and cross-modal evidence into a final assessment.
- Provide evidence supporting its classification.
- Handle uncertainty when available evidence is insufficient.
- Evaluate generalization to manipulation patterns or combinations not directly represented during training.

**Core Idea**

```text
Individual Modality Analysis
            ↓
Cross-Modal Reasoning
            ↓
Evidence Fusion & Trust Decision
```

**Individual Modality Analysis**

| Modality | Analysis |
|---|---|
| Image | Visual features and manipulation signals |
| Video | Frame-level and temporal analysis |
| Audio | Spoof / synthetic speech detection and transcription |
| Text | Semantic analysis and entity extraction |
| Metadata | EXIF and file-level consistency |

**Cross-Modal Reasoning**

TrustLayer compares:

| Pair | What is checked |
|---|---|
| Image ↔ Text | Does the image support the claim? |
| Image ↔ Video | Are they consistent in content and origin? |
| Video ↔ Audio | Do speech and visuals match? |
| Video ↔ Text | Does the video support the claim? |
| Audio ↔ Transcript | Does the transcribed speech agree with the claim? |
| Metadata ↔ Claims | Do time, location, and device match what is claimed? |

The system looks for **semantic, temporal, entity, metadata, and authenticity** inconsistencies.

**Evidence Fusion**

Individual and cross-modal signals are combined into an overall trust score with a confidence estimate. The first version uses **transparent weighted fusion** so every decision can be explained; a trained fusion classifier (Random Forest / XGBoost) is added once the case-level dataset is ready.


## Classification Outcomes

| Outcome | Meaning |
|---|---|
| **Authentic** | Evidence is generally consistent and no strong manipulation indicators are detected. |
| **Manipulated** | One or more artifacts show significant evidence of alteration or synthetic generation. |
| **Coordinated Synthetic** | Multiple artifacts show coordinated manipulation, synthetic generation, or strong cross-modal conflicts suggesting a fabricated narrative. |
| **Uncertain / Requires Verification** | Evidence is incomplete, conflicting, or insufficient for a confident classification. |

> **Design decision:** *Uncertain* is an **abstain output** driven by confidence and evidence-quality thresholds, not a fourth training class. Classification metrics (macro F1, confusion matrix) are computed on the three real classes (Authentic / Manipulated / Coordinated). Abstention rate and accuracy-when-not-abstaining are reported separately.

---

## System Architecture

```text
                         TRUSTLAYER
                              │
                              ▼
                    ┌──────────────────┐
                    │  React Frontend  │
                    │ Investigation UI │
                    └────────┬─────────┘
                             │ REST API
                             ▼
                    ┌──────────────────┐
                    │  FastAPI Backend │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │   Image   │      │   Video   │      │   Audio   │
    │  Analyzer │      │  Analyzer │      │  Analyzer │
    └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                    ┌────────────────┐
                    │ Text Analyzer  │
                    └───────┬────────┘
                            ▼
                ┌────────────────────────┐
                │ Cross-Modal Reasoning  │
                │        Engine          │
                └────────────┬───────────┘
                             ▼
                    ┌────────────────┐
                    │ Evidence Graph │
                    └───────┬────────┘
                            ▼
                    ┌────────────────┐
                    │ Evidence Fusion│
                    └───────┬────────┘
                            ▼
                    ┌────────────────┐
                    │ Trust Score +  │
                    │  Confidence    │
                    └───────┬────────┘
                            ▼
                    ┌────────────────┐
                    │ Explainable    │
                    │ Investigation  │
                    └────────────────┘
```


**Technology Stack**

**Frontend**

- React
- Vite
- Tailwind CSS
- Recharts/visualization components

**Backend**

- Python
- FastAPI
- Uvicorn
- SQLite (initial MVP, optional)

**AI / Machine Learning**

- PyTorch
- Hugging Face Transformers
- CLIP
- Wav2Vec2
- Whisper
- Sentence embeddings
- Scikit-learn
- XGBoost / Random Forest

**Media Processing**

- OpenCV
- FFmpeg
- Librosa
- EXIF / metadata extraction


**AI Model Strategy**

TrustLayer does **not** train every model from scratch. Overnight and hackathon-scale training of full detectors would neither be feasible nor generalize well, so the project follows:

```text
Pretrained Models
       +
Selective Fine-Tuning
       +
Custom Cross-Modal Reasoning
       +
Evidence Fusion
```

| Component | Approach |
|---|---|
| Image | Pretrained AI-image detector and CLIP features + lightweight classifier; ELA and EXIF checks |
| Video | OpenCV frame sampling run through the image pipeline (temporal classifier on the roadmap) |
| Audio | Pretrained Wav2Vec2-based spoof classifier |
| Speech | Whisper transcription |
| Text | Sentence embeddings + NLP / LLM-based claim extraction |
| Cross-modal reasoning | Custom evidence graph and consistency engine (CLIP similarity, transcript ↔ claim, metadata ↔ claim) |
| Final fusion | Transparent weighted fusion first; Random Forest / XGBoost once the case dataset exists |

The main original contribution is the **cross-modal reasoning and evidence-fusion layer**, not new single-modality detectors.

---

## Dataset Strategy

TrustLayer uses modality-specific public datasets for detector fine-tuning and evaluation, plus a **custom case-level multimodal dataset** that we generate ourselves, because no public dataset provides multimodal cases with Authentic / Manipulated / Coordinated labels.

> Some of these datasets require an access request and/or are large. The MVP relies on pretrained models and small subsets; full fine-tuning is part of the roadmap.

### FaceForensics++

Image/video manipulation detection and evaluation.
Focus: real videos, Deepfakes, Face2Face, FaceSwap, NeuralTextures.

### ASVspoof 2021

Audio spoof/deepfake detection.
Focus: bona fide speech and spoofed/synthetic speech.

### FakeAVCeleb

Multimodal audio-video deepfake research.
Focus: real video + real audio, fake video + real audio, real video + fake audio, fake video + synthesized audio.

### Custom TrustLayer Dataset

Case-level examples built programmatically from real media, AI-generated media, mismatched captions, and synthetic voice (TTS).

```text
trustlayer_dataset/
├── train/
│   ├── authentic/
│   ├── manipulated/
│   └── coordinated/
├── validation/
│   ├── authentic/
│   ├── manipulated/
│   └── coordinated/
└── test/
    ├── authentic/
    ├── manipulated/
    └── coordinated/
```

Each case can contain:

```text
case_001/
├── image.jpg
├── video.mp4
├── audio.wav
├── claim.txt
└── labels.json
```

A **held-out** set of manipulation techniques and combinations is reserved and never used for training or tuning, to measure generalization.

Large datasets and model weights are **not** committed to this repository.

---

## Project Structure

```text
TrustLayer/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── App.jsx
│   └── package.json
│
├── backend/
│   ├── api/
│   ├── analyzers/
│   │   ├── image/
│   │   ├── video/
│   │   ├── audio/
│   │   └── text/
│   ├── models/
│   ├── reasoning/
│   ├── database/
│   ├── utils/
│   ├── uploads/
│   └── main.py
│
├── training/
│   ├── image/
│   ├── video/
│   ├── audio/
│   └── fusion/
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── custom_cases/
│
├── experiments/
├── tests/
├── notebooks/
├── docs/
│   └── mock_response.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## API Overview

### MVP endpoint

The MVP exposes a single analysis call so the frontend and backend can be built in parallel.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/analyze` | Upload evidence (optional `image`, `video`, `audio`, plus a `claim` text field) and receive the full investigation result |

Example response:

```json
{
  "verdict": "coordinated_synthetic",
  "trust_score": 0.18,
  "confidence": 0.82,
  "modalities": {
    "image": { "synthetic_prob": 0.91, "signals": ["AI-generator likelihood high", "no camera EXIF"] },
    "audio": { "spoof_prob": 0.77, "transcript": "..." },
    "text":  { "claims": ["..."] }
  },
  "cross_modal": [
    { "pair": "image-text", "score": 0.21, "finding": "Image content does not match the claim" },
    { "pair": "audio-text", "score": 0.34, "finding": "Transcript contradicts the claim" }
  ],
  "evidence_graph": {
    "nodes": [{ "id": "image", "risk": 0.91 }, { "id": "audio", "risk": 0.77 }, { "id": "text", "risk": 0.30 }],
    "edges": [{ "from": "image", "to": "text", "conflict": 0.79 }]
  },
  "explanation": "Multiple artifacts show synthetic indicators and conflict with the claim."
}
```

A mock of this response lives in `docs/mock_response.json` so the UI can be developed independently of the models.

### Planned investigation API (post-MVP)

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/investigations` | Create investigation |
| `POST` | `/api/investigations/{id}/evidence` | Upload evidence |
| `POST` | `/api/investigations/{id}/analyze` | Start analysis |
| `GET` | `/api/investigations/{id}/status` | Check analysis status |
| `GET` | `/api/investigations/{id}/result` | Retrieve result |
| `GET` | `/api/investigations/{id}/evidence` | Retrieve evidence analysis |
| `GET` | `/api/investigations/{id}/report` | Generate report |

---

## MVP Scope

The first working version prioritizes a **working end-to-end demonstration** over training large models.

**Must have**

- Image upload and analysis (AI-generation signals, ELA, EXIF / metadata)
- Text claim input and semantic analysis
- Speech transcription (Whisper) and transcript ↔ claim checking
- Cross-modal consistency checks
- Evidence graph
- Evidence fusion, trust score, and confidence estimation
- Uncertain / Requires Verification outcome
- Explainable result dashboard (React + FastAPI)

**Should have**

- Audio upload with pretrained spoof detection
- Video upload with frame sampling through the image pipeline
- Small custom case set with confusion matrix and cross-modal ablation

**Roadmap (not in the MVP)**

- Trained temporal video classifier
- Fine-tuned Wav2Vec2 spoof classifier
- Trained fusion classifier (XGBoost / Random Forest)
- Investigation history, Docker, and deployment

---

## Development Roadmap

### Phase 1 — Working MVP

```text
React → FastAPI → File Upload → Pretrained AI Models
      → Cross-Modal Reasoning → Evidence Fusion → Dashboard
```

### Phase 2 — Modality-Specific Models

- Image manipulation classifier
- Video temporal classifier
- Audio spoof classifier
- Improved text consistency analysis

### Phase 3 — Multimodal Fusion

- Build case-level feature vectors
- Create the custom multimodal dataset
- Train the fusion classifier
- Evaluate Authentic / Manipulated / Coordinated classes

### Phase 4 — Generalization

- Test on manipulation techniques and combinations not directly represented during training

### Phase 5 — Explainability and Evaluation

- Evidence visualization
- Confidence calibration
- Confusion matrix
- Precision / Recall / F1
- Macro F1 for three-class classification
- Cross-modal ablation experiments

### Phase 6 — Deployment

- Dockerization
- Production API
- Frontend deployment
- Model optimization
- Final documentation

---

## Evaluation

**Individual models**

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC where applicable

**Multimodal system**

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Confusion matrix
- Abstention rate and accuracy-when-not-abstaining (for the Uncertain outcome)

### Key experiment

```text
Individual Modality Detection
            VS
Individual Detection + Cross-Modal Reasoning + Evidence Fusion
```

This evaluates whether cross-modal reasoning improves detection of **coordinated** manipulation, which single-modality detectors are expected to miss.

Evaluation also includes a **held-out generalization test** on manipulation types and combinations excluded from training. Results are reported honestly, including performance drops on unseen patterns.

---

## Responsible AI

TrustLayer is an **evidence-based assessment system, not an absolute truth oracle**. Predictions should not be treated as definitive proof of manipulation.

The system communicates:

- Strong evidence
- Weak evidence
- Conflicting evidence
- Insufficient evidence

The interface therefore presents evidence and confidence alongside every classification.

---

## Team and Development Principles

**Roles**

| Area | Responsibility |
|---|---|
| AI / ML (2 members) | Modality analyzers, cross-modal reasoning, fusion, custom cases, evaluation |
| UI / UX and Web (1 member) | React investigation interface, results dashboard, evidence graph, API integration |

**Principles**

- Do not commit large datasets to Git.
- Do not commit model weights unless explicitly approved.
- Use `.gitignore` for datasets, uploads, caches, secrets (`.env`), and virtual environments.
- Use separate branches for major features.
- Use meaningful commit messages.
- Keep AI analysis modules independent from the frontend.
- Document experiments and model versions.

---

## Current Status

**Project stage:** Initial architecture and development.
**Target:** Working multimodal TrustLayer MVP.

| Area | Status |
|---|---|
| Problem analysis and architecture | Done |
| Repository structure and `.gitignore` | In progress |
| API contract and mock response | In progress |
| Image/metadata analyzers | Planned |
| Audio transcription and spoof detection | Planned |
| Cross-modal reasoning engine | Planned |
| Evidence fusion and trust score | Planned |
| Frontend dashboard | Planned |
| Custom case dataset and evaluation | Planned |
| Video temporal model, Docker, deployment | Roadmap |

**Primary classification outcomes:** `AUTHENTIC` · `MANIPULATED` · `COORDINATED SYNTHETIC` (and `UNCERTAIN` when evidence is insufficient)

**Core innovation:** Cross-modal evidence reasoning for digital authenticity assessment.

> Update this table as work progresses so it always reflects what is actually built.


## License

This project is developed for the CodeGarage Maharashtra Round and for educational / research purposes.

Dataset licenses and usage terms are respected individually for each external dataset used by the project.
