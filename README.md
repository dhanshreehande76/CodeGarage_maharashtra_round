# CodeGarage_maharashtra_round
TrustLayer — AI-Powered Digital Authenticity and Trust

CodeGarage Maharashtra Round | Multimodal Digital Content Authenticity & Trust

TrustLayer is an AI-powered digital authenticity investigation platform designed to analyze multiple forms of digital content together and determine whether evidence is Authentic, Manipulated, or Coordinated Synthetic.

Unlike systems that inspect images, videos, audio, or text independently, TrustLayer focuses on cross-modal reasoning. It compares relationships between different evidence sources, detects inconsistencies, combines multiple signals, and provides an explainable trust assessment with confidence and supporting evidence.

Problem Statement

Generative AI has made digital content increasingly difficult to authenticate. Images, videos, audio, messages, and documents can individually appear convincing while inconsistencies between them may reveal manipulation or a coordinated synthetic narrative.

TrustLayer aims to:

Analyze multiple forms of digital content within a single investigation.

Detect manipulation or synthetic content across supported modalities.

Identify relationships and inconsistencies between different inputs.

Combine individual and cross-modal evidence into a final assessment.

Provide evidence supporting its classification.

Handle uncertainty when available evidence is insufficient.

Evaluate generalization to manipulation patterns or combinations not directly represented during training.

Core Idea

Individual Modality Analysis
            ↓
Cross-Modal Reasoning
            ↓
Evidence Fusion & Trust Decision

Individual Modality Analysis

Image → visual features and manipulation signals

Video → frame-level and temporal analysis

Audio → spoof/synthetic speech detection and transcription

Text → semantic analysis and entity extraction

Cross-Modal Reasoning

TrustLayer compares:

Image ↔ Text
Image ↔ Video
Video ↔ Audio
Video ↔ Text
Audio ↔ Transcript
Metadata ↔ Claims

The system looks for semantic, temporal, entity, metadata, and authenticity inconsistencies.

Evidence Fusion

Individual and cross-modal signals are combined into an overall assessment.

Target Classifications

Authentic

Evidence is generally consistent and no strong manipulation indicators are detected.

Uncertain / Requires Verification

Evidence is incomplete, conflicting, or insufficient for a confident classification.

Manipulated

One or more artifacts show significant evidence of alteration or synthetic generation.

Coordinated Synthetic

Multiple artifacts show coordinated manipulation, synthetic generation, or strong cross-modal conflicts suggesting a fabricated or manipulated narrative.

System Architecture

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

Technology Stack

Frontend

React

Vite

Tailwind CSS

Recharts / visualization components

Backend

Python

FastAPI

Uvicorn

SQLite for the initial MVP

AI / Machine Learning

PyTorch

Hugging Face Transformers

CLIP

Wav2Vec2

Whisper

Sentence Embeddings

Scikit-learn

XGBoost / Random Forest

Media Processing

OpenCV

FFmpeg

Librosa

EXIF / metadata extraction

AI Model Strategy

TrustLayer will not train every model from scratch.

The project follows:

Pretrained Models
       +
Selective Fine-Tuning
       +
Custom Cross-Modal Reasoning
       +
Evidence Fusion

Component

Approach

Image

CLIP features + lightweight classifier

Video

OpenCV frame sampling + CLIP + temporal classifier

Audio

Wav2Vec2-based spoof classifier

Speech

Whisper transcription

Text

Sentence embeddings + NLP

Cross-modal reasoning

Custom evidence graph and consistency engine

Final fusion

Random Forest / XGBoost or transparent weighted fusion

Dataset Strategy

TrustLayer uses modality-specific datasets plus a custom case-level multimodal dataset.

FaceForensics++

Used for image/video manipulation detection and evaluation.

Focus:

Real videos

Deepfakes

Face2Face

FaceSwap

NeuralTextures

ASVspoof 2021

Used for audio spoof/deepfake detection.

Focus:

Bonafide speech

Spoofed/synthetic speech

FakeAVCeleb

Used for multimodal audio-video deepfake research.

Focus:

Real video + real audio

Fake video + real audio

Real video + fake audio

Fake video + synthesized audio

Custom TrustLayer Dataset

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

Each case can contain:

case_001/
├── image.jpg
├── video.mp4
├── audio.wav
├── claim.txt
└── labels.json

Large datasets and model weights will not be committed to this GitHub repository.

Project Structure

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
│
├── requirements.txt
├── .gitignore
└── README.md

API Overview

Method

Endpoint

Purpose

POST

/api/investigations

Create investigation

POST

/api/investigations/{id}/evidence

Upload evidence

POST

/api/investigations/{id}/analyze

Start analysis

GET

/api/investigations/{id}/status

Check analysis status

GET

/api/investigations/{id}/result

Retrieve result

GET

/api/investigations/{id}/evidence

Retrieve evidence analysis

GET

/api/investigations/{id}/report

Generate report

MVP Scope

The first working version will focus on:

Image upload and analysis

Video upload and frame analysis

Audio upload and spoof analysis

Speech transcription

Text input and semantic analysis

Cross-modal consistency checks

Evidence graph

Evidence fusion

Trust score

Confidence estimation

Explainable result dashboard

Investigation history

The MVP prioritizes a working end-to-end demonstration over training large models from scratch.

Development Roadmap

Phase 1 — Working MVP

React
  ↓
FastAPI
  ↓
File Upload
  ↓
Pretrained AI Models
  ↓
Cross-Modal Reasoning
  ↓
Evidence Fusion
  ↓
Dashboard

Phase 2 — Modality-Specific Models

Image manipulation classifier

Video temporal classifier

Audio spoof classifier

Improved text consistency analysis

Phase 3 — Multimodal Fusion

Build case-level feature vectors

Create custom multimodal dataset

Train fusion classifier

Evaluate Authentic / Manipulated / Coordinated classes

Phase 4 — Generalization

Test on manipulation techniques and combinations not directly represented during training.

Phase 5 — Explainability & Evaluation

Evidence visualization

Confidence calibration

Confusion matrix

Precision / Recall / F1

Macro F1 for three-class classification

Cross-modal ablation experiments

Phase 6 — Deployment

Dockerization

Production API

Frontend deployment

Model optimization

Final documentation

Evaluation

Individual Models

Accuracy

Precision

Recall

F1-score

ROC-AUC where applicable

Multimodal System

Accuracy

Macro Precision

Macro Recall

Macro F1

Confusion Matrix

Key Experiment

Compare:

Individual Modality Detection
             VS
Individual Detection
+
Cross-Modal Reasoning
+
Evidence Fusion

This evaluates whether cross-modal reasoning improves detection of coordinated manipulation.

Responsible AI

TrustLayer is an evidence-based assessment system, not an absolute truth oracle.

Predictions should not be treated as definitive proof of manipulation. The system communicates:

Strong evidence

Weak evidence

Conflicting evidence

Insufficient evidence

The final interface therefore provides evidence and confidence alongside classifications.

Team Development Principles

Do not commit large datasets to Git.

Do not commit model weights unless explicitly approved.

Use .gitignore for datasets, uploads, caches, and virtual environments.

Use separate branches for major features.

Use meaningful commit messages.

Keep AI analysis modules independent from the frontend.

Document experiments and model versions.

Current Status

Project Stage: Initial Architecture / Development

Target: Working multimodal TrustLayer MVP

Primary Classification:

AUTHENTIC
MANIPULATED
COORDINATED SYNTHETIC

Core Innovation:

Cross-modal evidence reasoning for digital authenticity assessment.

License

This project is developed for the CodeGarage Maharashtra Round and educational/research purposes.

Dataset licenses and usage terms will be respected individually for each external dataset used by the project.
