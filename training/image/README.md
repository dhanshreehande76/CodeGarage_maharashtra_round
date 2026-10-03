# Image detector evaluation

Evaluates the pretrained detector (`buildborderless/CommunityForensics-DeepfakeDet-ViT`)
on a labelled dataset. **No results are shipped in the repository**; metrics are
only produced by running this script on data you provide.

## Dataset format

Put images in two folders (sub-folders are searched recursively):

```text
datasets/raw/image_eval/
├── real/    authentic images (label 0)
└── fake/    AI-generated or manipulated images (label 1, the positive class)
```

`datasets/raw/` is git-ignored, so datasets are never committed. Supported
formats: JPEG, PNG, WebP, BMP, TIFF. Unreadable or unsupported files are skipped
and listed in `metrics.json`.

For a meaningful evaluation use images the model was **not** trained on, and keep
real and fake images comparable in format and resolution. Otherwise the model
can score well by picking up on format differences (for example real = JPEG,
fake = PNG) instead of authenticity.

## Run

```bash
pip install -r requirements.txt
python training/image/evaluate_detector.py --data-dir datasets/raw/image_eval
```

Useful options: `--limit 200` (quick run), `--batch-size 32`, `--threshold 0.5`,
`--device cpu`, `--local-files-only` (never contact the Hub),
`--model <hf-id-or-local-path>`, `--output-dir <dir>`.

The model is downloaded once into the local Hugging Face cache
(`~/.cache/huggingface`) and reused afterwards.

## Output

Written to `experiments/image_detector_eval/` (override with `--output-dir`):

- `metrics.json`: accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
  threshold, class counts, skipped files and model/library versions.
- `predictions.csv`: per-image label, `ai_generated_probability` and `raw_logit`.

Exit codes: `0` success, `2` dataset missing or a class is empty, `3` model load
or inference failure.

## Notes

- Metrics use the positive class `fake`. ROC-AUC uses the raw
  `ai_generated_probability`; the other metrics use `--threshold` (default 0.5,
  the model card's value; it is not tuned on your data).
- The probabilities are uncalibrated model outputs; do not read 0.9 as "90%
  likely to be fake".
