import argparse
import json

from backend.pipeline import run_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--image")
parser.add_argument("--video")
parser.add_argument("--audio")
parser.add_argument("--claim")
args = parser.parse_args()

result = run_pipeline({
    "image": args.image,
    "video": args.video,
    "audio": args.audio,
    "claim": args.claim,
})
print(json.dumps(result, indent=2, default=str))