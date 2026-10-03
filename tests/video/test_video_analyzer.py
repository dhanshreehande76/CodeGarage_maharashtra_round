from pathlib import Path

import pytest

from backend.analyzers.video.analyzer import (
    VideoAnalysisError,
    analyze_video,
)


def test_missing_video_raises_error():
    with pytest.raises(VideoAnalysisError):
        analyze_video("does-not-exist.mp4")


def test_video_directory_raises_error(tmp_path: Path):
    with pytest.raises(VideoAnalysisError):
        analyze_video(tmp_path)


def test_video_analyzer_exports():
    from backend.analyzers.video.analyzer import analyze_video

    assert callable(analyze_video)