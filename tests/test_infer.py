import pytest
from pathlib import Path
from ultralytics import YOLO


def test_infer_handles_missing_weights():
	with pytest.raises(Exception):
		YOLO("nonexistent.pt")


def test_infer_script_arguments_present():
	# sanity: script file exists
	assert Path("src/infer.py").exists()


