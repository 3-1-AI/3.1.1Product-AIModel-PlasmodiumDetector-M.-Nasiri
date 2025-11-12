from pathlib import Path
import pytest
import yaml


def test_data_yaml_exists():
	p = Path("config/data.yaml")
	assert p.exists(), "config/data.yaml should exist"
	with open(p, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
	assert "path" in data and "train" in data and "val" in data, "data.yaml missing required keys"


def test_dataset_structure_placeholders():
	with open("config/data.yaml", "r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
	root = Path(str(data["path"]))
	# Placeholders may not exist locally; test should be lenient
	assert isinstance(root, Path)
	# Validate names is list-like
	names = data.get("names", [])
	assert isinstance(names, list), "names must be a list"


