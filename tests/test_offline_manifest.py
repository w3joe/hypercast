import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "web" / "public" / "offline-manifest.json"


def _generator_module():
    path = ROOT / "scripts" / "generate_offline_manifest.py"
    spec = importlib.util.spec_from_file_location("generate_offline_manifest", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_manifest_is_complete_and_current():
    checked_in = json.loads(MANIFEST.read_text(encoding="utf-8"))
    generated = _generator_module().build_manifest()
    assert checked_in == generated
    assert checked_in["reference_cell"] == {"window": 10, "horizon": 1}
    assert len(checked_in["presets"]) == 25
    assert checked_in["catalog"]["offline"]["preset_count"] == 25
    for preset_id, entry in checked_in["presets"].items():
        assert entry["graph"]["preset_id"] == preset_id
        assert entry["graph_nodes"]
        assert entry["parameters"] > 0
        assert entry["auto_fit_connections"]
        sample = entry["auto_fit_connections"][0]
        assert sample["algebras"]["quaternion"]["axis"] == -1
