"""Independent local integrity and metric audit for Crossformer baseline tuning."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np


def sha_bytes(value): return hashlib.sha256(value).hexdigest()


def audit(run):
    ledger=json.loads((run/"ledger.json").read_text());assert ledger["status"]=="complete"
    assert len(ledger["attempts"])==51 and all(a["status"]=="complete" for a in ledger["attempts"])
    actual=None;persistence=None;rows=[];errors=[]
    for attempt in ledger["attempts"]:
        try:
            index=attempt["index"];archive_path=run/f"job-{index:03d}.tar.gz";blob=archive_path.read_bytes()
            assert sha_bytes(blob)==attempt["archive_sha256"]
            with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
                status=json.load(archive.extractfile("status.json"));result=json.load(archive.extractfile("trial/result.json"))
                prediction_blob=archive.extractfile("trial/development_predictions.npz").read()
                checkpoint_blob=archive.extractfile("trial/best.pt").read()
            assert status["ok"] and result["checkpoint_replay_passed"]
            assert result["checkpoint_sha256"]==sha_bytes(checkpoint_blob)
            with np.load(io.BytesIO(prediction_blob),allow_pickle=False) as p:
                assert set(p.files)=={"prediction","actual","persistence","seasonal_24","target_start"}
                assert np.isfinite(p["prediction"]).all() and np.isfinite(p["actual"]).all()
                assert p["target_start"][0]==11520 and p["target_start"][-1]+4==14399
                if actual is None:actual=p["actual"].copy();persistence=p["persistence"].copy()
                else:
                    np.testing.assert_array_equal(p["actual"],actual);np.testing.assert_array_equal(p["persistence"],persistence)
                error=p["prediction"]-p["actual"]
                np.testing.assert_allclose(np.abs(error).mean(),result["development_mae"],rtol=1e-12)
                np.testing.assert_allclose(np.sqrt(np.square(error).mean()),result["development_rmse"],rtol=1e-12)
                np.testing.assert_allclose(error.mean(),result["development_bias"],rtol=1e-12)
                np.testing.assert_allclose(np.abs(p["persistence"]-p["actual"]).mean(),result["persistence_mae"],rtol=1e-12)
            rows.append(dict(index=index,stage=attempt["stage"],passed=True))
        except Exception as exc:errors.append(dict(index=attempt.get("index"),error=repr(exc)))
    report=dict(passed=not errors and len(rows)==51,audited=len(rows),errors=errors,
                tail_rows_loaded=False,development_last_target_row=14399)
    (run/"artifact-audit.json").write_text(json.dumps(report,indent=2));return report


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path)
    print(json.dumps(audit(parser.parse_args().run),indent=2))
