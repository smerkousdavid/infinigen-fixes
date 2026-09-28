"""Stamp a clean source checkout before copying it to a host without .git."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
from infinigen.p4d.runtime import source_identity

destination = root / "src/infinigen/p4d/source_manifest.json"
destination.unlink(missing_ok=True)
identity = source_identity()
if identity.get("dirty") or not identity.get("fork_commit"):
    raise SystemExit("commit the generator changes before stamping a source snapshot")
destination.write_text(json.dumps(identity, indent=2) + "\n")
print(json.dumps(identity))
