import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / source.name
    target.write_bytes(source.read_bytes())
    manifest = {"file": target.name, "sha256": sha256(target)}
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    sys.stdout.write(json.dumps(manifest, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
