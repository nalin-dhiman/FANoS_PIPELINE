
from __future__ import annotations

import zipfile
import shutil
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "RELEASE_RESULTS.zip"
SNAPSHOT = ROOT / "code_snapshot"

def create_snapshot():
    if SNAPSHOT.exists():
        shutil.rmtree(SNAPSHOT)
    SNAPSHOT.mkdir()
    
    # Copy python files to snapshot
    for p in ROOT.glob("*.py"):
        shutil.copy2(p, SNAPSHOT / p.name)
        
    # Copy fanos/ dir if it exists
    if (ROOT / "fanos").exists():
        shutil.copytree(ROOT / "fanos", SNAPSHOT / "fanos")

def add_path(zf: zipfile.ZipFile, path: Path, arc_name: str=None):
    if not path.exists():
        # Only warn if it's strictly expected
        # print(f"[package] warning: {path} does not exist")
        return
        
    if path.is_file():
        # If arc_name provided, use it. Else use filename.
        name = arc_name if arc_name else path.name
        zf.write(path, arcname=name)
    elif path.is_dir():
        for p in path.rglob("*"):
            if p.is_file():
                rel = p.relative_to(path.parent)
                # If arc_name provided validation?
                # If path is 'artifacts', p is 'artifacts/raw/foo.csv'. rel is 'artifacts/raw/foo.csv'.
                # We want just that.
                zf.write(p, arcname=str(p.relative_to(ROOT)))

def main():
    create_snapshot()
    
    if OUT.exists():
        OUT.unlink()
        
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add Scripts to Root
        for p in ROOT.glob("*.py"):
            zf.write(p, arcname=p.name)
            
        # Add Snapshot
        add_path(zf, SNAPSHOT)
        
        # Add Artifacts
        add_path(zf, ROOT / "artifacts")
        
        # Add Latex Snippets
        add_path(zf, ROOT / "latex_snippets")
        
        # Add MDs
        if (ROOT / "README.md").exists():
            zf.write(ROOT / "README.md", arcname="README.md")
        if (ROOT / "METHOD_SPEC.md").exists():
            zf.write(ROOT / "METHOD_SPEC.md", arcname="METHOD_SPEC.md")
        if (ROOT / "requirements.txt").exists():
            zf.write(ROOT / "requirements.txt", arcname="requirements.txt")
            
    print(f"[package] wrote {OUT}")

if __name__ == "__main__":
    main()
