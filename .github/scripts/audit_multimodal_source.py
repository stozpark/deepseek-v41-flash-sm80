#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

import vllm

root = Path(vllm.__file__).resolve().parent
files = {
    "v41_mm": root / "models/deepseek_v4_1/common/mm_preprocess.py",
    "v41_vl": root / "models/deepseek_v4_1/nvidia/vl_model.py",
    "shared_vision": root / "models/deepseek_v4/common/vision.py",
}
for label, path in files.items():
    if not path.is_file():
        raise SystemExit(f"ERROR: DeepSeek-V4.1 multimodal source missing ({label}): {path}")
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

mm_text = files["v41_mm"].read_text(encoding="utf-8")
vl_text = files["v41_vl"].read_text(encoding="utf-8")
for needle in (
    'IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"',
    "DeepseekV4VLMultiModalProcessor",
    "DeepseekV4VLProcessingInfo",
):
    if needle not in mm_text:
        raise SystemExit(f"ERROR: multimodal preprocessing invariant missing: {needle}")
for needle in (
    "SupportsMultiModal",
    "supports_encoder_tp_data = True",
    "requires_raw_input_tokens = True",
    "MULTIMODAL_REGISTRY.register_processor",
    "DeepseekV4ViT",
    "run_dp_sharded_vision_tower",
):
    if needle not in vl_text:
        raise SystemExit(f"ERROR: V4.1 vision-model invariant missing: {needle}")

vision_paths = {
    "models/deepseek_v4_1/common/mm_preprocess.py",
    "models/deepseek_v4_1/nvidia/vl_model.py",
    "models/deepseek_v4/common/vision.py",
}
manifest_candidates = [
    Path("/work/patches/generated/sm80-cu129-minimal.manifest.txt"),
    Path("/work/patches/generated/sm80-cu130-minimal.manifest.txt"),
]
found_manifest = False
for manifest in manifest_candidates:
    if not manifest.is_file():
        continue
    found_manifest = True
    patched = set(manifest.read_text(encoding="utf-8").splitlines())
    leaked = sorted(vision_paths & patched)
    if leaked:
        raise SystemExit(
            "ERROR: production SM80 patch should preserve official vision source unchanged: "
            + ", ".join(leaked)
        )
    print(f"vision_source_preservation_manifest={manifest.name}: OK")

if Path("/work/patches/generated").is_dir() and not found_manifest:
    raise SystemExit("ERROR: no recognized production minimal-patch manifest found")

print("DEEPSEEK-V4.1 MULTIMODAL SOURCE AUDIT OK")
print("official vision preprocessing/model wrapper preserved; encoder TP-data support present")
