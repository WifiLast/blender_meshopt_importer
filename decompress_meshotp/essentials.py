from pathlib import Path
from typing import Any
import json
import os
import shutil
import struct
import subprocess

import bpy


def get_classes(modules: tuple[Any]) -> tuple[type]:
    bases = {
        bpy.types.AddonPreferences,
        bpy.types.Menu,
        bpy.types.Operator,
        bpy.types.Panel,
        bpy.types.PropertyGroup,
        bpy.types.UIList,
    }

    classes = []

    for module in modules:
        for cls in module.__dict__.values():
            if isinstance(cls, type):
                for base in cls.__bases__:
                    if base in bases:
                        classes.append(cls)
                        break

    return tuple(classes)


def reload_recursive(path: Path, mods: dict[str, Any]) -> None:
    import importlib

    for child in path.iterdir():

        if child.is_file() and child.suffix == ".py" and not child.name.startswith("__") and child.stem in mods:
            importlib.reload(mods[child.stem])

        elif child.is_dir() and not child.name.startswith((".", "__")):

            if child.name in mods:
                reload_recursive(child, mods[child.name].__dict__)
                importlib.reload(mods[child.name])
                continue

            reload_recursive(child, mods)


def check_integrity(path: Path) -> None:
    """:raises FileNotFoundError:"""

    if not path.exists():
        raise FileNotFoundError("Incorrect package, follow installation guide")


def get_gltf_transform_command() -> tuple[list[str] | None, str]:
    """Resolve gltf-transform the same way as the backend: binary first, npx fallback."""
    gltf_transform = shutil.which("gltf-transform")
    if gltf_transform:
        return [gltf_transform], gltf_transform

    npx = shutil.which("npx")
    if npx:
        if os.name == "nt":
            return ["cmd", "/c", "npx", "--yes", "@gltf-transform/cli"], "npx @gltf-transform/cli"
        return [npx, "--yes", "@gltf-transform/cli"], "npx @gltf-transform/cli"

    return None, "missing"


def run_gltf_transform(input_path: Path, output_path: Path, command_name: str) -> None:
    command, display_name = get_gltf_transform_command()

    if command is None:
        raise FileNotFoundError(
            "gltf-transform is not available. Install '@gltf-transform/cli' globally "
            "or make 'npx' available in Blender's environment."
        )

    cmd = [*command, command_name, str(input_path), str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "Unknown error").strip()
        raise RuntimeError(f"{display_name} {command_name} failed: {details}")


def _load_gltf_json(input_path: Path) -> dict[str, Any]:
    if input_path.suffix.lower() == ".gltf":
        return json.loads(input_path.read_text(encoding="utf-8"))

    if input_path.suffix.lower() != ".glb":
        return {}

    with input_path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise RuntimeError(f"Invalid GLB header in {input_path}")

        magic, _, _ = struct.unpack("<4sII", header)
        if magic != b"glTF":
            raise RuntimeError(f"Invalid GLB magic in {input_path}")

        while True:
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            chunk_data = handle.read(chunk_length)
            if chunk_type == 0x4E4F534A:  # JSON
                return json.loads(chunk_data.decode("utf-8"))

    return {}


def detect_compression_extensions(input_path: Path) -> set[str]:
    gltf_json = _load_gltf_json(input_path)
    extensions_used = set(gltf_json.get("extensionsUsed", []))
    extensions_required = set(gltf_json.get("extensionsRequired", []))
    return extensions_used | extensions_required


def describe_import_extensions(input_path: Path) -> str:
    extensions = detect_compression_extensions(input_path)
    labels = []
    if "KHR_draco_mesh_compression" in extensions:
        labels.append("Draco")
    if "EXT_meshopt_compression" in extensions:
        labels.append("Meshopt")
    if "KHR_mesh_quantization" in extensions:
        labels.append("quantized geometry")
    if not labels:
        return "uncompressed glTF/GLB"
    return ", ".join(labels)


def normalize_model_for_import(input_path: Path, output_path: Path) -> None:
    """Normalize Draco, Meshopt, and quantized geometry into a Blender-readable GLB."""
    extensions = detect_compression_extensions(input_path)
    needs_geometry_decode = bool(
        {"KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_mesh_quantization"} & extensions
    )

    if not needs_geometry_decode:
        if input_path.suffix.lower() == ".glb":
            shutil.copy2(input_path, output_path)
        else:
            run_gltf_transform(input_path, output_path, "copy")
        return

    current_path = input_path
    if needs_geometry_decode:
        dequantized_path = output_path.with_name(f"{output_path.stem}_dequantized.glb")
        run_gltf_transform(current_path, dequantized_path, "dequantize")
        current_path = dequantized_path

    # Rewriting the file after dequantization expands accessor/layout details that
    # Blender still trips over on some Draco and Meshopt exports.
    run_gltf_transform(current_path, output_path, "dedup")
