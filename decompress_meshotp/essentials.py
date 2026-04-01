from pathlib import Path
from typing import Any
import json
import shutil
import struct

import bpy

from . import standalone_gltf
from .scripts.decode_meshopt import decode_file as decode_meshopt_file


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


def run_meshopt_decoder(input_path: Path, output_path: Path) -> None:
    decode_meshopt_file(input_path, output_path)


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


def can_normalize_with_python(input_path: Path) -> bool:
    extensions = detect_compression_extensions(input_path)
    return DRACO_EXTENSION not in extensions


DRACO_EXTENSION = "KHR_draco_mesh_compression"


def describe_import_backend(input_path: Path) -> str:
    extensions = detect_compression_extensions(input_path)
    if "EXT_meshopt_compression" in extensions:
        return "standalone Python"
    if DRACO_EXTENSION in extensions:
        return "unsupported Draco"
    return "standalone Python"


def normalize_model_for_import(input_path: Path, output_path: Path) -> Path:
    """Normalize Meshopt and quantized geometry into a Blender-readable glTF asset."""
    extensions = detect_compression_extensions(input_path)
    needs_geometry_decode = bool({DRACO_EXTENSION, "EXT_meshopt_compression", "KHR_mesh_quantization"} & extensions)

    if not needs_geometry_decode:
        if input_path.suffix.lower() == output_path.suffix.lower() == ".glb":
            shutil.copy2(input_path, output_path)
            return output_path
        else:
            return standalone_gltf.normalize_model_for_import(input_path, output_path)

    if "EXT_meshopt_compression" not in extensions and DRACO_EXTENSION not in extensions:
        return standalone_gltf.normalize_model_for_import(input_path, output_path)

    if "EXT_meshopt_compression" in extensions:
        decoded_path = output_path.with_name(f"{output_path.stem}_meshopt_decoded.gltf")
        run_meshopt_decoder(input_path, decoded_path)
        return standalone_gltf.normalize_model_for_import(decoded_path, output_path)

    raise RuntimeError("Draco-compressed files are not supported by this addon.")
