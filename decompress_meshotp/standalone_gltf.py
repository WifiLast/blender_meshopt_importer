from __future__ import annotations

import base64
import json
import math
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar


GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942

COMPONENT_TYPE_FORMAT = {
    5120: "b",
    5121: "B",
    5122: "h",
    5123: "H",
    5125: "I",
    5126: "f",
}

COMPONENT_TYPE_SIZE = {
    5120: 1,
    5121: 1,
    5122: 2,
    5123: 2,
    5125: 4,
    5126: 4,
}

TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

IMAGE_SUFFIXES = {
    "image/avif": ".avif",
    "image/jpeg": ".jpg",
    "image/ktx2": ".ktx2",
    "image/png": ".png",
    "image/webp": ".webp",
}

MESHOPT_EXTENSION = "EXT_meshopt_compression"
QUANTIZATION_EXTENSION = "KHR_mesh_quantization"
DRACO_EXTENSION = "KHR_draco_mesh_compression"


@dataclass
class GLTFDocument:
    json_data: dict[str, Any]
    buffers: list[bytes]
    base_dir: Path
    images_to_write: list[tuple[str, bytes]] = field(default_factory=list)


@dataclass
class AccessorPayload:
    accessor: dict[str, Any]
    blob: bytes


def normalize_model_for_import(input_path: Path, output_path: Path) -> Path:
    document = load_document(input_path)
    ensure_meshopt_is_decoded(document.json_data)

    strip_extension(document.json_data, MESHOPT_EXTENSION)
    strip_extension(document.json_data, QUANTIZATION_EXTENSION)
    extract_buffer_view_images(document, output_path.parent)

    accessors = document.json_data.get("accessors", [])
    dequantized_accessors = find_dequantized_accessor_indices(document.json_data)
    payloads = [materialize_accessor(document, index, index in dequantized_accessors) for index in range(len(accessors))]

    accessor_index_map, payloads = dedup_accessors(document.json_data, payloads)
    remap_accessor_references(document.json_data, accessor_index_map)

    texture_index_map = dedup_textures(document.json_data, document.base_dir, output_path.parent)
    remap_texture_references(document.json_data, texture_index_map)

    material_index_map = dedup_materials(document.json_data)
    remap_material_references(document.json_data, material_index_map)

    mesh_index_map = dedup_meshes(document.json_data)
    remap_mesh_references(document.json_data, mesh_index_map)

    skin_index_map = dedup_skins(document.json_data)
    remap_skin_references(document.json_data, skin_index_map)

    rebuild_accessor_storage(document.json_data, payloads)
    return write_gltf(document, output_path)


def ensure_meshopt_is_decoded(json_data: dict[str, Any]) -> None:
    for buffer_view in json_data.get("bufferViews", []):
        extensions = buffer_view.get("extensions", {})
        if MESHOPT_EXTENSION in extensions:
            raise RuntimeError("Standalone normalization expects decoded bufferViews. Decode Meshopt first.")


def load_document(input_path: Path) -> GLTFDocument:
    if input_path.suffix.lower() == ".gltf":
        json_data = json.loads(input_path.read_text(encoding="utf-8"))
        buffers = [read_resource(input_path.parent, buffer_def.get("uri")) for buffer_def in json_data.get("buffers", [])]
        return GLTFDocument(json_data=json_data, buffers=buffers, base_dir=input_path.parent)

    if input_path.suffix.lower() != ".glb":
        raise RuntimeError(f"Unsupported file type: {input_path.suffix}")

    with input_path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise RuntimeError(f"Invalid GLB header in {input_path}")
        magic, version, length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise RuntimeError(f"Invalid GLB header in {input_path}")

        json_chunk = None
        bin_chunk = b""
        bytes_read = 12
        while bytes_read < length:
            chunk_header = handle.read(8)
            if len(chunk_header) != 8:
                break
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            chunk_data = handle.read(chunk_length)
            bytes_read += 8 + chunk_length
            if chunk_type == GLB_JSON_CHUNK:
                json_chunk = chunk_data
            elif chunk_type == GLB_BIN_CHUNK:
                bin_chunk = chunk_data

    if json_chunk is None:
        raise RuntimeError(f"Missing JSON chunk in {input_path}")

    json_data = json.loads(json_chunk.decode("utf-8"))
    buffers = []
    bin_offset = 0
    for index, buffer_def in enumerate(json_data.get("buffers", [])):
        uri = buffer_def.get("uri")
        if uri:
            buffers.append(read_resource(input_path.parent, uri))
        else:
            byte_length = buffer_def.get("byteLength", 0)
            buffers.append(bin_chunk[bin_offset : bin_offset + byte_length])
            bin_offset += byte_length

    return GLTFDocument(json_data=json_data, buffers=buffers, base_dir=input_path.parent)


def read_resource(base_dir: Path, uri: str | None) -> bytes:
    if not uri:
        return b""
    if uri.startswith("data:"):
        _, encoded = uri.split(",", 1)
        return base64.b64decode(encoded)
    return (base_dir / uri).read_bytes()


def strip_extension(json_data: dict[str, Any], extension_name: str) -> None:
    for key in ("extensionsUsed", "extensionsRequired"):
        values = [value for value in json_data.get(key, []) if value != extension_name]
        if values:
            json_data[key] = values
        elif key in json_data:
            del json_data[key]

    for buffer_view in json_data.get("bufferViews", []):
        extensions = buffer_view.get("extensions")
        if not extensions:
            continue
        extensions.pop(extension_name, None)
        if not extensions:
            buffer_view.pop("extensions", None)

    for buffer_def in json_data.get("buffers", []):
        extensions = buffer_def.get("extensions")
        if not extensions:
            continue
        extensions.pop(extension_name, None)
        if not extensions:
            buffer_def.pop("extensions", None)


def extract_buffer_view_images(document: GLTFDocument, output_dir: Path) -> None:
    images = document.json_data.get("images", [])
    for index, image in enumerate(images):
        buffer_view_index = image.get("bufferView")
        if buffer_view_index is None:
            uri = image.get("uri")
            if not uri or uri.startswith("data:"):
                continue
            source = document.base_dir / uri
            target = output_dir / Path(uri).name
            if source != target:
                shutil.copy2(source, target)
            image["uri"] = target.name
            continue

        mime_type = image.get("mimeType")
        image_bytes = read_buffer_view(document, buffer_view_index)
        suffix = IMAGE_SUFFIXES.get(mime_type, ".bin")
        filename = f"image_{index}{suffix}"
        document.images_to_write.append((filename, image_bytes))
        image.pop("bufferView", None)
        image["uri"] = filename


def find_dequantized_accessor_indices(json_data: dict[str, Any]) -> set[int]:
    accessor_indices: set[int] = set()
    for mesh in json_data.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            for semantic, accessor_index in primitive.get("attributes", {}).items():
                if not semantic.startswith("JOINTS_"):
                    accessor_indices.add(accessor_index)
            for target in primitive.get("targets", []):
                for semantic, accessor_index in target.items():
                    if not semantic.startswith("JOINTS_"):
                        accessor_indices.add(accessor_index)
    return accessor_indices


def materialize_accessor(document: GLTFDocument, accessor_index: int, dequantize: bool) -> AccessorPayload:
    source = dict(document.json_data["accessors"][accessor_index])
    values = read_accessor_values(document, source)

    component_type = source["componentType"]
    normalized = bool(source.get("normalized"))

    if dequantize:
        flattened = [decode_normalized(value, component_type) if normalized else float(value) for value in values]
        component_type = 5126
        normalized = False
        blob = pack_values(flattened, component_type)
        values_for_bounds = flattened
    else:
        blob = pack_values(values, component_type)
        values_for_bounds = values

    source["componentType"] = component_type
    source["normalized"] = normalized
    source.pop("bufferView", None)
    source.pop("byteOffset", None)
    source.pop("sparse", None)
    update_accessor_bounds(source, values_for_bounds)
    return AccessorPayload(accessor=source, blob=blob)


def read_accessor_values(document: GLTFDocument, accessor: dict[str, Any]) -> list[float | int]:
    component_type = accessor["componentType"]
    component_count = TYPE_COMPONENTS[accessor["type"]]
    component_size = COMPONENT_TYPE_SIZE[component_type]
    element_size = component_count * component_size
    count = accessor["count"]

    if "bufferView" in accessor:
        buffer_view = document.json_data["bufferViews"][accessor["bufferView"]]
        raw = read_buffer_view(document, accessor["bufferView"])
        stride = buffer_view.get("byteStride", element_size)
        offset = accessor.get("byteOffset", 0)
        values = []
        for index in range(count):
            start = offset + index * stride
            chunk = raw[start : start + element_size]
            values.extend(unpack_values(chunk, component_type, component_count))
    else:
        values = [0] * (count * component_count)

    sparse = accessor.get("sparse")
    if sparse:
        apply_sparse_values(document, values, accessor, sparse)

    return values


def apply_sparse_values(
    document: GLTFDocument,
    values: list[float | int],
    accessor: dict[str, Any],
    sparse: dict[str, Any],
) -> None:
    component_count = TYPE_COMPONENTS[accessor["type"]]
    sparse_count = sparse["count"]

    index_component_type = sparse["indices"]["componentType"]
    index_buffer_view = sparse["indices"]["bufferView"]
    index_offset = sparse["indices"].get("byteOffset", 0)
    index_blob = read_buffer_view(document, index_buffer_view)[index_offset : index_offset + sparse_count * COMPONENT_TYPE_SIZE[index_component_type]]
    indices = unpack_values(index_blob, index_component_type, sparse_count)

    value_component_type = accessor["componentType"]
    value_buffer_view = sparse["values"]["bufferView"]
    value_offset = sparse["values"].get("byteOffset", 0)
    value_blob = read_buffer_view(document, value_buffer_view)[
        value_offset : value_offset + sparse_count * component_count * COMPONENT_TYPE_SIZE[value_component_type]
    ]
    sparse_values = unpack_values(value_blob, value_component_type, sparse_count * component_count)

    for sparse_index, accessor_index in enumerate(indices):
        accessor_index = int(accessor_index)
        start = accessor_index * component_count
        values[start : start + component_count] = sparse_values[
            sparse_index * component_count : (sparse_index + 1) * component_count
        ]


def read_buffer_view(document: GLTFDocument, buffer_view_index: int) -> bytes:
    buffer_view = document.json_data["bufferViews"][buffer_view_index]
    buffer_index = buffer_view["buffer"]
    buffer_bytes = document.buffers[buffer_index]
    start = buffer_view.get("byteOffset", 0)
    end = start + buffer_view["byteLength"]
    return buffer_bytes[start:end]


def unpack_values(blob: bytes, component_type: int, value_count: int) -> list[float | int]:
    if value_count == 0:
        return []
    fmt = "<" + COMPONENT_TYPE_FORMAT[component_type] * value_count
    return list(struct.unpack(fmt, blob[: struct.calcsize(fmt)]))


def pack_values(values: list[float | int], component_type: int) -> bytes:
    if not values:
        return b""
    fmt = "<" + COMPONENT_TYPE_FORMAT[component_type] * len(values)
    return struct.pack(fmt, *values)


def decode_normalized(value: float | int, component_type: int) -> float:
    if component_type == 5120:
        return max(float(value) / 127.0, -1.0)
    if component_type == 5121:
        return float(value) / 255.0
    if component_type == 5122:
        return max(float(value) / 32767.0, -1.0)
    if component_type == 5123:
        return float(value) / 65535.0
    if component_type == 5125:
        return float(value) / 4294967295.0
    return float(value)


def update_accessor_bounds(accessor: dict[str, Any], values: list[float | int]) -> None:
    component_count = TYPE_COMPONENTS[accessor["type"]]
    if accessor["count"] == 0 or component_count == 0:
        accessor.pop("min", None)
        accessor.pop("max", None)
        return

    mins = [math.inf] * component_count
    maxes = [-math.inf] * component_count
    for index in range(accessor["count"]):
        start = index * component_count
        for component in range(component_count):
            value = values[start + component]
            mins[component] = min(mins[component], value)
            maxes[component] = max(maxes[component], value)

    accessor["min"] = [float(value) if accessor["componentType"] == 5126 else int(value) for value in mins]
    accessor["max"] = [float(value) if accessor["componentType"] == 5126 else int(value) for value in maxes]


def dedup_accessors(
    json_data: dict[str, Any], payloads: list[AccessorPayload]
) -> tuple[dict[int, int], list[AccessorPayload]]:
    used_indices = find_accessor_references(json_data)
    duplicates: dict[int, int] = {}
    groups: dict[tuple[Any, ...], list[int]] = {}

    for accessor_index in used_indices:
        accessor = payloads[accessor_index].accessor
        key = (
            accessor["count"],
            accessor["type"],
            accessor["componentType"],
            bool(accessor.get("normalized")),
            bool(accessor.get("sparse")),
        )
        groups.setdefault(key, []).append(accessor_index)

    for group in groups.values():
        for index, accessor_index in enumerate(group):
            if accessor_index in duplicates:
                continue
            for other_index in group[index + 1 :]:
                if other_index in duplicates:
                    continue
                if payloads[accessor_index].blob == payloads[other_index].blob:
                    duplicates[other_index] = accessor_index

    return prune_items(payloads, duplicates)


def find_accessor_references(json_data: dict[str, Any]) -> set[int]:
    used: set[int] = set()
    for mesh in json_data.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            indices = primitive.get("indices")
            if indices is not None:
                used.add(indices)
            used.update(primitive.get("attributes", {}).values())
            for target in primitive.get("targets", []):
                used.update(target.values())

    for animation in json_data.get("animations", []):
        for sampler in animation.get("samplers", []):
            if "input" in sampler:
                used.add(sampler["input"])
            if "output" in sampler:
                used.add(sampler["output"])

    return used


def dedup_textures(json_data: dict[str, Any], source_dir: Path, output_dir: Path) -> dict[int, int]:
    textures = json_data.get("textures", [])
    duplicates: dict[int, int] = {}
    image_cache: dict[int, bytes] = {}

    for index, texture in enumerate(textures):
        if index in duplicates:
            continue
        for other_index in range(index + 1, len(textures)):
            if other_index in duplicates:
                continue
            other = textures[other_index]
            if texture.get("sampler") != other.get("sampler"):
                continue
            source_a = texture.get("source")
            source_b = other.get("source")
            if source_a is None or source_b is None:
                continue
            if image_mime(json_data, source_a) != image_mime(json_data, source_b):
                continue
            if load_image_bytes(json_data, source_a, source_dir, output_dir, image_cache) == load_image_bytes(
                json_data, source_b, source_dir, output_dir, image_cache
            ):
                duplicates[other_index] = index

    index_map, new_textures = prune_items(textures, duplicates)
    if new_textures or "textures" in json_data:
        json_data["textures"] = new_textures
    return index_map


def image_mime(json_data: dict[str, Any], image_index: int) -> str | None:
    images = json_data.get("images", [])
    if image_index >= len(images):
        return None
    return images[image_index].get("mimeType")


def load_image_bytes(
    json_data: dict[str, Any],
    image_index: int,
    source_dir: Path,
    output_dir: Path,
    cache: dict[int, bytes],
) -> bytes:
    if image_index in cache:
        return cache[image_index]

    image = json_data.get("images", [])[image_index]
    uri = image.get("uri")
    if not uri:
        cache[image_index] = b""
    elif uri.startswith("data:"):
        _, encoded = uri.split(",", 1)
        cache[image_index] = base64.b64decode(encoded)
    else:
        candidate = output_dir / uri
        cache[image_index] = candidate.read_bytes() if candidate.exists() else (source_dir / uri).read_bytes()
    return cache[image_index]


def dedup_materials(json_data: dict[str, Any]) -> dict[int, int]:
    materials = json_data.get("materials", [])
    duplicates: dict[int, int] = {}
    canonical_cache: dict[int, str] = {}

    for index, material in enumerate(materials):
        if index in duplicates:
            continue
        for other_index in range(index + 1, len(materials)):
            if other_index in duplicates:
                continue
            if canonical_material(material, canonical_cache, index) == canonical_material(
                materials[other_index], canonical_cache, other_index
            ):
                duplicates[other_index] = index

    index_map, new_materials = prune_items(materials, duplicates)
    if new_materials or "materials" in json_data:
        json_data["materials"] = new_materials
    return index_map


def canonical_material(material: dict[str, Any], cache: dict[int, str], index: int) -> str:
    if index not in cache:
        value = dict(material)
        value.pop("name", None)
        cache[index] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return cache[index]


def dedup_meshes(json_data: dict[str, Any]) -> dict[int, int]:
    meshes = json_data.get("meshes", [])
    duplicates: dict[int, int] = {}
    keys: dict[str, int] = {}

    for index, mesh in enumerate(meshes):
        parts = [create_primitive_key(primitive) for primitive in mesh.get("primitives", [])]
        mesh_key = ";".join(parts)
        if mesh_key in keys:
            duplicates[index] = keys[mesh_key]
        else:
            keys[mesh_key] = index

    index_map, new_meshes = prune_items(meshes, duplicates)
    if new_meshes or "meshes" in json_data:
        json_data["meshes"] = new_meshes
    return index_map


def create_primitive_key(primitive: dict[str, Any]) -> str:
    items = [f"{semantic}:{primitive['attributes'][semantic]}" for semantic in sorted(primitive.get("attributes", {}))]
    if "indices" in primitive:
        items.append(f"indices:{primitive['indices']}")
    if "material" in primitive:
        items.append(f"material:{primitive['material']}")
    items.append(f"mode:{primitive.get('mode', 4)}")
    for target in primitive.get("targets", []):
        target_items = [f"{semantic}:{target[semantic]}" for semantic in sorted(target)]
        items.append("target:" + ",".join(target_items))
    return ",".join(items)


def dedup_skins(json_data: dict[str, Any]) -> dict[int, int]:
    skins = json_data.get("skins", [])
    duplicates: dict[int, int] = {}
    canonical: dict[str, int] = {}

    for index, skin in enumerate(skins):
        value = dict(skin)
        joints = value.pop("joints", [])
        value.pop("name", None)
        key = json.dumps(value, sort_keys=True, separators=(",", ":")) + "|" + json.dumps(joints)
        if key in canonical:
            duplicates[index] = canonical[key]
        else:
            canonical[key] = index

    index_map, new_skins = prune_items(skins, duplicates)
    if new_skins or "skins" in json_data:
        json_data["skins"] = new_skins
    return index_map


T = TypeVar("T")


def prune_items(items: list[T], duplicates: dict[int, int]) -> tuple[dict[int, int], list[T]]:
    index_map: dict[int, int] = {}
    result: list[T] = []

    for index, item in enumerate(items):
        if index in duplicates:
            continue
        index_map[index] = len(result)
        result.append(item)

    for index, target in duplicates.items():
        index_map[index] = index_map[target]

    return index_map, result


def remap_accessor_references(json_data: dict[str, Any], index_map: dict[int, int]) -> None:
    for mesh in json_data.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if "indices" in primitive:
                primitive["indices"] = index_map[primitive["indices"]]
            primitive["attributes"] = {
                semantic: index_map[accessor_index] for semantic, accessor_index in primitive.get("attributes", {}).items()
            }
            for target in primitive.get("targets", []):
                for semantic, accessor_index in list(target.items()):
                    target[semantic] = index_map[accessor_index]

    for animation in json_data.get("animations", []):
        for sampler in animation.get("samplers", []):
            if "input" in sampler:
                sampler["input"] = index_map[sampler["input"]]
            if "output" in sampler:
                sampler["output"] = index_map[sampler["output"]]

    for skin in json_data.get("skins", []):
        if "inverseBindMatrices" in skin:
            skin["inverseBindMatrices"] = index_map[skin["inverseBindMatrices"]]


def remap_texture_references(json_data: dict[str, Any], index_map: dict[int, int]) -> None:
    for material in json_data.get("materials", []):
        remap_texture_info_dict(material, index_map)

    for animation in json_data.get("animations", []):
        for channel in animation.get("channels", []):
            target = channel.get("target", {})
            extensions = target.get("extensions", {})
            if not isinstance(extensions, dict):
                continue
            remap_texture_info_dict(extensions, index_map)


def remap_texture_info_dict(value: Any, index_map: dict[int, int]) -> None:
    if isinstance(value, dict):
        if "index" in value and isinstance(value["index"], int) and value["index"] in index_map:
            value["index"] = index_map[value["index"]]
        for child in value.values():
            remap_texture_info_dict(child, index_map)
    elif isinstance(value, list):
        for child in value:
            remap_texture_info_dict(child, index_map)


def remap_material_references(json_data: dict[str, Any], index_map: dict[int, int]) -> None:
    for mesh in json_data.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if "material" in primitive:
                primitive["material"] = index_map[primitive["material"]]


def remap_mesh_references(json_data: dict[str, Any], index_map: dict[int, int]) -> None:
    for node in json_data.get("nodes", []):
        if "mesh" in node:
            node["mesh"] = index_map[node["mesh"]]


def remap_skin_references(json_data: dict[str, Any], index_map: dict[int, int]) -> None:
    for node in json_data.get("nodes", []):
        if "skin" in node:
            node["skin"] = index_map[node["skin"]]


def rebuild_accessor_storage(json_data: dict[str, Any], payloads: list[AccessorPayload]) -> None:
    buffer_views = []
    buffer_blob = bytearray()
    accessors = []

    for payload in payloads:
        align_buffer(buffer_blob, 4)
        byte_offset = len(buffer_blob)
        buffer_blob.extend(payload.blob)
        buffer_view = {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(payload.blob),
        }
        buffer_views.append(buffer_view)
        accessor = dict(payload.accessor)
        accessor["bufferView"] = len(buffer_views) - 1
        accessor["byteOffset"] = 0
        accessors.append(accessor)

    json_data["bufferViews"] = buffer_views
    json_data["buffers"] = [{"uri": "scene.bin", "byteLength": len(buffer_blob)}]
    json_data["accessors"] = accessors
    json_data["_standalone_buffer_blob"] = bytes(buffer_blob)


def align_buffer(buffer_blob: bytearray, alignment: int) -> None:
    padding = (-len(buffer_blob)) % alignment
    if padding:
        buffer_blob.extend(b"\x00" * padding)


def write_gltf(document: GLTFDocument, output_path: Path) -> Path:
    output_path = output_path.with_suffix(".gltf")
    buffer_blob = document.json_data.pop("_standalone_buffer_blob", b"")
    buffer_name = document.json_data["buffers"][0]["uri"] if document.json_data.get("buffers") else "scene.bin"

    for filename, data in document.images_to_write:
        (output_path.parent / filename).write_bytes(data)

    if document.json_data.get("buffers"):
        (output_path.parent / buffer_name).write_bytes(buffer_blob)

    output_path.write_text(json.dumps(document.json_data, indent=2), encoding="utf-8")
    return output_path
