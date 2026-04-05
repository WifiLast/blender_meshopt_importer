#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942
EXTENSION_NAME = "EXT_meshopt_compression"


@dataclass
class MeshoptAsset:
    json_doc: dict
    buffers: list[bytearray]
    base_dir: Path


def assert_true(condition: bool, message: str = "Assertion failed") -> None:
    if not condition:
        raise ValueError(message)


def decode_vertex_buffer(target: bytearray, element_count: int, byte_stride: int, source: bytes, filter_name: str | None) -> None:
    assert_true(source[0] in (0xA0, 0xA1))
    version = source[0] & 0x0F

    max_block_elements = min((0x2000 // byte_stride) & ~0x000F, 0x100)
    deltas = bytearray(max_block_elements * byte_stride)

    tail_size = byte_stride if version == 0 else byte_stride + byte_stride // 4
    tail_data_offset = len(source) - tail_size
    temp_data = bytearray(source[tail_data_offset : tail_data_offset + byte_stride])
    channels = None if version == 0 else source[tail_data_offset + byte_stride : tail_data_offset + tail_size]

    source_offset = 1
    header_modes = (
        (0, 2, 4, 8),
        (0, 1, 2, 4),
        (1, 2, 4, 8),
    )

    for dst_elem_base in range(0, element_count, max_block_elements):
        attr_block_element_count = min(element_count - dst_elem_base, max_block_elements)
        group_count = ((attr_block_element_count + 0x0F) & ~0x0F) >> 4
        header_byte_count = ((group_count + 0x03) & ~0x03) >> 2

        control_bits_offset = source_offset
        source_offset += 0 if version == 0 else byte_stride // 4

        for index in range(len(deltas)):
            deltas[index] = 0

        for byte in range(byte_stride):
            delta_base = byte * attr_block_element_count
            control_mode = 0
            if version != 0:
                control_mode = (source[control_bits_offset + (byte >> 2)] >> ((byte & 0x03) << 1)) & 0x03

            if control_mode == 2:
                continue
            if control_mode == 3:
                deltas[delta_base : delta_base + attr_block_element_count] = source[source_offset : source_offset + attr_block_element_count]
                source_offset += attr_block_element_count
                continue

            header_bits_offset = source_offset
            source_offset += header_byte_count

            for group in range(group_count):
                mode = (source[header_bits_offset + (group >> 2)] >> ((group & 0x03) << 1)) & 0x03
                mode_bits = header_modes[0 if version == 0 else control_mode + 1][mode]
                delta_offset = delta_base + (group << 4)

                if mode_bits == 0:
                    continue
                if mode_bits == 1:
                    src_base = source_offset
                    source_offset += 0x02
                    for m in range(0x10):
                        shift = m & 0x07
                        delta = (source[src_base + (m >> 3)] >> shift) & 0x01
                        if delta == 1:
                            delta = source[source_offset]
                            source_offset += 1
                        deltas[delta_offset + m] = delta
                    continue
                if mode_bits == 2:
                    src_base = source_offset
                    source_offset += 0x04
                    for m in range(0x10):
                        shift = 6 - ((m & 0x03) << 1)
                        delta = (source[src_base + (m >> 2)] >> shift) & 0x03
                        if delta == 3:
                            delta = source[source_offset]
                            source_offset += 1
                        deltas[delta_offset + m] = delta
                    continue
                if mode_bits == 4:
                    src_base = source_offset
                    source_offset += 0x08
                    for m in range(0x10):
                        shift = 4 - ((m & 0x01) << 2)
                        delta = (source[src_base + (m >> 1)] >> shift) & 0x0F
                        if delta == 0x0F:
                            delta = source[source_offset]
                            source_offset += 1
                        deltas[delta_offset + m] = delta
                    continue

                deltas[delta_offset : delta_offset + 0x10] = source[source_offset : source_offset + 0x10]
                source_offset += 0x10

        for elem in range(attr_block_element_count):
            dst_elem = dst_elem_base + elem

            for byte_group in range(0, byte_stride, 4):
                channel_mode = 0 if version == 0 else channels[byte_group >> 2] & 0x03
                assert_true(channel_mode != 0x03)

                if channel_mode == 0:
                    for byte in range(byte_group, byte_group + 4):
                        val = deltas[byte * attr_block_element_count + elem]
                        delta = (val >> 1) ^ -(val & 1)
                        temp = (temp_data[byte] + delta) & 0xFF
                        dst_offset = dst_elem * byte_stride + byte
                        target[dst_offset] = temp
                        temp_data[byte] = temp
                elif channel_mode == 1:
                    for byte in range(byte_group, byte_group + 4, 2):
                        val = (
                            deltas[byte * attr_block_element_count + elem]
                            + (deltas[(byte + 1) * attr_block_element_count + elem] << 8)
                        )
                        delta = (val >> 1) ^ -(val & 1)
                        temp = temp_data[byte] + (temp_data[byte + 1] << 8)
                        temp = (temp + delta) & 0xFFFF
                        dst_offset = dst_elem * byte_stride + byte
                        target[dst_offset] = temp & 0xFF
                        target[dst_offset + 1] = (temp >> 8) & 0xFF
                        temp_data[byte] = target[dst_offset]
                        temp_data[byte + 1] = target[dst_offset + 1]
                elif channel_mode == 2:
                    byte = byte_group
                    delta = (
                        deltas[byte * attr_block_element_count + elem]
                        + (deltas[(byte + 1) * attr_block_element_count + elem] << 8)
                        + (deltas[(byte + 2) * attr_block_element_count + elem] << 16)
                        + (deltas[(byte + 3) * attr_block_element_count + elem] << 24)
                    ) & 0xFFFFFFFF
                    temp = (
                        temp_data[byte]
                        + (temp_data[byte + 1] << 8)
                        + (temp_data[byte + 2] << 16)
                        + (temp_data[byte + 3] << 24)
                    ) & 0xFFFFFFFF
                    rot = channels[byte_group >> 2] >> 4
                    if rot:
                        delta = ((delta >> rot) | ((delta << (32 - rot)) & 0xFFFFFFFF)) & 0xFFFFFFFF
                    temp ^= delta
                    dst_offset = dst_elem * byte_stride + byte
                    target[dst_offset] = temp & 0xFF
                    target[dst_offset + 1] = (temp >> 8) & 0xFF
                    target[dst_offset + 2] = (temp >> 16) & 0xFF
                    target[dst_offset + 3] = (temp >> 24) & 0xFF
                    temp_data[byte] = target[dst_offset]
                    temp_data[byte + 1] = target[dst_offset + 1]
                    temp_data[byte + 2] = target[dst_offset + 2]
                    temp_data[byte + 3] = target[dst_offset + 3]

    tail_size_padded = max(tail_size, 32 if version == 0 else 24)
    assert_true(source_offset == len(source) - tail_size_padded)

    if filter_name == "OCTAHEDRAL":
        apply_octahedral_filter(target, element_count, byte_stride)
    elif filter_name == "QUATERNION":
        apply_quaternion_filter(target, element_count)
    elif filter_name == "EXPONENTIAL":
        apply_exponential_filter(target, element_count, byte_stride)
    elif filter_name == "COLOR":
        apply_color_filter(target, element_count, byte_stride)


def apply_octahedral_filter(target: bytearray, element_count: int, byte_stride: int) -> None:
    assert_true(byte_stride in (4, 8))
    if byte_stride == 4:
        values = np.frombuffer(target, dtype=np.int8).reshape(element_count, 4)
        max_int = 127.0
        xy = values[:, :2].astype(np.float32)
        one = values[:, 2].astype(np.float32)
        safe_one = np.where(one == 0, 1.0, one)
        xy = xy / safe_one[:, None]
        z = 1.0 - np.abs(xy[:, 0]) - np.abs(xy[:, 1])
        t = np.maximum(-z, 0.0)
        xy[:, 0] -= np.where(xy[:, 0] >= 0, t, -t)
        xy[:, 1] -= np.where(xy[:, 1] >= 0, t, -t)
        h = max_int / np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2 + z**2)
        values[:, 0] = np.rint(xy[:, 0] * h).astype(np.int8)
        values[:, 1] = np.rint(xy[:, 1] * h).astype(np.int8)
        values[:, 2] = np.rint(z * h).astype(np.int8)
        return

    values = np.frombuffer(target, dtype=np.int16).reshape(element_count, 4)
    max_int = 32767.0
    xy = values[:, :2].astype(np.float32)
    one = values[:, 2].astype(np.float32)
    safe_one = np.where(one == 0, 1.0, one)
    xy = xy / safe_one[:, None]
    z = 1.0 - np.abs(xy[:, 0]) - np.abs(xy[:, 1])
    t = np.maximum(-z, 0.0)
    xy[:, 0] -= np.where(xy[:, 0] >= 0, t, -t)
    xy[:, 1] -= np.where(xy[:, 1] >= 0, t, -t)
    h = max_int / np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2 + z**2)
    values[:, 0] = np.rint(xy[:, 0] * h).astype(np.int16)
    values[:, 1] = np.rint(xy[:, 1] * h).astype(np.int16)
    values[:, 2] = np.rint(z * h).astype(np.int16)


def apply_quaternion_filter(target: bytearray, element_count: int) -> None:
    values = np.frombuffer(target, dtype=np.int16).reshape(element_count, 4)
    input_w = values[:, 3].astype(np.int32)
    max_component = input_w & 0x03
    scale = np.float32(math.sqrt(0.5)) / (input_w | 0x03)
    xyz = values[:, :3].astype(np.float32) * scale[:, None]
    w = np.sqrt(np.maximum(0.0, 1.0 - np.sum(xyz**2, axis=1)))
    output = np.empty_like(values)
    row_indices = np.arange(element_count)
    output[row_indices, (max_component + 0) % 4] = np.rint(w * 32767).astype(np.int16)
    output[row_indices, (max_component + 1) % 4] = np.rint(xyz[:, 0] * 32767).astype(np.int16)
    output[row_indices, (max_component + 2) % 4] = np.rint(xyz[:, 1] * 32767).astype(np.int16)
    output[row_indices, (max_component + 3) % 4] = np.rint(xyz[:, 2] * 32767).astype(np.int16)
    values[:, :] = output


def apply_exponential_filter(target: bytearray, element_count: int, byte_stride: int) -> None:
    assert_true((byte_stride & 0x03) == 0)
    ints = np.frombuffer(target, dtype=np.int32)
    exponents = ints >> 24
    mantissas = (ints << 8) >> 8
    floats = np.power(np.float32(2.0), exponents.astype(np.float32)) * mantissas.astype(np.float32)
    target[:] = floats.astype(np.float32).tobytes()


def apply_color_filter(target: bytearray, element_count: int, byte_stride: int) -> None:
    assert_true(byte_stride in (4, 8))
    max_int = (1 << (byte_stride * 2)) - 1

    if byte_stride == 4:
        data = np.frombuffer(target, dtype=np.uint8).reshape(element_count, 4)
        signed = np.frombuffer(target, dtype=np.int8).reshape(element_count, 4)
        y = data[:, 0].astype(np.int32)
        co = signed[:, 1].astype(np.int32)
        cg = signed[:, 2].astype(np.int32)
        alpha_input = data[:, 3].astype(np.int32)
        alpha_bits = np.zeros_like(alpha_input)
        non_zero = alpha_input > 0
        alpha_bits[non_zero] = np.floor(np.log2(alpha_input[non_zero])).astype(np.int32)
        alpha_scale = np.where(alpha_input > 0, (1 << (alpha_bits + 1)) - 1, 1)
        r = y + co - cg
        g = y + cg
        b = y - co - cg
        a = alpha_input & (alpha_scale >> 1)
        a = (a << 1) | (a & 1)
        scale = max_int / alpha_scale.astype(np.float32)
        data[:, 0] = np.rint(r * scale).clip(0, max_int).astype(np.uint8)
        data[:, 1] = np.rint(g * scale).clip(0, max_int).astype(np.uint8)
        data[:, 2] = np.rint(b * scale).clip(0, max_int).astype(np.uint8)
        data[:, 3] = np.rint(a * scale).clip(0, max_int).astype(np.uint8)
        return

    data = np.frombuffer(target, dtype=np.uint16).reshape(element_count, 4)
    signed = np.frombuffer(target, dtype=np.int16).reshape(element_count, 4)
    y = data[:, 0].astype(np.int32)
    co = signed[:, 1].astype(np.int32)
    cg = signed[:, 2].astype(np.int32)
    alpha_input = data[:, 3].astype(np.int32)
    alpha_bits = np.zeros_like(alpha_input)
    non_zero = alpha_input > 0
    alpha_bits[non_zero] = np.floor(np.log2(alpha_input[non_zero])).astype(np.int32)
    alpha_scale = np.where(alpha_input > 0, (1 << (alpha_bits + 1)) - 1, 1)
    r = y + co - cg
    g = y + cg
    b = y - co - cg
    a = alpha_input & (alpha_scale >> 1)
    a = (a << 1) | (a & 1)
    scale = max_int / alpha_scale.astype(np.float32)
    data[:, 0] = np.rint(r * scale).clip(0, max_int).astype(np.uint16)
    data[:, 1] = np.rint(g * scale).clip(0, max_int).astype(np.uint16)
    data[:, 2] = np.rint(b * scale).clip(0, max_int).astype(np.uint16)
    data[:, 3] = np.rint(a * scale).clip(0, max_int).astype(np.uint16)


def decode_index_buffer(target: bytearray, count: int, byte_stride: int, source: bytes) -> None:
    assert_true(source[0] == 0xE1)
    assert_true(count % 3 == 0)
    assert_true(byte_stride in (2, 4))

    tri_count = count // 3
    code_offset = 0x01
    data_offset = code_offset + tri_count
    codeaux_offset = len(source) - 0x10

    def read_leb128() -> int:
        nonlocal data_offset
        value = 0
        shift = 0
        while True:
            byte = source[data_offset]
            data_offset += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                return value
            shift += 7

    next_index = 0
    last = 0
    edgefifo = deque([0] * 32, maxlen=32)
    vertexfifo = deque([0] * 16, maxlen=16)
    decoded: list[int] = []

    def decode_index(value: int) -> int:
        nonlocal last
        last += (value >> 1) ^ -(value & 1)
        return last

    for _ in range(tri_count):
        code = source[code_offset]
        code_offset += 1
        b0 = code >> 4
        b1 = code & 0x0F

        if b0 < 0x0F:
            a = edgefifo[(b0 << 1) + 0]
            b = edgefifo[(b0 << 1) + 1]
            if b1 == 0x00:
                c = next_index
                next_index += 1
                vertexfifo.appendleft(c)
            elif b1 < 0x0D:
                c = vertexfifo[b1]
            elif b1 == 0x0D:
                last -= 1
                c = last
                vertexfifo.appendleft(c)
            elif b1 == 0x0E:
                last += 1
                c = last
                vertexfifo.appendleft(c)
            else:
                c = decode_index(read_leb128())
                vertexfifo.appendleft(c)

            edgefifo.appendleft(b)
            edgefifo.appendleft(c)
            edgefifo.appendleft(c)
            edgefifo.appendleft(a)
            decoded.extend((a, b, c))
            continue

        if b1 < 0x0E:
            e = source[codeaux_offset + b1]
            z = e >> 4
            w = e & 0x0F
            a = next_index
            next_index += 1
            b = next_index if z == 0x00 else vertexfifo[z - 1]
            if z == 0x00:
                next_index += 1
            c = next_index if w == 0x00 else vertexfifo[w - 1]
            if w == 0x00:
                next_index += 1
            vertexfifo.appendleft(a)
            if z == 0x00:
                vertexfifo.appendleft(b)
            if w == 0x00:
                vertexfifo.appendleft(c)
        else:
            e = source[data_offset]
            data_offset += 1
            if e == 0x00:
                next_index = 0
            z = e >> 4
            w = e & 0x0F
            if b1 == 0x0E:
                a = next_index
                next_index += 1
            else:
                a = decode_index(read_leb128())

            if z == 0x00:
                b = next_index
                next_index += 1
            elif z == 0x0F:
                b = decode_index(read_leb128())
            else:
                b = vertexfifo[z - 1]

            if w == 0x00:
                c = next_index
                next_index += 1
            elif w == 0x0F:
                c = decode_index(read_leb128())
            else:
                c = vertexfifo[w - 1]

            vertexfifo.appendleft(a)
            if z in (0x00, 0x0F):
                vertexfifo.appendleft(b)
            if w in (0x00, 0x0F):
                vertexfifo.appendleft(c)

        edgefifo.appendleft(a)
        edgefifo.appendleft(b)
        edgefifo.appendleft(b)
        edgefifo.appendleft(c)
        edgefifo.appendleft(c)
        edgefifo.appendleft(a)
        decoded.extend((a, b, c))

    dtype = np.uint16 if byte_stride == 2 else np.uint32
    target[:] = np.asarray(decoded, dtype=dtype).tobytes()


def decode_index_sequence(target: bytearray, count: int, byte_stride: int, source: bytes) -> None:
    assert_true(source[0] == 0xD1)
    assert_true(byte_stride in (2, 4))

    data_offset = 0x01

    def read_leb128() -> int:
        nonlocal data_offset
        value = 0
        shift = 0
        while True:
            byte = source[data_offset]
            data_offset += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                return value
            shift += 7

    last = [0, 0]
    decoded = []
    for _ in range(count):
        value = read_leb128()
        bucket = value & 0x01
        val = value >> 1
        delta = (val >> 1) ^ -(val & 1)
        last[bucket] += delta
        decoded.append(last[bucket])

    dtype = np.uint16 if byte_stride == 2 else np.uint32
    target[:] = np.asarray(decoded, dtype=dtype).tobytes()


def decode_gltf_buffer(count: int, size: int, source: bytes, mode: str, filter_name: str | None) -> bytes:
    target = bytearray(count * size)
    if mode == "ATTRIBUTES":
        decode_vertex_buffer(target, count, size, source, filter_name)
    elif mode == "TRIANGLES":
        decode_index_buffer(target, count, size, source)
    elif mode == "INDICES":
        decode_index_sequence(target, count, size, source)
    else:
        raise ValueError(f"Unsupported Meshopt mode: {mode}")
    return bytes(target)


def read_resource(base_dir: Path, uri: str | None) -> bytes:
    if not uri:
        return b""
    if uri.startswith("data:"):
        _, encoded = uri.split(",", 1)
        return base64.b64decode(encoded)
    return (base_dir / uri).read_bytes()


def load_asset(input_path: Path) -> MeshoptAsset:
    if input_path.suffix.lower() == ".gltf":
        json_doc = json.loads(input_path.read_text(encoding="utf-8"))
        base_dir = input_path.parent
        buffers = [bytearray(read_resource(base_dir, buffer_def.get("uri"))) for buffer_def in json_doc.get("buffers", [])]
        return MeshoptAsset(json_doc=json_doc, buffers=buffers, base_dir=base_dir)

    if input_path.suffix.lower() != ".glb":
        raise ValueError(f"Unsupported file type: {input_path.suffix.lower()}")

    payload = input_path.read_bytes()
    if len(payload) < 20:
        raise ValueError(f"Invalid GLB header in {input_path}")

    magic = payload[:4]
    version = int(np.frombuffer(payload[4:8], dtype=np.uint32)[0])
    if magic != b"glTF" or version != 2:
        raise ValueError(f"Invalid GLB header in {input_path}")

    offset = 12
    json_chunk = None
    bin_chunk = b""
    while offset + 8 <= len(payload):
        chunk_header = np.frombuffer(payload[offset : offset + 8], dtype=np.uint32)
        chunk_length = int(chunk_header[0])
        chunk_type = int(chunk_header[1])
        chunk_data = payload[offset + 8 : offset + 8 + chunk_length]
        if chunk_type == GLB_JSON_CHUNK:
            json_chunk = chunk_data
        elif chunk_type == GLB_BIN_CHUNK:
            bin_chunk = chunk_data
        offset += 8 + chunk_length

    if json_chunk is None:
        raise ValueError(f"Missing JSON chunk in {input_path}")

    json_doc = json.loads(json_chunk.decode("utf-8"))
    buffers: list[bytearray] = []
    for index, buffer_def in enumerate(json_doc.get("buffers", [])):
        if buffer_def.get("uri"):
            buffers.append(bytearray(read_resource(input_path.parent, buffer_def["uri"])))
        elif index == 0:
            buffers.append(bytearray(bin_chunk))
        else:
            buffers.append(bytearray(buffer_def.get("byteLength", 0)))
    return MeshoptAsset(json_doc=json_doc, buffers=buffers, base_dir=input_path.parent)


def decode_meshopt(json_doc: dict, buffers: list[bytearray]) -> None:
    buffer_views = json_doc.get("bufferViews", [])
    buffer_defs = json_doc.get("buffers", [])

    for buffer_view in buffer_views:
        meshopt_ext = buffer_view.get("extensions", {}).get(EXTENSION_NAME)
        if not meshopt_ext:
            continue

        source_buffer = buffers[meshopt_ext["buffer"]]
        source_offset = meshopt_ext.get("byteOffset", 0)
        source_length = meshopt_ext.get("byteLength", 0)
        source = bytes(source_buffer[source_offset : source_offset + source_length])
        target = decode_gltf_buffer(
            meshopt_ext["count"],
            meshopt_ext["byteStride"],
            source,
            meshopt_ext["mode"],
            meshopt_ext.get("filter"),
        )

        fallback_buffer_index = buffer_view["buffer"]
        fallback_buffer_length = buffer_defs[fallback_buffer_index].get("byteLength", 0)
        if len(buffers[fallback_buffer_index]) < fallback_buffer_length:
            buffers[fallback_buffer_index] = bytearray(fallback_buffer_length)

        fallback_offset = buffer_view.get("byteOffset", 0)
        buffers[fallback_buffer_index][fallback_offset : fallback_offset + len(target)] = target

        extensions = buffer_view.get("extensions", {})
        extensions.pop(EXTENSION_NAME, None)
        if extensions:
            buffer_view["extensions"] = extensions
        elif "extensions" in buffer_view:
            del buffer_view["extensions"]

    for buffer_def in buffer_defs:
        extensions = buffer_def.get("extensions", {})
        extensions.pop(EXTENSION_NAME, None)
        if extensions:
            buffer_def["extensions"] = extensions
        elif "extensions" in buffer_def:
            del buffer_def["extensions"]

    for key in ("extensionsUsed", "extensionsRequired"):
        if key not in json_doc:
            continue
        json_doc[key] = [name for name in json_doc[key] if name != EXTENSION_NAME]
        if not json_doc[key]:
            del json_doc[key]


def write_asset(output_path: Path, json_doc: dict, buffers: list[bytearray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for index, buffer_data in enumerate(buffers):
        buffer_name = f"buffer_{index}.bin"
        json_doc["buffers"][index]["uri"] = buffer_name
        json_doc["buffers"][index]["byteLength"] = len(buffer_data)
        (output_path.parent / buffer_name).write_bytes(buffer_data)
    output_path.write_text(json.dumps(json_doc, indent=2), encoding="utf-8")


def decode_file(input_path: Path, output_path: Path) -> Path:
    asset = load_asset(input_path)
    decode_meshopt(asset.json_doc, asset.buffers)
    write_asset(output_path, asset.json_doc, asset.buffers)
    return output_path


def decode_asset(input_path: Path) -> MeshoptAsset:
    asset = load_asset(input_path)
    decode_meshopt(asset.json_doc, asset.buffers)
    return asset


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("Usage: decode_meshopt.py <input.{gltf|glb}> <output.gltf>")

    input_path = Path(argv[1]).resolve()
    output_path = Path(argv[2]).resolve()
    decode_file(input_path, output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
