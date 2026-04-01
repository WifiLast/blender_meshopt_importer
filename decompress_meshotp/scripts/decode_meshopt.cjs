#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const MeshoptDecoder = require("../vendor/meshopt_decoder.cjs");

const GLB_JSON_CHUNK = 0x4e4f534a;
const GLB_BIN_CHUNK = 0x004e4942;
const EXTENSION_NAME = "EXT_meshopt_compression";

async function main() {
  const [, , inputArg, outputArg] = process.argv;
  if (!inputArg || !outputArg) {
    throw new Error("Usage: node decode_meshopt.cjs <input.{gltf|glb}> <output.gltf>");
  }

  await MeshoptDecoder.ready;

  const inputPath = path.resolve(inputArg);
  const outputPath = path.resolve(outputArg);
  const { json, buffers, baseDir } = loadAsset(inputPath);

  decodeMeshopt(json, buffers);
  writeAsset(outputPath, json, buffers, baseDir);
}

function loadAsset(inputPath) {
  const ext = path.extname(inputPath).toLowerCase();
  if (ext === ".gltf") {
    const json = JSON.parse(fs.readFileSync(inputPath, "utf8"));
    const baseDir = path.dirname(inputPath);
    const buffers = (json.buffers || []).map((bufferDef) => readResource(baseDir, bufferDef.uri));
    return { json, buffers, baseDir };
  }

  if (ext !== ".glb") {
    throw new Error(`Unsupported file type: ${ext}`);
  }

  const payload = fs.readFileSync(inputPath);
  if (payload.length < 20) {
    throw new Error(`Invalid GLB header in ${inputPath}`);
  }

  const magic = payload.slice(0, 4).toString("ascii");
  const version = payload.readUInt32LE(4);
  if (magic !== "glTF" || version !== 2) {
    throw new Error(`Invalid GLB header in ${inputPath}`);
  }

  let offset = 12;
  let jsonChunk = null;
  let binChunk = Buffer.alloc(0);

  while (offset + 8 <= payload.length) {
    const chunkLength = payload.readUInt32LE(offset);
    const chunkType = payload.readUInt32LE(offset + 4);
    const chunkData = payload.subarray(offset + 8, offset + 8 + chunkLength);
    if (chunkType === GLB_JSON_CHUNK) {
      jsonChunk = chunkData;
    } else if (chunkType === GLB_BIN_CHUNK) {
      binChunk = chunkData;
    }
    offset += 8 + chunkLength;
  }

  if (!jsonChunk) {
    throw new Error(`Missing JSON chunk in ${inputPath}`);
  }

  const json = JSON.parse(jsonChunk.toString("utf8"));
  const buffers = [];
  let binOffset = 0;
  for (let i = 0; i < (json.buffers || []).length; i++) {
    const bufferDef = json.buffers[i];
    if (bufferDef.uri) {
      buffers.push(readResource(path.dirname(inputPath), bufferDef.uri));
      continue;
    }

    if (i === 0) {
      buffers.push(binChunk);
      binOffset = binChunk.length;
      continue;
    }

    buffers.push(Buffer.alloc(bufferDef.byteLength || 0));
  }

  return { json, buffers, baseDir: path.dirname(inputPath) };
}

function readResource(baseDir, uri) {
  if (!uri) {
    return Buffer.alloc(0);
  }
  if (uri.startsWith("data:")) {
    const encoded = uri.slice(uri.indexOf(",") + 1);
    return Buffer.from(encoded, "base64");
  }
  return fs.readFileSync(path.resolve(baseDir, uri));
}

function decodeMeshopt(json, buffers) {
  const bufferViews = json.bufferViews || [];
  const bufferDefs = json.buffers || [];

  for (const bufferView of bufferViews) {
    const meshoptExt = bufferView.extensions && bufferView.extensions[EXTENSION_NAME];
    if (!meshoptExt) {
      continue;
    }

    const sourceBuffer = buffers[meshoptExt.buffer];
    if (!sourceBuffer) {
      throw new Error(`Missing source buffer ${meshoptExt.buffer} for Meshopt decode.`);
    }

    const sourceOffset = meshoptExt.byteOffset || 0;
    const sourceLength = meshoptExt.byteLength || 0;
    const source = sourceBuffer.subarray(sourceOffset, sourceOffset + sourceLength);
    const targetLength = meshoptExt.count * meshoptExt.byteStride;
    const target = new Uint8Array(targetLength);
    MeshoptDecoder.decodeGltfBuffer(target, meshoptExt.count, meshoptExt.byteStride, new Uint8Array(source), meshoptExt.mode, meshoptExt.filter);

    const fallbackBufferIndex = bufferView.buffer;
    const fallbackBufferDef = bufferDefs[fallbackBufferIndex];
    if (!buffers[fallbackBufferIndex] || buffers[fallbackBufferIndex].length < (fallbackBufferDef.byteLength || 0)) {
      buffers[fallbackBufferIndex] = Buffer.alloc(fallbackBufferDef.byteLength || 0);
    }

    const fallbackOffset = bufferView.byteOffset || 0;
    Buffer.from(target).copy(buffers[fallbackBufferIndex], fallbackOffset);

    delete bufferView.extensions[EXTENSION_NAME];
    if (Object.keys(bufferView.extensions).length === 0) {
      delete bufferView.extensions;
    }
  }

  for (const bufferDef of bufferDefs) {
    if (!bufferDef.extensions || !bufferDef.extensions[EXTENSION_NAME]) {
      continue;
    }
    delete bufferDef.extensions[EXTENSION_NAME];
    if (Object.keys(bufferDef.extensions).length === 0) {
      delete bufferDef.extensions;
    }
  }

  for (const key of ["extensionsUsed", "extensionsRequired"]) {
    if (!json[key]) {
      continue;
    }
    json[key] = json[key].filter((name) => name !== EXTENSION_NAME);
    if (json[key].length === 0) {
      delete json[key];
    }
  }
}

function writeAsset(outputPath, json, buffers) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  for (let i = 0; i < (json.buffers || []).length; i++) {
    const bufferName = `buffer_${i}.bin`;
    json.buffers[i].uri = bufferName;
    json.buffers[i].byteLength = buffers[i].length;
    fs.writeFileSync(path.join(path.dirname(outputPath), bufferName), buffers[i]);
  }

  fs.writeFileSync(outputPath, JSON.stringify(json, null, 2), "utf8");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exit(1);
});
