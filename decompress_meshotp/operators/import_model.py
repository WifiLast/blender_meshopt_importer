from pathlib import Path
import shutil
import tempfile
import threading

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from .. import essentials


class IMPORT_SCENE_OT_dequantize_gltf(Operator, ImportHelper):
    bl_idname = "import_scene.dequantize_gltf"
    bl_label = "Import GLTF/GLB (Meshopt)"
    bl_description = "Normalize Meshopt and quantized geometry before importing"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".glb"
    filter_glob: StringProperty(
        default="*.glb;*.gltf",
        options={"HIDDEN"},
    )
    keep_dequantized_file: BoolProperty(
        name="Keep Temporary GLB",
        description="Preserve the generated normalized GLB in the system temp directory",
        default=False,
    )

    def invoke(self, context, event):
        self.keep_dequantized_file = context.scene.decompress_draco_meshotp.keep_dequantized_file
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if self._worker is not None and self._worker.is_alive():
            return {"RUNNING_MODAL"}

        self._finish(context)
        return self._result

    def execute(self, context):
        source_path = Path(self.filepath)
        if not source_path.exists():
            self.report({"ERROR"}, f"Input file not found: {source_path}")
            return {"CANCELLED"}

        extensions = essentials.detect_compression_extensions(source_path)
        if essentials.DRACO_EXTENSION in extensions:
            self.report(
                {"ERROR"},
                "Draco-compressed files are not supported by this addon.",
            )
            return {"CANCELLED"}

        compression_label = essentials.describe_import_extensions(source_path)
        temp_dir = Path(tempfile.mkdtemp(prefix="decompress_draco_meshopt_"))
        normalized_path = temp_dir / f"{source_path.stem}_normalized{source_path.suffix.lower()}"
        self._display_name = essentials.describe_import_backend(source_path)
        self._compression_label = compression_label
        self._temp_dir = temp_dir
        self._normalized_path = normalized_path
        self._import_path = normalized_path
        self._error = None
        self._result = {"RUNNING_MODAL"}
        self._worker = threading.Thread(
            target=self._normalize_in_background,
            args=(source_path, normalized_path),
            daemon=True,
        )
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        self._worker.start()
        self.report({"INFO"}, f"Preparing {source_path.name} ({compression_label}) with {self._display_name}...")
        return {"RUNNING_MODAL"}

    def _normalize_in_background(self, source_path, normalized_path):
        try:
            self._import_path = essentials.normalize_model_for_import(source_path, normalized_path)
        except Exception as exc:
            self._error = str(exc)

    def _finish(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        try:
            if self._error is not None:
                self.report({"ERROR"}, self._error)
                self._result = {"CANCELLED"}
                return

            bpy.ops.import_scene.gltf(filepath=str(self._import_path))
            self.report({"INFO"}, f"Imported {self._compression_label} model via {self._display_name}")
            self._result = {"FINISHED"}
        finally:
            if self.keep_dequantized_file:
                self.report({"INFO"}, f"Temporary normalized asset kept at {self._import_path}")
            else:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
