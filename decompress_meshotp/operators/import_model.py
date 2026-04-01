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
    bl_label = "glTF 2.0 (.glb/.gltf) With Meshopt"
    bl_description = "Extend Blender's glTF import with Meshopt and quantized geometry support"
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

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "keep_dequantized_file")
        layout.label(text="Meshopt decode: standalone Python", icon="TOOL_SETTINGS")
        layout.label(text="Dequantize/dedup: standalone Python", icon="TOOL_SETTINGS")
        layout.label(text="Draco: unsupported", icon="INFO")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if self._worker is not None and self._worker.is_alive():
            return {"RUNNING_MODAL"}

        if self._error is None and self._stage == "meshopt_decode":
            self._start_normalize_stage()
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
        self._source_path = source_path
        self._decoded_document = None
        self._needs_meshopt_decode = "EXT_meshopt_compression" in extensions
        self._error = None
        self._result = {"RUNNING_MODAL"}
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, f"Preparing {source_path.name} ({compression_label}) with {self._display_name}...")
        if self._needs_meshopt_decode:
            self._start_worker(
                "meshopt_decode",
                "Decoding Meshopt geometry...",
                "_decoded_document",
                essentials.decode_meshopt_for_import,
                source_path,
            )
        else:
            self._start_normalize_stage()
        return {"RUNNING_MODAL"}

    def _run_stage(self, result_attr, task, *args):
        try:
            setattr(self, result_attr, task(*args))
        except Exception as exc:
            self._error = str(exc)

    def _start_worker(self, stage, report_message, result_attr, task, *args):
        self._stage = stage
        self._worker = threading.Thread(
            target=self._run_stage,
            args=(result_attr, task, *args),
            daemon=True,
        )
        self._worker.start()
        self.report({"INFO"}, report_message)

    def _start_normalize_stage(self):
        self._start_worker(
            "normalize",
            "Dequantizing and de-duplicating geometry...",
            "_import_path",
            essentials.normalize_document_for_import if self._needs_meshopt_decode else essentials.normalize_model_for_import,
            self._decoded_document if self._needs_meshopt_decode else self._source_path,
            self._normalized_path,
        )

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
