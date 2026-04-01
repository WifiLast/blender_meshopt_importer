import bpy
from bpy.types import Panel

from . import essentials


class Sidebar:
    bl_category = "GLB Import"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context):
        return True


class VIEW3D_PT_glb_import(Sidebar, Panel):
    bl_label = "Compressed GLB Import"

    def draw(self, context):
        layout = self.layout
        props = context.scene.decompress_draco_meshotp

        import_col = layout.column(align=True)
        import_col.operator("import_scene.dequantize_gltf")
        import_col.prop(props, "keep_dequantized_file")
        import_col.label(text="Meshopt decode: standalone Python", icon="TOOL_SETTINGS")
        import_col.label(text="Dequantize/dedup: standalone Python", icon="TOOL_SETTINGS")
        import_col.label(text="Draco: unsupported", icon="INFO")
