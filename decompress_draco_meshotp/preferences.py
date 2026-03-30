import bpy
from bpy.props import BoolProperty
from bpy.types import PropertyGroup


class SceneProperties(PropertyGroup):
    keep_dequantized_file: BoolProperty(
        name="Keep Temporary GLB",
        description="Keep the normalized GLB generated during import",
        default=False,
    )
