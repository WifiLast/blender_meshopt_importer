if "bpy" in locals():
    from pathlib import Path
    essentials.reload_recursive(Path(__file__).parent, locals())
else:
    import bpy
    from bpy.props import PointerProperty

    from . import essentials, operators, preferences, ui


classes = essentials.get_classes((operators, preferences, ui))


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.decompress_draco_meshotp = PointerProperty(type=preferences.SceneProperties)


def unregister():
    del bpy.types.Scene.decompress_draco_meshotp

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
