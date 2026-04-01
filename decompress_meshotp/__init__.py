if "bpy" in locals():
    from pathlib import Path
    essentials.reload_recursive(Path(__file__).parent, locals())
else:
    import bpy

    from . import essentials, operators, preferences, ui


classes = essentials.get_classes((operators, preferences, ui))


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(ui.menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(ui.menu_func_import)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
