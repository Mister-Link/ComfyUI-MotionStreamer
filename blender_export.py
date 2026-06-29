import argparse
from pathlib import Path
import sys

import bpy


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Import a BVH and export an FBX.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_anim.bvh(filepath=str(input_path))
    bpy.ops.export_scene.fbx(
        filepath=str(output_path),
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=True,
    )


if __name__ == "__main__":
    main()
