# ruff: noqa: T201
"""Convert a GLiNER model to OpenVINO IR format."""

from pathlib import Path
from argparse import Namespace, ArgumentParser

from gliner import GLiNER


def main(args: Namespace) -> Path:
    """Export a GLiNER checkpoint and convert it to OpenVINO IR."""
    gliner_model = GLiNER.from_pretrained(args.model_path)
    exported = gliner_model.export_to_openvino(
        save_dir=args.save_path,
        openvino_filename=args.file_name,
    )
    return Path(exported["openvino_path"])


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="logs/model_12000")
    parser.add_argument("--save_path", default="model/")
    parser.add_argument("--file_name", default="model.xml")
    output_path = main(parser.parse_args())
    print(f"Saved OpenVINO model to {output_path} and {output_path.with_suffix('.bin')}")
