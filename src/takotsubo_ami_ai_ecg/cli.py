"""Command-line interface; parsing/help do not import the ML dependencies."""

import argparse
import csv
import json
from pathlib import Path
import sys

from .settings import MODEL_NAMES, InferenceSettings, ModelSettings

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def build_parser(fixed_model=None):
    parser = argparse.ArgumentParser(description="Research ECG image inference with paper TTA/logit defaults.")
    if fixed_model is None:
        parser.add_argument("--model", choices=[*MODEL_NAMES, "ensemble"], default="ensemble")
    else:
        parser.set_defaults(model=fixed_model)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--image", action="append", help="Local image path; repeat to preserve an explicit input order.")
    sources.add_argument("--input-dir", help="Read supported images recursively in relative-path sorted order.")
    sources.add_argument("--input-csv", help="CSV with a path column; relative paths resolve beside the CSV; row order is preserved.")
    parser.add_argument("--weights-dir", default="weights", help="Root containing vit/, convnextv2/, mlpmixer/.")
    for name in MODEL_NAMES:
        parser.add_argument(f"--{name}-weights", help=f"Override the {name} safetensors file or its folder.")
        defaults = getattr(InferenceSettings(), name)
        parser.add_argument(f"--{name}-tta", type=int, default=defaults.tta, help=f"Total views including identity (default {defaults.tta}); 1 disables TTA.")
        parser.add_argument(f"--{name}-magnitude", type=int, default=defaults.magnitude, help=f"RandAugment magnitude 0..9, ten bins (default {defaults.magnitude}).")
    if fixed_model in MODEL_NAMES:
        parser.add_argument("--tta", type=int, help="Override TTA count for this single model.")
        parser.add_argument("--tta-magnitude", type=int, help="Override magnitude for this single model.")
    parser.add_argument("--ensemble-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("VIT", "CONVNEXTV2", "MIXER"))
    parser.add_argument("--threshold", type=float, default=None, help="Explicit AMI probability cutoff. Default: ensemble >=0.5; single models argmax (ties=tako).")
    parser.add_argument("--seed", type=int, default=1234, help="First augmented pass seed; increments each pass (default 1234).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32", help="ViT stays float32; bfloat16 applies to timm models and requires CUDA support.")
    parser.add_argument("--input-policy", choices=["strict480", "resize"], default="strict480", help="strict480 preserves prepared 480x480 input; resize explicitly changes other images using bilinear resizing.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default 4 as in archived evaluators; 0 is a portable override).")
    parser.add_argument("--output", required=True, help="New .json or .csv result file; CSV also gets a .metadata.json sidecar.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing prediction output.")
    return parser


def collect_images(args):
    if args.image:
        paths = [Path(value).expanduser().resolve() for value in args.image]
    elif args.input_dir:
        root = Path(args.input_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("The input directory does not exist.")
        paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.relative_to(root).as_posix())
    else:
        source = Path(args.input_csv).expanduser().resolve()
        try:
            with source.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or "path" not in reader.fieldnames:
                    raise ValueError("The input CSV must contain a path column.")
                paths = []
                for index, row in enumerate(reader, 2):
                    if not row.get("path", "").strip():
                        raise ValueError(f"Missing image path on input CSV row {index}.")
                    path = Path(row["path"]).expanduser()
                    paths.append((path if path.is_absolute() else source.parent / path).resolve())
        except OSError:
            raise ValueError("Cannot read the input CSV.") from None
    if not paths:
        raise ValueError("No input images were found.")
    return paths


def settings_from_args(args):
    specifications = {}
    for name in MODEL_NAMES:
        tta = getattr(args, f"{name}_tta")
        magnitude = getattr(args, f"{name}_magnitude")
        if args.model == name:
            tta = getattr(args, "tta", None) if getattr(args, "tta", None) is not None else tta
            magnitude = getattr(args, "tta_magnitude", None) if getattr(args, "tta_magnitude", None) is not None else magnitude
        specifications[name] = ModelSettings(tta, magnitude)
    settings = InferenceSettings(**specifications, ensemble_weights=tuple(args.ensemble_weights),
        threshold=args.threshold, seed=args.seed, device=args.device, precision=args.precision,
        input_policy=args.input_policy, batch_size=args.batch_size, num_workers=args.num_workers)
    settings.validate()
    return settings


def output_paths(path):
    path = Path(path)
    if path.suffix.lower() not in (".json", ".csv"):
        raise ValueError("Output filename must end in .json or .csv.")
    return [path, path.with_suffix(".metadata.json")] if path.suffix.lower() == ".csv" else [path]


def write_result(result, output, overwrite=False):
    destinations = output_paths(output)
    for path in destinations:
        if path.exists() and not overwrite:
            raise ValueError("Prediction output already exists; choose a new name or use --overwrite.")
        path.parent.mkdir(parents=True, exist_ok=True)
    primary = destinations[0]
    if primary.suffix.lower() == ".json":
        primary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        models = list(result["predictions"][0]["models"])
        fields = ["sample_id", "input_sha256", "pred_id", "pred_label", "logit_tako", "logit_AMI", "prob_tako", "prob_AMI"]
        for name in models:
            fields += [f"{name}_{field}" for field in ("logit_tako", "logit_AMI", "prob_tako", "prob_AMI")]
        with primary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in result["predictions"]:
                values = {key: row[key] for key in ("sample_id", "input_sha256", "pred_id", "pred_label")}
                values.update(dict(zip(("logit_tako", "logit_AMI", "prob_tako", "prob_AMI"), row["logits"] + row["probabilities"])))
                for name in models:
                    model_values = row["models"][name]
                    values.update(dict(zip([f"{name}_{key}" for key in ("logit_tako", "logit_AMI", "prob_tako", "prob_AMI")], model_values["logits"] + model_values["probabilities"])))
                writer.writerow(values)
        destinations[1].write_text(json.dumps(result["metadata"], indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None, fixed_model=None):
    parser = build_parser(fixed_model)
    args = parser.parse_args(argv)
    try:
        settings = settings_from_args(args)
        images = collect_images(args)
        destinations = output_paths(Path(args.output).expanduser().resolve())
        protected = set(images)
        if args.input_csv:
            protected.add(Path(args.input_csv).expanduser().resolve())
        weight_root = Path(args.weights_dir).expanduser().resolve()
        protected.update((weight_root / name / "model.safetensors").resolve() for name in MODEL_NAMES)
        protected.add((weight_root / "vit" / "config.json").resolve())
        protected.add((weight_root / "vit" / "preprocessor_config.json").resolve())
        protected.add((weight_root / "manifest.json").resolve())
        overrides = {name: getattr(args, f"{name}_weights") for name in MODEL_NAMES if getattr(args, f"{name}_weights")}
        for value in overrides.values():
            candidate = Path(value).expanduser().resolve()
            weight_path = candidate / "model.safetensors" if candidate.is_dir() else candidate
            protected.add(weight_path)
            protected.add((weight_path.parent / "config.json").resolve())
            protected.add((weight_path.parent / "preprocessor_config.json").resolve())
        if any(path in protected for path in destinations):
            raise ValueError("Prediction output must not overwrite an input image, input CSV or model file.")
        if any(path.exists() for path in destinations) and not args.overwrite:
            raise ValueError("Prediction output already exists; choose a new name or use --overwrite.")
        from .inference import predict
        result = predict(images, args.weights_dir, args.model, settings, overrides)
        write_result(result, destinations[0], args.overwrite)
    except ImportError:
        parser.exit(2, "Missing inference dependency. Install a compatible torch/torchvision pair, then requirements-inference.txt.\n")
    except (ValueError, OSError) as error:
        # Do not print source paths or full exception tracebacks by default.
        message = str(error) if isinstance(error, ValueError) else "Cannot access an input or output file."
        parser.exit(2, f"Error: {message}\n")
    print(f"Saved {len(result['predictions'])} prediction(s); class order: tako, AMI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

