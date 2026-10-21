"""Command line: build a catalog, recognize a shelf photo, or launch the web demo."""

import argparse
import json
from pathlib import Path

from skuscope.config import DETECTOR_WEIGHTS, EMBEDDERS

DEFAULT_INDEX = Path("index")
# DINOv2 scores even different products at 0.6-0.8, so the bar sits high.
DEFAULT_THRESHOLD = 0.8


def cmd_build_catalog(args: argparse.Namespace) -> None:
    from skuscope.detector import ProductDetector
    from skuscope.embedder import Embedder
    from skuscope.pipeline import build_catalog

    detector = None if args.no_crop else ProductDetector(args.detector)
    catalog = build_catalog(args.photos, Embedder(args.embedder), detector)
    catalog.save(args.index)
    skus = sorted(set(catalog.skus))
    print(f"Catalog saved to {args.index}: {len(catalog.skus)} photos of {len(skus)} SKUs ({', '.join(skus)})")


def cmd_recognize(args: argparse.Namespace) -> None:
    from skuscope.catalog import Catalog
    from skuscope.detector import ProductDetector
    from skuscope.draw import annotate, palette_for
    from skuscope.embedder import Embedder
    from skuscope.pipeline import count_skus, load_photo, recognize

    catalog = Catalog.load(args.index)
    image = load_photo(args.image)
    detections = recognize(
        image, ProductDetector(args.detector, args.conf), Embedder(catalog.embedder), catalog, args.threshold
    )

    out = args.out or Path("outputs") / f"{args.image.stem}_skuscope.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    annotate(image, detections, palette_for(catalog.skus)).save(out, quality=92)

    counts = count_skus(detections)
    print(f"{len(detections)} products detected → {out}")
    for sku, n in counts.most_common():
        print(f"  {sku:<30} {n:>4}")

    if args.json:
        rows = [
            {"sku": d.sku, "score": round(d.score, 4), "box": [d.box.x1, d.box.y1, d.box.x2, d.box.y2]}
            for d in detections
        ]
        args.json.write_text(json.dumps({"counts": dict(counts), "detections": rows}, indent=2, ensure_ascii=False))


def cmd_app(args: argparse.Namespace) -> None:
    from skuscope.app import launch

    launch(args.index, args.detector, DEFAULT_THRESHOLD, args.share)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="skuscope", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-catalog", help="Embed reference photos of each SKU")
    build.add_argument("photos", type=Path, help="Folder laid out as <photos>/<sku>/<photo>.jpg")
    build.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Where to save the catalog")
    build.add_argument("--embedder", choices=sorted(EMBEDDERS), default="dinov2-small")
    build.add_argument("--detector", choices=sorted(DETECTOR_WEIGHTS), default="s")
    build.add_argument("--no-crop", action="store_true", help="Embed whole photos instead of the largest detected product")
    build.set_defaults(func=cmd_build_catalog)

    rec = sub.add_parser("recognize", help="Find and identify the SKUs on a shelf photo")
    rec.add_argument("image", type=Path)
    rec.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    rec.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum cosine similarity to accept a SKU")
    rec.add_argument("--conf", type=float, default=0.25, help="Detector confidence")
    rec.add_argument("--detector", choices=sorted(DETECTOR_WEIGHTS), default="s")
    rec.add_argument("--out", type=Path, help="Annotated image (default: outputs/<image>_skuscope.jpg)")
    rec.add_argument("--json", type=Path, help="Also write detections as JSON")
    rec.set_defaults(func=cmd_recognize)

    app = sub.add_parser("app", help="Launch the Gradio web demo")
    app.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    app.add_argument("--detector", choices=sorted(DETECTOR_WEIGHTS), default="s")
    app.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    app.set_defaults(func=cmd_app)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
