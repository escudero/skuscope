"""Command line: build a catalog, recognize a shelf photo or video, or launch the web demo."""

import argparse
import json
from pathlib import Path

from skuscope.config import DETECTOR_WEIGHTS, EMBEDDERS
from skuscope.pipeline import DEFAULT_MARGIN, DEFAULT_THRESHOLD

DEFAULT_INDEX = Path("index")


def cmd_build_catalog(args: argparse.Namespace) -> None:
    from skuscope.detector import ProductDetector
    from skuscope.embedder import Embedder
    from skuscope.pipeline import build_catalog

    detector = None if args.no_crop else ProductDetector(args.detector)
    catalog = build_catalog(args.photos, Embedder(args.embedder), detector)
    catalog.save(args.index)
    skus = catalog.sku_names
    print(f"Catalog saved to {args.index}: {len(catalog.skus)} photos of {len(skus)} SKUs ({', '.join(skus)})")
    if catalog.families:
        print(f"  {len(catalog.families)} families: {', '.join(sorted(catalog.families))}")


def cmd_recognize(args: argparse.Namespace) -> None:
    from skuscope.catalog import Catalog
    from skuscope.detector import ProductDetector
    from skuscope.draw import annotate, palette_for
    from skuscope.embedder import Embedder
    from skuscope.pipeline import count_skus, load_photo, recognize

    catalog = Catalog.load(args.index)
    image = load_photo(args.image)
    detections = recognize(
        image, ProductDetector(args.detector, args.conf), Embedder(catalog.embedder), catalog, args.threshold, args.margin
    )

    out = args.out or Path("outputs") / f"{args.image.stem}_skuscope.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    annotate(image, detections, palette_for(catalog.labels)).save(out, quality=92)

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


def cmd_video(args: argparse.Namespace) -> None:
    from skuscope.catalog import Catalog
    from skuscope.detector import ProductDetector
    from skuscope.draw import palette_for
    from skuscope.embedder import Embedder
    from skuscope.video import recognize_video

    catalog = Catalog.load(args.index)
    out = args.out or Path("outputs") / f"{args.video.stem}_skuscope.mp4"

    def progress(frame: int, total: int) -> None:
        print(f"\r  frame {frame}/{total}", end="", flush=True)

    result = recognize_video(
        args.video,
        ProductDetector(args.detector, args.conf),
        Embedder(catalog.embedder),
        catalog,
        args.threshold,
        args.margin,
        out,
        palette_for(catalog.labels),
        progress,
        tuple(args.motion_region),
    )
    print(f"\n{result.frames} frames, {result.seen} different products seen, "
          f"{sum(result.counts.values())} counted → {out}")
    for label, n in result.counts.most_common():
        print(f"  {label:<30} {n:>4}")

    if args.json:
        summary = {"frames": result.frames, "seen": result.seen, "counts": dict(result.counts)}
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_app(args: argparse.Namespace) -> None:
    from skuscope.app import launch

    launch(args.index, args.detector, DEFAULT_THRESHOLD, DEFAULT_MARGIN, args.share)


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

    def add_matching_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--index", type=Path, default=DEFAULT_INDEX)
        command.add_argument(
            "--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum cosine similarity to accept a SKU"
        )
        command.add_argument(
            "--margin", type=float, default=DEFAULT_MARGIN, help="How much the best SKU must beat the second best"
        )
        command.add_argument("--conf", type=float, default=0.25, help="Detector confidence")
        command.add_argument("--detector", choices=sorted(DETECTOR_WEIGHTS), default="s")

    rec = sub.add_parser("recognize", help="Find and identify the SKUs on a shelf photo")
    rec.add_argument("image", type=Path)
    add_matching_options(rec)
    rec.add_argument("--out", type=Path, help="Annotated image (default: outputs/<image>_skuscope.jpg)")
    rec.add_argument("--json", type=Path, help="Also write detections as JSON")
    rec.set_defaults(func=cmd_recognize)

    video = sub.add_parser("video", help="Track the products in a video and count each one once")
    video.add_argument("video", type=Path)
    add_matching_options(video)
    video.add_argument("--out", type=Path, help="Annotated video (default: outputs/<video>_skuscope.mp4)")
    video.add_argument("--json", type=Path, help="Also write the counts as JSON")
    video.add_argument(
        "--motion-region",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("LEFT", "RIGHT"),
        help="Estimate the camera motion only from this part of the width, as fractions (e.g. 0 0.5 for the left shelf)",
    )
    video.set_defaults(func=cmd_video)

    app = sub.add_parser("app", help="Launch the Gradio web demo")
    app.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    app.add_argument("--detector", choices=sorted(DETECTOR_WEIGHTS), default="s")
    app.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    app.set_defaults(func=cmd_app)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
