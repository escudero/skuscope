"""Gradio web demo: upload a shelf photo, see every product labeled."""

from pathlib import Path

from PIL import Image

from skuscope.catalog import Catalog
from skuscope.detector import ProductDetector
from skuscope.draw import annotate, palette_for
from skuscope.embedder import Embedder
from skuscope.pipeline import count_skus, recognize


def launch(index: Path, detector_size: str, threshold: float, share: bool) -> None:
    import gradio as gr

    catalog = Catalog.load(index)
    detector = ProductDetector(detector_size)
    embedder = Embedder(catalog.embedder)
    colors = palette_for(catalog.skus)

    def run(image: Image.Image | None, threshold: float, conf: float):
        if image is None:
            return None, []
        detector.conf = conf
        detections = recognize(image.convert("RGB"), detector, embedder, catalog, threshold)
        counts = count_skus(detections)
        return annotate(image, detections, colors), [[sku, n] for sku, n in counts.most_common()]

    with gr.Blocks(title="SkuScope") as demo:
        gr.Markdown(
            "# 🔎 SkuScope\n"
            "**YOLO** finds every product on the shelf, **DINO** tells which SKU each one is. "
            f"Catalog: {len(set(catalog.skus))} SKUs, {len(catalog.skus)} reference photos, `{catalog.embedder}`."
        )
        with gr.Row():
            photo = gr.Image(type="pil", label="Shelf photo")
            result = gr.Image(type="pil", label="Recognized SKUs")
        with gr.Row():
            threshold_slider = gr.Slider(0.0, 1.0, value=threshold, step=0.01, label="Similarity threshold (DINO)")
            conf_slider = gr.Slider(0.05, 0.9, value=detector.conf, step=0.05, label="Detection confidence (YOLO)")
        button = gr.Button("Recognize", variant="primary")
        table = gr.Dataframe(headers=["SKU", "Count"], label="Counts")
        button.click(run, [photo, threshold_slider, conf_slider], [result, table])

    demo.launch(share=share)
