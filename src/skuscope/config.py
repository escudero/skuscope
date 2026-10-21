"""Pinned model sources."""

# YOLO11 trained on SKU-110K: one class ("object"), it finds products but does
# not know which product is which. That part is DINO's job.
DETECTOR_REPO = "chistopat/sku110k-yolo11-object-detector"
DETECTOR_REVISION = "ee1b8ac34eb3b68969ffa8165e50c43457fe4e35"

# ONNX exports instead of .pt checkpoints: unpickling a third-party .pt can run
# arbitrary code, loading an ONNX graph cannot. Checksums come from the repo.
DETECTOR_WEIGHTS = {
    "n": (
        "weights/sku110k-yolo11-n640.onnx",
        "5810269bf9687ca93b0d4e1bc91cb83ac4311cd48d91c4f6091777721ba083c5",
    ),
    "s": (
        "weights/sku110k-yolo11-s640.onnx",
        "e8bc019d4241cf9486c4b6aaf50d51b652b3a0d641fce1189fbba12136794cd1",
    ),
}

EMBEDDERS = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    # Gated on Hugging Face: accept the license on the model page and run
    # `hf auth login` before using it.
    "dinov3-small": "facebook/dinov3-vits16-pretrain-lvd1689m",
}
