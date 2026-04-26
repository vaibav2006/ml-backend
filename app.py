from __future__ import annotations

"""Main FastAPI service for training, prediction, explainability, and assistant endpoints.

This file orchestrates the full backend flow:
1) load/train the manual CNN,
2) preprocess image and run prediction,
3) build explainable payloads (maps, matrices, Grad-CAM, logs),
4) return frontend-ready JSON.
"""

import json
import os
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np

from dataset_builder import LABELS, load_or_generate_dataset
from image_utils import (
    array_to_base64_png,
    assess_cloud_image,
    encode_feature_maps,
    heatmap_to_base64_png,
    matrix_preview,
    overlay_heatmap_on_grayscale,
    preprocess_cloud_image,
    rgb_array_to_base64_png,
    stats_for
)
from manual_cnn import ManualRainfallCNN
from novelty_utils import build_farmer_alert
from assistant_service import build_dynamic_assistant_response


app = FastAPI(title="V2S Rainfall Prediction ML Service")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for testing (later restrict to your frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
WEIGHTS_PATH = ARTIFACTS_DIR / "manual_cnn_weights.npz"
METADATA_PATH = ARTIFACTS_DIR / "manual_cnn_metadata.json"
PROJECT_PAPER_URL = "/v2s-ieee-paper.html"
BASE_PAPER_URL = "https://www.techscience.com/cmc/v72n2/47227/html"

MODEL: ManualRainfallCNN | None = None
MODEL_METADATA: dict | None = None


def log(message: str, payload: dict | None = None) -> None:
    """Structured logger used across backend pipeline steps."""
    if payload is None:
        print(f"[ml-service] {message}", flush=True)
        return
    print(f"[ml-service] {message}: {json.dumps(payload)}", flush=True)


def default_train_options(payload: dict | None = None) -> dict:
    """Build training options with sensible defaults for demo usage."""
    payload = payload or {}
    return {
        "epochs": int(payload.get("epochs", 6)),
        "learningRate": float(payload.get("learningRate", 0.01)),
        "datasetPath": (payload.get("datasetPath") or "").strip() or None,
        "samplesPerClass": int(payload.get("samplesPerClass", 24))
    }


def training_notes(model: ManualRainfallCNN, source: str) -> list[str]:
    """Generate human-readable notes shown in paper/model info sections."""
    return [
        "The attached PDF could not be text-extracted directly because most text was converted to vector drawing commands.",
        "This implementation follows the accessible paper structure available online: grayscale 28x28 input, scale-aware preprocessing, CNN convolution, ReLU, pooling, and softmax classification.",
        f"The current trainer is using the '{source}' dataset source.",
        "All forward and backward passes are coded directly in numpy. No TensorFlow, PyTorch, Keras, or sklearn estimators are used for the CNN."
    ]


def build_metadata(model: ManualRainfallCNN, options: dict, dataset: dict, history: list[dict]) -> dict:
    """Assemble metadata saved alongside model weights."""
    return {
        "modelName": "V2S Rainfall Prediction And Crop Safety Portal",
        "paperAlignment": {
            "referenceUrl": PROJECT_PAPER_URL,
            "basePaperUrl": BASE_PAPER_URL,
            "notes": training_notes(model, dataset["source"])
        },
        "architecture": model.architecture_summary(),
        "formulas": model.formulas,
        "trainingConfig": options,
        "history": history,
        "datasetSource": dataset["source"],
        "generatedDataPath": str(GENERATED_DIR),
        "classLabels": LABELS
    }


def load_saved_model() -> bool:
    """Load existing model artifacts from disk if available."""
    global MODEL, MODEL_METADATA
    if not WEIGHTS_PATH.exists() or not METADATA_PATH.exists():
        return False

    MODEL, MODEL_METADATA = ManualRainfallCNN.load(str(WEIGHTS_PATH), str(METADATA_PATH))
    MODEL_METADATA = normalize_metadata_links(MODEL_METADATA)
    log("Loaded saved CNN model", {"weights": str(WEIGHTS_PATH)})
    return True


def normalize_metadata_links(metadata: dict) -> dict:
    """Ensure paper links always point to project-specific and base-paper URLs."""
    alignment = metadata.setdefault("paperAlignment", {})
    alignment["referenceUrl"] = PROJECT_PAPER_URL
    alignment["basePaperUrl"] = BASE_PAPER_URL
    return metadata


def ensure_model(auto_train: bool = False) -> tuple[ManualRainfallCNN | None, dict | None, list[str]]:
    """Return a ready model, loading or auto-training when requested."""
    global MODEL, MODEL_METADATA
    logs = []

    if MODEL is not None and MODEL_METADATA is not None:
        return MODEL, MODEL_METADATA, logs

    if load_saved_model():
        logs.append("Loaded the saved CNN weights from the local artifacts folder.")
        return MODEL, MODEL_METADATA, logs

    if auto_train:
        logs.append("No saved model was available, so the service started a fresh training run.")
        result = train_model({})
        logs.extend(result["logs"])
        return MODEL, MODEL_METADATA, logs

    return None, None, ["No trained model is available yet."]


def train_model(payload: dict | None) -> dict:
    """Train model on real/synthetic dataset and persist weights+metadata."""
    global MODEL, MODEL_METADATA
    options = default_train_options(payload)
    log("Training options", options)

    dataset = load_or_generate_dataset(
        dataset_root=options["datasetPath"],
        generated_root=str(GENERATED_DIR),
        image_size=28,
        samples_per_class=options["samplesPerClass"]
    )

    model = ManualRainfallCNN()
    train_result = model.train(
        dataset["trainX"],
        dataset["trainY"],
        dataset["valX"],
        dataset["valY"],
        epochs=options["epochs"],
        learning_rate=options["learningRate"]
    )

    logs = []
    logs.extend(dataset["logs"])
    logs.append(
        f"Training used {len(dataset['trainX'])} training examples and {len(dataset['valX'])} validation examples."
    )

    for epoch in train_result["history"]:
        line = (
            f"Epoch {epoch['epoch']}: "
            f"train loss {epoch['trainLoss']:.4f}, train accuracy {epoch['trainAccuracy']:.4f}, "
            f"validation loss {epoch['validationLoss']:.4f}, validation accuracy {epoch['validationAccuracy']:.4f}"
        )
        logs.append(line)
        log("Epoch summary", epoch)

    metadata = build_metadata(model, options, dataset, train_result["history"])
    metadata = normalize_metadata_links(metadata)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(WEIGHTS_PATH), str(METADATA_PATH), metadata)

    MODEL = model
    MODEL_METADATA = metadata
    log("Training complete", {"validationAccuracy": train_result["finalValidationAccuracy"]})

    return {
        "ok": True,
        "logs": logs,
        "history": train_result["history"],
        "modelInfo": metadata
    }


def dense_stage_payload(name: str, vector: np.ndarray, formula: str, description: str) -> dict:
    """Format dense-layer vectors into frontend-friendly summary cards."""
    vector = np.asarray(vector, dtype=np.float64)
    top_indices = np.argsort(vector)[::-1][:8]
    return {
        "name": name,
        "formula": formula,
        "description": description,
        "values": [round(float(value), 6) for value in vector.tolist()],
        "matrixPreview": matrix_preview(vector, rows=1, cols=12),
        "topActivations": [
            {"index": int(index), "value": round(float(vector[index]), 6)} for index in top_indices
        ]
    }


def rounded_matrix(array: np.ndarray) -> list[list[float]]:
    """Convert numpy matrix into rounded nested list for JSON transport."""
    matrix = np.asarray(array, dtype=np.float64)
    return [[round(float(value), 6) for value in row] for row in matrix.tolist()]


def matrix_progression_payload(progression: list[dict]) -> list[dict]:
    """Package 6x6->4x4->3x3 matrix progression with stats and images."""
    payload = []
    for item in progression:
        matrix_6 = np.asarray(item["sixBySix"], dtype=np.float64)
        matrix_4 = np.asarray(item["fourByFour"], dtype=np.float64)
        matrix_3 = np.asarray(item["threeByThree"], dtype=np.float64)
        payload.append(
            {
                "filterIndex": int(item["filterIndex"]),
                "method": item["method"],
                "formulas": item["formulas"],
                "sixBySix": {
                    "matrix": rounded_matrix(matrix_6),
                    "stats": stats_for(matrix_6),
                    "imageBase64": array_to_base64_png(matrix_6)
                },
                "fourByFour": {
                    "matrix": rounded_matrix(matrix_4),
                    "stats": stats_for(matrix_4),
                    "imageBase64": array_to_base64_png(matrix_4)
                },
                "threeByThree": {
                    "matrix": rounded_matrix(matrix_3),
                    "stats": stats_for(matrix_3),
                    "imageBase64": array_to_base64_png(matrix_3)
                }
            }
        )
    return payload


def summarize_focus_regions(heatmap: np.ndarray) -> list[dict]:
    """Rank Grad-CAM quadrant regions by average attention score."""
    height, width = heatmap.shape
    half_h = height // 2
    half_w = width // 2
    regions = [
        ("Upper Left", heatmap[:half_h, :half_w]),
        ("Upper Right", heatmap[:half_h, half_w:]),
        ("Lower Left", heatmap[half_h:, :half_w]),
        ("Lower Right", heatmap[half_h:, half_w:])
    ]

    summaries = []
    for name, region in regions:
        summaries.append(
            {
                "name": name,
                "score": round(float(np.mean(region)), 6)
            }
        )

    return sorted(summaries, key=lambda item: item["score"], reverse=True)


def build_xai_payload(processed_image: np.ndarray, grad_cam: dict) -> dict:
    """Build complete explainability payload (heatmap, overlay, top filters)."""
    heatmap = grad_cam["heatmap"]
    overlay = overlay_heatmap_on_grayscale(processed_image, heatmap, alpha=0.48)
    region_summary = summarize_focus_regions(heatmap)
    top_regions = region_summary[:2]
    top_filters = np.argsort(np.abs(np.asarray(grad_cam["alphas"])))[::-1][:3]

    focus_summary = (
        "The model focused on these cloud regions (dark cumulonimbus formations), especially "
        + " and ".join(region["name"].lower() for region in top_regions)
        + "."
    )

    return {
        "title": "Why did AI predict rain? - Explainable AI",
        "highlight": "The model focused on these cloud regions (dark cumulonimbus formations).",
        "summary": focus_summary,
        "innovationBadge": "First free XAI rainfall tool",
        "sdgBadge": "SDG 13 - Climate Action",
        "formula": "L_GradCAM^c = ReLU(sum_k alpha_k^c * A^k), alpha_k^c = avg(dy^c / dA^k)",
        "heatmapBase64": heatmap_to_base64_png(heatmap),
        "overlayBase64": rgb_array_to_base64_png(overlay),
        "heatmapStats": stats_for(heatmap),
        "legend": [
            {"label": "Low attention", "color": "#1d4ed8"},
            {"label": "Medium attention", "color": "#22c55e"},
            {"label": "High attention", "color": "#facc15"},
            {"label": "Strongest attention", "color": "#ef4444"}
        ],
        "alphaWeights": grad_cam["alphas"],
        "topFilters": [
            {
                "filter": int(index) + 1,
                "weight": round(float(grad_cam["alphas"][index]), 6)
            }
            for index in top_filters
        ],
        "focusRegions": region_summary
    }


def build_math_console_lines(layer_math_trace: list[dict]) -> list[str]:
    """Flatten layer math trace into line-by-line console logs."""
    lines = []
    for item in layer_math_trace:
        lines.append(f"{item['name']}: {item['equation']}")
        for detail in item.get("details", []):
            lines.append(f"  - {detail}")
        for contribution in item.get("topContributions", []):
            lines.append(
                "  - Dense contribution index "
                f"{contribution['index']}: weight={contribution['weight']}, "
                f"activation={contribution['activation']}, contribution={contribution['contribution']}"
            )
    return lines


def build_backend_readiness_notes(metadata: dict | None, model_logs: list[str]) -> list[str]:
    """Compose backend readiness status lines displayed in system notes card."""
    notes = [
        "Backend services completed the prediction request successfully.",
        "Image preprocessing, inference, and response generation are operational."
    ]
    if metadata:
        architecture = metadata.get("architecture", {})
        input_size = architecture.get("inputSize")
        pool_size = architecture.get("poolSize")
        pool_stride = architecture.get("poolStride")
        if input_size is not None:
            notes.append(f"Configured input size: {input_size}x{input_size}.")
        if pool_size is not None and pool_stride is not None:
            notes.append(f"Pooling: {pool_size}x{pool_size} with stride {pool_stride}.")

    loaded_message = next(
        (line for line in model_logs if "Loaded the saved CNN weights" in line),
        None
    )
    if loaded_message:
        notes.append(loaded_message)
    return notes


def build_training_lifecycle_notes(model_logs: list[str]) -> list[str]:
    """Extract training lifecycle updates from runtime model logs."""
    training_lines = []
    for line in model_logs:
        normalized = line.lower()
        if "training" in normalized or "epoch" in normalized or "dataset" in normalized:
            training_lines.append(line)

    if training_lines:
        return training_lines

    if any("loaded the saved cnn weights" in line.lower() for line in model_logs):
        return ["No retraining was required. The backend reused the saved CNN weights."]

    return ["Training lifecycle did not change during this prediction request."]


def error_response(message: str, status_code: int) -> JSONResponse:
    """Standardized error payload used by gateway and frontend."""
    return JSONResponse(status_code=status_code, content={"error": message})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return upload/body validation errors in consistent JSON format."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed.",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled errors so clients never receive plain text/HTML 500 responses."""
    log("Unhandled exception", {"type": exc.__class__.__name__, "message": str(exc)})
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal server error: {str(exc)}"}
    )


@app.get("/")
def root() -> dict:
    """Small root endpoint so Render URL checks don't show plain Not Found."""
    return {
        "ok": True,
        "service": "ml_service",
        "framework": "FastAPI",
        "health": "/health"
    }


@app.get("/health")
def health() -> dict:
    """Health endpoint: confirms service status and model readiness."""
    model, metadata, logs = ensure_model(auto_train=False)
    return {
        "ok": True,
        "modelReady": model is not None,
        "logs": logs,
        "referencePaper": PROJECT_PAPER_URL,
        "architecture": metadata["architecture"] if metadata else None
    }


@app.get("/model-info")
def model_info() -> dict:
    """Expose architecture/formulas and paper alignment details for frontend."""
    model, metadata, logs = ensure_model(auto_train=False)
    if model is None or metadata is None:
        preview_model = ManualRainfallCNN()
        return {
            "ok": True,
            "modelReady": False,
            "logs": logs,
            "modelInfo": {
                "modelName": "V2S Rainfall Prediction And Crop Safety Portal",
                "architecture": preview_model.architecture_summary(),
                "formulas": preview_model.formulas,
                "paperAlignment": {
                    "referenceUrl": PROJECT_PAPER_URL,
                    "basePaperUrl": BASE_PAPER_URL,
                    "notes": [
                        "The app is ready, but the model has not been trained yet.",
                        "Upload an image and click predict to auto-initialize the model, or call /api/train for manual retraining."
                    ]
                },
                "classLabels": LABELS
            }
        }

    metadata = normalize_metadata_links(metadata)
    return {"ok": True, "modelReady": True, "logs": logs, "modelInfo": metadata}


@app.post("/train")
def train(payload: dict | None = Body(default=None)) -> dict:
    """Manual train endpoint for explicit retraining requests."""
    result = train_model(payload or {})
    return result


@app.post("/predict")
async def predict(image: UploadFile | None = File(default=None), cropType: str = Form(default="Rice")) -> dict:
    """Main inference endpoint: validate, preprocess, predict, explain, and respond."""
    if image is None:
        return error_response("Upload a sky/cloud/atmosphere image with the form field name 'image'.", 400)

    model, metadata, model_logs = ensure_model(auto_train=True)
    if model is None or metadata is None:
        return error_response("Model initialization failed.", 500)

    file_bytes = await image.read()
    if not file_bytes:
        return error_response("The uploaded image was empty.", 400)
    crop_type = cropType or "Rice"

    upload_validation = assess_cloud_image(file_bytes)

    preprocess_payload = preprocess_cloud_image(file_bytes, image_size=28)
    tensor = preprocess_payload["processed"][np.newaxis, :, :]
    inference = model.predict_with_intermediates(tensor)

    rain_probability = float(inference["probabilities"][1])
    no_rain_probability = float(inference["probabilities"][0])
    predicted_index = int(inference["classIndex"])
    confidence = float(inference["probabilities"][predicted_index])
    xai_payload = build_xai_payload(preprocess_payload["processed"], inference["gradCam"])
    farmer_alert = build_farmer_alert(
        crop_type=crop_type,
        prediction_label=LABELS[predicted_index],
        confidence=confidence,
        rain_probability=rain_probability,
        no_rain_probability=no_rain_probability
    )
    math_console_logs = build_math_console_lines(inference["layerMathTrace"])

    layer_groups = [
        encode_feature_maps(
            "Convolution Layer",
            inference["convMaps"],
            model.formulas["convolution"],
            "Each filter slides over the processed cloud image to detect edges and cloud textures."
        ),
        encode_feature_maps(
            "ReLU Activation",
            inference["reluMaps"],
            model.formulas["relu"],
            "Negative responses are clipped to zero so strong cloud patterns stand out."
        ),
        encode_feature_maps(
            "Max Pooling",
            inference["poolMaps"],
            model.formulas["pool"],
            "The strongest response in each local window is preserved, reducing spatial size while keeping dominant structure."
        ),
        encode_feature_maps(
            "Flatten Layer",
            [inference["flattened"].reshape(1, -1)],
            "f = vec(P), where P in R^(4x13x13), so f in R^676",
            "All 4 x 13 x 13 pooled maps are unrolled into one ordered 1D vector of 676 values."
        )
    ]

    # Expose full 13x13 max-pooling matrices so users can inspect exact pooled outputs.
    for index, pooling_map in enumerate(layer_groups[2]["maps"]):
        pooling_map["fullMatrix"] = rounded_matrix(inference["poolMaps"][index])

    # Expose full flatten vector (1x676) so users can inspect complete ordered flatten output.
    flatten_matrix = inference["flattened"].reshape(1, -1)
    layer_groups[3]["maps"][0]["fullMatrix"] = rounded_matrix(flatten_matrix)
    layer_groups[3]["maps"][0]["fullVector"] = [round(float(value), 6) for value in inference["flattened"].tolist()]

    dense_groups = [
        dense_stage_payload(
            "Hidden Dense Activations",
            inference["denseVector"],
            model.formulas["dense"],
            "The pooled feature maps are flattened and projected into a dense reasoning layer."
        ),
        dense_stage_payload(
            "Softmax Logits",
            inference["logits"],
            "z = W_hidden * h + b",
            "Final class scores before normalization into probabilities."
        )
    ]

    logs = []
    logs.extend(model_logs)
    logs.append(upload_validation["message"])
    logs.extend(preprocess_payload["logs"])
    logs.append(
        f"Convolution created {model.conv_filters} feature maps of size {model.input_size}x{model.input_size}."
    )
    logs.append(
        f"Max pooling reduced each map to {model.pooled_size}x{model.pooled_size} before dense classification."
    )
    logs.append(
        f"Flatten converted 4x{model.pooled_size}x{model.pooled_size} into a 1x{len(inference['flattened'])} vector."
    )
    logs.append(
        f"Softmax probabilities: no-rain={no_rain_probability:.4f}, rain={rain_probability:.4f}."
    )
    logs.append(
        f"Predicted label '{LABELS[predicted_index]}' with confidence {confidence:.4f}."
    )
    logs.append(
        "User selection received: "
        f"crop='{farmer_alert['userSelection']['cropType']}'."
    )
    logs.append(
        f"Rain confidence is {rain_probability:.4f}; alert threshold is {farmer_alert['alertThreshold']:.2f}."
    )
    logs.append(xai_payload["summary"])
    logs.append(
        f"Smart Crop Rain Advisory generated for crop '{farmer_alert['cropType']}' with risk band '{farmer_alert['riskBand']}'."
    )
    logs.append(
        f"Confidence chart prepared with no-rain={farmer_alert['confidenceChart']['points'][0]['value']:.4f}, "
        f"rain={farmer_alert['confidenceChart']['points'][1]['value']:.4f}, "
        f"prediction={farmer_alert['confidenceChart']['points'][2]['value']:.4f}."
    )
    logs.append(
        "Matrix mentor trace prepared: 6x6 activation patch -> 4x4 split matrix -> 3x3 split matrix "
        "for each of the four filters."
    )
    if farmer_alert["shouldNotify"]:
        logs.append("Rain confidence crossed 70%, so the toast alert workflow is enabled.")
    else:
        logs.append("Rain confidence stayed below the 70% alert threshold, so no toast alert is triggered.")

    for line in logs:
        log(line)
    for line in math_console_logs:
        log(f"Math trace :: {line}")

    system_notes = {
        "backendReadiness": build_backend_readiness_notes(metadata, model_logs),
        "trainingLog": build_training_lifecycle_notes(model_logs),
        "predictionLog": logs
    }

    return {
        "ok": True,
        "prediction": {
            "label": LABELS[predicted_index],
            "classIndex": predicted_index,
            "confidence": round(confidence, 6),
            "probabilities": [
                {"label": LABELS[0], "value": round(no_rain_probability, 6)},
                {"label": LABELS[1], "value": round(rain_probability, 6)}
            ]
        },
        "logs": logs,
        "modelLifecycleLogs": model_logs,
        "systemNotes": system_notes,
        "modelInfo": metadata,
        "uploadValidation": upload_validation,
        "uploadedPreviewBase64": preprocess_payload["originalPreviewBase64"],
        "preprocessingStages": preprocess_payload["preprocessingStages"],
        "layerGroups": layer_groups,
        "denseGroups": dense_groups,
        "xai": xai_payload,
        "farmerAlert": farmer_alert,
        "confidenceChart": farmer_alert["confidenceChart"],
        "mathematics": {
            "inputTensorPreview": matrix_preview(preprocess_payload["processed"]),
            "inputTensor28x28": rounded_matrix(preprocess_payload["processed"]),
            "flattenVector676": [round(float(value), 6) for value in inference["flattened"].tolist()],
            "flattenVectorShape": [1, int(len(inference["flattened"]))],
            "flattenVectorImageBase64": array_to_base64_png(inference["flattened"].reshape(1, -1)),
            "flattenFormula": "f = vec(P), where P in R^(4x13x13), so f in R^676",
            "softmaxFormula": model.formulas["softmax"],
            "lossFormula": model.formulas["loss"],
            "layerMathTrace": inference["layerMathTrace"],
            "consoleLines": math_console_logs,
            "matrixProgression": matrix_progression_payload(inference["matrixProgression"]),
            "denseExplanation": inference["denseExplanation"]
        }
    }


@app.post("/assistant")
def assistant(payload: dict | None = Body(default=None)) -> dict:
    """Assistant endpoint: dynamic contextual answers with optional web snippets."""
    payload = payload or {}
    question = str(payload.get("question") or "").strip()
    context = payload.get("context") or {}
    if not question:
        return error_response("Assistant question is required.", 400)

    answer_payload = build_dynamic_assistant_response(question, context)
    return {
        "ok": True,
        "answer": answer_payload.get("answer", ""),
        "sources": answer_payload.get("sources", [])
    }


if __name__ == "__main__":
    """Run FastAPI app via uvicorn when executed directly."""
    import uvicorn

    host = os.environ.get("ML_SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("ML_SERVICE_PORT", "8000"))
    debug = os.environ.get("ML_SERVICE_DEBUG", "false").strip().lower() in {"1", "true", "yes"}
    uvicorn.run(app, host=host, port=port, reload=debug)

