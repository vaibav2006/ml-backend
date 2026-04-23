from __future__ import annotations

"""Manual CNN implementation for rainfall classification.

Important for study:
- The network is implemented directly with NumPy (no deep-learning framework).
- Forward and backward passes are written explicitly for transparency.
- Intermediate tensors are exposed for explainability in the UI.
"""

import json
from pathlib import Path

import numpy as np

from image_utils import matrix_preview


class ManualRainfallCNN:
    """Small educational CNN that predicts `no-rain` vs `rain` from 28x28 input."""

    def __init__(
        self,
        input_size: int = 28,
        conv_filters: int = 4,
        kernel_size: int = 3,
        pool_size: int = 4,
        pool_stride: int = 2,
        hidden_units: int = 24,
        class_count: int = 2,
        seed: int = 7
    ) -> None:
        self.input_size = input_size
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.hidden_units = hidden_units
        self.class_count = class_count
        self.seed = seed

        rng = np.random.default_rng(seed)
        conv_scale = np.sqrt(2.0 / (kernel_size * kernel_size))
        self.conv_weights = rng.normal(
            loc=0.0,
            scale=conv_scale,
            size=(conv_filters, 1, kernel_size, kernel_size)
        )
        self.conv_bias = np.full(conv_filters, 0.2, dtype=np.float64)

        pooled_size = ((input_size - pool_size) // pool_stride) + 1
        self.pooled_size = pooled_size
        flattened_size = conv_filters * pooled_size * pooled_size

        dense_scale = np.sqrt(2.0 / flattened_size)
        self.fc1_weights = rng.normal(
            loc=0.0,
            scale=dense_scale,
            size=(hidden_units, flattened_size)
        )
        self.fc1_bias = np.full(hidden_units, 0.1, dtype=np.float64)

        fc2_scale = np.sqrt(2.0 / hidden_units)
        self.fc2_weights = rng.normal(
            loc=0.0,
            scale=fc2_scale,
            size=(class_count, hidden_units)
        )
        self.fc2_bias = np.zeros(class_count, dtype=np.float64)

        self.training_history: list[dict] = []

    @property
    def formulas(self) -> dict:
        """Expose core equations used in each CNN stage for UI/paper display."""
        return {
            "convolution": "Z_k(i,j) = sum_c sum_u sum_v X_c(i+u,j+v) * W_k,c(u,v) + b_k",
            "relu": "A_k(i,j) = max(0, Z_k(i,j))",
            "pool": "P_k(i,j) = max_{u,v in window} A_k(stride*i+u, stride*j+v)",
            "dense": "h = W_flat * vec(P) + b",
            "softmax": "p_c = exp(z_c) / sum_j exp(z_j)",
            "loss": "L = -sum_c y_c log(p_c)"
        }

    def _conv_forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        """Convolution forward pass with edge padding and learned kernels."""
        padding = self.kernel_size // 2
        padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="edge")
        output = np.zeros((self.conv_filters, self.input_size, self.input_size), dtype=np.float64)

        for filt in range(self.conv_filters):
            for row in range(self.input_size):
                for col in range(self.input_size):
                    value = 0.0
                    for channel in range(x.shape[0]):
                        patch = padded[channel, row : row + self.kernel_size, col : col + self.kernel_size]
                        value += float(np.sum(patch * self.conv_weights[filt, channel]))
                    output[filt, row, col] = value + self.conv_bias[filt]

        cache = (x, padded)
        return output, cache

    def _conv_backward(self, gradient: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backpropagate through convolution and return input/weight/bias gradients."""
        x, padded = cache
        padding = self.kernel_size // 2
        dx_padded = np.zeros_like(padded, dtype=np.float64)
        dw = np.zeros_like(self.conv_weights, dtype=np.float64)
        db = np.zeros_like(self.conv_bias, dtype=np.float64)

        for filt in range(self.conv_filters):
            db[filt] = np.sum(gradient[filt])
            for row in range(self.input_size):
                for col in range(self.input_size):
                    grad_value = gradient[filt, row, col]
                    for channel in range(x.shape[0]):
                        patch = padded[channel, row : row + self.kernel_size, col : col + self.kernel_size]
                        dw[filt, channel] += grad_value * patch
                        dx_padded[channel, row : row + self.kernel_size, col : col + self.kernel_size] += (
                            grad_value * self.conv_weights[filt, channel]
                        )

        dx = dx_padded[:, padding:-padding, padding:-padding]
        return dx, dw, db

    def _relu_forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ReLU activation: keep positive values, zero-out negative values."""
        return np.maximum(0.0, x), x

    def _relu_backward(self, gradient: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """Backpropagate through ReLU using the cached pre-activation values."""
        dx = gradient.copy()
        dx[cache <= 0.0] = 0.0
        return dx

    def _pool_forward(self, x: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        """Max-pooling forward pass with saved argmax positions for backprop."""
        output_size = self.pooled_size
        output = np.zeros((x.shape[0], output_size, output_size), dtype=np.float64)
        positions: list[tuple[int, int, int]] = []

        for channel in range(x.shape[0]):
            for row in range(output_size):
                for col in range(output_size):
                    row_start = row * self.pool_stride
                    col_start = col * self.pool_stride
                    window = x[
                        channel,
                        row_start : row_start + self.pool_size,
                        col_start : col_start + self.pool_size
                    ]
                    max_index = int(np.argmax(window))
                    max_row = row_start + (max_index // self.pool_size)
                    max_col = col_start + (max_index % self.pool_size)
                    output[channel, row, col] = window.reshape(-1)[max_index]
                    positions.append((channel, max_row, max_col))

        return output, positions

    def _pool_backward(self, gradient: np.ndarray, positions: list[tuple[int, int, int]], input_shape: tuple[int, int, int]) -> np.ndarray:
        """Route pooling gradients only to winner positions recorded in forward pass."""
        dx = np.zeros(input_shape, dtype=np.float64)
        cursor = 0
        for channel in range(gradient.shape[0]):
            for row in range(gradient.shape[1]):
                for col in range(gradient.shape[2]):
                    pos_channel, max_row, max_col = positions[cursor]
                    dx[pos_channel, max_row, max_col] += gradient[channel, row, col]
                    cursor += 1
        return dx

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax for two-class probability output."""
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials)

    def forward(self, x: np.ndarray, capture_intermediates: bool = False) -> tuple[np.ndarray, dict]:
        """Run complete forward pass and optionally return all intermediate tensors."""
        conv_z, conv_cache = self._conv_forward(x)
        relu_a, relu_cache = self._relu_forward(conv_z)
        pooled, pool_positions = self._pool_forward(relu_a)

        flattened = pooled.reshape(-1)
        dense_z = self.fc1_weights @ flattened + self.fc1_bias
        dense_a = np.maximum(0.0, dense_z)
        logits = self.fc2_weights @ dense_a + self.fc2_bias
        probabilities = self._softmax(logits)

        cache = {
            "convCache": conv_cache,
            "convZ": conv_z,
            "reluCache": relu_cache,
            "reluA": relu_a,
            "poolPositions": pool_positions,
            "pooled": pooled,
            "flattened": flattened,
            "denseZ": dense_z,
            "denseA": dense_a,
            "logits": logits
        }

        if capture_intermediates:
            return probabilities, cache

        return probabilities, cache

    def train(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        epochs: int = 10,
        learning_rate: float = 0.02
    ) -> dict:
        """Train with per-sample SGD updates and return epoch history."""
        history = []
        rng = np.random.default_rng(self.seed)

        for epoch in range(epochs):
            indices = np.arange(len(train_x))
            rng.shuffle(indices)

            epoch_loss = 0.0
            correct = 0

            for index in indices:
                x = train_x[index]
                y = int(train_y[index])

                probabilities, cache = self.forward(x, capture_intermediates=True)
                loss = -np.log(float(probabilities[y]) + 1e-12)
                epoch_loss += loss
                correct += int(np.argmax(probabilities) == y)

                d_logits = probabilities.copy()
                d_logits[y] -= 1.0

                d_fc2_w = d_logits[:, np.newaxis] @ cache["denseA"][np.newaxis, :]
                d_fc2_b = d_logits

                d_dense_a = self.fc2_weights.T @ d_logits
                d_dense_z = d_dense_a.copy()
                d_dense_z[cache["denseZ"] <= 0.0] = 0.0

                d_fc1_w = d_dense_z[:, np.newaxis] @ cache["flattened"][np.newaxis, :]
                d_fc1_b = d_dense_z
                d_flattened = self.fc1_weights.T @ d_dense_z
                d_pooled = d_flattened.reshape(cache["pooled"].shape)

                d_relu_a = self._pool_backward(d_pooled, cache["poolPositions"], cache["reluA"].shape)
                d_conv_z = self._relu_backward(d_relu_a, cache["reluCache"])
                _, d_conv_w, d_conv_b = self._conv_backward(d_conv_z, cache["convCache"])

                self.fc2_weights -= learning_rate * d_fc2_w
                self.fc2_bias -= learning_rate * d_fc2_b
                self.fc1_weights -= learning_rate * d_fc1_w
                self.fc1_bias -= learning_rate * d_fc1_b
                self.conv_weights -= learning_rate * d_conv_w
                self.conv_bias -= learning_rate * d_conv_b

            train_accuracy = correct / max(len(train_x), 1)
            val_loss, val_accuracy = self.evaluate(val_x, val_y)

            record = {
                "epoch": epoch + 1,
                "trainLoss": round(epoch_loss / max(len(train_x), 1), 6),
                "trainAccuracy": round(train_accuracy, 6),
                "validationLoss": round(val_loss, 6),
                "validationAccuracy": round(val_accuracy, 6)
            }
            history.append(record)

        self.training_history = history
        return {
            "history": history,
            "finalValidationAccuracy": history[-1]["validationAccuracy"] if history else 0.0
        }

    def evaluate(self, x_set: np.ndarray, y_set: np.ndarray) -> tuple[float, float]:
        """Evaluate average loss and accuracy on a dataset split."""
        if len(x_set) == 0:
            return 0.0, 0.0

        total_loss = 0.0
        correct = 0
        for x, y in zip(x_set, y_set):
            probabilities, _ = self.forward(x, capture_intermediates=True)
            total_loss += -np.log(float(probabilities[int(y)]) + 1e-12)
            correct += int(np.argmax(probabilities) == int(y))

        return total_loss / len(x_set), correct / len(x_set)

    def predict_with_intermediates(self, x: np.ndarray) -> dict:
        """Predict class and return rich explainability payload for frontend cards."""
        probabilities, cache = self.forward(x, capture_intermediates=True)
        prediction = int(np.argmax(probabilities))
        grad_cam = self._grad_cam_from_cache(cache, prediction)
        layer_math_trace = self._build_layer_math_trace(probabilities, cache, grad_cam)
        matrix_progression = self._build_matrix_progression(cache)
        dense_explanation = self._build_dense_explanation(cache)
        return {
            "classIndex": prediction,
            "probabilities": probabilities,
            "convMaps": cache["convZ"],
            "reluMaps": cache["reluA"],
            "poolMaps": cache["pooled"],
            "denseVector": cache["denseA"],
            "logits": cache["logits"],
            "denseZ": cache["denseZ"],
            "flattened": cache["flattened"],
            "gradCam": grad_cam,
            "layerMathTrace": layer_math_trace,
            "matrixProgression": matrix_progression,
            "denseExplanation": dense_explanation
        }

    @staticmethod
    def _max_pool_stride1(matrix: np.ndarray, window_size: int) -> np.ndarray:
        """Utility pooling used for the mentor-requested 6x6->4x4->3x3 progression."""
        rows, cols = matrix.shape
        out_rows = rows - window_size + 1
        out_cols = cols - window_size + 1
        output = np.zeros((out_rows, out_cols), dtype=np.float64)
        for row in range(out_rows):
            for col in range(out_cols):
                window = matrix[row : row + window_size, col : col + window_size]
                output[row, col] = float(np.max(window))
        return output

    @staticmethod
    def _peak_centered_patch(matrix: np.ndarray, patch_size: int = 6) -> np.ndarray:
        """Extract a patch centered near the strongest activation in a feature map."""
        rows, cols = matrix.shape
        peak_row, peak_col = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        start_row = max(0, min(rows - patch_size, peak_row - (patch_size // 2)))
        start_col = max(0, min(cols - patch_size, peak_col - (patch_size // 2)))
        return matrix[start_row : start_row + patch_size, start_col : start_col + patch_size]

    def _build_matrix_progression(self, cache: dict) -> list[dict]:
        """Build per-filter matrix traces used in the explainable CNN section."""
        progression = []
        for filter_index in range(self.conv_filters):
            relu_map = np.asarray(cache["reluA"][filter_index], dtype=np.float64)
            matrix_6 = self._peak_centered_patch(relu_map, patch_size=6)
            matrix_4 = self._max_pool_stride1(matrix_6, window_size=3)
            matrix_3 = self._max_pool_stride1(matrix_4, window_size=2)

            progression.append(
                {
                    "filterIndex": filter_index + 1,
                    "method": (
                        "Peak-centered 6x6 activation crop from ReLU map, then stride-1 max pooling "
                        "with 3x3 window to produce 4x4, followed by stride-1 max pooling with 2x2 "
                        "window to produce 3x3."
                    ),
                    "formulas": {
                        "sixBySix": "M6 = ReLU_map[r:r+6, c:c+6] around strongest activation",
                        "fourByFour": "M4(i,j) = max(M6[i:i+3, j:j+3])",
                        "threeByThree": "M3(i,j) = max(M4[i:i+2, j:j+2])"
                    },
                    "sixBySix": matrix_6,
                    "fourByFour": matrix_4,
                    "threeByThree": matrix_3
                }
            )
        return progression

    def _build_dense_explanation(self, cache: dict) -> list[dict]:
        """Explain top hidden-neuron contributions from flattened pooled features."""
        dense_values = np.asarray(cache["denseA"], dtype=np.float64)
        dense_pre_activation = np.asarray(cache["denseZ"], dtype=np.float64)
        flattened = np.asarray(cache["flattened"], dtype=np.float64)
        top_neurons = np.argsort(np.abs(dense_values))[::-1][:3]

        details = []
        for neuron_index in top_neurons:
            contributions = self.fc1_weights[neuron_index] * flattened
            top_indices = np.argsort(np.abs(contributions))[::-1][:8]
            details.append(
                {
                    "neuronIndex": int(neuron_index),
                    "preActivation": round(float(dense_pre_activation[neuron_index]), 6),
                    "postActivation": round(float(dense_values[neuron_index]), 6),
                    "bias": round(float(self.fc1_bias[neuron_index]), 6),
                    "equation": (
                        f"h_{int(neuron_index)} = ReLU(sum_j(w_{int(neuron_index)},j * flat_j) + b_{int(neuron_index)})"
                    ),
                    "topContributions": [
                        {
                            "flatIndex": int(idx),
                            "weight": round(float(self.fc1_weights[neuron_index, idx]), 6),
                            "flatValue": round(float(flattened[idx]), 6),
                            "contribution": round(float(contributions[idx]), 6)
                        }
                        for idx in top_indices
                    ]
                }
            )
        return details

    def _grad_cam_from_cache(self, cache: dict, class_index: int) -> dict:
        """Compute Grad-CAM style heatmap using cached activations and class gradient."""
        class_signal = np.zeros(self.class_count, dtype=np.float64)
        class_signal[class_index] = 1.0

        grad_dense_a = self.fc2_weights.T @ class_signal
        grad_dense_z = grad_dense_a.copy()
        grad_dense_z[cache["denseZ"] <= 0.0] = 0.0

        grad_flattened = self.fc1_weights.T @ grad_dense_z
        grad_pooled = grad_flattened.reshape(cache["pooled"].shape)
        grad_relu_a = self._pool_backward(grad_pooled, cache["poolPositions"], cache["reluA"].shape)

        alphas = np.mean(grad_relu_a, axis=(1, 2))
        heatmap = np.zeros(cache["reluA"][0].shape, dtype=np.float64)
        for index, alpha in enumerate(alphas):
            heatmap += float(alpha) * cache["reluA"][index]

        heatmap = np.maximum(heatmap, 0.0)
        maximum = float(np.max(heatmap))
        if maximum > 1e-12:
            heatmap = heatmap / maximum

        return {
            "alphas": [round(float(value), 6) for value in alphas.tolist()],
            "heatmap": heatmap,
            "gradientMaps": grad_relu_a
        }

    def _build_layer_math_trace(self, probabilities: np.ndarray, cache: dict, grad_cam: dict) -> list[dict]:
        """Assemble human-readable numeric examples from each model layer."""
        _, padded = cache["convCache"]
        conv_patch = padded[0, 0 : self.kernel_size, 0 : self.kernel_size]
        conv_kernel = self.conv_weights[0, 0]
        conv_weighted_sum = float(np.sum(conv_patch * conv_kernel))
        conv_result = float(cache["convZ"][0, 0, 0])

        relu_input = conv_result
        relu_output = float(cache["reluA"][0, 0, 0])

        pool_window = cache["reluA"][0, 0 : self.pool_size, 0 : self.pool_size]
        pool_max = float(cache["pooled"][0, 0, 0])
        pool_position = np.unravel_index(int(np.argmax(pool_window)), pool_window.shape)

        contributions = self.fc1_weights[0] * cache["flattened"]
        top_contribution_indices = np.argsort(np.abs(contributions))[::-1][:6]
        top_contributions = [
            {
                "index": int(index),
                "weight": round(float(self.fc1_weights[0, index]), 6),
                "activation": round(float(cache["flattened"][index]), 6),
                "contribution": round(float(contributions[index]), 6)
            }
            for index in top_contribution_indices
        ]

        shifted_logits = cache["logits"] - np.max(cache["logits"])
        exponentials = np.exp(shifted_logits)

        grad_cam_weights = grad_cam["alphas"]
        top_grad_filters = np.argsort(np.abs(np.asarray(grad_cam_weights)))[::-1][:3]

        return [
            {
                "name": "Convolution Example",
                "formula": self.formulas["convolution"],
                "equation": (
                    f"Z_1(0,0) = sum(patch * kernel) + b = {conv_weighted_sum:.4f} + "
                    f"{float(self.conv_bias[0]):.4f} = {conv_result:.4f}"
                ),
                "details": [
                    "Patch and kernel are both 3x3 for the first convolution filter.",
                    f"The first filter bias is {float(self.conv_bias[0]):.4f}.",
                    "This numeric example comes from filter 1 at output position (0, 0)."
                ],
                "patchPreview": matrix_preview(conv_patch, rows=self.kernel_size, cols=self.kernel_size),
                "kernelPreview": matrix_preview(conv_kernel, rows=self.kernel_size, cols=self.kernel_size),
                "result": round(conv_result, 6)
            },
            {
                "name": "ReLU Example",
                "formula": self.formulas["relu"],
                "equation": f"A_1(0,0) = max(0, {relu_input:.4f}) = {relu_output:.4f}",
                "details": [
                    "ReLU keeps positive activations and removes negative responses.",
                    "This helps the CNN preserve strong cloud textures while suppressing weak inverse responses."
                ],
                "result": round(relu_output, 6)
            },
            {
                "name": "Max Pooling Example",
                "formula": self.formulas["pool"],
                "equation": (
                    f"P_1(0,0) = max(window) = {pool_max:.4f} at local window position "
                    f"({pool_position[0]}, {pool_position[1]})"
                ),
                "details": [
                    f"The pooling window is {self.pool_size}x{self.pool_size} with stride {self.pool_stride}.",
                    "Only the strongest activation in the window is preserved."
                ],
                "windowPreview": matrix_preview(pool_window, rows=self.pool_size, cols=self.pool_size),
                "result": round(pool_max, 6)
            },
            {
                "name": "Dense Layer Example",
                "formula": self.formulas["dense"],
                "equation": (
                    f"h_1 = sum_j (w_1j * flat_j) + b_1 = {float(cache['denseZ'][0]):.4f}, "
                    f"ReLU(h_1) = {float(cache['denseA'][0]):.4f}"
                ),
                "details": [
                    "The pooled feature maps are flattened into one long vector.",
                    "The top contributors below are the strongest numeric terms for hidden neuron 1."
                ],
                "topContributions": top_contributions,
                "result": round(float(cache["denseA"][0]), 6)
            },
            {
                "name": "Softmax Output Example",
                "formula": self.formulas["softmax"],
                "equation": (
                    f"softmax([{float(cache['logits'][0]):.4f}, {float(cache['logits'][1]):.4f}]) = "
                    f"[{float(probabilities[0]):.4f}, {float(probabilities[1]):.4f}]"
                ),
                "details": [
                    f"Shifted logits: {[round(float(value), 6) for value in shifted_logits.tolist()]}",
                    f"Exponentials: {[round(float(value), 6) for value in exponentials.tolist()]}"
                ],
                "result": [round(float(value), 6) for value in probabilities.tolist()]
            },
            {
                "name": "Grad-CAM Explainability",
                "formula": "L_GradCAM^c = ReLU(sum_k alpha_k^c * A^k), alpha_k^c = avg(dy^c / dA^k)",
                "equation": (
                    "Top alpha weights for the predicted class: "
                    + ", ".join(
                        f"filter {int(index) + 1} -> {float(grad_cam_weights[index]):.4f}"
                        for index in top_grad_filters
                    )
                ),
                "details": [
                    "A larger positive alpha means the corresponding feature map contributed more strongly to the chosen class.",
                    "The weighted sum of those feature maps forms the attention heatmap."
                ],
                "result": grad_cam_weights
            }
        ]

    def save(self, weights_path: str, metadata_path: str, metadata: dict) -> None:
        """Persist model weights and metadata JSON for fast reload in future runs."""
        np.savez(
            weights_path,
            conv_weights=self.conv_weights,
            conv_bias=self.conv_bias,
            fc1_weights=self.fc1_weights,
            fc1_bias=self.fc1_bias,
            fc2_weights=self.fc2_weights,
            fc2_bias=self.fc2_bias
        )
        Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, weights_path: str, metadata_path: str) -> tuple["ManualRainfallCNN", dict]:
        """Load trained weights and metadata to reconstruct the same architecture."""
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        model = cls(
            input_size=metadata["architecture"]["inputSize"],
            conv_filters=metadata["architecture"]["convFilters"],
            kernel_size=metadata["architecture"]["kernelSize"],
            pool_size=metadata["architecture"]["poolSize"],
            pool_stride=metadata["architecture"]["poolStride"],
            hidden_units=metadata["architecture"]["hiddenUnits"],
            class_count=metadata["architecture"]["classCount"],
            seed=metadata["architecture"]["seed"]
        )

        payload = np.load(weights_path)
        model.conv_weights = payload["conv_weights"]
        model.conv_bias = payload["conv_bias"]
        model.fc1_weights = payload["fc1_weights"]
        model.fc1_bias = payload["fc1_bias"]
        model.fc2_weights = payload["fc2_weights"]
        model.fc2_bias = payload["fc2_bias"]
        model.training_history = metadata.get("history", [])
        return model, metadata

    def architecture_summary(self) -> dict:
        """Return model configuration fields consumed by API/frontend."""
        return {
            "inputSize": self.input_size,
            "convFilters": self.conv_filters,
            "kernelSize": self.kernel_size,
            "poolSize": self.pool_size,
            "poolStride": self.pool_stride,
            "hiddenUnits": self.hidden_units,
            "classCount": self.class_count,
            "seed": self.seed,
            "pooledSize": self.pooled_size
        }
