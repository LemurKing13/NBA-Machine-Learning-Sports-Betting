import numpy as np


def multiclass_brier_score(y_true, probabilities, num_classes=None):
    """Compute multiclass Brier score (lower is better)."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array of shape (n_samples, n_classes).")

    if num_classes is None:
        num_classes = probabilities.shape[1]
    if probabilities.shape[1] != num_classes:
        raise ValueError("probabilities second dimension must match num_classes.")

    one_hot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true, probabilities, n_bins=10):
    """Compute ECE using top-class confidence bins."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array of shape (n_samples, n_classes).")
    if len(y_true) == 0:
        return float("nan")

    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]

        if idx == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        prop = np.mean(in_bin)
        if prop == 0:
            continue

        avg_conf = np.mean(confidences[in_bin])
        avg_acc = np.mean(correct[in_bin])
        ece += prop * abs(avg_acc - avg_conf)

    return float(ece)
