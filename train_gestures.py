"""
Stage 2: train a small neural network to recognize your gestures.

This replaces the hand-written rules in app.js with a learned model, which is what the original
team did (they reported 99% accuracy on their test gestures).

Steps:
  1. Run the app, open "Train gestures" at the bottom right.
  2. Pick a label (draw, hover, pinch, palm, fist), hold your hand in that gesture,
     and hold the R key for a few seconds. Move your hand around and tilt it a bit.
     Aim for 300+ samples per gesture.
  3. Click Download. Move gestures.json into this folder.
  4. Run:  python train_gestures.py
  5. Restart server.py and reload the page. The pill at the top will say "neural network".
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
samples = json.loads((ROOT / "gestures.json").read_text())
X = np.array([s["features"] for s in samples])
y = np.array([s["label"] for s in samples])
print("Samples per gesture:", dict(Counter(y)))

# Hold back 20% to test on data the model never saw
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

scaler = StandardScaler().fit(X_train)
net = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=0)
net.fit(scaler.transform(X_train), y_train)

print(classification_report(y_test, net.predict(scaler.transform(X_test))))

# Save in a simple format app.js can read: weights and biases for each layer
model = {
    "labels": list(net.classes_),
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in zip(net.coefs_, net.intercepts_)],
}
(ROOT / "gesture_model.json").write_text(json.dumps(model))
print("Saved gesture_model.json. Restart server.py and reload the page.")
