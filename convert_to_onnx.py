"""
Run this ONCE locally to convert the Keras model to ONNX format for Vercel.

    pip install tf2onnx
    python convert_to_onnx.py
"""
import tensorflow as tf
import tf2onnx
import numpy as np

MODEL_PATH  = "lstm_sign_model.keras"
OUTPUT_PATH = "lstm_sign_model.onnx"

model = tf.keras.models.load_model(MODEL_PATH)
seq_len    = 30
n_features = int(model.input_shape[-1])

input_signature = [
    tf.TensorSpec(shape=(None, seq_len, n_features), dtype=tf.float32, name="input")
]

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

with open(OUTPUT_PATH, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Saved {OUTPUT_PATH}  ({len(model_proto.SerializeToString())//1024} KB)")
print(f"Input shape : (batch, {seq_len}, {n_features})")
print(f"Output shape: (batch, {len(np.load('actions.npy', allow_pickle=True))})")
