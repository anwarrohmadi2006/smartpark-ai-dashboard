import pickle
import traceback
from pathlib import Path
try:
    import tensorflow as tf
except Exception as e:
    print("TF Import Error:", e)

@tf.keras.utils.register_keras_serializable()
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1, name='score')

    def call(self, x):
        e = self.score(x)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

base_dir = Path("models")
try:
    with open(base_dir / "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    print("Scaler X OK")
    
    model = tf.keras.models.load_model(str(base_dir / "best_model.keras"), custom_objects={
        'TemporalAttention': TemporalAttention,
        'Orthogonal': tf.keras.initializers.Orthogonal,
        'GlorotUniform': tf.keras.initializers.GlorotUniform,
        'Zeros': tf.keras.initializers.Zeros,
        'Ones': tf.keras.initializers.Ones
    }, compile=False)
    print("Model Loaded OK")
except Exception as e:
    print("Error Loading:")
    traceback.print_exc()
