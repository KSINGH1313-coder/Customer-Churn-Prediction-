import tensorflow as tf

print(tf.__version__)

model = tf.keras.models.load_model("model.keras")
print("Loaded successfully!")

model.summary()