import tensorflow as tf

model = tf.keras.models.load_model("model.keras")

model.save("model_new.keras")

print("Done!")