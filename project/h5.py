import tensorflow as tf

# Load the original model
model = tf.keras.models.load_model("project/model/audio_emotion_rnn.h5")

# Re-save it in the SavedModel format
model.save("model/audio_emotion_rnn_saved_model", save_format="tf")
