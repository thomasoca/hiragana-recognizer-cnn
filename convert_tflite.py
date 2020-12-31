import tensorflow as tf

# print(tf.__version__)
# print(help(tf.lite.TFLiteConverter))
# convert into tflite
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('hiragana_model') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('hiragana.tflite', 'wb') as f:
  f.write(tflite_model)