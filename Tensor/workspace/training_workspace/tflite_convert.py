import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('TFLite_inference_graph_2/saved_model/') # path to the SavedModel directory
# enable TensorFlow ops along with enable TensorFlow Lite ops
#converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]
tflite_model = converter.convert()

# Save the model.
with open('fackup_003.tflite', 'wb') as f:
  f.write(tflite_model)