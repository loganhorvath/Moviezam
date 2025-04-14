import tensorflow as tf

# Check if TensorFlow is installed and working
print("TensorFlow version:", tf.__version__)

# Create a simple constant tensor
hello = tf.constant('Hello, TensorFlow!')

# Start a TensorFlow session and run the tensor
tf.print(hello)