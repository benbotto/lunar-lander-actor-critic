
##
# This script is used to verify that TensorFlow can use the GPU.
##
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

# This is an example of some debugging code that shows where each TF operation
# runs.
#tf.debugging.set_log_device_placement(True)
#print(tf.reduce_sum(tf.random.normal([1000, 1000])))
