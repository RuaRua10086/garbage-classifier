import tensorflow as tf
print("GPU 可用:", tf.test.is_gpu_available())
print("GPU 详情:", tf.config.list_physical_devices('GPU'))
