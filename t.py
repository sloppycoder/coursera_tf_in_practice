import tensorflow as tf
print(f'******** { tf.__version__ } ********')
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

