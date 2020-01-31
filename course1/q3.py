import tensorflow as tf
print(tf.__version__)
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3),
                           activation=tf.nn.relu,
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.998:
            self.model.stop_training = True
            print("\nReached 99.8% accuracy so cancelling training!")


callbacks = MyCallback()

history = model.fit(training_images, training_labels,
                    epochs=20,
                    callbacks=[callbacks])
print(f'stop training after { len(history.epoch) } epochs')

test_loss = model.evaluate(test_images, test_labels)
