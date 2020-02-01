# import tensorflow as tf
import tensorflow.compat.v1 as tf
import os

tf.disable_v2_behavior()

DESIRED_ACCURACY = 0.999


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        if logs.get('acc') >= DESIRED_ACCURACY:
            self.model.stop_training = True
            print("\nReached 99.8% accuracy so cancelling training!")


callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu,
#                            input_shape=(150, 150, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation=tf.nn.relu),
#     tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])
print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['acc'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    os.path.expanduser('~/tmp/h-or-s/'),
    target_size=(150, 150),
    batch_size=128,
    class_mode='binary')
# Expected output: 'Found 80 images belonging to 2 classes'

history = model.fit(train_generator,
                    steps_per_epoch=8,
                    epochs=15,
                    verbose=1)
# callbacks=[callbacks])

print(f"accuracy = { history.history['acc'][-1] }\n")
