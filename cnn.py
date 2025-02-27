# https://www.tensorflow.org/tutorials/images/cnn
# good examples about plotting too

#%% imports
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#%% Check if GPU is being used!
gpu_list = tf.config.experimental.list_physical_devices('GPU')

if len(gpu_list)==0:
    print("FYI:no GPU is being used by tensorflow")
else:
    print(f"GPUs used: {gpu_list}")


#%% Download at prepare dataset (this is what will be changed)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


#%% create model

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# model.summary() # shows the architecture of the model

#%% Compile and train model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


#%% Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#%% Check accuracy of model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)


#%%
