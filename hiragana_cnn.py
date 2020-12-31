import numpy as np
from numpy import mean
from numpy import std
import tensorflow as tf
import skimage.transform
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os

config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8), device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# Resize the image into 32 X 32 pixels
# We have total of 71 hiragana characters
n_char = 71
# input image dimensions
img_rows, img_cols = 32, 32
# Load hiragana from .npz file and preprocess it

hiragana = np.load("hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32)
hiragana = hiragana / np.max(hiragana)
X_df = np.zeros([n_char * 160, img_rows, img_cols], dtype=np.float32)
y_df = np.repeat(np.arange(n_char), 160)
# Sanity check to see the data we've extracted
# plt.figure(figsize=(6,6))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(hiragana[-i], cmap=plt.cm.binary)
#     plt.xlabel(y_df[-i])

# plt.show()

print(tf.keras.backend.image_data_format())
for i in range(n_char * 160):
    X_df[i] = skimage.transform.resize(hiragana[i], (img_rows, img_cols))



# Split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
# Data augmentation, we use shifting and zoom here
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5, width_shift_range=0.3, height_shift_range=0.3)
datagen.fit(X_train)


# Create the CNN model
def define_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(96, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(n_char))
    model.summary()
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def check_prediction(index, predictions, test_labels, test_images):
    print(predictions[index])
    print(np.argmax(predictions[index]))
    print(np.argmax(predictions[index])==test_labels[index])
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(index, predictions[index], test_labels, test_images)
    # plt.subplot(1,2,2)
    # plot_value_array(index, predictions[index],  test_labels)
    plt.show()

# evaluate a model 
def evaluate_model(X_train, X_test, y_train, y_test, n_folds=5):
    scores, histories = list(), list()
    # define model
    model = define_model()
    # fit model
    history = model.fit(datagen.flow(X_train, y_train,shuffle=True),epochs=30,validation_data=(X_test, y_test),
                callbacks = [tf.keras.callbacks.EarlyStopping(patience=8,verbose=0,restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=3,verbose=0)])
    # evaluate model
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # save model if the accuracy above 97%
    if (acc * 100.0) > 97.0:
        probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
        predictions = probability_model.predict(X_test)
        check_prediction(10, predictions, y_test, X_test)
        probability_model.save("hiragana_model")
    # stores scores
    scores.append(acc)
    histories.append(history)
    return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.legend()
    plt.show()
    plt.savefig('evaluation',format='png')
 
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	#plt.boxplot(scores)
	#plt.show()

# Train CNN and evaluate model
def run_training():
    scores, histories = evaluate_model(X_train, X_test, y_train, y_test)
    # learning curves
    summarize_diagnostics(histories)
	# summarize estimated performance
    summarize_performance(scores)

run_training()
tf.keras.backend.clear_session()
