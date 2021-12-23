from tensorflow.keras.layers import Input, Dense, Conv2D, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np

class CharClassifier:
  def __init__(self):
    self.labels = [str(idx) for idx in range(10)] + [chr(idx) for idx in range(65, 65+26)]
    self.label_map = {label: idx for idx, label in enumerate(self.labels)}
  
  def build_model(self):
    input_layer = Input(shape=[48, 48, 3])
    conv2d_layer_1 = Conv2D(filters=8, kernel_size=3, strides=1, activation='relu')(input_layer)
    conv2d_layer_2 = Conv2D(filters=8, kernel_size=3, strides=2, activation='relu')(conv2d_layer_1)
    conv2d_layer_3 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu')(conv2d_layer_2)
    conv2d_layer_4 = Conv2D(filters=16, kernel_size=3, strides=2, activation='relu')(conv2d_layer_3)
    conv2d_layer_5 = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(conv2d_layer_4)
    conv2d_layer_6 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu')(conv2d_layer_5)
    flatten_layer = Flatten()(conv2d_layer_6)
    dense_layer = Dense(len(self.labels), activation='softmax')(flatten_layer)
    self.model = Model(input_layer, dense_layer)
    self.model.summary()

    loss = SparseCategoricalCrossentropy()
    metric = SparseCategoricalAccuracy()
    optimizer = Adam(learning_rate=1e-3)

    self.model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

  def load_model(self):
    self.model = load_model('models/char_classifier.h5')

  def train(self):
    train_generator = DataGenerator(32, '/content/train', self.label_map)
    valid_generator = DataGenerator(32, '/content/valid', self.label_map)
    save_best = ModelCheckpoint(filepath='/content/drive/MyDrive/YoloV3/keras-yolo3-master/char_classifier.h5', save_best_only=True, verbose=1)
    self.model.fit(train_generator, validation_data= valid_generator, epochs=20, callbacks=[save_best])

  def predict(self, image):
    resized_image = np.array(image.resize([256, 256]))
    output = self.model.predict(np.array([resized_image]))[0]
    label = self.labels[np.argmax(output)]
    return label