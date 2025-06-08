
from keras import layers, Model
import tensorflow as tf
import os

class Crnn_64_256:
  model: Model

  _WEIGHTS_PATH = "ocr_crnn_64_256.weights.h5"

  def __init__(self, alphabet, verbose = False):
    inp = layers.Input(shape=(64,256,1), name="image")
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPool2D((2,2))(x)  # now 32×128
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D((2,2))(x)  # now 16×64
    # collapse height dimension:
    x = layers.Permute((2,1,3))(x)      # shape: (width=64, height=16, channels)
    x = layers.Reshape((64, 16*64))(x)   # sequence length=64 time steps
    # recurrent:
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    # output:
    logits = layers.Dense(len(alphabet)+1, activation="linear")(x)
    model = Model(inputs=inp, outputs=logits)

    if (verbose): 
      model.summary()

    model.compile(
      optimizer="adam",
      loss=self.__ctc_loss
    )
    
    if os.path.exists(self._WEIGHTS_PATH):
      print("🛠 Loading weights from", self._WEIGHTS_PATH)
      model.load_weights(self._WEIGHTS_PATH)

    self.model = model

  @tf.function
  def __ctc_loss(self, y_true, y_pred):
    # y_pred: [batch, time, num_classes]
    # y_true: [batch, max_label_len]
    batch_size = tf.shape(y_pred)[0]
    time_steps = tf.shape(y_pred)[1]

    input_len = tf.fill([batch_size], time_steps)
    label_len = tf.reduce_sum(
      tf.cast(tf.not_equal(y_true, 0), tf.int32),
      axis=1
    )

    sparse_labels = self.__dense_to_sparse(y_true)

    loss = tf.nn.ctc_loss(
      labels=sparse_labels,
      logits=y_pred,
      label_length=label_len,
      logit_length=input_len,
      logits_time_major=False,
      blank_index=0
    )
    
    return tf.reduce_mean(loss)
  
  def __dense_to_sparse(self, labels):
    # labels: [batch, max_label_len], padded with 0=blank
    labels = tf.cast(labels, tf.int32)
    indices = tf.where(tf.not_equal(labels, 0))
    values  = tf.gather_nd(labels, indices)
    dense_shape = tf.cast(tf.shape(labels), tf.int64)
    return tf.SparseTensor(indices, values, dense_shape)
  
  def __to_training_batch(self, imgs, labels):
    return {"image": imgs}, labels
  
  def train(self, dataset):
    train_ds = dataset.map(self.__to_training_batch).take(1000)
    self.model.fit(train_ds, epochs=1, steps_per_epoch=1000)

    self.model.save_weights(self._WEIGHTS_PATH)
    print("💾 Saved weights to", self._WEIGHTS_PATH)

  def predict(self, x):
    return self.model.predict(x)