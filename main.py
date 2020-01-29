"""Main for nin"""
import tensorflow as tf

from absl import app
from absl import flags

from model import define_nin
from flags import define_flags
from utils import normalize_img


def run(flags_obj):
  """Run nin model"""
  if flags_obj.dataset == 'cifar-10':
    nin = define_nin(input_shape=(32, 32, 3))
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  elif flags.FLAGS.dataset == 'mnist':
    nin = define_nin(input_shape=(28, 28, 1))
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  else:
    raise Exception('Dataset must be "cifar-10" or "mnist"')
  train_size = x_train.shape[0]
  test_size = x_test.shape[0]

  x_train = normalize_img(x_train.astype('float32'))
  x_train = normalize_img(x_train.astype('float32'))
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  train_dataset = train_dataset.shuffle(10000).batch(
      flags_obj.batch_size).repeat()
  test_dataset = test_dataset.batch(flags_obj.batch_size).repeat()

  nin.compile(
      optimizer=tf.keras.optimizers.Adam(lr=flags_obj.learning_rate),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

  nin.fit(train_dataset,
          epochs=flags_obj.epoch,
          steps_per_epoch=train_size // flags_obj.batch_size,
          validation_data=test_dataset,
          validation_steps=test_size // flags_obj.batch_size)


def main(_):
  """main func"""
  run(flags.FLAGS)


if __name__ == '__main__':
  define_flags()
  app.run(main)
