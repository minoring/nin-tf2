"""Main for nin"""
import tensorflow as tf

from absl import app
from absl import flags

from model import define_nin
from flags import define_flags


def run(flags_obj):
  """Run nin model"""
  if flags_obj.dataset == 'cifar-10':
    nin = define_nin(input_shape=(32, 32, 3))
  elif flags.FLAGS.dataset == 'mnist':
    nin = define_nin(input_shape=(28, 28, 1))
  else:
    raise Exception('Dataset must be "cifar-10" or "mnist"')


def main(_):
  """main func"""
  run(flags.FLAGS)


if __name__ == '__main__':
  define_flags()
  app.run(main)
