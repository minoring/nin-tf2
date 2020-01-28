"""Utility for flags"""
from absl import flags


def define_flags():
  """Define flags for nin"""
  flags.DEFINE_integer("epoch", 200, "The number of epochs. [50]")
  flags.DEFINE_string("dataset", "cifar-10",
                      "The name of dataset. [cifar-10, mnist]")
  flags.DEFINE_integer("batch_size", 128, "The size of batch. [128]")
