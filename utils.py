"""Utility functions for nin"""


def normalize_img(input_img):
  """Normalize given image"""
  # TODO: Modify as global contrast normalization and ZCA whitening.
  # Refer to: https://arxiv.org/pdf/1302.4389.pdf
  return input_img / 127.5 - 1.0
