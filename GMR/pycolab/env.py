# made by kyx 202001112

"""Pycolab env scrolly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pycolab import rendering
from pycolab.examples.scrolly_maze import *
from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import drapes as prefab_drapes
from pycolab.prefab_parts import sprites as prefab_sprites


class PycolabEnvironment(object):
    def __init__(self,):
        self._game=