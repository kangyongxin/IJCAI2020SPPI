# test program 

import collections
import numpy as np
PolicyOutputs = collections.namedtuple(
    'PolicyOutputs', ['policy', 'action', 'baseline'])

P = PolicyOutputs(1,2,3)
print(P.policy)

read_strengths= np.array([[1,2],[3,4],[5,6]])
print(read_strengths)
read_strengths = tf.expand_dims(read_strengths, axis=-1)  # [B, H, 1]
print(read_strengths)