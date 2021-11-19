from flwr.common import Weights
from functools import reduce
from typing import List, Tuple
import numpy as np
def Weighted_Aggregate(results: List[Tuple[Weights, int]], learning_rate=1) -> Weights:
  """Compute weighted average."""
  # Calculate the total number of examples used during training
  num_examples_total = sum([num_examples for _, num_examples in results])

  # Create a list of weights, each multiplied by the related number of examples
  weighted_weights = [
      [layer * num_examples for layer in weights] for weights, num_examples in results
  ]

  # Compute average weights of each layer
  weights_prime: Weights = [
       learning_rate*(reduce(np.add, layer_updates) / num_examples_total)
      for layer_updates in zip(*weighted_weights)
  ]
  return weights_prime