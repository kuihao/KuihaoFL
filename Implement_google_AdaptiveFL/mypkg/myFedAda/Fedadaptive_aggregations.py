from flwr.common import Weights
from typing import List, Tuple, Optional
import numpy as np
from myFedAda.weighted_aggregate import Weighted_Aggregate

def FedAdagrad_Aggregate(
    current_weights: Weights,
    results: List[Tuple[Weights, int]], 
    eta=1.0,
    tau=1e-2,
    ) -> Weights:
    '''
    current_weights(model weights): The current newest global model weights .
    results: [(model weights, dataset size)].
    learning_rate(float): eta, Server-side learning rate.
    tau(float): Controls the algorithm's degree of adaptability.
    '''
    v_t: Optional[Weights] = None


    # fedavg algo.
    fedavg_weights = Weighted_Aggregate(current_weights=current_weights, 
                                        results=results, 
                                        learning_rate=1.0)
    if fedavg_weights is None: return None

    # Get difference of delta weights
    aggregated_updates = [
            subset_weights - current_weights[idx]
            for idx, subset_weights in enumerate(fedavg_weights)
        ]

    # Adagrad
    delta_t = aggregated_updates
    if not v_t:
        v_t = [np.zeros_like(subset_weights) for subset_weights in delta_t]

    v_t = [
        v_t[idx] + np.multiply(subset_weights, subset_weights)
        for idx, subset_weights in enumerate(delta_t)
    ]

    weights_prime = [
        current_weights[idx]
        + eta * delta_t[idx] / (np.sqrt(v_t[idx]) + tau)
        for idx in range(len(delta_t))
    ]
    
    return weights_prime

def FedAdam_Aggregate(
    current_weights: Weights,
    results: List[Tuple[Weights, int]], 
    eta=1.0,
    tau=1e-1,
    beta_1=0.9,
    beta_2=0.99,
    ) -> Weights:
    '''
    current_weights(model weights): The current newest global model weights .
    results: [(model weights, dataset size)].
    learning_rate(float): eta, Server-side learning rate.
    tau(float): Controls the algorithm's degree of adaptability.
    beta_1 (float): Momentum parameter. Defaults to 0.9.
    beta_2 (float): Second moment parameter. Defaults to 0.99.
    '''
    delta_t: Optional[Weights] = None
    m_t: Optional[Weights] = None
    v_t: Optional[Weights] = None
    # fedavg algo.
    fedavg_weights = Weighted_Aggregate(current_weights=current_weights, 
                                        results=results, 
                                        learning_rate=1.0)
    if fedavg_weights is None: return None

    # Get difference of delta weights
    aggregated_updates = [
            subset_weights - current_weights[idx]
            for idx, subset_weights in enumerate(fedavg_weights)
        ]
    
    # Adam
    delta_t = aggregated_updates

    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]

    m_t = [
        beta_1 * x + (1.0 - beta_1) * y
        for x, y in zip(m_t, delta_t)
    ]

    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]

    v_t = [
        beta_2 * x + (1.0 - beta_2) * np.multiply(y, y)
        for x, y in zip(v_t, delta_t)
    ]

    weights_prime = [
        x + eta * y / (np.sqrt(z) + tau)
        for x, y, z in zip(current_weights, m_t, v_t)
    ]

    return weights_prime

def FedYogi_Aggregate(
    current_weights: Weights,
    results: List[Tuple[Weights, int]], 
    eta=1.0,
    tau=1e-1,
    beta_1=0.9,
    beta_2=0.99,
    ) -> Weights:
    '''
    current_weights(model weights): The current newest global model weights .
    results: [(model weights, dataset size)].
    learning_rate(float): eta, Server-side learning rate.
    tau(float): Controls the algorithm's degree of adaptability.
    beta_1 (float): Momentum parameter. Defaults to 0.9.
    beta_2 (float): Second moment parameter. Defaults to 0.99.
    '''
    delta_t: Optional[Weights] = None
    m_t: Optional[Weights] = None
    v_t: Optional[Weights] = None
    # fedavg algo.
    fedavg_weights = Weighted_Aggregate(current_weights=current_weights, 
                                        results=results, 
                                        learning_rate=1.0)
    if fedavg_weights is None: return None

    # Get difference of delta weights
    aggregated_updates = [
            subset_weights - current_weights[idx]
            for idx, subset_weights in enumerate(fedavg_weights)
        ]
    
    # Yogi
    delta_t = aggregated_updates

    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    
    m_t = [
        beta_1 * x + (1.0 - beta_1) * y
        for x, y in zip(m_t, delta_t)
    ]

    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]

    v_t = [
        x - (1.0 - beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y))
        for x, y in zip(v_t, delta_t)
    ]

    weights_prime = [
        x + eta * y / (np.sqrt(z) + tau)
        for x, y, z in zip(current_weights, m_t, v_t)
    ]

    return weights_prime