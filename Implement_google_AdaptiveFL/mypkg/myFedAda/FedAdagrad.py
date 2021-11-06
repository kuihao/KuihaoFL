'''
# No Change
引入套件
'''
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.fedopt import FedOpt


class MyFedAdagrad(FedOpt):
    """
    Flower: https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedadagrad.py 
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        tau: float = 1e-9,
    ) -> None:
        """Federated learning strategy using Adagrad on server-side.
        Implementation based on https://arxiv.org/abs/2003.00295
        Args:
            fraction_fit (float, optional): Fraction of clients used during
                training. Defaults to 0.1.
            fraction_eval (float, optional): Fraction of clients used during
                validation. Defaults to 0.1.
            min_fit_clients (int, optional): Minimum number of clients used
                during training. Defaults to 2.
            min_eval_clients (int, optional): Minimum number of clients used
                during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total
                clients in the system. Defaults to 2.
            eval_fn (Callable[[Weights], Optional[Tuple[float, float]]], optional):
                Function used for validation. Defaults to None.
            on_fit_config_fn (Callable[[int], Dict[str, str]], optional):
                Function used to configure training. Defaults to None.
            on_evaluate_config_fn (Callable[[int], Dict[str, str]], optional):
                Function used to configure validation. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds
                containing failures. Defaults to True.
            initial_parameters (Parameters): Initial set of parameters from the server.
            eta (float, optional): Server-side learning rate. Defaults to 1e-1.
            eta_l (float, optional): Client-side learning rate. Defaults to 1e-1.
            tau (float, optional): Controls the algorithm's degree of adaptability.
                Defaults to 1e-9.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            eta=eta,
            eta_l=eta_l,
            beta_1=0.0,
            beta_2=0.0,
            tau=tau,
        )
        self.v_t: Optional[Weights] = None

    def __repr__(self) -> str:
        rep = f"FedAdagrad(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        rnd: int,
        K_defining_eta: float,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        self.eta = K_defining_eta
        
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            rnd=rnd, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_weights(fedavg_parameters_aggregated)
        aggregated_updates = [
            subset_weights - self.current_weights[idx]
            for idx, subset_weights in enumerate(fedavg_weights_aggregate)
        ]

        # Adagrad
        delta_t = aggregated_updates
        if not self.v_t:
            self.v_t = [np.zeros_like(subset_weights) for subset_weights in delta_t]

        self.v_t = [
            self.v_t[idx] + np.multiply(subset_weights, subset_weights)
            for idx, subset_weights in enumerate(delta_t)
        ]

        new_weights = [
            self.current_weights[idx]
            + self.eta * delta_t[idx] / (np.sqrt(self.v_t[idx]) + self.tau)
            for idx in range(len(delta_t))
        ]
        self.current_weights = new_weights

        return weights_to_parameters(self.current_weights), metrics_aggregated