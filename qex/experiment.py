"""
Experiment abstraction: parametric circuit builder.
"""

from typing import Callable, Dict, Any, Sequence, Union
import cirq


class Experiment:
    """
    An experiment defines a parametric quantum circuit builder.

    An experiment has a name and a function that builds a Cirq circuit
    from a set of qubits and parameters. Circuits may use 1 or more qubits.

    Attributes:
        name: Unique identifier for the experiment.
        builder: Function (qubits, params) -> Cirq Circuit.
    """

    def __init__(
        self,
        name: str,
        builder: Callable[[Sequence[cirq.Qid], Dict[str, Any]], cirq.Circuit],
    ):
        """
        Initialize an experiment.

        Args:
            name: Unique name for the experiment.
            builder: (qubits: Sequence[cirq.Qid], params: Dict[str, Any]) -> cirq.Circuit
        """
        self.name = name
        self.builder = builder

    def build_circuit(
        self,
        qubits: Union[cirq.Qid, Sequence[cirq.Qid]],
        params: Dict[str, Any],
    ) -> cirq.Circuit:
        """
        Build the circuit for this experiment with given qubits and parameters.

        Args:
            qubits: Single qubit or sequence of qubits to operate on.
            params: Parameter dictionary for the experiment.

        Returns:
            A Cirq Circuit operating on the given qubits.
        """
        if isinstance(qubits, cirq.Qid):
            qubits = (qubits,)
        else:
            qubits = tuple(qubits)
        return self.builder(qubits, params)
