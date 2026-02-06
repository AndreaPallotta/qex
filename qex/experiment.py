"""
Experiment abstraction: parametric circuit builder.
"""

from typing import Callable, Dict, Any
import cirq


class Experiment:
    """
    An experiment defines a parametric quantum circuit builder.
    
    An experiment has a name and a function that builds a Cirq circuit
    from parameters. The circuit must be 1-qubit for MVP.
    
    Attributes:
        name: Unique identifier for the experiment.
        builder: Function that takes parameters and returns a Cirq Circuit.
                 Must accept a single qubit as first argument.
    """
    
    def __init__(self, name: str, builder: Callable[[cirq.Qid, Dict[str, Any]], cirq.Circuit]):
        """
        Initialize an experiment.
        
        Args:
            name: Unique name for the experiment.
            builder: Function signature: (qubit: cirq.Qid, params: Dict[str, Any]) -> cirq.Circuit
                     The builder must create a 1-qubit circuit.
        """
        self.name = name
        self.builder = builder
    
    def build_circuit(self, qubit: cirq.Qid, params: Dict[str, Any]) -> cirq.Circuit:
        """
        Build the circuit for this experiment with given parameters.
        
        Args:
            qubit: The single qubit to operate on.
            params: Parameter dictionary for the experiment.
            
        Returns:
            A Cirq Circuit operating on the given qubit.
        """
        return self.builder(qubit, params)
