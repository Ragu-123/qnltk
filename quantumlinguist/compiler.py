from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import pennylane as qml
import numpy as np
import logging
from quantumlinguist.parser import ParseNode

logger = logging.getLogger(__name__)

RULESET_CONSTITUENCY_V1 = "constituency_v1"
RULESET_DEPENDENCY_V1 = "dependency_v1"
RULESET_MINIMAL_V1 = "minimal_v1"


@dataclass
class CircuitMetadata:
    n_qubits: int
    depth: int
    n_params: int
    sentence: str
    ruleset: str
    parse_tree_str: str


def _assign_qubit_indices(node: ParseNode) -> dict:
    qubit_map = {}
    counter = [0]

    def _walk(n: ParseNode) -> None:
        if n.is_leaf:
            qubit_map[id(n)] = counter[0]
            counter[0] += 1
        else:
            for child in n.children:
                _walk(child)

    _walk(node)
    return qubit_map


def _apply_constituency_v1(
    node: ParseNode, qubit_map: dict, ops: list, params: list
) -> None:
    for child in node.children:
        _apply_constituency_v1(child, qubit_map, ops, params)

    label = node.label.upper() if node.label else ""

    if node.is_leaf:
        ops.append(("RY", qubit_map[id(node)]))
        params.append(None)

    elif label == "NP":
        child_indices = []
        for child in node.children:
            if child.is_leaf:
                child_indices.append(qubit_map[id(child)])
            else:
                leaves = []
                _collect_leaves(child, qubit_map, leaves)
                child_indices.extend(leaves)
        for i in range(len(child_indices) - 1):
            ops.append(("CNOT", [child_indices[i], child_indices[i + 1]]))
            ops.append(("RZ", child_indices[i + 1]))
            params.append(None)

    elif label == "VP":
        all_indices = []
        for child in node.children:
            if child.is_leaf:
                all_indices.append(qubit_map[id(child)])
            else:
                leaves = []
                _collect_leaves(child, qubit_map, leaves)
                all_indices.extend(leaves)
        if len(all_indices) >= 2:
            verb_idx = all_indices[0]
            obj_idx = all_indices[1]
            ops.append(("CNOT", [verb_idx, obj_idx]))
            ops.append(("RY", verb_idx))
            params.append(None)

    elif label == "PP":
        all_indices = []
        for child in node.children:
            if child.is_leaf:
                all_indices.append(qubit_map[id(child)])
            else:
                leaves = []
                _collect_leaves(child, qubit_map, leaves)
                all_indices.extend(leaves)
        if len(all_indices) >= 2:
            prep_idx = all_indices[0]
            np_idx = all_indices[1]
            ops.append(("CRY", [prep_idx, np_idx]))
            params.append(None)

    elif label == "S":
        all_indices = []
        _collect_leaves(node, qubit_map, all_indices)
        for qubit_idx in all_indices:
            ops.append(("RZ", qubit_idx))
            params.append(None)


def _collect_leaves(node: ParseNode, qubit_map: dict, result: list) -> None:
    if node.is_leaf:
        result.append(qubit_map[id(node)])
    else:
        for child in node.children:
            _collect_leaves(child, qubit_map, result)


def _apply_minimal_v1(n_qubits: int, ops: list, params: list) -> None:
    for i in range(n_qubits):
        ops.append(("RY", i))
        params.append(None)
    for i in range(n_qubits - 1):
        ops.append(("CNOT", [i, i + 1]))
    for i in range(n_qubits):
        ops.append(("RZ", i))
        params.append(None)


def _build_qnode(ops: list, n_qubits: int, backend: str) -> Tuple[Callable, int]:
    """Build a PennyLane QNode from an ops list; return (qnode, n_params)."""
    dev = qml.device(backend, wires=n_qubits)

    parametric = {"RY", "RZ", "CRY"}
    n_params = sum(1 for op, *_ in ops if op in parametric)

    @qml.qnode(dev, interface="torch")
    def circuit(params: np.ndarray):
        param_idx = 0
        for entry in ops:
            gate = entry[0]
            target = entry[1]
            if gate == "RY":
                qml.RY(params[param_idx], wires=target)
                param_idx += 1
            elif gate == "RZ":
                qml.RZ(params[param_idx], wires=target)
                param_idx += 1
            elif gate == "CNOT":
                qml.CNOT(wires=target)
            elif gate == "CRY":
                qml.CRY(params[param_idx], wires=target)
                param_idx += 1
            else:
                logger.warning("Unknown gate '%s'; skipping.", gate)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit, n_params


def _count_leaves(node: ParseNode) -> int:
    if node.is_leaf:
        return 1
    return sum(_count_leaves(c) for c in node.children)


def _tree_str(node: ParseNode, indent: int = 0) -> str:
    prefix = "  " * indent
    if node.is_leaf:
        return f"{prefix}[{node.label} '{node.word}']"
    children_str = "\n".join(_tree_str(c, indent + 1) for c in node.children)
    return f"{prefix}({node.label}\n{children_str})"


def _collect_words(node: ParseNode) -> list:
    if node.is_leaf:
        return [node.word or ""]
    words = []
    for c in node.children:
        words.extend(_collect_words(c))
    return words


def compile(
    node: ParseNode,
    ruleset: str = RULESET_CONSTITUENCY_V1,
    backend: str = "default.qubit",
) -> Tuple[Callable, CircuitMetadata]:
    """Compile a ParseNode tree into a PennyLane QNode and CircuitMetadata."""
    n_qubits = _count_leaves(node)
    if n_qubits == 0:
        raise ValueError("ParseNode tree contains no leaf nodes.")

    ops: list = []
    params: list = []

    if ruleset == RULESET_CONSTITUENCY_V1:
        qubit_map = _assign_qubit_indices(node)
        _apply_constituency_v1(node, qubit_map, ops, params)
    elif ruleset == RULESET_MINIMAL_V1:
        _apply_minimal_v1(n_qubits, ops, params)
    elif ruleset == RULESET_DEPENDENCY_V1:
        # Full spaCy dependency parsing is future work; falls back to sequential.
        _apply_minimal_v1(n_qubits, ops, params)
    else:
        raise ValueError(f"Unknown ruleset: '{ruleset}'")

    qnode, n_params = _build_qnode(ops, n_qubits, backend)

    sentence = " ".join(_collect_words(node))
    parse_tree_str = _tree_str(node)

    depth = len(ops)

    metadata = CircuitMetadata(
        n_qubits=n_qubits,
        depth=depth,
        n_params=n_params,
        sentence=sentence,
        ruleset=ruleset,
        parse_tree_str=parse_tree_str,
    )

    logger.debug(
        "Compiled circuit: qubits=%d depth=%d params=%d ruleset=%s",
        n_qubits,
        depth,
        n_params,
        ruleset,
    )

    return qnode, metadata
