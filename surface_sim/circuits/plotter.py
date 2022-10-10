from typing import Iterable, List, Optional
import re
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from copy import deepcopy
from matplotlib.patches import Rectangle, Ellipse, Arc

from ..layouts import Layout
from ..circuits import Circuit, Gate


def plot(
    circuit: Circuit,
    layout: Optional[Layout] = None,
    *,
    ax: Optional[plt.Axes] = None,
    qubit_order: Optional[List[str]] = None,
) -> Figure:
    plotter = MatplotlibPlotter(circuit, layout, ax, qubit_order)
    return plotter.plot()


class MatplotlibPlotter:
    zorders = {
        "line": 1,
        "marker": 1,
        "circle": 5,
        "box": 10,
        "text": 20,
    }

    def __init__(
        self,
        circuit: Circuit,
        layout: Optional[Layout] = None,
        ax: Optional[Axes] = None,
        qubit_order: Optional[List[str]] = None,
    ):
        self.circuit = circuit
        self.layout = layout

        if ax is not None:
            self.fig = None
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(figsize=(7, len(circuit.qubits)))

        if isinstance(qubit_order, Iterable):
            self.qubits = list(qubit_order)
        elif qubit_order is None:
            self.qubits = list(reversed(circuit.qubits))
        else:
            raise ValueError("Qubit order must be a list or None")

        if layout is not None:
            layout_qubits = self.layout.get_qubits()
            if not all(q in layout_qubits for q in self.qubits):
                raise ValueError("Not all qubits in circuit are present in layout")

        y_pad = 1
        x_pad = 1
        self.ax.set_ylim(-y_pad, len(self.qubits) + y_pad)
        self.ax.set_xlim(
            circuit.time - x_pad,
            circuit.time + circuit.depth + x_pad,
        )
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def plot(self) -> None:
        for qubit in self.circuit.qubits:
            self._draw_qubit(qubit)
            self._annotate_qubit(qubit)
            self._plot_qubit_line(qubit)
        for gate in self.circuit.gates:
            self._plot_gate(gate)

        return self.fig

    def _plot_gate(self, gate: Gate) -> None:
        metadata = deepcopy(gate._plot_metadata)

        time = gate.time
        if gate.label == "cnot":  # FIXME
            ctrl_q, target_q = gate.qubits
            q_start = self.qubits.index(ctrl_q)
            q_end = self.qubits.index(target_q)
            self.ax.plot(
                (time, time),
                (q_start, q_end),
                color=metadata.get("color", "black"),
                linewidth=2,
                zorder=self.zorders["circle"],
            )

            self.ax.add_patch(
                Ellipse(
                    (time, q_start),
                    0.2,
                    0.2,
                    facecolor=metadata.get("facecolor", "black"),
                    edgecolor=metadata.get("edgecolor", "black"),
                    linewidth=metadata.get("linewidth", 0),
                    zorder=metadata.get("zorder", self.zorders["circle"]),
                )
            )
            self.ax.add_patch(
                Ellipse(
                    (time, q_end),
                    0.3,
                    0.3,
                    facecolor=metadata.get("facecolor", "black"),
                    edgecolor=metadata.get("edgecolor", "black"),
                    linewidth=metadata.get("linewidth", 2),
                    zorder=metadata.get("zorder", self.zorders["circle"]),
                )
            )

            self.ax.plot(
                (time, time),
                (q_end - 0.1, q_end + 0.1),
                color=metadata.get("color", "white"),
                linewidth=2,
                zorder=self.zorders["circle"],
            )
            self.ax.plot(
                (time - 0.1, time + 0.1),
                (q_end, q_end),
                color=metadata.get("color", "white"),
                linewidth=2,
                zorder=self.zorders["circle"],
            )

        elif gate.label == "cphase":
            ctrl_q, target_q = gate.qubits
            q_start = self.qubits.index(ctrl_q)
            q_end = self.qubits.index(target_q)
            self.ax.plot(
                (time, time),
                (q_start, q_end),
                color=metadata.get("color", "black"),
                linewidth=2,
                zorder=self.zorders["circle"],
            )

            self.ax.add_patch(
                Ellipse(
                    (time, q_start),
                    0.2,
                    0.2,
                    facecolor=metadata.get("facecolor", "black"),
                    edgecolor=metadata.get("edgecolor", "black"),
                    linewidth=metadata.get("linewidth", 0),
                    zorder=metadata.get("zorder", self.zorders["circle"]),
                )
            )
            self.ax.add_patch(
                Ellipse(
                    (time, q_end),
                    0.2,
                    0.2,
                    facecolor=metadata.get("facecolor", "black"),
                    edgecolor=metadata.get("edgecolor", "black"),
                    linewidth=metadata.get("linewidth", 0),
                    zorder=metadata.get("zorder", self.zorders["circle"]),
                )
            )

        elif gate.label == "measure":
            ind = self.qubits.index(gate.qubits[0])
            rect = Rectangle(
                (time - 0.3, ind - 0.3),
                0.6,
                0.6,
                linewidth=2,
                fc=metadata.get("facecolor", "white"),
                ec=metadata.get("edgecolor", "black"),
                zorder=self.zorders["box"],
            )
            self.ax.add_patch(rect)

            self.ax.add_patch(
                Arc(
                    (time, ind - 0.05),
                    0.4,
                    0.3,
                    theta1=0,
                    theta2=180,
                    lw=1,
                    color="black",
                    zorder=self.zorders["text"],
                )
            )
            self.ax.arrow(
                time,
                ind - 0.05,
                0.15,
                0.15,
                color="black",
                head_length=0.1,
                head_width=0.1,
                lw=0.5,
                zorder=self.zorders["text"],
            )
        else:
            inds = [self.qubits.index(q) for q in gate.qubits]
            q_start, q_end = min(inds), max(inds)
            rect = Rectangle(
                (time - 0.3, q_start - 0.3),
                0.6,
                q_end - q_start + 0.6,
                linewidth=2,
                fc=metadata.get("facecolor", "white"),
                ec=metadata.get("edgecolor", "black"),
                zorder=self.zorders["box"],
            )
            self.ax.add_patch(rect)
            if gate.label is not None:
                self.ax.text(
                    time,
                    q_start + 0.5 * (q_end - q_start),
                    metadata.get("label", r"$\mathcal{G}$"),
                    ha="center",
                    va="center",
                    zorder=self.zorders["text"],
                )

    def _draw_qubit(self, qubit: str) -> None:
        ind = self.qubits.index(qubit)

        metadata = {}

        if self.layout is not None:
            role = self.layout.param("role", qubit)
            if role == "data":
                freq_group = self.layout.param("freq_group", qubit)
                default_facecolor = "#d32f2f" if freq_group == "high" else "#e57373"
            else:
                stab_type = self.layout.param("stab_type", qubit)
                default_facecolor = "#2196f3" if stab_type == "x_type" else "#4caf50"
        else:
            default_facecolor = "gray"
        circ = Ellipse(
            (-0.6, ind),
            0.3,
            0.3,
            facecolor=metadata.get("facecolor", default_facecolor),
            edgecolor=metadata.get("edgecolor", "black"),
            linewidth=metadata.get("linewidth", 1),
            zorder=metadata.get("zorder", self.zorders["circle"]),
        )
        self.ax.add_patch(circ)

    def _annotate_qubit(self, qubit: str) -> None:
        n = self.qubits.index(qubit)
        self.ax.text(
            -1,
            n,
            self._latexify_label(str(qubit)),
            ha="center",
            va="center",
            zorder=self.zorders["text"],
            weight="bold",
        )

    def _plot_qubit_line(self, qubit: str) -> None:
        n = self.qubits.index(qubit)
        self.ax.axhline(
            n, xmin=0.05, xmax=1, color="black", zorder=self.zorders["line"], lw=1
        )

    def _latexify_label(self, label: str) -> str:
        int_char = re.search("\d", label)
        if int_char is not None:
            start_ind = int_char.start()
            if re.search("\s", label[start_ind:]) is None:
                label = r"$" + label[:start_ind] + "_{" + label[start_ind:] + "}$"
        return label
