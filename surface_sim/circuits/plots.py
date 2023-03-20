import re
from itertools import combinations
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

from qec_util import Layout

RE_FILTER = re.compile("([a-zA-Z]+)([0-9]+)")

SQ_GATE_PARAMS = dict(linewidth=2, facecolor="white", edgecolor="black")

WIDTHS = dict(sq_gate=0.7)

ZORDERS = dict(patch=1, cricle=4, line=3, gate=4, label=5)

LABEL_PARAMS = dict(
    horizontalalignment="center",
    verticalalignment="center",
    weight="bold",
)

QB_CIRCLE_PARAMS = dict(radius=0.3, linewidth=1, edgecolor="black")
CZ_CIRCLE_PARAMS = dict(radius=0.2, linewidth=1, edgecolor="black", facecolor="black")

TQ_LINE_PARAMS = dict(color="black", linestyle="-", linewidth=2)

PATCH_PARAMS = dict(linewidth=0, alpha=0.5)

STAB_COLORS = dict(x_type="#2196f3", z_type="#4caf50")
DATA_COLOR = "#fafafa"

TQ_INSTRUCTIONS = {
    "CZ",
}
ERROR_INSTRUCTIONS = {"DEPOLARIZE1", "DEPOLARIZE2"}

SKIPPED_INSTRUCTIONS = {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE"}

LIM_PAD = 1


def _reverse(coords):
    x, y = coords
    return y, x


def plot_layout(
    axis,
    layout: Layout,
    draw_qubits: bool = True,
    label_qubits: bool = True,
    draw_patches: bool = True,
):
    if draw_qubits:
        plot_qubits(axis, layout, label_qubits)
    if draw_patches:
        plot_patches(axis, layout)


def get_label(qubit: str) -> None:
    match = RE_FILTER.match(qubit)
    if match is None:
        raise ValueError(f"Unexpected qubit label {qubit}")
    label, ind = match.groups()
    return f"${label}_\\mathrm{{{ind}}}$"


def plot_qubits(axis, layout, label_qubits: Optional[bool] = True) -> None:
    qubits = layout.get_qubits()
    data = dict(layout.graph.nodes(data=True))

    init_qubit = qubits.pop()
    drawn_qubits = set()

    def _dfs_draw(qubit: str) -> None:
        if qubit not in drawn_qubits:
            role = data[qubit]["role"]
            if role == "data":
                color = DATA_COLOR
            else:
                stab_type = data[qubit]["stab_type"]
                color = STAB_COLORS[stab_type]

            y, x = data[qubit]["coords"]
            qubit_circ = Circle(
                (x, y), facecolor=color, zorder=ZORDERS["cricle"], **QB_CIRCLE_PARAMS
            )
            axis.add_artist(qubit_circ)
            if label_qubits:
                label = get_label(qubit)
                axis.text(x, y, label, zorder=ZORDERS["label"], **LABEL_PARAMS)
            drawn_qubits.add(qubit)

            neighbors = layout.get_neighbors(qubit)
            for neighbour in neighbors:
                _dfs_draw(neighbour)

    _dfs_draw(init_qubit)


def plot_patches(axis, layout):
    anc_qubits = layout.get_qubits(role="anc")
    data = dict(layout.graph.nodes(data=True))

    for anc_qubit in anc_qubits:
        stab_type = data[anc_qubit]["stab_type"]
        color = STAB_COLORS[stab_type]

        anc_coords = _reverse(data[anc_qubit]["coords"])
        neigbors = layout.get_neighbors(anc_qubit)
        for start_q, end_q in combinations(neigbors, 2):
            start_coords = _reverse(data[start_q]["coords"])
            end_coords = _reverse(data[end_q]["coords"])

            dist = sum([abs(i - j) for i, j in zip(start_coords, end_coords)])
            if dist <= 2:
                patch_coords = [anc_coords, start_coords, end_coords]
                patch = Polygon(
                    patch_coords,
                    color=color,
                    zorder=ZORDERS["patch"],
                    **PATCH_PARAMS,
                )
                axis.add_artist(patch)


def plot_sq_gate(axis, coords, name: str) -> None:
    y, x = coords
    width = WIDTHS["sq_gate"]
    half_width = 0.5 * width
    center = (x - half_width, y - half_width)
    rect = Rectangle(
        center, width=width, height=width, zorder=ZORDERS["gate"], **SQ_GATE_PARAMS
    )
    axis.add_patch(rect)
    axis.text(x, y, name, zorder=ZORDERS["label"], **LABEL_PARAMS)


def plot_cz_gate(axis, control_coords, target_coords) -> None:
    for y, x in (control_coords, target_coords):
        circle = Circle((x, y), zorder=ZORDERS["cricle"], **CZ_CIRCLE_PARAMS)
        axis.add_patch(circle)

    ys, xs = zip(control_coords, target_coords)
    axis.plot(xs, ys, zorder=ZORDERS["line"], **TQ_LINE_PARAMS)


def plot_layer(instruction, layout, axis=None):
    name = instruction.name
    if name in SKIPPED_INSTRUCTIONS:
        return None
    if axis:
        fig, ax = None, axis
    else:
        fig, ax = plt.subplots()

    ax.set_aspect("equal")
    ax.axis("off")

    node_coords = layout.graph.nodes(data="coords")
    _, coords = zip(*node_coords)

    for ind, ax_coords in enumerate(zip(*coords)):
        ax_min, ax_max = min(ax_coords), max(ax_coords)
        pass
        if ind:
            ax.set_xlim(ax_min - LIM_PAD, ax_max + LIM_PAD)
        else:
            ax.set_ylim(ax_min - LIM_PAD, ax_max + LIM_PAD)

    plot_layout(ax, layout, label_qubits=False)

    targets = instruction.targets_copy()
    inds = [target.value for target in targets]
    if name in TQ_INSTRUCTIONS:
        if name != "CZ":
            raise NotImplementedError("Other two-qubit gates not yet supported.")
        ind_iter = iter(inds)
        pair_iter = zip(ind_iter, ind_iter, strict=True)

        for ctrl_ind, tar_ind in pair_iter:
            ctrl_coords = coords[ctrl_ind]
            tar_coords = coords[tar_ind]
            plot_cz_gate(axis, ctrl_coords, tar_coords)
    else:
        if name not in ("H", "M", "R"):
            raise NotImplementedError("Only 'H', 'M' and 'R' gates are supported.")
        for ind in inds:
            plot_sq_gate(axis, coords[ind], name)
    return fig, ax
