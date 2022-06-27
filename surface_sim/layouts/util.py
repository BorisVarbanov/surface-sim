from .layout import Layout


def set_coords(layout: Layout):
    """
    set_coords Automatically sets the qubit coordinates, if they are not already set

    Parameters
    ----------
    layout : Layout
        The layout of the qubit device.
    """

    qubits = layout.get_qubits()
    for qubit in qubits:
        q_coords = layout.param("coords", qubit)
        if q_coords is not None:
            raise ValueError(
                "'set_coords' only works on layout where none of the qubits have their coordinates"
                f" set, instead qubit {qubit} has coordinates {q_coords}"
            )
    init_qubit = qubits.pop()
    init_cords = (0, 0)

    set_qubits = set()

    def dfs_position(qubit, x_pos, y_pos):
        if qubit not in set_qubits:
            layout.set_param("coords", qubit, (x_pos, y_pos))
            set_qubits.add(qubit)

            neighbors_dict = layout.param("neighbors", qubit)
            for con_dir, neighbour in neighbors_dict.items():
                if neighbour:
                    ver_dir, hor_dir = con_dir.split("_")
                    x_shift = -1 if hor_dir == "west" else 1
                    y_shift = -1 if ver_dir == "south" else 1

                    dfs_position(neighbour, x_pos + x_shift, y_pos + y_shift)

    dfs_position(init_qubit, *init_cords)
