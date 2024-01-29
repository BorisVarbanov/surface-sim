# Creating and configuring a `Setup`

This file contains information on how to create and configure a `Setup` object a the noise model `Model` from (1) a YAML file, and (2) a `dict` object. It explains the structure of these two inputs and the use of *free parameters* and *global parameters*. 

## Structure of the configuration for a `Setup`

The configuration input must contain the following blocks:

- `gate_durations`: data in dictionary format specifying the durations of the gates used in the quantum circuit. The gate names follow the same convention as Stim. *Note: in some noise models this block is not needed.*
- `setup`: data about the noise parameters for each qubit. Some noise parameters are:
    - `sq_error_prob`: float,
    - `cz_error_prob`: float,
    - `meas_error_prob`: float, # quantum measurement errors
    - `assign_error_prob`: float, # classical measurement error
    - `reset_error_prob`: float,
    - `idle_error_prob`: float,
    - `T1`: float,
    - `T2`: float,

*Note: not all parameters are required for each noise model*

*Note: the units of T1, T2 and gate durations must match*

The configuration input can also contain a `name` and a `description`. 

The parameters are classified into three categories:

- **local parameter**: float value that is defined for an specific qubit or pair of qubits for the case of two-qubit gates
- **global parameter**: float value that is defined for all qubits or all two-qubit gates
- **free parameter**: string name that can be set up and modified for an specific qubit or all qubits. These parameters can be setup using the `Setup.set_var_param` function. 

Examples:
```
# local parameters
setup = [
    {
        "qubit": "D1", 
        "sq_error_prob": 0.001 
    },
    {
        "qubits": ["D1", "D2"], 
        "cz_error_prob": 0.01
    },
]
```

```
# global parameters
setup = [
    {
        "sq_error_prob": 0.001 
        "cz_error_prob": 0.01 
    },
]
```

```
# free parameters
setup = [
    {
        # global free parameter
        "sq_error_prob": "param1"
    },
    {
        # local free parameter
        "qubits": ["D1", "D2"], 
        "cz_error_prob": "param2" 
    },
]
```

## Loading `Setup` from YAML file

```
from surface_sim import Setup

setup = Setup.from_yaml("path/to/yaml/file.yaml")
```

Examples of YAML file can be found in `docs`: `circ_level_noise.yaml` and `biased_circ_level_noise`.

## Creating `Setup` from `dict`

```
from surface_sim import Setup

setup_dict = {
    "gate_durations": {
        "X": 3.2,
        "Z": 1,
        "H": 1,
        "CZ": 1,
        "M": 1,
        "R": 1,
    },
    "setup": [
        {
            "cz_error_prob": 0.1,
            "meas_error_prob": 0.1,
            "assign_error_flag": True,
            "assign_error_prob": 0.1,
            "reset_error_prob": 0.1,
            "idle_error_prob": 0.1,
            "T1": 1,
            "T2": 1,
        },
        {
            "qubit": "D1", 
            "sq_error_prob": 0.001 
        }
        {
            "qubit": "D2", 
            "sq_error_prob": 0.002 
        }
    ],
    "name": "test",
    "description": "test description",
}

setup = Setup(setup_dict)
```