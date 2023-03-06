
# Tools

## `muspinsim-gen`

`muspinsim-gen` is a command line tool made to help construct an input file given a cell structure file as input. At the moment it can take an input structure (defined in a file supported by `ase` e.g. a .cell, .cif or .magres), find a requested number of nearest neighbour atoms to the muon, and then output the muspinsim config defining these atoms and any interactions requested (currently either dipole between the muon and atom, or quadrupole interactions for the atoms).

The available command line options can be found by invoking `muspinsim-gen --help`.

### Example
Taking a cell file named `V3Si.cell` you may use
```bash
muspinsim-gen ./V3Si.cell 6 --dipolar --out V3Si.in
```

!!! note
    The muon is assumed to be defined in the structure as `H`, this can be modified by the command line option `--muon_symbol MUON_SYMBOL`.

This takes the cell file, then iteratively expands the structure outwards to find the closest 6 atoms to the muon. Then it computes vectors between each of the found atoms and the muon before outputting the muspinsim config with dipole interactions between the muon and each of the atoms into the file `V3Si.in`. This output looks like

```plaintext
spins
    mu V V V V Si Si
dipolar 1 2
    5.8867821595143255e-06 -1.2410298376667999 1.2417097773151198
dipolar 1 3
    -1.2410421175142399 1.7637655919999997e-05 -1.2417029559655202
dipolar 1 4
    1.2410539563139198 1.7614965359999998e-05 -1.2417021305964
dipolar 1 5
    5.8811095198230134e-06 1.2410652462856788 1.2417072770990398
dipolar 1 6
    5.84849183971059e-06 -2.377947784716 -1.18857807822936
dipolar 1 7
    -2.37795896407128 1.786314336e-05 1.18858129319808
```

!!! warning
    By default, MuSpinSim assumes that the atoms present have a non-zero spin (in this case, 29Si with a spin of 1/2). To avoid interactions between the muon and Si (effectively treating it as the more abundant, spin-zero 28Si) we can use `--ignore_symbol Si`. Additional ignored symbols can be added by repeating this option.

#### Adding quadrupole interactions
To add quadrupole interactions you have two options. First you may use a Magres file containing the calculated EFG tensors from CASTEP as your input structure file. In this case you can simply add `--quadrupolar` e.g.

```bash
muspinsim-gen ./V3Si.magres 6 --dipolar --quadrupolar --out V3Si.in
```

Alternatively if you are using GIPAW from Quantum ESPRESSO to calculate the EFG tensors, you may use `--quadrupolar GIPAW_FILEPATH`. This file should contain the output of GIPAW from calculating the EFG tensors on the same structure. These will then be matched to the found atoms and will be output into the file. e.g.

```bash
muspinsim-gen ./V3Si.cell 6 --dipolar --quadrupolar V3Si_EFGs.out --out V3Si.in
```

#### Adding dipole interactions between the atoms

The above example only adds dipole interactions between the muon and each found atom from the structure. You may also wish to add dipole interactions between each of these atoms themselves i.e. between V-V, V-Si and Si-Si in the example above. To do this simply add `--include_interatomic` to the command.

### Advanced usage as a library

When using muspinsim as a library it is possible to use this tool and modify its behaviour for making adjustments to the selected atoms. This may be useful in cases where DFT may underestimate the distance between the muon and its nearest neighbours. In this case you may use code like the following to make adjustments before the tool generates the output.

```python

import sys

from muspinsim.tools.generator import GeneratorToolParams, _run_generator_tool

def select_atoms(params: GeneratorToolParams):
    # Select the closest atoms
    structure = params.structure

    selected_atoms = structure.compute_closest(
        number=params.number_closest,
        ignored_symbols=structure.symbols_zero_spin + params.additional_ignored_symbols,
        max_layer=params.max_layer,
    )

    muon = structure.muon

    # Move the 2 closest to be 1.21 Angstrom away from the muon
    structure.move_atom(muon, selected_atoms[0], 1.21)
    structure.move_atom(muon, selected_atoms[1], 1.21)

    return selected_atoms

def main():
    """Entrypoint for command line tool"""

    _run_generator_tool(sys.argv[1:], select_atoms)


if __name__ == "__main__":
    main()
```

This will retain the command line functionality, but moves the first two found atoms to be a distance of 1.21 Angstrom from the muon.