
# Tools

## `muspinsim-gen`

`muspinsim-gen` is a command line tool made to help construct an input file given a cell structure file as input. At the moment it can take an input structure (defined in a file supported by `ase` e.g. a .cell or .cif), find a requested number of nearest neighbour atoms to the muon, and then output the muspinsim config defining these atoms and the dipolar interactions with the muon. It can also (optionally) take a file containing output defining EFG tensors from `GIPAW` in order to include quadrupole interactions.

The available command line options can be found by invoking `muspinsim-gen --help`.

### Example
Taking a cell file named `V3Si.cell` you may use
```bash
muspinsim-gen ./V3Si.cell 6 --dipolar --out V3Si.in
```

!!! note
    The muon is assumed to be defined in the structure as `H`, this can be modified by the command line option `--muon_symbol MUON_SYMBOL`.

This takes the cell file, then iteratively expands the structure outwards to find the closest 6 atoms to the muon. Then it computes vectors between each of the found atoms and the muon before outputting the muspinsim config into the file `V3Si.in` which looks like

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

In this case, we may also decide to ignore the silicon atoms as they have a spin of 0. This can be achieved by using `--ignore_symbol Si`. Additional ignored symbols can be added by repeating this option. To add quadrupole interactions you may use `--quadrupolar GIPAW_FILEPATH`. This file should contain the output of GIPAW from calculating the EFG tensors on the same structure. These will then be matched to the found atoms and will be output into the file.