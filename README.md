# Replication of Sosis et al 2007

Conceptual replication of "Scars for war: Evaluating alternative signaling explanations to cross-cultural variance in ritual costs" (Sosis et al. 2007) using the Standard Cross-Cultural Sample of Religion (SCCSR.v3) from the Database of Religious History (https://religiondatabase.org)

# Reproduce

To reproduce the results, follow these steps:

0. Activate environment: `mamba env create -f environment.yml` (`mamba activate drh-env`)
1. Unzip `drh_tables.zip` in the `data/raw` folder. Alternatively, this file can be obtained from the official SCCSR.v3 (https://zenodo.org/records/18394095).
2. Follow `/preprocessing` steps
3. Follow `/analysis` steps (scripts ending in `hraf.py` can be skipped)

# Funding

The DRH has enjoyed generous support from the John Templeton Foundation (JTF), Templeton Religion Trust (TRT), and Canada’s SSHRC. Thanks are owed to the hundreds of experts who have contributed entries and our editorial team.

# Contact

Victor Møller Poulsen (victormoeller@gmail.com). Feel free to open an issue.
