# Replication of Sosis et al 2007
Conceptual replication of "Scars for war: Evaluating alternative signaling explanations to cross-cultural variance in ritual costs" (Sosis et al. 2007) using the SCCSR.v1 from the Database of Religious History (https://religiondatabase.org)

# Reproduce
To reproduce the results, follow these steps: 
1. Unzip drh_tables.zip in the `data/raw` folder. Alternatively, this file can be optained from the official SCCSR.v1 (https://zenodo.org/records/12572187) where it is located in `data/clean` within this archive. The relevant tables from the SCCSR.v1 are `answerset.csv` (too large to push to GitHub), `entry_data.csv`, `questionrelation.csv`, and `region_data.csv`. These need to be in `data/raw` to reproduce the analysis. 
2. Follow `/preprocessing` steps 
3. Follow `/analysis` steps (scripts ending in `hraf.py` can be skipped)

# Funding 
The DRH has enjoyed generous support from the John Templeton Foundation (JTF), Templeton Religion Trust (TRT), and Canada’s SSHRC. Thanks are owed to the hundreds of experts who have contributed entries and our editorial team. 

# Contact 
Victor Møller Poulsen (victormoeller@gmail.com). Feel free to open an issue. 