"""
vmp — orphan rescue (STANDALONE, not wired into main pipeline yet).
Reads tip_map_orphans.csv, proposes a tip for the recoverable ones (verified
against the tree), and writes a single `orphans` file for expert confirmation.
"""
import pandas as pd
from Bio import Phylo

orphans = pd.read_csv("data/tip_map_orphans.csv")   # adjust path
tree    = Phylo.read("../data/jaeger2018/world.tre", "newick")
tree_tips = {t.name for t in tree.get_terminals()}

# proposed coding for some.
PROPOSALS = {
    # Arabic and Greek.
    "Arabic":                          ("AA.SEMITIC.STANDARD_ARABIC", "confirm coding"),
    "Greek":                           ("IE.GREEK.GREEK_ANCIENT", "confirm coding"),
    "Attic":                           ("IE.GREEK.GREEK_ANCIENT", "confirm coding"),
    "Doric":                           ("IE.GREEK.GREEK_ANCIENT", "confirm coding"),
    "Medieval Greek":                  ("IE.GREEK.GREEK_ANCIENT", "confirm coding"),
    "Koineic Greek":                   ("IE.GREEK.GREEK_ANCIENT", "confirm coding"),
    "Classical-Middle-Modern Sinitic": ("ST.CHINESE.MIDDLE_CHINESE", "confirm coding"),
    "Literary Chinese":                ("ST.CHINESE.MIDDLE_CHINESE", "confirm coding"),
    "Sinitic":                         ("ST.CHINESE.MIDDLE_CHINESE", "confirm coding"),
    "Vietic":                          ("AuA.VIET_MUONG.VIETNAMESE", "confirm coding"),
    "Viet-Muong":                      ("AuA.VIET_MUONG.VIETNAMESE", "confirm coding"),
    # Category 3: needs decision (Yoruba, English) or language (not family)
    "Yoruba; English":                ("NC.DEFOID.YORUBA", "confirm coding"), 
    "Afro-Asiatic":                    ("", "needs expert: find family"),
    "Indo-European":                   ("", "needs expert: find family"),
    "Sino-Tibetan":                    ("", "needs expert: find family"),
    "Classical Indo-European":         ("", "needs expert: find family"),
    # Category 4: unknown or not in tree.
    "Unknown":                         ("", "cannot be rescued"),
    "Pidgin":                          ("", "cannot be rescued"),
}

def propose(tag):
    return PROPOSALS.get(tag, ("", "needs expert: find nearest available tip"))

orphans[["PROPOSED","PROPOSED_REASON"]] = orphans["deepest_tag"].apply(
    lambda t: pd.Series(propose(t)))

# columns 
out = pd.DataFrame({
    "entry_id":    orphans["entry_id"],
    "entry_name":  orphans["entry_name"],
    "deepest_tag": orphans["deepest_tag"],
    "PROPOSED_tip": orphans["PROPOSED"], 
    "PROPOSED_reason": orphans["PROPOSED_REASON"],
    "CONFIRMED": "",
    "CODE": "",
    "CODER": "",
})

# custom priority by reason, then entry_id
REASON_ORDER = {
    "confirm coding": 0, # quick confirms (top)
    "needs expert: find family": 1, # needs expert: find family/language
    "needs expert: find nearest available tip": 2,
    "cannot be rescued": 3, # bottom
}
out["_rank"] = out["PROPOSED_reason"].map(REASON_ORDER)
out = (out.sort_values(["_rank", "deepest_tag"], ascending=[True, True])
          .drop(columns="_rank"))
out.groupby("PROPOSED_reason").size()

'''
CONFIRM CODING: n=72
Needs expert (family): n=5
Needs expert (tip): n=9
Cannot be rescued: n=6
'''

out.to_csv("data/orphans_to_confirm.csv", index=False)