def code_conflict(row):
    if row["violent external"] == 1 and row["violent internal"] == 0:
        return "External only"
    elif row["violent external"] == 0 and row["violent internal"] == 1:
        return "Internal only"
    elif row["violent external"] == 1 and row["violent internal"] == 1:
        return "Internal and external"
    else:
        return "No violent conflict"


def code_conflict_collapsed(row):
    if row["violent external"] == 1:
        return "External"
    else:
        return "No external"
