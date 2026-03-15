#!/usr/bin/env python3
"""Generate a Gen 3-accurate pokemon_base_stats.csv.

Uses the current CSV as a base and applies corrections from Pokemon Showdown's
authoritative generation mod data (gen5/pokedex.ts and gen6/pokedex.ts) to
revert post-Gen 3 stat and type changes.

Sources:
- gen5/pokedex.ts: Pre-Gen 6 stats (Pokemon that changed in Gen 6 XY)
- gen6/pokedex.ts: Pre-Gen 7 stats (Pokemon that changed in Gen 7 SM)
- No Gen 1-3 Pokemon had stat changes in Gen 4, 5, 8, or 9

Run once to regenerate the CSV:
    python scripts/generate_gen3_pokedex.py
"""

import csv
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "pokedex" / "pokemon_base_stats.csv"

# ── Stat corrections ──────────────────────────────────────────────────────
# Format: "Name" -> (HP, Atk, Def, SpA, SpD, Spe)
# From gen5/pokedex.ts: Pokemon whose stats changed in Gen 6 (XY)
# These are the pre-Gen 6 values = Gen 3 values
STAT_CORRECTIONS_GEN6 = {
    "Butterfree":  (60, 45, 50, 80, 80, 70),
    "Beedrill":    (65, 80, 40, 45, 80, 75),
    "Pidgeot":     (83, 80, 75, 70, 70, 91),
    "Pikachu":     (35, 55, 30, 50, 40, 90),
    "Raichu":      (60, 90, 55, 90, 80, 100),
    "Nidoqueen":   (90, 82, 87, 75, 85, 76),
    "Nidoking":    (81, 92, 77, 85, 75, 85),
    "Clefable":    (95, 70, 73, 85, 90, 60),
    "Vileplume":   (75, 80, 85, 100, 90, 50),
    "Poliwrath":   (90, 85, 95, 70, 90, 70),
    "Alakazam":    (55, 50, 45, 135, 85, 120),
    "Victreebel":  (80, 105, 65, 100, 60, 70),
    "Golem":       (80, 110, 130, 55, 65, 45),
    "Ampharos":    (90, 75, 75, 115, 90, 55),
    "Bellossom":   (75, 80, 85, 90, 100, 50),
    "Azumarill":   (100, 50, 80, 50, 80, 50),
    "Jumpluff":    (75, 55, 70, 55, 85, 110),
    "Wigglytuff":  (140, 70, 45, 75, 50, 45),
    "Beautifly":   (60, 70, 50, 90, 50, 65),
    "Exploud":     (104, 91, 63, 91, 63, 68),
}

# From gen6/pokedex.ts: Pokemon whose stats changed in Gen 7 (SM)
# These are the pre-Gen 7 values = Gen 3 values
STAT_CORRECTIONS_GEN7 = {
    "Arbok":       (60, 85, 69, 65, 79, 80),
    "Dugtrio":     (35, 80, 50, 50, 70, 120),
    "Farfetch'd":  (52, 65, 55, 58, 62, 60),
    "Dodrio":      (60, 110, 70, 60, 60, 100),
    "Electrode":   (60, 50, 70, 80, 80, 140),
    "Exeggutor":   (95, 95, 85, 125, 65, 55),
    "Noctowl":     (100, 50, 50, 76, 96, 70),
    "Ariados":     (70, 90, 70, 60, 60, 40),
    "Qwilfish":    (65, 95, 75, 55, 55, 85),
    "Magcargo":    (50, 50, 120, 80, 80, 30),
    "Corsola":     (55, 55, 85, 65, 85, 35),
    "Mantine":     (65, 40, 70, 80, 140, 70),
    "Swellow":     (60, 85, 60, 50, 50, 125),
    "Pelipper":    (60, 50, 100, 85, 70, 65),
    "Masquerain":  (70, 60, 62, 80, 82, 60),
    "Delcatty":    (70, 65, 65, 55, 55, 70),
    "Volbeat":     (65, 73, 55, 47, 75, 85),
    "Illumise":    (65, 47, 55, 73, 75, 85),
    "Lunatone":    (70, 55, 65, 95, 85, 70),
    "Solrock":     (70, 95, 85, 55, 65, 70),
    "Chimecho":    (65, 50, 70, 95, 80, 65),
}

ALL_STAT_CORRECTIONS = {**STAT_CORRECTIONS_GEN6, **STAT_CORRECTIONS_GEN7}

# ── Type corrections ──────────────────────────────────────────────────────
# Fairy type didn't exist in Gen 3. These Pokemon had type changes in Gen 6.
# Format: "Name" -> (Type1, Type2)
TYPE_CORRECTIONS = {
    "Cleffa":     ("Normal", " "),
    "Clefairy":   ("Normal", " "),
    "Clefable":   ("Normal", " "),
    "Igglybuff":  ("Normal", " "),
    "Jigglypuff": ("Normal", " "),
    "Wigglytuff": ("Normal", " "),
    "Togepi":     ("Normal", " "),
    "Togetic":    ("Normal", "Flying"),
    "Mr. Mime":   ("Psychic", " "),
    "Snubbull":   ("Normal", " "),
    "Granbull":   ("Normal", " "),
    "Ralts":      ("Psychic", " "),
    "Kirlia":     ("Psychic", " "),
    "Gardevoir":  ("Psychic", " "),
    "Azurill":    ("Normal", " "),
    "Marill":     ("Water", " "),
    "Azumarill":  ("Water", " "),
    "Mawile":     ("Steel", " "),
}


def main():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            name = row["Name"].strip().strip('"')
            form = row["Form"].strip().strip('"')

            # Only process Gen 1-3 Pokemon
            gen = int(row.get("Generation", "0").strip().strip('"'))
            if gen > 3:
                continue

            # Apply stat corrections (only to base forms, not alternate forms)
            if name in ALL_STAT_CORRECTIONS and form.strip() in ("", " "):
                hp, atk, dfn, spa, spd, spe = ALL_STAT_CORRECTIONS[name]
                row["HP"] = str(hp)
                row["Attack"] = str(atk)
                row["Defense"] = str(dfn)
                row["Sp. Atk"] = str(spa)
                row["Sp. Def"] = str(spd)
                row["Speed"] = str(spe)
                row["Total"] = str(hp + atk + dfn + spa + spd + spe)

            # Apply type corrections
            if name in TYPE_CORRECTIONS and form.strip() in ("", " "):
                type1, type2 = TYPE_CORRECTIONS[name]
                row["Type1"] = type1
                row["Type2"] = type2

            rows.append(row)

    # Write corrected CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Print summary
    stat_changes = [n for n in ALL_STAT_CORRECTIONS if any(
        r["Name"].strip().strip('"') == n for r in rows
    )]
    type_changes = [n for n in TYPE_CORRECTIONS if any(
        r["Name"].strip().strip('"') == n for r in rows
    )]
    print(f"Wrote {len(rows)} entries to {CSV_PATH}")
    print(f"Applied {len(stat_changes)} stat corrections")
    print(f"Applied {len(type_changes)} type corrections")

    # Print details
    print("\nStat corrections applied:")
    for name, stats in sorted(ALL_STAT_CORRECTIONS.items()):
        print(f"  {name}: HP={stats[0]} Atk={stats[1]} Def={stats[2]} "
              f"SpA={stats[3]} SpD={stats[4]} Spe={stats[5]}")

    print("\nType corrections applied:")
    for name, types in sorted(TYPE_CORRECTIONS.items()):
        t2 = types[1] if types[1].strip() else "—"
        print(f"  {name}: {types[0]}/{t2}")


if __name__ == "__main__":
    main()
