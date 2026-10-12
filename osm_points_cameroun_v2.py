"""
AquiferX — Extraction des points d'eau OpenStreetMap sur l'Extreme-Nord Cameroun.
Version 2 : corrige le 406 (User-Agent) et gere le 429 (relances + miroirs).

ATTENTION - limite methodologique : ces points sont "presence-only". Ils disent
ou il Y A de l'eau, jamais ou il n'y en a PAS (aucun forage sec). Ils servent a
VALIDER (le modele place-t-il l'eau connue en zone favorable ?), pas a
recalibrer. Pour ca, il faut des fiches de forage avec les echecs.

Lancer depuis D:\\AquiferX :  python osm_points_cameroun_v2.py
Sortie : data/osm_water_points_cameroun.csv
"""
import os
import sys
import time

import pandas as pd
import requests

# =============================================================================
# CONFIG
# =============================================================================
# Zone identique a Config.TRANSFER_AOI_BOUNDS d'aquiferx_poc.py
AOI = [13.5, 10.2, 14.8, 11.4]          # [lon_min, lat_min, lon_max, lat_max]
OUT_CSV = os.path.join("data", "osm_water_points_cameroun.csv")

# Overpass exige un User-Agent identifiable : sans lui -> 406 Not Acceptable.
HEADERS = {
    "User-Agent": "AquiferX-research/0.2 (groundwater potential mapping; Python)",
    "Accept": "application/json",
}

# Plusieurs instances publiques : si l'une refuse, on passe a la suivante.
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]
MAX_RETRIES = 2                          # essais par serveur

# Tags OSM d'interet. La categorie suit la convention WPdx d'aquiferx_poc
# (GROUNDWATER_SOURCES = ["Well", "Spring"]).
TAGS = [
    ("man_made", "water_well",      "Well"),
    ("man_made", "borehole",        "Well"),
    ("man_made", "water_works",     "Other"),
    ("natural",  "spring",          "Spring"),
    ("amenity",  "water_point",     "Other"),
    ("amenity",  "drinking_water",  "Other"),
    ("man_made", "water_tap",       "Other"),
]


# =============================================================================
# 1. REQUETE OVERPASS
# =============================================================================
def build_query(aoi):
    """Construit la requete Overpass QL. Bbox Overpass = (Sud,Ouest,Nord,Est)."""
    lon_min, lat_min, lon_max, lat_max = aoi
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    lines = []
    for key, val, _cat in TAGS:
        lines.append(f'  node["{key}"="{val}"]({bbox});')
        lines.append(f'  way["{key}"="{val}"]({bbox});')
    body = "\n".join(lines)
    return f"[out:json][timeout:180];\n(\n{body}\n);\nout center;"


def fetch(query):
    """Interroge Overpass : plusieurs serveurs, plusieurs essais, attente sur 429."""
    for i, url in enumerate(ENDPOINTS, 1):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[i] Overpass {i}/{len(ENDPOINTS)} (essai {attempt}) : {url}")
                r = requests.post(url, data={"data": query},
                                  headers=HEADERS, timeout=200)

                if r.status_code == 429:               # rate limit -> on patiente
                    wait = 25 * attempt
                    print(f"[!] 429 (trop de requetes). Attente {wait} s...")
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                return r.json().get("elements", [])

            except Exception as e:
                print(f"[!] Echec ({e}).")
                if attempt < MAX_RETRIES:
                    time.sleep(8)
        print("    -> serveur suivant.\n")
    return None


# =============================================================================
# 2. PARSING
# =============================================================================
def categorize(tags):
    """Rend (categorie, tag_source) selon les tags OSM de l'element."""
    for key, val, cat in TAGS:
        if tags.get(key) == val:
            return cat, f"{key}={val}"
    return "Other", "?"


def parse(elements):
    rows = []
    for el in elements:
        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:                                    # way -> centroide (out center)
            c = el.get("center") or {}
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue

        tags = el.get("tags", {})
        cat, src = categorize(tags)

        # Un tag pump signe presque toujours un forage = eau souterraine
        if cat == "Other" and tags.get("pump"):
            cat = "Well"

        rows.append({
            "lat_deg": lat,
            "lon_deg": lon,
            "water_source_category": cat,
            "osm_tag": src,
            "osm_id": el.get("id"),
            "osm_type": el.get("type"),
            "name": tags.get("name", ""),
            "pump": tags.get("pump", ""),
            "status_clean": "Unknown",           # OSM ne dit rien de la fonctionnalite
            "clean_adm1": "Extreme-Nord",
        })
    return pd.DataFrame(rows)


# =============================================================================
# 3. MAIN
# =============================================================================
def main():
    print(f"[i] Zone : lon {AOI[0]}-{AOI[2]} / lat {AOI[1]}-{AOI[3]} "
          "(Extreme-Nord, Cameroun)\n")

    elements = fetch(build_query(AOI))

    if elements is None:
        print("\n[X] Aucun serveur Overpass n'a repondu.")
        print("    -> Relance dans 10-15 min (les instances publiques saturent),")
        print("       ou passe au plan B : le fichier HOT/OSM du Cameroun sur HDX.")
        sys.exit(1)

    print(f"\n[OK] {len(elements)} elements bruts recus.\n")

    if not elements:
        print("[!] Aucun point d'eau trouve sur cette zone dans OpenStreetMap.")
        print("    -> Zone peu cartographiee. Pistes : elargir l'AOI, ou HDX.")
        return

    df = parse(elements)
    if df.empty:
        print("[!] Elements recus mais aucune coordonnee exploitable.")
        return

    # --- Recapitulatif ---
    print("--- Repartition par categorie ---")
    print(df["water_source_category"].value_counts().to_string())
    print("\n--- Repartition par tag OSM ---")
    print(df["osm_tag"].value_counts().to_string())

    gw = df[df["water_source_category"].isin(["Well", "Spring"])]
    print(f"\n[i] Points d'eau SOUTERRAINE (Well/Spring) : {len(gw)} / {len(df)}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[OK] Sauvegarde -> {OUT_CSV}")

    # --- Lecture honnete du resultat ---
    print("\n--- Ce que ca vaut ---")
    if len(gw) >= 100:
        print(f"  {len(gw)} points souterrains : de quoi tenter une vraie")
        print("  validation du transfert (tombent-ils en zone favorable ?).")
    elif len(gw) >= 20:
        print(f"  {len(gw)} points souterrains : peu, mais suffisant pour un")
        print("  premier controle de coherence. A interpreter avec prudence.")
    else:
        print(f"  Seulement {len(gw)} points souterrains : trop peu pour conclure.")
        print("  La zone est mal cartographiee dans OSM.")
    print("  Rappel : presence-only, aucun forage sec -> impossible de mesurer")
    print("  les faux positifs. Ca ne remplace pas des fiches de forage.")


if __name__ == "__main__":
    main()
