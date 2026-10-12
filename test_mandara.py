"""
AquiferX — Proximite geologique contre volume de donnees.

QUESTION : vaut-il mieux entrainer sur 1000 puits a 550 km (Kano/Katsina) ou sur
287 puits a 90 km, dans les MEMES Monts Mandara que la zone cible ?

Trois bras d'entrainement, evalues sur le MEME jeu de test camerounais
(251 forages OSM vs 251 pseudo-absences appariees) — seule la source
d'entrainement change :

  1. Kano/Katsina           ~1000 pts, socle nigerian, 550 km a l'ouest
  2. Mandara nigerian        287 pts, MEME massif, 90 km
  3. Kano/Katsina + Mandara ~1287 pts, volume + proximite

Chaque region genere ses propres pseudo-absences et est echantillonnee sur son
propre stack (evite une AOI geante et un composite S2 inutilement large).

Reference a battre : protocole C (apparie + blocs spatiaux) = 0.718 avec Kano/Katsina.

Prealable :  python osm_points_cameroun_v2.py
Lancer depuis D:\\AquiferX :  python test_mandara.py
Duree : ~8-10 min (trois entrainements).
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

import aquiferx_poc as aq

OSM_CSV = os.path.join("data", "osm_water_points_cameroun.csv")

# Versant nigerian des Monts Mandara : Gwoza, Michika, Madagali.
# Colle a l'AOI camerounaise, qui demarre a 13.5 E.
MANDARA_BOX = [12.9, 10.0, 13.9, 11.6]          # [lon_min, lat_min, lon_max, lat_max]


# =============================================================================
# CHARGEMENT
# =============================================================================
def _clean(path=aq.Config.WPDX_CSV):
    """Lecture + nettoyage WPdx, identiques a load_water_points."""
    df = pd.read_csv(path, low_memory=False)
    df["lat_deg"] = pd.to_numeric(df["lat_deg"], errors="coerce")
    df["lon_deg"] = pd.to_numeric(df["lon_deg"], errors="coerce")
    df = df.dropna(subset=["lat_deg", "lon_deg"])
    return df[df["water_source_category"].isin(aq.Config.GROUNDWATER_SOURCES)]


def load_states(df, states, cap=aq.Config.MAX_PRESENCE):
    sub = df[df["clean_adm1"].isin(states)]
    if cap and len(sub) > cap:
        sub = sub.sample(n=cap, random_state=aq.Config.RANDOM_STATE)
    return _to_gdf(sub, f"{'/'.join(states)}")


def load_bbox(df, bbox):
    lo1, la1, lo2, la2 = bbox
    sub = df[df["lon_deg"].between(lo1, lo2) & df["lat_deg"].between(la1, la2)]
    return _to_gdf(sub, "Mandara")


def _to_gdf(sub, tag):
    gdf = gpd.GeoDataFrame(
        sub, geometry=gpd.points_from_xy(sub["lon_deg"], sub["lat_deg"]),
        crs=aq.Config.CRS,
    )
    gdf["label"] = 1
    print(f"    [OK] {tag} : {len(gdf)} presences.")
    return gdf


# =============================================================================
# OUTILS
# =============================================================================
def sample_at_points(ee, stack, lons, lats, tag=""):
    feats = [ee.Feature(ee.Geometry.Point([float(lo), float(la)]))
             for lo, la in zip(lons, lats)]
    sampled = stack.sampleRegions(
        collection=ee.FeatureCollection(feats),
        scale=aq.Config.SCALE, geometries=True, tileScale=4
    ).getInfo()["features"]
    if not sampled:
        print(f"    [!] {tag} : aucun point echantillonne.")
        return None
    rows = []
    for f in sampled:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        p["x"], p["y"] = lon, lat
        rows.append(p)
    df = pd.DataFrame(rows).dropna()
    print(f"    [OK] {tag} : {len(df)} points.")
    return df if len(df) else None


def build_training_df(ee, regions):
    """
    regions : liste de (nom, gdf_presences).
    Chaque region : pseudo-absences + stack + echantillonnage sur SON emprise,
    puis concatenation. Les absences restent ainsi locales a leur region.
    """
    dfs = []
    for name, pres in regions:
        print(f"  [region] {name}")
        np.random.seed(aq.Config.RANDOM_STATE)     # reproductible (cf. remarque)
        absence = aq.generate_pseudo_absences(pres)
        truth = aq.build_truth_set(pres, absence)
        stack = aq.build_feature_stack(ee, aq.get_aoi(ee, pres))
        d = aq.sample_features(ee, stack, truth)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def eval_cameroun(model, feature_cols, df_pos, df_neg):
    """Protocole B (apparie) et C (apparie + blocs spatiaux). Rend (aucB, aucC)."""
    p_pos = model.predict_proba(df_pos[feature_cols].values)[:, 1]
    p_neg = model.predict_proba(df_neg[feature_cols].values)[:, 1]
    y = np.r_[np.ones(len(p_pos)), np.zeros(len(p_neg))]
    aucB = roc_auc_score(y, np.r_[p_pos, p_neg])
    _, pval = mannwhitneyu(p_pos, p_neg, alternative="greater")
    print(f"    B (apparie)          AUC = {aucB:.3f}   (p = {pval:.1e})")

    df = pd.concat([df_pos.assign(label=1), df_neg.assign(label=0)],
                   ignore_index=True)
    df["blk"] = aq.spatial_blocks(df, n_blocks=4)
    aucs = []
    for b in sorted(df["blk"].unique()):
        sub = df[df["blk"] == b]
        if sub["label"].nunique() < 2 or len(sub) < 20:
            continue
        aucs.append(roc_auc_score(
            sub["label"].values,
            model.predict_proba(sub[feature_cols].values)[:, 1]))
    if not aucs:
        return aucB, None
    aucs = np.array(aucs)
    print(f"    C (apparie+spatial)  AUC = {aucs.mean():.3f} +/- {aucs.std():.3f}"
          f"   {np.round(aucs, 3).tolist()}")
    return aucB, aucs.mean()


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.exists(OSM_CSV):
        print(f"[X] {OSM_CSV} introuvable. Lance osm_points_cameroun_v2.py d'abord.")
        return

    ee = aq.init_gee()

    # --- Jeu de test camerounais : construit UNE fois, partage par les 3 bras ---
    print("\n=== Jeu de test : Extreme-Nord Cameroun ===")
    osm = pd.read_csv(OSM_CSV)
    osm_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(osm["lon_deg"], osm["lat_deg"]),
        crs=aq.Config.CRS,
    )
    stack_cam = aq.build_feature_stack(
        ee, ee.Geometry.Rectangle(aq.Config.TRANSFER_AOI_BOUNDS))

    df_pos = sample_at_points(ee, stack_cam, osm["lon_deg"].values,
                              osm["lat_deg"].values, tag="forages OSM")
    if df_pos is None:
        return

    np.random.seed(aq.Config.RANDOM_STATE)
    abs_cam = aq.generate_pseudo_absences(osm_gdf, n=len(osm_gdf))
    df_neg = sample_at_points(ee, stack_cam, abs_cam.geometry.x.values,
                              abs_cam.geometry.y.values, tag="pseudo-absences")
    if df_neg is None:
        return
    print("  -> jeu de test fige : identique pour les trois bras.")

    # --- Definition des trois bras ---
    raw = _clean()
    print("\n=== Sources d'entrainement ===")
    pres_kk = load_states(raw, aq.Config.ZONE_STATES)
    pres_md = load_bbox(raw, MANDARA_BOX)

    if len(pres_md) < 50:
        print(f"[!] Seulement {len(pres_md)} points dans les Mandara : "
              "trop peu pour entrainer. Test abandonne.")
        return

    arms = [
        (f"1. Kano/Katsina  ({len(pres_kk)} pts, 550 km)",
         [("Kano/Katsina", pres_kk)]),
        (f"2. Mandara NG    ({len(pres_md)} pts, 90 km, MEME socle)",
         [("Mandara", pres_md)]),
        (f"3. Les deux      ({len(pres_kk)+len(pres_md)} pts)",
         [("Kano/Katsina", pres_kk), ("Mandara", pres_md)]),
    ]

    # --- Boucle sur les bras ---
    results = {}
    feature_cols = None
    for label, regions in arms:
        print("\n" + "=" * 62)
        print(f"BRAS {label}")
        print("=" * 62)
        try:
            df_train = build_training_df(ee, regions)
            model, feature_cols = aq.train_model(df_train)
            print("  --- transfert vers le Cameroun ---")
            results[label] = eval_cameroun(model, feature_cols, df_pos, df_neg)
        except Exception as e:
            print(f"  [X] Bras echoue : {e}")
            results[label] = (None, None)

    # =========================================================================
    # SYNTHESE
    # =========================================================================
    print("\n" + "=" * 62)
    print("SYNTHESE — transfert vers l'Extreme-Nord Cameroun")
    print("  (meme jeu de test partout ; seul l'entrainement change)")
    print("=" * 62)
    print(f"  {'Entrainement':<44s} {'B':>7s} {'C':>7s}")
    for label, (b, c) in results.items():
        fb = "--" if b is None else f"{b:.3f}"
        fc = "--" if c is None else f"{c:.3f}"
        print(f"  {label:<44s} {fb:>7s} {fc:>7s}")
    print("\n  C = apparie + blocs spatiaux = le chiffre a citer.")

    # --- Lecture ---
    cs = {k: v[1] for k, v in results.items() if v[1] is not None}
    if len(cs) < 2:
        print("\n[!] Pas assez de bras exploitables pour conclure.")
        return

    best = max(cs, key=cs.get)
    ref = list(cs.values())[0]                  # bras 1 = reference
    print("\n--- Lecture ---")
    print(f"  Meilleur : {best}  (C = {cs[best]:.3f})")
    spread = max(cs.values()) - min(cs.values())
    if spread < 0.03:
        print(f"  Ecart max entre bras : {spread:.3f} — NEGLIGEABLE.")
        print("  Ni la proximite ni le volume ne changent quoi que ce soit.")
        print("  -> Garde Kano/Katsina et arrete de creuser ici : le plafond")
        print("     n'est pas dans le choix de la zone d'entrainement.")
    else:
        print(f"  Ecart max entre bras : {spread:.3f}")
        if cs[best] > ref + 0.03:
            print(f"  Le bras '{best}' bat la reference de {cs[best]-ref:+.3f}.")
            print("  -> Gain reel. A basculer dans Config.")
        else:
            print("  La reference tient. La proximite geologique ne compense pas")
            print("  le manque de volume.")

    print("\n  Rappel constant : aucun forage sec ici. Ces AUC mesurent 'eau")
    print("  connue vs fond de carte', pas 'productif vs sec'. Aucun de ces")
    print("  bras ne repond a la seule question qui compte pour un foreur.")


if __name__ == "__main__":
    main()
