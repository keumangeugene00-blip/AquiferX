"""
AquiferX — Validation du transfert, version 2 : protocole APPARIE.

La v1 comparait les points d'eau OSM a 1000 points tires au hasard dans TOUT le
rectangle de l'AOI -> AUC 0.838. Probleme : les points OSM sont agglutines
(biais de cartographie HOT), alors que le fond aleatoire balaie tout le paysage.
La tache devient "es-tu pres d'un village ?" au lieu de "es-tu sur un forage ?".
Ce n'est PAS comparable a l'AUC du Nigeria (validation croisee spatiale, 0.78).

Cette v2 lance TROIS protocoles sur les memes points, du plus laxiste au plus
severe, et affiche l'ecart. C'est l'ecart qui est informatif.

  A. Fond uniforme sur tout l'AOI          <- la v1 (optimiste)
  B. Fond APPARIE : meme methode qu'au Nigeria (generate_pseudo_absences :
     aleatoire dans l'emprise DES POINTS, a >=1 km de tout point connu)
  C. B + validation croisee spatiale       <- le plus honnete

Prealable :  python osm_points_cameroun_v2.py
Lancer depuis D:\\AquiferX :  python validation_cameroun_v2.py
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

import aquiferx_poc as aq

OSM_CSV = os.path.join("data", "osm_water_points_cameroun.csv")


# =============================================================================
# OUTILS
# =============================================================================
def sample_at_points(ee, stack, lons, lats, feature_cols, tag=""):
    """Echantillonne le stack a des coordonnees libres -> DataFrame de features."""
    feats = [ee.Feature(ee.Geometry.Point([float(lo), float(la)]))
             for lo, la in zip(lons, lats)]
    sampled = stack.sampleRegions(
        collection=ee.FeatureCollection(feats),
        scale=aq.Config.SCALE, geometries=True, tileScale=4
    ).getInfo()["features"]

    if not sampled:
        print(f"[!] {tag} : aucun point echantillonne.")
        return None

    rows = []
    for f in sampled:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        p["x"], p["y"] = lon, lat
        rows.append(p)
    df = pd.DataFrame(rows)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[!] {tag} : bandes absentes {missing}.")
        return None

    df = df.dropna(subset=feature_cols)
    print(f"[OK] {tag} : {len(df)} points exploitables.")
    return df if len(df) else None


def auc_of(model, df_pos, df_neg, feature_cols):
    """AUC presences vs absences, sans decoupage spatial."""
    p_pos = model.predict_proba(df_pos[feature_cols].values)[:, 1]
    p_neg = model.predict_proba(df_neg[feature_cols].values)[:, 1]
    y = np.r_[np.ones(len(p_pos)), np.zeros(len(p_neg))]
    auc = roc_auc_score(y, np.r_[p_pos, p_neg])
    _, pval = mannwhitneyu(p_pos, p_neg, alternative="greater")
    print(f"     presences n={len(p_pos):4d} moy={p_pos.mean():.3f} | "
          f"absences n={len(p_neg):4d} moy={p_neg.mean():.3f}")
    print(f"     AUC = {auc:.3f}  (p = {pval:.1e})")
    return auc


def auc_spatial(model, df_pos, df_neg, feature_cols, n_blocks=4):
    """
    AUC par blocs spatiaux : on evalue bloc par bloc puis on moyenne.
    Neutralise l'autocorrelation spatiale — le plus severe des trois.
    """
    df = pd.concat([df_pos.assign(label=1), df_neg.assign(label=0)],
                   ignore_index=True)
    df["blk"] = aq.spatial_blocks(df, n_blocks=n_blocks)

    aucs = []
    for b in sorted(df["blk"].unique()):
        sub = df[df["blk"] == b]
        if sub["label"].nunique() < 2 or len(sub) < 20:
            continue
        p = model.predict_proba(sub[feature_cols].values)[:, 1]
        aucs.append(roc_auc_score(sub["label"].values, p))

    if not aucs:
        print("     [!] Blocs spatiaux inexploitables.")
        return None
    aucs = np.array(aucs)
    print(f"     AUC par bloc : {np.round(aucs, 3).tolist()}")
    print(f"     AUC spatiale = {aucs.mean():.3f} +/- {aucs.std():.3f} "
          f"({len(aucs)} blocs)")
    return aucs.mean()


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.exists(OSM_CSV):
        print(f"[X] {OSM_CSV} introuvable. Lance osm_points_cameroun_v2.py d'abord.")
        return

    ee = aq.init_gee()

    # --- 1. Entrainement sur le Nigeria (pipeline inchange) ---
    print("\n[1/5] Entrainement sur le Nigeria...")
    presence = aq.load_water_points()
    absence = aq.generate_pseudo_absences(presence)
    truth = aq.build_truth_set(presence, absence)
    stack_nga = aq.build_feature_stack(ee, aq.get_aoi(ee, presence))
    df_train = aq.sample_features(ee, stack_nga, truth)
    model, feature_cols = aq.train_model(df_train)

    # --- 2. Points OSM du Cameroun ---
    print("\n[2/5] Chargement des points OSM...")
    osm = pd.read_csv(OSM_CSV)
    osm_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(osm["lon_deg"], osm["lat_deg"]),
        crs=aq.Config.CRS,
    )
    print(f"[OK] {len(osm_gdf)} points d'eau.")

    # --- Diagnostic d'agglutination ---
    lon_min, lat_min, lon_max, lat_max = aq.Config.TRANSFER_AOI_BOUNDS
    xmin, ymin, xmax, ymax = osm_gdf.total_bounds
    frac = ((xmax - xmin) * (ymax - ymin)) / \
           ((lon_max - lon_min) * (lat_max - lat_min))
    print("\n--- Agglutination des points OSM ---")
    print(f"  Emprise des points : lon {xmin:.2f}-{xmax:.2f} / "
          f"lat {ymin:.2f}-{ymax:.2f}")
    print(f"  AOI complet        : lon {lon_min}-{lon_max} / lat {lat_min}-{lat_max}")
    print(f"  Boite englobante   : {100*frac:.0f} % de la surface de l'AOI")

    # Indice de Clark-Evans : distance moyenne au plus proche voisin, rapportee
    # a ce qu'on attendrait si les points etaient repartis au hasard sur l'AOI.
    #   R ~ 1  -> repartition aleatoire
    #   R < 1  -> agglutines   |   R > 1 -> reguliers
    try:
        from scipy.spatial import cKDTree
        utm = osm_gdf.estimate_utm_crs()
        g = osm_gdf.to_crs(utm)
        xy = np.c_[g.geometry.x.values, g.geometry.y.values]
        d, _ = cKDTree(xy).query(xy, k=2)
        obs = d[:, 1].mean()                       # voisin le plus proche (hors soi)

        lat_mid = 0.5 * (lat_min + lat_max)
        area = ((lon_max - lon_min) * 111320 * np.cos(np.radians(lat_mid))) * \
               ((lat_max - lat_min) * 110540)
        exp = 0.5 / np.sqrt(len(osm_gdf) / area)
        R = obs / exp
        print(f"  Plus proche voisin : observe {obs/1000:.2f} km  vs  "
              f"attendu au hasard {exp/1000:.2f} km")
        print(f"  Indice de Clark-Evans R = {R:.2f}")
        if R < 0.5:
            print("     -> AGGLUTINATION FORTE. Le fond uniforme (protocole A)")
            print("        est forcement optimiste : B est le seul juge valable.")
        elif R < 0.8:
            print("     -> Agglutination moderee. A est un peu optimiste.")
        else:
            print("     -> Repartition proche de l'aleatoire : A n'est pas si")
            print("        biaise que ca. Le 0.84 serait alors plus credible.")
    except Exception as e:
        print(f"  [!] Indice d'agglutination non calcule ({e}).")

    stack_cam = aq.build_feature_stack(ee, ee.Geometry.Rectangle(
        aq.Config.TRANSFER_AOI_BOUNDS))

    # --- 3. Features aux points d'eau ---
    print("\n[3/5] Echantillonnage aux points d'eau...")
    df_pos = sample_at_points(ee, stack_cam, osm["lon_deg"].values,
                              osm["lat_deg"].values, feature_cols,
                              tag="Points d'eau OSM")
    if df_pos is None:
        return

    # --- 4. Protocole A : fond uniforme sur tout l'AOI ---
    print("\n[4/5] PROTOCOLE A — fond uniforme sur tout l'AOI (la v1)")
    rng = np.random.default_rng(aq.Config.RANDOM_STATE)
    df_negA = sample_at_points(ee, stack_cam,
                               rng.uniform(lon_min, lon_max, 1000),
                               rng.uniform(lat_min, lat_max, 1000),
                               feature_cols, tag="  fond uniforme")
    aucA = auc_of(model, df_pos, df_negA, feature_cols) if df_negA is not None else None

    # --- 5. Protocole B : pseudo-absences APPARIEES (methode Nigeria) ---
    print("\n[5/5] PROTOCOLE B — pseudo-absences appariees (methode Nigeria)")
    absB = aq.generate_pseudo_absences(osm_gdf, n=max(len(osm_gdf), 250))
    df_negB = sample_at_points(ee, stack_cam,
                               absB.geometry.x.values, absB.geometry.y.values,
                               feature_cols, tag="  fond apparie")
    aucB = auc_of(model, df_pos, df_negB, feature_cols) if df_negB is not None else None

    # --- Protocole C : B + blocs spatiaux ---
    aucC = None
    if df_negB is not None:
        print("\n     PROTOCOLE C — apparie + blocs spatiaux")
        aucC = auc_spatial(model, df_pos, df_negB, feature_cols)

    # =========================================================================
    # SYNTHESE
    # =========================================================================
    print("\n" + "=" * 62)
    print("SYNTHESE — AUC de transfert sur l'Extreme-Nord Cameroun")
    print("=" * 62)
    def line(tag, v):
        print(f"  {tag:<46s} {'--' if v is None else f'{v:.3f}'}")
    line("A. fond uniforme sur tout l'AOI  (optimiste)", aucA)
    line("B. fond apparie, methode Nigeria (comparable)", aucB)
    line("C. B + blocs spatiaux            (le plus severe)", aucC)
    print(f"  {'Reference Nigeria (CV spatiale, cf. plus haut)':<46s} ~0.78")

    print("\n--- Lecture ---")
    if aucA and aucB:
        drop = aucA - aucB
        print(f"  Ecart A -> B : {drop:+.3f}")
        if drop > 0.12:
            print("  L'essentiel du 0.84 venait de l'agglutination des points OSM.")
            print("  Le chiffre a defendre est B, pas A.")
        elif drop > 0.05:
            print("  Une partie du 0.84 venait de l'agglutination. B est le chiffre")
            print("  a citer.")
        else:
            print("  L'agglutination n'explique presque rien : le signal tient")
            print("  meme a protocole apparie. Resultat solide.")

    ref = aucC or aucB
    if ref is not None:
        print()
        if ref < 0.55:
            print(f"  {ref:.3f} ~ hasard. A protocole egal, le modele NE TRANSFERE")
            print("  PAS. -> reentrainer sur Borno / Adamawa (memes Monts Mandara).")
        elif ref < 0.65:
            print(f"  {ref:.3f} : signal faible. Le transfert tient a peine.")
            print("  A ne pas presenter comme un outil de decision.")
        elif ref < 0.75:
            print(f"  {ref:.3f} : vrai signal, en retrait du Nigeria — degradation")
            print("  attendue en transfert. Resultat honnete et defendable.")
        else:
            print(f"  {ref:.3f} : le transfert tient a protocole egal. C'est un")
            print("  argument fort pour aller chercher les fiches de forage.")

    print("\n  Dans TOUS les cas : aucun forage sec ici. On mesure 'eau connue vs")
    print("  fond de carte', jamais 'productif vs sec'. Le vrai taux de faux")
    print("  positifs reste inconnu tant que tu n'as pas les echecs de forage.")


if __name__ == "__main__":
    main()
