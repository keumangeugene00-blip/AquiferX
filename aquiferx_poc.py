"""
==============================================================================
 AquiferX — Pipeline PoC de dé-risquage du forage d'eau souterraine
==============================================================================
Objectif : produire une carte de probabilité de présence d'eau souterraine
            sur une zone pilote, à partir de données satellites gratuites et
            d'un inventaire de points d'eau (vérité terrain).

Méthode : groundwater potential mapping (présence / pseudo-absence) avec un
          modèle d'ensemble (Random Forest / Gradient Boosting), validation
          croisée SPATIALE et explicabilité SHAP.

Étapes du pipeline :
    1. Initialisation Google Earth Engine
    2. Vérité terrain  : points d'eau (WPdx) + génération de pseudo-absences
    3. Features         : stack multibande GEE (relief, sols, S2, pluie, ...)
    4. Modèle           : entraînement + CV spatiale + SHAP
    5. Prédiction       : carte raster de probabilité + validation

Dépendances :
    earthengine-api, geemap, geopandas, shapely, numpy, pandas,
    scikit-learn, shap, rasterio, matplotlib

    pip install earthengine-api geemap geopandas scikit-learn shap rasterio matplotlib

Auteur : KEUMANG TCHATAT Eugène
==============================================================================
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Earth Engine (importé dans init_gee pour permettre l'exécution partielle)
# import ee

# Machine learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # --- Projet Earth Engine ---
    EE_PROJECT = "static-lead-444117-q5"     # ton projet GEE non commercial

    # --- Zone pilote : socle cristallin du nord Nigeria (Kano / Katsina) ---
    # Même contexte sahélien / socle que l'Extrême-Nord camerounais, mais avec
    # des données WPdx abondantes. On entraine ici, puis on transfere au Cameroun.
    ZONE_STATES = ["Kano", "Katsina"]        # filtre sur la colonne clean_adm1
    GROUNDWATER_SOURCES = ["Well", "Spring"] # eau souterraine (exclut Piped Water)
    AOI_MARGIN_DEG = 0.1                      # marge (deg) autour des points -> AOI
    MAX_PRESENCE = 1000                       # sous-échantillon (vitesse / limite getInfo)

    # --- Zone de TRANSFERT : Extrême-Nord Cameroun (prédiction uniquement) ---
    # Le modèle entraîné sur le socle nigérian y est appliqué. Résultat à
    # valider avec des données de terrain camerounaises (None = désactivé).
    TRANSFER_AOI_BOUNDS = [13.5, 10.2, 14.8, 11.4]   # autour de Maroua / Mokolo
    CRS = "EPSG:4326"
    SCALE = 100                              # résolution d'échantillonnage (m) — 100m suffit
                                             # pour des facteurs contextuels et accélère bcp GEE

    # --- Fenêtre temporelle pour l'imagerie ---
    DATE_START = "2023-01-01"
    DATE_END   = "2023-12-31"
    S2_CLOUD_MAX = 20                        # % nuages max pour Sentinel-2

    # --- Vérité terrain ---
    WPDX_CSV = "data/wpdx_nga.csv"           # export wpdx_nga : data.humdata.org/dataset/wpdx_nga
    MIN_DIST_PSEUDO_ABS = 1000               # distance min (m) présence <-> absence
    N_PSEUDO_ABSENCE = None                  # None = autant que de présences

    # --- Modèle ---
    MODEL = "rf"                             # "rf" ou "gbm"
    N_SPATIAL_FOLDS = 5
    RANDOM_STATE = 2026


# =============================================================================
# 1. INITIALISATION EARTH ENGINE
# =============================================================================
def init_gee():
    """Authentifie et initialise Earth Engine. Authenticate() une seule fois."""
    import ee
    try:
        ee.Initialize(project=Config.EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=Config.EE_PROJECT)
    print("[OK] Earth Engine initialisé.")
    return ee


def get_aoi(ee, points_gdf, margin=None):
    """Calcule l'emprise (AOI) à partir de l'enveloppe des points + une marge,
    pour que le feature stack couvre exactement la zone des données."""
    if margin is None:
        margin = Config.AOI_MARGIN_DEG
    xmin, ymin, xmax, ymax = points_gdf.total_bounds
    return ee.Geometry.Rectangle(
        [xmin - margin, ymin - margin, xmax + margin, ymax + margin]
    )


# =============================================================================
# 2. VÉRITÉ TERRAIN  (présences + pseudo-absences)
# =============================================================================
def load_water_points(path=Config.WPDX_CSV):
    """
    Charge l'inventaire WPdx, nettoie, et garde les points d'eau SOUTERRAINE
    (Well + Spring) situés dans les États de la zone pilote.
    """
    df = pd.read_csv(path, low_memory=False)

    # Coordonnées -> numérique : élimine au passage la ligne de balises HXL
    df["lat_deg"] = pd.to_numeric(df["lat_deg"], errors="coerce")
    df["lon_deg"] = pd.to_numeric(df["lon_deg"], errors="coerce")
    df = df.dropna(subset=["lat_deg", "lon_deg"])

    # Eau souterraine uniquement (exclut Piped Water / eau de surface)
    df = df[df["water_source_category"].isin(Config.GROUNDWATER_SOURCES)]

    # Zone pilote : États du socle nord-nigérian
    df = df[df["clean_adm1"].isin(Config.ZONE_STATES)]
    print(f"[i] {len(df)} points d'eau souterraine dans {Config.ZONE_STATES}.")

    # Sous-échantillonnage (vitesse du PoC + limite getInfo)
    if Config.MAX_PRESENCE and len(df) > Config.MAX_PRESENCE:
        df = df.sample(n=Config.MAX_PRESENCE, random_state=Config.RANDOM_STATE)
        print(f"[i] Sous-échantillon a {Config.MAX_PRESENCE} presences.")

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon_deg"], df["lat_deg"]),
        crs=Config.CRS,
    )
    gdf["label"] = 1
    print(f"[OK] {len(gdf)} présences retenues.")
    return gdf


def generate_pseudo_absences(presence_gdf, n=None,
                             min_dist=Config.MIN_DIST_PSEUDO_ABS):
    """
    Génère des pseudo-absences : points aléatoires dans l'emprise des présences,
    éloignés d'au moins `min_dist` de tout point d'eau connu. Proxy standard du
    potential mapping (à remplacer plus tard par les vrais forages secs).
    """
    if n is None:
        n = len(presence_gdf)

    xmin, ymin, xmax, ymax = presence_gdf.total_bounds
    utm = presence_gdf.estimate_utm_crs()      # zone UTM locale automatique
    forbidden = presence_gdf.to_crs(utm).geometry.buffer(min_dist).union_all()

    pts = []
    while len(pts) < n:
        lons = np.random.uniform(xmin, xmax, n * 2)
        lats = np.random.uniform(ymin, ymax, n * 2)
        cand = gpd.GeoSeries(gpd.points_from_xy(lons, lats), crs=Config.CRS)
        keep = ~cand.to_crs(utm).intersects(forbidden)   # hors des buffers
        for p in cand[keep]:
            pts.append(p)
            if len(pts) >= n:
                break

    absence = gpd.GeoDataFrame({"label": [0] * len(pts)}, geometry=pts,
                               crs=Config.CRS)
    print(f"[OK] {len(absence)} pseudo-absences générées.")
    return absence


def build_truth_set(presence_gdf, absence_gdf):
    """Concatène présences + absences en un seul GeoDataFrame étiqueté."""
    truth = pd.concat([presence_gdf[["label", "geometry"]],
                       absence_gdf[["label", "geometry"]]], ignore_index=True)
    return gpd.GeoDataFrame(truth, crs=Config.CRS)


# =============================================================================
# 3. FEATURES  (stack multibande Earth Engine)
# =============================================================================
def build_feature_stack(ee, aoi):
    """
    Construit l'image multibande des variables explicatives.
    Chaque bande = un facteur de contrôle du potentiel en eau souterraine.
    """
    bands = []

    # --- 3.1 Relief (Copernicus GLO-30 DEM) ---
    dem_col = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    # mosaic() perd la projection native -> on la réimpose (~30 m). Sans ça,
    # ee.Terrain.slope retourne du vide sur les petites zones (ex. Maroua).
    dem_proj = dem_col.filterBounds(aoi).first().projection()
    dem = dem_col.mosaic().setDefaultProjection(dem_proj) \
                 .clip(aoi).rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    bands += [dem, slope]

    # TODO: indice d'humidité topographique (TWI) et densité de drainage
    #       -> nécessite flow accumulation. Piste : package gee 'TerrainAnalysis'
    #          ou pré-calcul SAGA/WhiteboxTools puis upload en asset GEE.

    # --- 3.2 Linéaments / fractures (CLÉ en zone de socle) ---
    # En socle cristallin, l'eau loge dans les fractures -> facteur n°1.
    # (a) courbure (proxy local) ; (b) densité de linéaments par détection
    # d'arêtes (Canny) sur l'ombrage du relief, puis densité par voisinage.
    curvature = dem.convolve(ee.Kernel.laplacian8()).rename("curvature")
    bands.append(curvature)
    # NB: densité de linéaments (Canny sur ombrage du relief) testée
    #     -> importance ~0.001. Le proxy DEM 30m capte du bruit topographique,
    #     pas les vraies fractures. Retiré (coûteux). Un vrai layer exigerait
    #     une carte structurale des failles ou un traitement radar Sentinel-1.

    # --- 3.3 Hydrologie : accumulation d'écoulement, TWI, position topographique ---
    # Facteurs de contrôle directs de l'eau souterraine peu profonde :
    # convergence de l'écoulement et zones d'accumulation (signal PHYSIQUE,
    # moins biaisé que les indices de surface ndvi/ndwi).
    acc = ee.Image("WWF/HydroSHEDS/15ACC").clip(aoi)          # nb de cellules drainées
    flow_acc = acc.add(1).log().rename("flow_acc")            # log -> convergence
    slope_rad = slope.multiply(0.0174533)                     # degrés -> radians
    acc1  = acc.add(1).rename("x")                            # noms alignés pour divide
    denom = slope_rad.tan().add(0.001).rename("x")
    twi = acc1.divide(denom).log().rename("twi")              # humidité topographique
    tpi = dem.subtract(
        dem.focal_mean(300, "circle", "meters")              # élévation - moyenne 300m
    ).rename("tpi")                                           # <0 = dépression (accumule l'eau)
    bands += [flow_acc, twi, tpi]

    # --- 3.3b Regolith / sol : composition (perméabilité de l'altération) ---
    # Sable -> perméabilité / infiltration ; argile -> rétention. Proxy de la
    # nature du regolith au-dessus du socle (OpenLandMap, ~250m).
    sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02") \
             .select("b0").clip(aoi).rename("sand")
    clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02") \
             .select("b0").clip(aoi).rename("clay")
    bands += [sand, clay]

    # --- 3.4 Sentinel-2 : indices de végétation et d'eau ---
    def mask_s2(img):
        scl = img.select("SCL")
        good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(good)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi)
          .filterDate(Config.DATE_START, Config.DATE_END)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", Config.S2_CLOUD_MAX))
          .map(mask_s2)
          .median()
          .clip(aoi))

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("ndvi")
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("ndwi")   # eau de surface
    bands += [ndvi, ndwi]

    # --- 3.5 Pluviométrie (CHIRPS) : recharge potentielle ---
    rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD")
            .filterDate(Config.DATE_START, Config.DATE_END)
            .sum().clip(aoi).rename("rainfall"))
    bands.append(rain)

    stack = ee.Image.cat(bands).toFloat()
    print(f"[OK] Feature stack : {len(bands)} bandes.")
    return stack


def sample_features(ee, feature_img, truth_gdf):
    """
    Échantillonne les valeurs du stack aux points (présences + absences)
    et retourne un DataFrame (features + label + x + y).
    """
    feats = []
    for _, row in truth_gdf.iterrows():
        feats.append(ee.Feature(
            ee.Geometry.Point([row.geometry.x, row.geometry.y]),
            {"label": int(row["label"])}
        ))
    fc = ee.FeatureCollection(feats)

    sampled = feature_img.sampleRegions(
        collection=fc, scale=Config.SCALE, geometries=True, tileScale=4
    )

    # Récupération côté client (PoC : OK si quelques milliers de points)
    data = sampled.getInfo()["features"]
    rows = []
    for f in data:
        props = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        props["x"], props["y"] = lon, lat
        rows.append(props)

    df = pd.DataFrame(rows).dropna()
    print(f"[OK] {len(df)} points échantillonnés, {df.shape[1]-3} features.")
    return df


# =============================================================================
# 4. MODÈLE  (entraînement + validation spatiale + SHAP)
# =============================================================================
def spatial_blocks(df, n_blocks=Config.N_SPATIAL_FOLDS):
    """
    Attribue chaque point à un bloc spatial (grille) pour la CV spatiale.
    Évite la fuite spatiale : des points voisins ne doivent pas se retrouver
    à la fois en train et en test (sinon l'AUC est sur-optimiste).
    """
    qx = pd.qcut(df["x"], q=int(np.sqrt(n_blocks)), labels=False, duplicates="drop")
    qy = pd.qcut(df["y"], q=int(np.sqrt(n_blocks)), labels=False, duplicates="drop")
    return (qx.astype(str) + "_" + qy.astype(str)).astype("category").cat.codes


def train_model(df):
    """Entraîne le modèle avec validation croisée spatiale et retourne le modèle."""
    feature_cols = [c for c in df.columns if c not in ("label", "x", "y")]
    X = df[feature_cols].values
    y = df["label"].values
    groups = spatial_blocks(df).values

    if Config.MODEL == "gbm":
        model = GradientBoostingClassifier(random_state=Config.RANDOM_STATE)
    else:
        model = RandomForestClassifier(
            n_estimators=500, random_state=Config.RANDOM_STATE, n_jobs=-1
        )

    # --- Validation croisée SPATIALE ---
    gkf = GroupKFold(n_splits=min(Config.N_SPATIAL_FOLDS, len(np.unique(groups))))
    auc = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring="roc_auc")
    print(f"[CV spatiale] AUC = {auc.mean():.3f} ± {auc.std():.3f}")

    model.fit(X, y)
    return model, feature_cols


def explain_model(model, df, feature_cols, out_png="outputs/shap_summary.png"):
    """Explicabilité : sauvegarde un graphique SHAP et affiche l'importance
    des variables (quels facteurs pilotent la prédiction)."""
    import os
    import matplotlib
    matplotlib.use("Agg")                 # backend non interactif (pas de fenêtre)
    import matplotlib.pyplot as plt
    import shap

    X = df[feature_cols]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Classifieur binaire -> on garde la classe positive (présence d'eau)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    if hasattr(sv, "ndim") and sv.ndim == 3:      # forme (n, features, classes)
        sv = sv[:, :, 1]

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    shap.summary_plot(sv, X, feature_names=feature_cols, show=False)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Graphique SHAP sauvegardé -> {out_png}")

    # Importance des variables (Random Forest / GBM)
    if hasattr(model, "feature_importances_"):
        imp = sorted(zip(feature_cols, model.feature_importances_),
                     key=lambda t: t[1], reverse=True)
        print("[Importance des variables]")
        for name, val in imp:
            print(f"    {name:14s} {val:.3f}")
    return shap_values


# =============================================================================
# 5. PRÉDICTION  (carte raster de probabilité)
# =============================================================================
def _plot_boundaries(ax, geo, **kw):
    """Trace les contours d'une géométrie GeoJSON (Polygon / MultiPolygon)."""
    if not geo:
        return
    t = geo.get("type")
    coords = geo.get("coordinates", [])
    polys = [coords] if t == "Polygon" else coords if t == "MultiPolygon" else []
    for poly in polys:
        for ring in poly:
            ax.plot([c[0] for c in ring], [c[1] for c in ring], **kw)


def predict_surface(ee, model, feature_img, aoi, feature_cols,
                    grid_step=0.04, out_png="outputs/aquiferx_map.png",
                    presence_gdf=None, title=None, cities=None,
                    boundary_adm0="Nigeria", boundary_adm1_list=None):
    """
    Produit la carte de probabilité de présence d'eau souterraine sur l'AOI :
    grille régulière -> échantillonnage des features -> prédiction sklearn -> PNG.
    grid_step en degrés (0.04 ~= 4 km). À réduire pour une carte plus fine.
    title / cities / boundary_* permettent d'adapter l'habillage à la zone.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")                      # backend non interactif
    import matplotlib.pyplot as plt

    # 1. Grille régulière sur l'emprise de l'AOI
    coords = aoi.bounds().getInfo()["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    gx = np.arange(xmin, xmax, grid_step)
    gy = np.arange(ymin, ymax, grid_step)
    XX, YY = np.meshgrid(gx, gy)
    feats = [ee.Feature(ee.Geometry.Point([float(x), float(y)]))
             for x, y in zip(XX.ravel(), YY.ravel())]
    fc = ee.FeatureCollection(feats)
    print(f"[i] Grille de {len(feats)} points sur l'AOI (échantillonnage GEE)...")

    # 2. Échantillonnage des features sur la grille
    sampled = feature_img.sampleRegions(
        collection=fc, scale=Config.SCALE, geometries=True, tileScale=4
    ).getInfo()["features"]

    if not sampled:
        print(f"[!] Aucune donnée échantillonnée sur cette zone — carte {out_png} ignorée.")
        return None
    rows = []
    for f in sampled:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        p["x"], p["y"] = lon, lat
        rows.append(p)
    grid = pd.DataFrame(rows)
    missing = [c for c in feature_cols if c not in grid.columns]
    if missing:
        present = [c for c in grid.columns if c not in ("x", "y")]
        print(f"[!] Bandes absentes sur cette zone : {missing}")
        print(f"    Bandes présentes : {present}")
        print(f"    -> carte {out_png} ignorée.")
        return None
    grid = grid.dropna(subset=feature_cols)
    print(f"[OK] {len(grid)} points de grille échantillonnés.")

    # 3. Prédiction de probabilité
    grid["proba"] = model.predict_proba(grid[feature_cols].values)[:, 1]

    # 4. Lissage spatial (interpolation) + carte habillée
    from scipy.interpolate import griddata

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    pts = grid[["x", "y"]].values
    xi = np.linspace(grid["x"].min(), grid["x"].max(), 300)
    yi = np.linspace(grid["y"].min(), grid["y"].max(), 300)
    XI, YI = np.meshgrid(xi, yi)
    try:
        ZI = griddata(pts, grid["proba"].values, (XI, YI), method="cubic")
    except Exception:
        ZI = griddata(pts, grid["proba"].values, (XI, YI), method="linear")
    ZI = np.clip(ZI, 0, 1)                       # borne les éventuels dépassements

    fig, ax = plt.subplots(figsize=(10, 8))
    cf = ax.contourf(XI, YI, ZI, levels=15, cmap="RdYlBu", vmin=0, vmax=1)

    # Contours administratifs (FAO/GAUL via GEE ; optionnel)
    try:
        f_adm = ee.FeatureCollection("FAO/GAUL/2015/level1") \
                  .filter(ee.Filter.eq("ADM0_NAME", boundary_adm0))
        if boundary_adm1_list:
            f_adm = f_adm.filter(ee.Filter.inList("ADM1_NAME", boundary_adm1_list))
        else:
            f_adm = f_adm.filterBounds(aoi)      # toutes les régions du cadre
        _plot_boundaries(ax, f_adm.geometry().getInfo(),
                         color="black", linewidth=1.3)
        print("[OK] Contours administratifs ajoutés.")
    except Exception as e:
        print(f"[!] Contours administratifs ignorés ({e}).")

    # Villes principales
    if cities is None:
        cities = {"Kano": (8.5167, 12.0), "Katsina": (7.6004, 12.9908)}
    for name, (lon, lat) in cities.items():
        ax.plot(lon, lat, marker="*", color="black", markersize=13, zorder=5)
        ax.annotate(name, (lon, lat), textcoords="offset points",
                    xytext=(7, 4), fontsize=10, fontweight="bold", zorder=5)

    # Points d'eau connus (surimpression légère)
    if presence_gdf is not None:
        ax.scatter(presence_gdf.geometry.x, presence_gdf.geometry.y,
                   c="black", s=1.5, alpha=0.12, zorder=3)

    cb = plt.colorbar(cf, ax=ax, shrink=0.85)
    cb.set_label("Probabilité de présence d'eau souterraine")
    if title is None:
        title = ("AquiferX — Potentiel en eau souterraine\n"
                 "États de Kano et Katsina (Nigeria)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(grid["x"].min(), grid["x"].max())
    ax.set_ylim(grid["y"].min(), grid["y"].max())
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] Carte sauvegardée -> {out_png}")
    return grid


def validate_holdout(model, holdout_df, feature_cols):
    """Valide sur des points tenus en réserve (la preuve à montrer)."""
    X = holdout_df[feature_cols].values
    y = holdout_df["label"].values
    proba = model.predict_proba(X)[:, 1]
    print(f"[Holdout] AUC = {roc_auc_score(y, proba):.3f}")
    return proba


# =============================================================================
# ORCHESTRATION
# =============================================================================
def main():
    np.random.seed(Config.RANDOM_STATE)

    # 1. GEE
    ee = init_gee()

    # 2. Vérité terrain
    presence = load_water_points()
    absence  = generate_pseudo_absences(presence, n=Config.N_PSEUDO_ABSENCE)
    truth    = build_truth_set(presence, absence)

    # 3. AOI (déduite des points) + features
    aoi = get_aoi(ee, truth)
    feature_img = build_feature_stack(ee, aoi)
    df = sample_features(ee, feature_img, truth)

    # 4. Modèle
    model, feature_cols = train_model(df)
    explain_model(model, df, feature_cols)

    # 5a. Carte de la zone d'entraînement (Kano / Katsina, Nigeria)
    predict_surface(ee, model, feature_img, aoi, feature_cols,
                    presence_gdf=presence,
                    out_png="outputs/aquiferx_map.png",
                    title="AquiferX — Potentiel en eau souterraine\n"
                          "États de Kano et Katsina (Nigeria)",
                    cities={"Kano": (8.5167, 12.0), "Katsina": (7.6004, 12.9908)},
                    boundary_adm0="Nigeria", boundary_adm1_list=Config.ZONE_STATES)

    # 5b. Carte de TRANSFERT : Extrême-Nord Cameroun (estimation à valider)
    if Config.TRANSFER_AOI_BOUNDS:
        xmin, ymin, xmax, ymax = Config.TRANSFER_AOI_BOUNDS
        aoi_cam = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
        print("\n[i] Transfert vers l'Extrême-Nord camerounais...")
        feat_cam = build_feature_stack(ee, aoi_cam)
        predict_surface(ee, model, feat_cam, aoi_cam, feature_cols,
                        presence_gdf=None,
                        out_png="outputs/aquiferx_map_cameroun.png",
                        title="AquiferX — Extrême-Nord, Cameroun (transfert)\n"
                              "MODÈLE NON CALIBRÉ LOCALEMENT — à valider avec des forages",
                        cities={"Maroua": (14.3155, 10.5910),
                                "Mokolo": (13.8003, 10.7404)},
                        boundary_adm0="Cameroon", boundary_adm1_list=None)

    print("\n[FINI] PoC AquiferX : pipeline exécuté.")


if __name__ == "__main__":
    main()
