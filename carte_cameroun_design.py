# -*- coding: utf-8 -*-
"""
AquiferX — Carte du potentiel en eau souterraine, Extrême-Nord Cameroun.
VERSION DESIGN : même donnée, même honnêteté, présentation soignée.

Ce qui change par rapport à carte_cameroun_v2.py :
  - ombrage du relief (hillshade) calculé depuis le MNT déjà échantillonné :
    les Monts Mandara ressortent, la carte gagne en profondeur ;
  - palette dessinée terre brûlée -> sable -> eau profonde (au lieu du RdYlBu
    par défaut de matplotlib) ;
  - encart de localisation Nigeria -> Cameroun : raconte le transfert d'un
    coup d'oeil (zone d'entraînement / zone de prédiction) ;
  - typographie et mise en page nettoyées, cadre allégé.

Ce qui NE change pas : la probabilité affichée, et la note de bas de figure
qui assume la limite. Une belle carte ne doit jamais dire plus que la donnée.

Prérequis :  python osm_points_cameroun_v2.py
Lancer depuis D:\\AquiferX :  python carte_cameroun_design.py
Sortie : outputs/aquiferx_cameroun_design.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata

import aquiferx_poc as aq

OSM_CSV = os.path.join("data", "osm_water_points_cameroun.csv")
OUT_PNG = os.path.join("outputs", "aquiferx_cameroun_design.png")

GRID_STEP = 0.025
CITIES = {"Maroua": (14.3155, 10.5910), "Mokolo": (13.8003, 10.7404)}

# Palette : sec -> eau. Séquentielle en ressenti, sans midpoint trompeur.
AQUA = LinearSegmentedColormap.from_list("aquiferx", [
    (0.00, "#6E3327"),      # terre brûlée
    (0.20, "#B8703F"),
    (0.38, "#DFC79B"),      # sable
    (0.50, "#EFE8D5"),      # bascule
    (0.62, "#B4D0DA"),
    (0.80, "#4E8FAE"),
    (1.00, "#0D3550"),      # eau profonde
])

INK = "#1B2A33"
MUTED = "#5A6B75"


def halo(w=2.6, c="white"):
    return [pe.withStroke(linewidth=w, foreground=c)]


def main():
    ee = aq.init_gee()

    # --- 1. Modèle ---
    print("\n[1/4] Entraînement sur le Nigeria...")
    presence = aq.load_water_points()
    np.random.seed(aq.Config.RANDOM_STATE)
    absence = aq.generate_pseudo_absences(presence)
    truth = aq.build_truth_set(presence, absence)
    stack_nga = aq.build_feature_stack(ee, aq.get_aoi(ee, presence))
    model, feature_cols = aq.train_model(aq.sample_features(ee, stack_nga, truth))
    nga_bounds = presence.total_bounds                       # pour l'encart

    # --- 2. Grille ---
    print("\n[2/4] Prédiction sur l'Extrême-Nord...")
    lon_min, lat_min, lon_max, lat_max = aq.Config.TRANSFER_AOI_BOUNDS
    aoi = ee.Geometry.Rectangle(aq.Config.TRANSFER_AOI_BOUNDS)
    stack = aq.build_feature_stack(ee, aoi)

    xs = np.arange(lon_min, lon_max, GRID_STEP)
    ys = np.arange(lat_min, lat_max, GRID_STEP)
    pts = [(float(x), float(y)) for y in ys for x in xs]
    print(f"[i] Grille de {len(pts)} points...")
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(list(p))) for p in pts])
    sampled = stack.sampleRegions(collection=fc, scale=aq.Config.SCALE,
                                  geometries=True, tileScale=4).getInfo()["features"]
    rows = []
    for f in sampled:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        p["x"], p["y"] = lon, lat
        rows.append(p)
    grid = pd.DataFrame(rows).dropna(subset=feature_cols)
    print(f"[OK] {len(grid)} points échantillonnés.")
    grid["proba"] = model.predict_proba(grid[feature_cols].values)[:, 1]

    # --- 3. Interpolation : probabilité ET relief ---
    print("\n[3/4] Interpolation et ombrage du relief...")
    N = 500
    xi = np.linspace(grid["x"].min(), grid["x"].max(), N)
    yi = np.linspace(grid["y"].min(), grid["y"].max(), N)
    XI, YI = np.meshgrid(xi, yi)
    P = grid[["x", "y"]].values

    def interp(col):
        try:
            Z = griddata(P, grid[col].values, (XI, YI), method="cubic")
        except Exception:
            Z = griddata(P, grid[col].values, (XI, YI), method="linear")
        bad = np.isnan(Z)
        if bad.any():                                   # bouche les trous
            Z[bad] = griddata(P, grid[col].values, (XI[bad], YI[bad]),
                              method="nearest")
        return Z

    ZI = np.clip(interp("proba"), 0, 1)
    ZE = interp("elevation")                            # MNT déjà dans le stack

    lat_mid = 0.5 * (lat_min + lat_max)
    dx = (xi[-1] - xi[0]) * 111320 * np.cos(np.radians(lat_mid)) / N
    dy = (yi[-1] - yi[0]) * 110540 / N
    hs = LightSource(azdeg=315, altdeg=42).hillshade(ZE, vert_exag=3.0,
                                                     dx=dx, dy=dy)

    # --- 4. Rendu ---
    print("\n[4/4] Rendu...")
    ext = [xi[0], xi[-1], yi[0], yi[-1]]
    fig = plt.figure(figsize=(11.5, 10.2), facecolor="#FBFAF7")
    ax = fig.add_axes([0.07, 0.16, 0.78, 0.72])
    ax.set_facecolor("#FBFAF7")

    ax.imshow(hs, cmap="gray", extent=ext, origin="lower",
              zorder=1, interpolation="bilinear")
    im = ax.imshow(ZI, cmap=AQUA, extent=ext, origin="lower", vmin=0, vmax=1,
                   alpha=0.76, zorder=2, interpolation="bilinear")
    # Liseré sur le seuil de bascule : sépare le favorable du reste
    ax.contour(XI, YI, ZI, levels=[0.5], colors="#0D3550",
               linewidths=0.7, alpha=0.5, zorder=3)

    # Frontières
    try:
        adm = ee.FeatureCollection("FAO/GAUL/2015/level1") \
                .filter(ee.Filter.eq("ADM0_NAME", "Cameroon")).filterBounds(aoi)
        aq._plot_boundaries(ax, adm.geometry().getInfo(), color=INK,
                            linewidth=1.6, zorder=4, alpha=0.85)
        print("[OK] Frontières ajoutées.")
    except Exception as e:
        print(f"[!] Frontières ignorées ({e}).")

    # Forages connus
    n_osm = 0
    if os.path.exists(OSM_CSV):
        osm = pd.read_csv(OSM_CSV)
        n_osm = len(osm)
        ax.scatter(osm["lon_deg"], osm["lat_deg"], s=17, c="#101820",
                   edgecolors="white", linewidths=0.5, alpha=0.9, zorder=6,
                   label=f"{n_osm} forages connus")
        leg = ax.legend(loc="upper right", fontsize=10, framealpha=0.92,
                        edgecolor="none", facecolor="#FBFAF7")
        leg.set_zorder(7)
    else:
        print(f"[!] {OSM_CSV} absent : forages non superposés.")

    # Villes
    for name, (lon, lat) in CITIES.items():
        ax.plot(lon, lat, marker="o", ms=7, mfc="white", mec=INK, mew=1.8,
                zorder=7)
        ax.annotate(name, (lon, lat), textcoords="offset points", xytext=(11, 6),
                    fontsize=12, fontweight="bold", color=INK, zorder=7,
                    path_effects=halo(3.2))

    m = 1.5 * GRID_STEP
    ax.set_xlim(grid["x"].min() + m, grid["x"].max() - m)
    ax.set_ylim(grid["y"].min() + m, grid["y"].max() - m)
    ax.set_xlabel("Longitude", fontsize=10, color=MUTED)
    ax.set_ylabel("Latitude", fontsize=10, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#D8D2C4")
        s.set_linewidth(0.8)

    # Barre de couleur : fine, sans cadre
    cax = fig.add_axes([0.875, 0.16, 0.022, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Probabilité de présence d'eau souterraine", fontsize=10,
                 color=MUTED, labelpad=12)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=9)

    # --- Encart de localisation : raconte le transfert ---
    try:
        ins = fig.add_axes([0.085, 0.635, 0.20, 0.20])
        gaul0 = ee.FeatureCollection("FAO/GAUL/2015/level0")
        for country, fill in [("Nigeria", "#E8E2D4"), ("Cameroon", "#DCE6EA")]:
            geo = gaul0.filter(ee.Filter.eq("ADM0_NAME", country)) \
                       .geometry().simplify(8000).getInfo()
            aq._plot_boundaries(ins, geo, color="#8A8577", linewidth=0.7)
        # Zone d'entraînement (Nigeria) et zone cible (Cameroun)
        nx0, ny0, nx1, ny1 = nga_bounds
        ins.add_patch(Rectangle((nx0, ny0), nx1 - nx0, ny1 - ny0,
                                fc="#B8703F", ec="#6E3327", lw=1.0, alpha=0.75,
                                zorder=4))
        ins.add_patch(Rectangle((lon_min, lat_min), lon_max - lon_min,
                                lat_max - lat_min, fc="#0D3550", ec="#0D3550",
                                lw=1.0, alpha=0.85, zorder=4))
        ins.annotate("", xy=(lon_min, lat_mid),
                     xytext=(0.5 * (nx0 + nx1), 0.5 * (ny0 + ny1)),
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.3,
                                     connectionstyle="arc3,rad=-0.25"), zorder=5)
        ins.text(0.5 * (nx0 + nx1), ny1 + 0.9, "entraînement", ha="center",
                 fontsize=7.5, color="#6E3327", fontweight="bold")
        ins.text(0.5 * (lon_min + lon_max), lat_max + 0.9, "prédiction",
                 ha="center", fontsize=7.5, color="#0D3550", fontweight="bold")
        ins.set_xlim(2, 17)
        ins.set_ylim(3, 15)
        ins.set_xticks([]); ins.set_yticks([])
        ins.set_facecolor("#FBFAF7")
        for s in ins.spines.values():
            s.set_color("#D8D2C4")
        print("[OK] Encart de localisation ajouté.")
    except Exception as e:
        print(f"[!] Encart ignoré ({e}).")

    # --- Titres ---
    fig.text(0.07, 0.945, "Potentiel en eau souterraine", fontsize=23,
             fontweight="bold", color=INK, ha="left")
    fig.text(0.07, 0.915, "Extrême-Nord, Cameroun · Maroua / Mokolo · socle des Monts Mandara",
             fontsize=12.5, color=MUTED, ha="left")
    fig.text(0.07, 0.892, "AquiferX — imagerie satellitaire et apprentissage automatique",
             fontsize=9.5, color="#8A8577", ha="left", style="italic")

    # --- Note : la carte ne doit jamais dire plus que la donnée ---
    note = (
        "Estimation par transfert : le modèle est entraîné uniquement sur le socle du nord du Nigeria — aucune donnée camerounaise "
        "n'entre dans l'apprentissage.\n"
        f"Validation indépendante sur {n_osm} forages connus de la zone (points noirs) : AUC de 0,75 à 0,80 selon le protocole "
        "d'évaluation. Le transfert géographique tient.\n"
        "Limite assumée : ces forages sont des présences seules. Le taux de faux positifs — la probabilité qu'un site désigné se révèle "
        "sec — reste inconnu tant que des fiches\nde forage incluant les échecs ne sont pas disponibles. C'est précisément l'objet de la "
        "collaboration recherchée. Cette carte oriente ; elle ne décide pas."
    )
    fig.text(0.07, 0.025, note, ha="left", va="bottom", fontsize=8.4,
             color=MUTED, linespacing=1.6)

    os.makedirs("outputs", exist_ok=True)
    plt.savefig(OUT_PNG, dpi=220, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Carte -> {OUT_PNG}")


if __name__ == "__main__":
    main()
