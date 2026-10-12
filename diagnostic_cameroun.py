"""
Diagnostic AquiferX — pourquoi le stack est-il vide sur l'Extrême-Nord Cameroun ?
Interroge chaque bande du feature stack au point de Maroua et affiche sa valeur.
Une valeur None = bande absente sur la zone = bande coupable.

Lancer depuis D:\\AquiferX :  python diagnostic_cameroun.py
"""
import ee
import aquiferx_poc as aq          # réutilise EXACTEMENT ton stack de features

ee.Initialize(project=aq.Config.EE_PROJECT)
print("[OK] Earth Engine initialisé.\n")

# Construit le stack sur la zone de transfert (Cameroun)
aoi_cam = ee.Geometry.Rectangle(aq.Config.TRANSFER_AOI_BOUNDS)
stack = aq.build_feature_stack(ee, aoi_cam)

try:
    print("Bandes du stack :", stack.bandNames().getInfo(), "\n")
except Exception as e:
    print("ERREUR bandNames :", e)

# Valeur de chaque bande au point de Maroua (au cœur de la zone)
maroua = ee.Geometry.Point([14.3155, 10.5910])
print("--- Valeurs au point de Maroua (14.32 E, 10.59 N) ---")
try:
    vals = stack.reduceRegion(ee.Reducer.first(), maroua, 100,
                              bestEffort=True).getInfo()
    if not vals:
        print("  (dictionnaire vide — aucune bande n'a de valeur ici)")
    for k, v in vals.items():
        flag = "  <-- VIDE" if v is None else ""
        print(f"  {k:18s} {v}{flag}")
except Exception as e:
    print("  ERREUR reduceRegion :", e)

# Comparaison : combien d'images Sentinel-2 sur la zone ?
print("\n--- Sentinel-2 disponibles sur la zone ---")
try:
    n = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
         .filterBounds(aoi_cam)
         .filterDate(aq.Config.DATE_START, aq.Config.DATE_END)
         .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", aq.Config.S2_CLOUD_MAX))
         .size().getInfo())
    print(f"  {n} images Sentinel-2 (filtre nuages < {aq.Config.S2_CLOUD_MAX} %)")
except Exception as e:
    print("  ERREUR :", e)
