"""Phase 2, étape 1 : inspection du jeu Kaggle avant tout entraînement (PC uniquement).

Affiche un bilan chiffré et enregistre des figures dans docs/figures/phase2/inspection/ :
  histogrammes.png       forme, taille et position des cadres ; hauteur des plaques après letterbox
  exemples.jpg           24 images tirées au hasard, avec leur cadre
  couchees_N.jpg         images dont le cadre est plus haut que large, avec les deux redressements possibles
  quasi_doublons_N.jpg   paires d'images proches selon le dHash, avec leur distance
  cadres_extremes.jpg    plus petites plaques et cadres très allongés

Utilisation, depuis la racine du projet :
    python -m entrainement.inspecter_donnees
"""

import collections
import hashlib
import random
import xml.etree.ElementTree as ET

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import matplotlib

matplotlib.use("Agg")  # figures enregistrées dans des fichiers, sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from entrainement.donnees_kaggle import (DOSSIER_KAGGLE, PARTIES, RACINE, assembler_planche, dessiner_cadres,
                                         distances_hamming, empreinte_dhash, faire_vignette, grouper,
                                         lister_images, lire_cadres, tourner_image_et_cadres)

DOSSIER_FIGURES = RACINE / "docs" / "figures" / "phase2" / "inspection"
TAILLES_ENTREE = (320, 416, 640)
DISTANCE_MAX_AFFICHEE = 12  # on montre les paires jusqu'à 12 bits d'écart, pour choisir le seuil à l'œil


# ------------------------------------------------------------ Collecte

def lire_details_xml(chemin_image):
    """Taille déclarée dans le XML, noms des classes et nombre de cadres marqués « tronqués »."""
    racine_xml = ET.parse(chemin_image.with_suffix(".xml")).getroot()
    taille = (int(float(racine_xml.findtext("size/width", "0"))), int(float(racine_xml.findtext("size/height", "0"))))
    noms = [objet.findtext("name") for objet in racine_xml.findall("object")]
    tronques = sum(objet.findtext("truncated", "0") == "1" for objet in racine_xml.findall("object"))
    return taille, noms, tronques


def collecter():
    """Lit toutes les images et leurs annotations ; renvoie une liste de fiches (une par image)."""
    fiches = []
    for partie in PARTIES:
        for chemin in lister_images(partie):
            with Image.open(chemin) as image_pil:  # en-tête seulement : taille et balise EXIF
                largeur, hauteur = image_pil.size
                orientation = image_pil.getexif().get(0x0112, 1)  # balise « Orientation » : 1 = normale
            with open(chemin, "rb") as fichier:
                md5 = hashlib.md5(fichier.read()).hexdigest()
            taille_xml, noms, tronques = lire_details_xml(chemin)
            fiches.append({
                "partie": partie, "chemin": chemin, "largeur": largeur, "hauteur": hauteur,
                "orientation": orientation, "taille_xml": taille_xml, "classes": noms, "tronques": tronques,
                "cadres": lire_cadres(chemin), "md5": md5, "dhash": empreinte_dhash(cv2.imread(str(chemin))),
            })
    return fiches


def rapport_cadre(cadre):
    x0, y0, x1, y1 = cadre
    return (x1 - x0) / (y1 - y0)


# --------------------------------------------------------------- Bilan

def afficher_bilan(fiches):
    for partie in PARTIES:
        dossier = DOSSIER_KAGGLE / partie
        noms_images = {p.stem for p in dossier.glob("*.jpg")}
        noms_xml = {p.stem for p in dossier.glob("*.xml")}
        lot = [f for f in fiches if f["partie"] == partie]
        print(f"\n=== {partie} : {len(lot)} images")
        print(f"  images sans XML : {len(noms_images - noms_xml)} ; XML sans image : {len(noms_xml - noms_images)}")
        print("  classes :", dict(collections.Counter(n for f in lot for n in f["classes"])))
        print("  plaques par image :", dict(sorted(collections.Counter(len(f["cadres"]) for f in lot).items())))
        tailles = collections.Counter((f["largeur"], f["hauteur"]) for f in lot)
        print(f"  tailles d'image : {len(tailles)} différentes ; les plus fréquentes : {tailles.most_common(4)}")
        print("  orientation EXIF :", dict(collections.Counter(f["orientation"] for f in lot)), "(1 = normale)")
        print("  taille XML différente de la taille réelle :",
              sum(f["taille_xml"] != (f["largeur"], f["hauteur"]) for f in lot))
        invalides = sum(1 for f in lot for x0, y0, x1, y1 in f["cadres"]
                        if x0 < 0 or y0 < 0 or x1 > f["largeur"] or y1 > f["hauteur"] or x1 <= x0 or y1 <= y0)
        print(f"  cadres hors image ou vides : {invalides} ; cadres marqués tronqués : {sum(f['tronques'] for f in lot)}")

    rapports = np.array([rapport_cadre(c) for f in fiches for c in f["cadres"]])
    largeurs = np.array([(c[2] - c[0]) / f["largeur"] for f in fiches for c in f["cadres"]])
    print("\n=== Cadres (toutes parties), percentiles 5 / 25 / 50 / 75 / 95")
    print("  rapport largeur/hauteur :", np.round(np.percentile(rapports, [5, 25, 50, 75, 95]), 2),
          "(plaque vue de face : 4,5)")
    print("  largeur du cadre / largeur de l'image :", np.round(np.percentile(largeurs, [5, 25, 50, 75, 95]), 3))
    for taille in TAILLES_ENTREE:
        hauteurs = np.array([(c[3] - c[1]) * taille / max(f["largeur"], f["hauteur"]) for f in fiches for c in f["cadres"]])
        print(f"  entrée {taille} : hauteur de plaque 5 % / médiane = {np.percentile(hauteurs, 5):.0f} / "
              f"{np.median(hauteurs):.0f} px ; moins de 8 px : {np.mean(hauteurs < 8) * 100:.1f} %")


# ------------------------------------------------------------- Figures

def figure_histogrammes(fiches):
    rapports = [rapport_cadre(c) for f in fiches for c in f["cadres"]]
    largeurs = [(c[2] - c[0]) / f["largeur"] for f in fiches for c in f["cadres"]]
    centres = np.array([((c[0] + c[2]) / 2 / f["largeur"], (c[1] + c[3]) / 2 / f["hauteur"])
                        for f in fiches for c in f["cadres"]])
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(rapports, bins=40, color="tab:blue")
    axes[0, 0].axvline(4.5, color="tab:green", linestyle="--", label="plaque vue de face (4,5)")
    axes[0, 0].axvline(1.0, color="tab:red", linestyle="--", label="carré (1,0)")
    axes[0, 0].set(title="Rapport largeur / hauteur des cadres", xlabel="largeur / hauteur", ylabel="nombre de cadres")
    axes[0, 0].legend()

    axes[0, 1].hist(largeurs, bins=40, color="tab:blue")
    axes[0, 1].set(title="Largeur du cadre / largeur de l'image", xlabel="fraction", ylabel="nombre de cadres")

    for taille in TAILLES_ENTREE:
        hauteurs = [(c[3] - c[1]) * taille / max(f["largeur"], f["hauteur"]) for f in fiches for c in f["cadres"]]
        axes[1, 0].hist(hauteurs, bins=range(0, 122, 2), histtype="step", linewidth=1.5, label=f"entrée {taille}")
    axes[1, 0].axvline(8, color="tab:red", linestyle="--", label="case la plus fine de YOLO (8 px)")
    axes[1, 0].set(title="Hauteur des plaques après letterbox", xlabel="pixels", ylabel="nombre de cadres")
    axes[1, 0].legend()

    axes[1, 1].scatter(centres[:, 0], centres[:, 1], s=6, alpha=0.5)
    axes[1, 1].set(title="Position du centre des plaques dans l'image", xlabel="x relatif", ylabel="y relatif",
                   xlim=(0, 1), ylim=(1, 0))

    figure.tight_layout()
    figure.savefig(DOSSIER_FIGURES / "histogrammes.png", dpi=120)
    plt.close(figure)


def planche_exemples(fiches):
    vignettes = []
    for fiche in random.Random(0).sample(fiches, 24):
        image = dessiner_cadres(cv2.imread(str(fiche["chemin"])), fiche["cadres"])
        vignettes.append(faire_vignette(image, 320, 240, f"{fiche['partie']}/{fiche['chemin'].name}"))
    cv2.imwrite(str(DOSSIER_FIGURES / "exemples.jpg"), assembler_planche(vignettes, 6))


def planches_couchees(fiches):
    """Images dont un cadre est plus haut que large : candidates au statut « image couchée ».
    Pour chacune : l'image telle quelle, puis tournée dans les deux sens (cadres recalculés).
    Le bon sens est celui où la scène est à l'endroit ; si le cadre ne retombe pas sur la plaque
    après rotation, c'est que le calcul des cadres est faux."""
    candidates = sorted((f for f in fiches if any(rapport_cadre(c) < 1.0 for c in f["cadres"])),
                        key=lambda f: min(rapport_cadre(c) for c in f["cadres"]))
    print(f"\n=== Candidates « couchées » (cadre plus haut que large) : {len(candidates)}")
    vignettes = []
    for fiche in candidates:
        nom = f"{fiche['partie']}/{fiche['chemin'].name}"
        rapport = min(rapport_cadre(c) for c in fiche["cadres"])
        print(f"  {nom:16s} rapport {rapport:.2f}")
        image = cv2.imread(str(fiche["chemin"]))
        vignettes.append(faire_vignette(dessiner_cadres(image.copy(), fiche["cadres"]), 240, 240, f"{nom} telle quelle"))
        for sens in ("horaire", "antihoraire"):
            tournee, cadres = tourner_image_et_cadres(image, fiche["cadres"], sens)
            vignettes.append(faire_vignette(dessiner_cadres(tournee, cadres), 240, 240, sens))
    par_page = 3 * 8  # 8 images par page, 3 vignettes chacune
    for page, debut in enumerate(range(0, len(vignettes), par_page), start=1):
        cv2.imwrite(str(DOSSIER_FIGURES / f"couchees_{page}.jpg"), assembler_planche(vignettes[debut:debut + par_page], 3))


def planches_quasi_doublons(fiches):
    """Paires d'images proches selon le dHash, de la plus proche à la moins proche, pour choisir le seuil."""
    md5 = collections.defaultdict(list)
    for fiche in fiches:
        md5[fiche["md5"]].append(f"{fiche['partie']}/{fiche['chemin'].name}")
    exacts = [noms for noms in md5.values() if len(noms) > 1]
    print(f"\n=== Doublons exacts (même contenu de fichier) : {len(exacts)} groupes")
    for noms in exacts:
        print("  ", " = ".join(noms))

    distances = distances_hamming([f["dhash"] for f in fiches])
    nb = len(fiches)
    paires = sorted((distances[i, j], i, j) for i in range(nb) for j in range(i + 1, nb)
                    if distances[i, j] <= DISTANCE_MAX_AFFICHEE)
    print(f"=== Paires proches (dHash <= {DISTANCE_MAX_AFFICHEE} bits) : {len(paires)}")
    for distance in range(DISTANCE_MAX_AFFICHEE + 1):
        lot = [(i, j) for d, i, j in paires if d == distance]
        croisees = sum(fiches[i]["partie"] != fiches[j]["partie"] for i, j in lot)
        if lot:
            print(f"  distance {distance:2d} : {len(lot):3d} paires, dont {croisees} entre train et test")

    vignettes = vignettes_paires(fiches, paires)
    par_page = 6 * 8  # 3 paires par ligne, 8 lignes
    for page, debut in enumerate(range(0, len(vignettes), par_page), start=1):
        cv2.imwrite(str(DOSSIER_FIGURES / f"quasi_doublons_{page}.jpg"),
                    assembler_planche(vignettes[debut:debut + par_page], 6))

    # Paires à cheval sur train et test : une même voiture des deux côtés serait une fuite vers le test.
    croisees = [(d, i, j) for d, i, j in paires if fiches[i]["partie"] != fiches[j]["partie"]]
    if croisees:
        cv2.imwrite(str(DOSSIER_FIGURES / "paires_train_test.jpg"),
                    assembler_planche(vignettes_paires(fiches, croisees), 6))

    etudier_seuils_regroupement(fiches, distances)


def vignettes_paires(fiches, paires):
    """Deux vignettes côte à côte par paire, légendées avec la distance dHash."""
    vignettes = []
    for distance, i, j in paires:
        for k in (i, j):
            image = cv2.imread(str(fiches[k]["chemin"]))
            vignettes.append(faire_vignette(image, 240, 180, f"d={distance} {fiches[k]['partie']}/{fiches[k]['chemin'].name}"))
    return vignettes


def etudier_seuils_regroupement(fiches, distances):
    """Pour plusieurs seuils : combien d'images d'entraînement se retrouvent regroupées, et la taille du
    plus gros groupe. Un seuil large évite les fuites, mais un groupe énorme déséquilibrerait le découpage."""
    indices = [k for k, f in enumerate(fiches) if f["partie"] == "train"]
    print("\n=== Regroupement des images d'entraînement selon le seuil dHash")
    for seuil in (2, 5, 8, 10, 11, 12):
        paires = [(a, b) for a in range(len(indices)) for b in range(a + 1, len(indices))
                  if distances[indices[a], indices[b]] <= seuil]
        groupes = [g for g in grouper(len(indices), paires) if len(g) > 1]
        plus_gros = max((len(g) for g in groupes), default=1)
        print(f"  seuil {seuil:2d} : {len(groupes):3d} groupes de 2 images ou plus, "
              f"{sum(len(g) for g in groupes):3d} images regroupées, plus gros groupe : {plus_gros}")


def planche_cadres_extremes(fiches):
    """Les 12 plus petites plaques (hauteur à l'entrée 416) et les cadres très allongés (rapport > 6)."""
    cadres = [(f, c) for f in fiches for c in f["cadres"]]
    petits = sorted(cadres, key=lambda fc: (fc[1][3] - fc[1][1]) * 416 / max(fc[0]["largeur"], fc[0]["hauteur"]))[:12]
    allonges = [fc for fc in cadres if rapport_cadre(fc[1]) > 6]
    vignettes = []
    for fiche, (x0, y0, x1, y1) in petits + allonges:
        image = dessiner_cadres(cv2.imread(str(fiche["chemin"])), [(x0, y0, x1, y1)])
        marge = int(max(x1 - x0, y1 - y0))  # zoom sur la plaque, avec un peu de contexte
        zoom = image[max(0, int(y0) - marge):int(y1) + marge, max(0, int(x0) - marge):int(x1) + marge]
        hauteur_416 = (y1 - y0) * 416 / max(fiche["largeur"], fiche["hauteur"])
        texte = f"{fiche['partie']}/{fiche['chemin'].name} h={hauteur_416:.0f}px r={rapport_cadre((x0, y0, x1, y1)):.1f}"
        vignettes.append(faire_vignette(zoom, 320, 200, texte))
    cv2.imwrite(str(DOSSIER_FIGURES / "cadres_extremes.jpg"), assembler_planche(vignettes, 4))


def main():
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    fiches = collecter()
    afficher_bilan(fiches)
    figure_histogrammes(fiches)
    planche_exemples(fiches)
    planches_couchees(fiches)
    planches_quasi_doublons(fiches)
    planche_cadres_extremes(fiches)
    print(f"\nFigures enregistrées dans {DOSSIER_FIGURES}")


if __name__ == "__main__":
    main()
