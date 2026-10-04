"""Phase 2, étape 2 : conversion du jeu Kaggle au format YOLO dans donnees/yolo/ (PC uniquement).

1. Doublons : deux images à 2 bits de dHash ou moins sont la même photo (vérifié à l'oeil pendant
   l'inspection) ; on n'en garde qu'une, sinon elle compterait deux fois.
2. Images couchées : redressées d'un quart de tour selon corrections_kaggle.csv (sens vérifié à l'oeil).
3. dataset/train est découpé en entraînement (80 %) et validation (20 %) avec une graine fixe, par
   groupes : deux images à 12 bits de dHash ou moins restent du même côté, pour qu'une même voiture
   ne se retrouve pas à la fois en entraînement et en validation (fuite de données).
4. dataset/test est copié tel quel, sans aucune correction : il ne sert qu'à l'évaluation finale.
5. Contrôle : les cadres sont relus depuis les fichiers YOLO et redessinés sur les images.

Utilisation, depuis la racine du projet :
    python -m entrainement.preparer_yolo
"""

import csv
import random
import shutil

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
from PIL import Image

from entrainement.donnees_kaggle import (RACINE, assembler_planche, dessiner_cadres, distances_hamming,
                                         empreinte_dhash, faire_vignette, grouper, lister_images, lire_cadres,
                                         lire_corrections, tourner_image_et_cadres)

GRAINE = 42
PART_VALIDATION = 0.20
SEUIL_DOUBLON = 2         # bits de dHash : même photo (inspection : 0 à 2 bits = photos identiques)
SEUIL_REGROUPEMENT = 12   # bits de dHash : même groupe au découpage (inspection : plus gros groupe = 11 images)
NOM_CLASSE = "plaque"
DOSSIER_YOLO = RACINE / "donnees" / "yolo"
DOSSIER_FIGURES = RACINE / "docs" / "figures" / "phase2" / "preparation"


def paires_proches(distances, seuil):
    """Paires d'indices (i, j) dont les empreintes diffèrent d'au plus `seuil` bits."""
    nb = len(distances)
    return [(i, j) for i in range(nb) for j in range(i + 1, nb) if distances[i, j] <= seuil]


def dedoublonner(distances):
    """Dans chaque groupe de copies d'une même photo, garde la première (par numéro).
    Renvoie (indices gardés, {indice retiré: indice gardé})."""
    gardes, retires = [], {}
    for groupe in grouper(len(distances), paires_proches(distances, SEUIL_DOUBLON)):
        groupe.sort()
        gardes.append(groupe[0])
        for indice in groupe[1:]:
            retires[indice] = groupe[0]
    return sorted(gardes), retires


def decouper(distances):
    """Découpe en entraînement / validation par groupes d'images proches.
    Renvoie (indices d'entraînement, indices de validation)."""
    groupes = grouper(len(distances), paires_proches(distances, SEUIL_REGROUPEMENT))
    groupes.sort(key=min)                  # ordre fixe avant le tirage : le résultat ne dépend que de la graine
    random.Random(GRAINE).shuffle(groupes)
    cible = round(PART_VALIDATION * len(distances))
    entrainement, validation = [], []
    for groupe in groupes:
        if len(validation) < cible:
            validation.extend(groupe)
        else:
            entrainement.extend(groupe)
    return sorted(entrainement), sorted(validation)


def convertir(chemin, partie_kaggle, partie_yolo, corrections):
    """Copie une image dans donnees/yolo/images/<partie> et écrit ses étiquettes YOLO.
    Renvoie le sens de rotation appliqué (ou None)."""
    cadres = lire_cadres(chemin)
    destination = DOSSIER_YOLO / "images" / partie_yolo / chemin.name
    sens = corrections.get((partie_kaggle, chemin.name)) if partie_kaggle == "train" else None
    if sens:
        image, cadres = tourner_image_et_cadres(cv2.imread(str(chemin)), cadres, sens)
        cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        hauteur, largeur = image.shape[:2]
    else:
        shutil.copy2(chemin, destination)  # copie à l'identique
        with Image.open(chemin) as image_pil:
            largeur, hauteur = image_pil.size

    # Format YOLO : une ligne par objet, « classe xc yc l h », coordonnées divisées par la taille de l'image
    lignes = ["0 {:.6f} {:.6f} {:.6f} {:.6f}".format((x0 + x1) / 2 / largeur, (y0 + y1) / 2 / hauteur,
                                                     (x1 - x0) / largeur, (y1 - y0) / hauteur)
              for x0, y0, x1, y1 in cadres]
    chemin_etiquettes = DOSSIER_YOLO / "labels" / partie_yolo / (chemin.stem + ".txt")
    chemin_etiquettes.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return sens


def ecrire_yaml():
    """Fichier de description du jeu, lu par Ultralytics (chemin absolu : aucune ambiguïté)."""
    contenu = ("# Généré par entrainement/preparer_yolo.py : ne pas modifier à la main.\n"
               f"path: {DOSSIER_YOLO.as_posix()}\n"
               "train: images/train\n"
               "val: images/val\n"
               "test: images/test\n"
               "names:\n"
               f"  0: {NOM_CLASSE}\n")
    (DOSSIER_YOLO / "plaques.yaml").write_text(contenu, encoding="utf-8")


def controler(images_tournees):
    """Relit les étiquettes YOLO et redessine les cadres : s'ils retombent sur les plaques,
    la conversion (et la rotation) est juste. Toutes les images tournées + 15 tirées au hasard."""
    toutes = sorted((DOSSIER_YOLO / "images").glob("*/*.jpg"))
    choix = [c for c in toutes if c.name in images_tournees and c.parent.name != "test"]
    choix += random.Random(0).sample([c for c in toutes if c not in choix], 15)
    vignettes = []
    for chemin in choix:
        image = cv2.imread(str(chemin))
        hauteur, largeur = image.shape[:2]
        cadres = []
        chemin_etiquettes = DOSSIER_YOLO / "labels" / chemin.parent.name / (chemin.stem + ".txt")
        for ligne in chemin_etiquettes.read_text(encoding="utf-8").splitlines():
            _, xc, yc, l, h = (float(v) for v in ligne.split())
            cadres.append(((xc - l / 2) * largeur, (yc - h / 2) * hauteur, (xc + l / 2) * largeur, (yc + h / 2) * hauteur))
        vignettes.append(faire_vignette(dessiner_cadres(image, cadres, (0, 255, 0)), 240, 240,
                                        f"{chemin.parent.name}/{chemin.name}"))
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DOSSIER_FIGURES / "controle_etiquettes_yolo.jpg"), assembler_planche(vignettes, 6))


def main():
    # On repart de zéro à chaque exécution : le résultat ne dépend que du jeu brut, des corrections et de la graine.
    if DOSSIER_YOLO.exists():
        shutil.rmtree(DOSSIER_YOLO)
    for partie in ("train", "val", "test"):
        (DOSSIER_YOLO / "images" / partie).mkdir(parents=True)
        (DOSSIER_YOLO / "labels" / partie).mkdir(parents=True)
    corrections = lire_corrections()

    # 1-3. Doublons et découpage, sur dataset/train uniquement
    chemins = lister_images("train")
    distances = distances_hamming([empreinte_dhash(cv2.imread(str(c))) for c in chemins])
    gardes, retires = dedoublonner(distances)
    distances_gardes = distances[gardes][:, gardes]
    indices_entrainement, indices_validation = decouper(distances_gardes)
    partie_de = {}
    for i in indices_entrainement:
        partie_de[gardes[i]] = "train"
    for i in indices_validation:
        partie_de[gardes[i]] = "val"

    # 4. Écriture des images et des étiquettes, et du tableau de traçabilité
    tournees = set()
    with open(DOSSIER_YOLO / "decoupage.csv", "w", encoding="utf-8", newline="") as fichier:
        tableau = csv.writer(fichier)
        tableau.writerow(["image", "partie_kaggle", "partie_yolo", "rotation", "remarque"])
        for indice, chemin in enumerate(chemins):
            if indice in retires:
                original = chemins[retires[indice]].name
                tableau.writerow([chemin.name, "train", "", "", f"doublon de {original}, retiré"])
                continue
            sens = convertir(chemin, "train", partie_de[indice], corrections)
            if sens:
                tournees.add(chemin.name)
            tableau.writerow([chemin.name, "train", partie_de[indice], sens or "", ""])
        for chemin in lister_images("test"):
            convertir(chemin, "test", "test", corrections)
            tableau.writerow([chemin.name, "test", "test", "", "copie à l'identique"])
    ecrire_yaml()

    # 5. Contrôle visuel et bilan
    controler(tournees)
    nb = {partie: len(list((DOSSIER_YOLO / "images" / partie).glob("*.jpg"))) for partie in ("train", "val", "test")}
    print(f"Doublons retirés ({len(retires)}) :",
          ", ".join(f"{chemins[i].name} (= {chemins[j].name})" for i, j in sorted(retires.items())))
    print(f"Images redressées ({len(tournees)}) :", ", ".join(sorted(tournees, key=lambda n: n.zfill(10))))
    print(f"Groupes pour le découpage : seuil {SEUIL_REGROUPEMENT} bits, graine {GRAINE}")
    print(f"Entraînement : {nb['train']} ; validation : {nb['val']} "
          f"({nb['val'] / (nb['train'] + nb['val']) * 100:.1f} %) ; test : {nb['test']}")
    print(f"Jeu YOLO : {DOSSIER_YOLO} ; contrôle : {DOSSIER_FIGURES / 'controle_etiquettes_yolo.jpg'}")


if __name__ == "__main__":
    main()
