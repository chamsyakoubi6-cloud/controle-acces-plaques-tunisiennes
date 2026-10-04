"""Phase 2, étape 3 : entraînement de YOLOv8n à une classe (« plaque ») sur donnees/yolo/ (PC, GPU).

Point de départ : les poids yolov8n.pt pré-entraînés sur COCO (apprentissage par transfert : 439 images
ne suffisent pas pour apprendre à partir de zéro). Seule la taille d'entrée change d'un entraînement
à l'autre, pour pouvoir comparer 320, 416 et 640 à réglages identiques.

Utilisation, depuis la racine du projet :
    python -m entrainement.entrainer_yolo --taille 416
Résultats : sorties/runs/detection/yolov8n_<taille>/ (poids best.pt et last.pt, courbes, args.yaml).
"""

import os

# Ultralytics ne doit rien installer tout seul (à définir avant son import).
os.environ["YOLO_AUTOINSTALL"] = "False"

import argparse

import systeme  # réglages d'OpenCV, avant tout import de cv2 (Ultralytics l'importe)
from ultralytics import YOLO

from entrainement.donnees_kaggle import RACINE

POIDS_INITIAUX = RACINE / "modeles" / "yolov8n.pt"   # téléchargés automatiquement s'ils sont absents
FICHIER_DONNEES = RACINE / "donnees" / "yolo" / "plaques.yaml"
DOSSIER_RESULTATS = RACINE / "sorties" / "runs" / "detection"

REGLAGES = {
    "epochs": 100,          # largement suffisant en partant de poids pré-entraînés...
    "patience": 30,         # ...et arrêt anticipé si la validation ne progresse plus pendant 30 époques
    "batch": 16,            # tient sous 4 Go de mémoire GPU
    "workers": 4,           # processus de chargement des images (sous Windows, chacun coûte de la RAM)
    "optimizer": "auto",    # choix d'Ultralytics selon la durée de l'entraînement (noté dans le journal)
    "amp": True,            # précision mixte : plus rapide et plus économe en mémoire sur une RTX
    "seed": 0,              # graine fixe...
    "deterministic": True,  # ...et calculs déterministes : entraînement reproductible
    # Augmentations : chacune doit produire une image que la caméra du parking pourrait voir.
    "fliplr": 0.0,          # PAS de miroir : une plaque inversée (chiffres et texte à l'envers) n'existe pas
    "flipud": 0.0,          # pas de retournement vertical non plus
    "degrees": 5.0,         # rotation de +/- 5 degrés : caméra ou plaque légèrement inclinée
    "mosaic": 1.0,          # assemble 4 images : plus de contextes et d'échelles, utile avec peu de données
    "close_mosaic": 10,     # mosaïque coupée pour les 10 dernières époques : fin sur des images réalistes
    "scale": 0.5,           # zoom de +/- 50 % : distance variable à la caméra
    "translate": 0.1,       # décalage de +/- 10 % : position variable dans l'image
    "hsv_h": 0.015,         # teinte : décalage faible (une plaque bleue de location reste bleue)
    "hsv_s": 0.7,           # saturation
    "hsv_v": 0.4,           # luminosité : jour, nuit, contre-jour
    "mixup": 0.0,           # mélange de deux images : irréaliste, désactivé
    "copy_paste": 0.0,      # copier-coller d'objets : irréaliste, désactivé
}


def main():
    parseur = argparse.ArgumentParser(description="Entraîne YOLOv8n (1 classe : plaque).")
    parseur.add_argument("--taille", type=int, required=True, choices=(320, 416, 640), help="taille d'entrée")
    arguments = parseur.parse_args()

    modele = YOLO(str(POIDS_INITIAUX))
    modele.train(data=str(FICHIER_DONNEES), imgsz=arguments.taille, device=0,
                 project=str(DOSSIER_RESULTATS), name=f"yolov8n_{arguments.taille}", exist_ok=True,
                 plots=True, **REGLAGES)


if __name__ == "__main__":
    main()
