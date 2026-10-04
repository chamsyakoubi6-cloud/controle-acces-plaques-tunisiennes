"""Phase 2, étape 6 : vérifie que notre chaîne NumPy (systeme/detection.py) reproduit Ultralytics (PC).

Le même fichier modeles/detecteur.onnx est exécuté sur les 142 images de test :
  - par notre chaîne : letterbox, ONNX Runtime, décodage et NMS en NumPy ;
  - par Ultralytics, avec sa propre chaîne de pré- et post-traitement.
Avec les mêmes seuils (confiance 0,25 et IoU 0,7, les valeurs par défaut d'Ultralytics), les deux
doivent trouver les mêmes boîtes : même nombre, IoU > 0,99 entre boîtes correspondantes, scores
identiques à 1e-3 près. Ensuite, aux seuils de fonctionnement, le taux de détection de notre chaîne.

Utilisation, depuis la racine du projet :
    python -m entrainement.verifier_detection_numpy
"""

import os

os.environ["YOLO_AUTOINSTALL"] = "False"  # Ultralytics ne doit rien installer tout seul

import sys

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import numpy as np
from ultralytics import YOLO

from entrainement.donnees_kaggle import RACINE
from entrainement.evaluer_yolo import CONFIANCE, IOU_CORRECTE, IOU_NMS, lire_etiquettes
from systeme.detection import DetecteurPlaques, iou

MODELE = RACINE / "modeles" / "detecteur.onnx"
CONFIANCE_ULTRALYTICS, IOU_ULTRALYTICS = 0.25, 0.7  # réglages par défaut de model.predict()


def comparer_a_ultralytics(images):
    print(f"1. Notre chaîne NumPy contre Ultralytics (même ONNX, confiance {CONFIANCE_ULTRALYTICS}, "
          f"IoU {IOU_ULTRALYTICS})")
    notre = DetecteurPlaques(str(MODELE), CONFIANCE_ULTRALYTICS, IOU_ULTRALYTICS)
    reference = YOLO(str(MODELE), task="detect")
    images_differentes, nb_boites, iou_minimale, ecart_score_max = 0, 0, 1.0, 0.0
    for chemin in images:
        image = cv2.imread(str(chemin))
        nos_detections = notre.detecter(image)
        resultat = reference.predict(image, imgsz=notre.taille, conf=CONFIANCE_ULTRALYTICS, iou=IOU_ULTRALYTICS,
                                     device="cpu", verbose=False)[0]
        leurs_boites = resultat.boxes.xyxy.numpy()
        leurs_scores = resultat.boxes.conf.numpy()
        if len(nos_detections) != len(leurs_boites):
            images_differentes += 1
            print(f"   {chemin.name} : {len(nos_detections)} boîte(s) chez nous, {len(leurs_boites)} chez Ultralytics")
            continue
        for x1, y1, x2, y2, score in nos_detections:
            recouvrements = iou(np.array([x1, y1, x2, y2]), leurs_boites)
            k = int(recouvrements.argmax())
            nb_boites += 1
            iou_minimale = min(iou_minimale, float(recouvrements[k]))
            ecart_score_max = max(ecart_score_max, abs(float(score) - float(leurs_scores[k])))
    print(f"   {len(images)} images : {images_differentes} avec un nombre de boîtes différent ; {nb_boites} boîtes "
          f"comparées ; IoU minimale {iou_minimale:.4f} (seuil 0,99) ; écart de score maximal {ecart_score_max:.1e} "
          f"(seuil 1e-3)")
    return images_differentes == 0 and iou_minimale > 0.99 and ecart_score_max < 1e-3


def taux_de_detection(images):
    print(f"\n2. Taux de détection de notre chaîne (confiance {CONFIANCE}, IoU de NMS {IOU_NMS})")
    detecteur = DetecteurPlaques(str(MODELE), CONFIANCE, IOU_NMS)
    nb_plaques = nb_trouvees = nb_fausses = 0
    for chemin in images:
        image = cv2.imread(str(chemin))
        detections = detecteur.detecter(image)
        vrais = lire_etiquettes(chemin, image.shape[1], image.shape[0])
        justes = set()
        for vrai in vrais:
            nb_plaques += 1
            if len(detections):
                recouvrements = iou(vrai, detections[:, :4])
                if recouvrements.max() >= IOU_CORRECTE:
                    nb_trouvees += 1
                    justes.add(int(recouvrements.argmax()))
        nb_fausses += len(detections) - len(justes)
    print(f"   {nb_trouvees}/{nb_plaques} plaques trouvées ({nb_trouvees / nb_plaques * 100:.1f} %) ; "
          f"{nb_fausses} fausse(s) détection(s)")


def main():
    images = sorted((RACINE / "donnees" / "yolo" / "images" / "test").glob("*.jpg"), key=lambda c: c.stem.zfill(10))
    identique = comparer_a_ultralytics(images)
    taux_de_detection(images)
    print(f"\nBilan : [{'OK' if identique else 'ECHEC':<5}] notre chaîne NumPy reproduit Ultralytics")
    return 0 if identique else 1


if __name__ == "__main__":
    sys.exit(main())
