"""Phase 2, étape 5 : export des modèles de détection en ONNX opset 12, entrée fixe, et vérifications (PC).

- Les trois modèles (320, 416, 640) -> sorties/verification/jetson/detecteur_<taille>.onnx :
  le test anticipé sur la Jetson mesurera la latence réelle de chaque taille.
- Le modèle retenu (modeles/detecteur.pt) -> modeles/detecteur.onnx, utilisé par le système.
Réglages d'export : opset 12 (TensorRT 8.2), entrée fixe (TensorRT construit un moteur optimisé pour une
taille connue), pas de NMS dans le graphe (elle est écrite en NumPy), simplification par onnxslim, sur CPU
(sur GPU, Ultralytics exigerait onnxruntime-gpu).
Vérifications de chaque fichier : opset, version IR, formes, écart ONNX Runtime / PyTorch sur une image de
test. La latence est mesurée à la fin par outils/mesurer_latence.py, dans un processus séparé : PyTorch,
chargé ici, fausserait la mesure en disputant les coeurs du processeur à ONNX Runtime.

Utilisation, depuis la racine du projet :
    python -m entrainement.exporter_yolo
"""

import os

os.environ["YOLO_AUTOINSTALL"] = "False"  # Ultralytics ne doit rien installer tout seul

import shutil
import subprocess
import sys
from pathlib import Path

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from entrainement.donnees_kaggle import RACINE
from entrainement.verifier_export_onnx import decrire_onnx
from systeme.detection import preparer_entree

OPSET = 12
TAILLES = (320, 416, 640)
DOSSIER_RUNS = RACINE / "sorties" / "runs" / "detection"
DOSSIER_JETSON = RACINE / "sorties" / "verification" / "jetson"
DETECTEUR_PT = RACINE / "modeles" / "detecteur.pt"
DETECTEUR_ONNX = RACINE / "modeles" / "detecteur.onnx"


def exporter(chemin_pt, taille):
    """Export ONNX ; Ultralytics écrit le fichier à côté du .pt et renvoie son chemin."""
    return Path(YOLO(str(chemin_pt)).export(format="onnx", opset=OPSET, imgsz=taille, dynamic=False,
                                            simplify=True, nms=False, half=False, batch=1, device="cpu"))


def verifier(chemin_pt, chemin_onnx, taille, image_essai):
    """Contrôles d'un fichier exporté ; renvoie True s'ils sont tous réussis."""
    opset = decrire_onnx(chemin_onnx)
    entree, _, _ = preparer_entree(image_essai, taille)  # notre pré-traitement NumPy (letterbox)
    session = ort.InferenceSession(str(chemin_onnx), providers=["CPUExecutionProvider"])
    nom_entree = session.get_inputs()[0].name
    sortie_onnx = session.run(None, {nom_entree: entree})[0]

    # Même entrée dans le modèle PyTorch d'origine : les sorties doivent coïncider
    reseau = YOLO(str(chemin_pt)).model.float().eval()
    with torch.no_grad():
        sortie_torch = reseau(torch.from_numpy(entree))
    if isinstance(sortie_torch, (list, tuple)):
        sortie_torch = sortie_torch[0]
    sortie_torch = sortie_torch.numpy()
    ecart = float(np.abs(sortie_torch - sortie_onnx).max() / np.abs(sortie_torch).max())

    # Une boîte candidate par case des trois grilles de YOLOv8 (cases de 8, 16 et 32 pixels)
    nb_candidats = sum((taille // pas) ** 2 for pas in (8, 16, 32))
    forme_ok = sortie_onnx.shape == (1, 5, nb_candidats)
    print(f"   sortie {sortie_onnx.shape} (attendue (1, 5, {nb_candidats})) ; écart relatif ONNX Runtime / PyTorch "
          f"{ecart:.1e} (tolérance 1e-3)")
    return opset == OPSET and forme_ok and ecart < 1e-3


def main():
    DOSSIER_JETSON.mkdir(parents=True, exist_ok=True)
    image_essai = cv2.imread(str(sorted((RACINE / "donnees" / "yolo" / "images" / "test").glob("*.jpg"))[0]))
    taille_retenue = YOLO(str(DETECTEUR_PT)).ckpt["train_args"]["imgsz"]  # taille d'entraînement du modèle retenu

    bilan = {}
    for taille in TAILLES:
        print(f"\n=== Modèle entraîné en {taille}")
        chemin_pt = DOSSIER_RUNS / f"yolov8n_{taille}" / "weights" / "best.pt"
        destination = DOSSIER_JETSON / f"detecteur_{taille}.onnx"
        shutil.copy2(exporter(chemin_pt, taille), destination)
        bilan[taille] = verifier(chemin_pt, destination, taille, image_essai)

    # Le modèle du système est le fichier exporté pour la taille retenue
    shutil.copy2(DOSSIER_JETSON / f"detecteur_{taille_retenue}.onnx", DETECTEUR_ONNX)

    print("\nBilan")
    for taille, reussi in bilan.items():
        print(f"  [{'OK' if reussi else 'ECHEC':<5}] entrée {taille}")
    print(f"Modèle du système (entrée {taille_retenue}) : {DETECTEUR_ONNX}")

    # flush : vide le tampon d'affichage avant de lancer le sous-processus, sinon ses lignes passent devant
    print("\nLatence d'inférence ONNX Runtime sur le CPU du PC (processus séparé, sans PyTorch) :", flush=True)
    fichiers = [str(DOSSIER_JETSON / f"detecteur_{taille}.onnx") for taille in TAILLES]
    subprocess.run([sys.executable, "-m", "outils.mesurer_latence"] + fichiers, cwd=str(RACINE))

    print("\nSur la Jetson (test anticipé), dans le dossier copié :")
    for taille in TAILLES:
        print(f"  /usr/src/tensorrt/bin/trtexec --onnx=detecteur_{taille}.onnx --fp16")
    return 0 if all(bilan.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
