"""Vérifie la chaîne d'export ONNX en opset 12 (exigence de TensorRT 8.2 sur la Jetson).

PC uniquement (Python 3.11). Deux essais, sans aucun entraînement :
  1. mini-CNN Keras (32x32x1 -> 11 classes, poids aléatoires)
     -> SavedModel -> tf2onnx -> cnn_test.onnx
  2. YOLOv8n pré-entraîné (Ultralytics) -> yolov8n_test.onnx
Pour chaque fichier : opset, version IR, producteur, entrées et sorties ;
puis exécution avec ONNX Runtime et comparaison avec le modèle d'origine.

Les deux ONNX sont gardés dans sorties/verification/jetson/ pour un test
avec trtexec sur la Jetson (voir README.md, « Test anticipé sur Jetson »).

Utilisation, depuis la racine du projet :
    python -m entrainement.verifier_export_onnx
"""

import os

# À définir AVANT d'importer TensorFlow et Ultralytics :
# - moins de messages techniques de TensorFlow ;
# - Ultralytics ne doit rien installer tout seul (il tenterait d'ajouter des paquets au venv).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["YOLO_AUTOINSTALL"] = "False"

import shutil
import subprocess
import sys
import traceback
from pathlib import Path

# Réglages d'environnement d'OpenCV (systeme/__init__.py) : avant tout import de cv2.
import systeme
import cv2
import numpy as np
import onnx
import onnxruntime as ort

OPSET = 12
RACINE = Path(__file__).resolve().parent.parent
DOSSIER = RACINE / "sorties" / "verification"
DOSSIER_JETSON = DOSSIER / "jetson"


# ------------------------------------------------------ Lecture d'un ONNX

def forme(info):
    """Forme d'une entrée ou sortie ONNX, ex. (1, 32, 32, 1) ; nom ou '?' si la dimension est variable."""
    dimensions = info.type.tensor_type.shape.dim
    return tuple(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in dimensions)


def decrire_onnx(chemin):
    """Affiche les caractéristiques d'un fichier ONNX et renvoie son opset."""
    modele = onnx.load(str(chemin))
    onnx.checker.check_model(modele)  # structure conforme à la spécification ONNX
    # Opset du domaine standard (nom de domaine vide ou "ai.onnx")
    opset = next(o.version for o in modele.opset_import if o.domain in ("", "ai.onnx"))
    # Les poids (initialiseurs) peuvent apparaître parmi les entrées : on ne garde que les vraies entrées.
    initialiseurs = {init.name for init in modele.graph.initializer}
    print(f"   fichier    : {chemin.relative_to(RACINE)} ({chemin.stat().st_size / 1e6:.1f} Mo)")
    print(f"   opset      : {opset} (attendu : {OPSET})")
    print(f"   version IR : {modele.ir_version}")
    print(f"   producteur : {modele.producer_name} {modele.producer_version}")
    for entree in modele.graph.input:
        if entree.name not in initialiseurs:
            print(f"   entrée     : {entree.name} {forme(entree)}")
    for sortie in modele.graph.output:
        print(f"   sortie     : {sortie.name} {forme(sortie)}")
    return opset


# ------------------------------------------------------ Essai 1 : CNN Keras

def construire_mini_cnn():
    """Petit CNN du même type que celui de la phase 4 (poids aléatoires, non entraîné)."""
    import keras
    from keras import layers

    return keras.Sequential([
        keras.Input(shape=(32, 32, 1), name="entree"),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(11, activation="softmax"),
    ])


def essai_cnn():
    print("\n1. Mini-CNN Keras -> SavedModel -> tf2onnx -> ONNX opset 12")
    import tensorflow as tf

    modele = construire_mini_cnn()

    # Keras 3 -> SavedModel, le format TensorFlow standard que lit tf2onnx.
    # Taille de lot fixée à 1 dans la signature d'export (sinon elle reste variable) :
    # l'entrée ONNX sera (1, 32, 32, 1), que trtexec lira sans option de forme.
    dossier_savedmodel = DOSSIER / "cnn_savedmodel"
    if dossier_savedmodel.exists():
        shutil.rmtree(dossier_savedmodel)
    signature = [tf.TensorSpec(shape=(1, 32, 32, 1), dtype=tf.float32, name="entree")]
    modele.export(str(dossier_savedmodel), verbose=False, input_signature=signature)

    # SavedModel -> ONNX avec l'outil en ligne de commande de tf2onnx
    chemin_onnx = DOSSIER_JETSON / "cnn_test.onnx"
    commande = [sys.executable, "-m", "tf2onnx.convert",
                "--saved-model", str(dossier_savedmodel),
                "--opset", str(OPSET),
                "--output", str(chemin_onnx)]
    resultat = subprocess.run(commande, capture_output=True, text=True, errors="replace")
    if resultat.returncode != 0:
        print(resultat.stderr[-3000:])
        raise RuntimeError(f"tf2onnx a échoué (code {resultat.returncode})")

    opset = decrire_onnx(chemin_onnx)

    # Même entrée pour Keras et ONNX Runtime : les sorties doivent coïncider.
    session = ort.InferenceSession(str(chemin_onnx), providers=["CPUExecutionProvider"])
    nom_entree = session.get_inputs()[0].name
    generateur = np.random.default_rng(0)
    ecart_max = 0.0
    for _ in range(8):
        image = generateur.random((1, 32, 32, 1), dtype=np.float32)
        sortie_keras = modele.predict(image, verbose=0)
        sortie_onnx = session.run(None, {nom_entree: image})[0]
        ecart_max = max(ecart_max, float(np.abs(sortie_keras - sortie_onnx).max()))
    print(f"   écart max Keras / ONNX Runtime sur 8 images : {ecart_max:.1e} (tolérance 1e-4)")

    return opset == OPSET and ecart_max < 1e-4


# ------------------------------------------------------ Essai 2 : YOLOv8n

def essai_yolo():
    print("\n2. YOLOv8n pré-entraîné -> Ultralytics -> ONNX opset 12")
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    # Poids pré-entraînés sur COCO (80 classes), téléchargés au premier lancement (~6 Mo)
    detecteur = YOLO(str(DOSSIER / "yolov8n.pt"))

    # Image de test fournie avec Ultralytics (un bus et des piétons), ramenée à 640x640
    image_bgr = cv2.imread(str(ASSETS / "bus.jpg"))
    image_rgb = cv2.cvtColor(cv2.resize(image_bgr, (640, 640)), cv2.COLOR_BGR2RGB)
    # HWC uint8 -> NCHW float32 dans [0, 1] : le format d'entrée de YOLO
    entree = image_rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    entree = np.ascontiguousarray(entree)

    # Sortie de référence : le modèle PyTorch d'origine
    reseau = detecteur.model.float().eval()
    with torch.no_grad():
        sortie_torch = reseau(torch.from_numpy(entree))
    if isinstance(sortie_torch, (list, tuple)):
        sortie_torch = sortie_torch[0]
    sortie_torch = sortie_torch.numpy()

    # Export : sur CPU (sur GPU, Ultralytics exigerait onnxruntime-gpu), taille fixe 640x640
    chemin_export = detecteur.export(format="onnx", opset=OPSET, imgsz=640, device="cpu",
                                     simplify=True, dynamic=False)
    chemin_onnx = DOSSIER_JETSON / "yolov8n_test.onnx"
    Path(chemin_export).replace(chemin_onnx)

    opset = decrire_onnx(chemin_onnx)

    session = ort.InferenceSession(str(chemin_onnx), providers=["CPUExecutionProvider"])
    sortie_onnx = session.run(None, {session.get_inputs()[0].name: entree})[0]
    forme_ok = sortie_onnx.shape == (1, 84, 8400)
    print(f"   forme de la sortie : {sortie_onnx.shape} (attendue : (1, 84, 8400))")

    ecart_relatif = float(np.abs(sortie_torch - sortie_onnx).max() / np.abs(sortie_torch).max())
    print(f"   écart relatif PyTorch / ONNX Runtime : {ecart_relatif:.1e} (tolérance 1e-3)")

    # Sortie (1, 84, 8400) : 8400 boîtes candidates ; lignes 0-3 = boîte, lignes 4-83 = scores des 80 classes
    scores = sortie_onnx[0, 4:, :]
    classe = int(scores.max(axis=1).argmax())
    print(f"   meilleur score : {scores.max():.2f} pour la classe « {detecteur.names[classe]} »")

    return opset == OPSET and forme_ok and ecart_relatif < 1e-3


# ------------------------------------------------------------------ Bilan

def main():
    DOSSIER_JETSON.mkdir(parents=True, exist_ok=True)
    bilan = {}
    for nom, essai in (("CNN Keras -> ONNX", essai_cnn), ("YOLOv8n -> ONNX", essai_yolo)):
        try:
            bilan[nom] = essai()
        except Exception:
            traceback.print_exc()
            bilan[nom] = False

    print("\nBilan")
    for nom, reussi in bilan.items():
        statut = "OK" if reussi else "ECHEC"
        print(f"  [{statut:<5}] {nom}")

    if not all(bilan.values()):
        return 1
    print(f"\nFichiers à copier sur la Jetson : {DOSSIER_JETSON}")
    print("Puis, sur la Jetson, dans ce dossier (swap activé) :")
    print("  /usr/src/tensorrt/bin/trtexec --onnx=cnn_test.onnx --fp16")
    print("  /usr/src/tensorrt/bin/trtexec --onnx=yolov8n_test.onnx --fp16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
