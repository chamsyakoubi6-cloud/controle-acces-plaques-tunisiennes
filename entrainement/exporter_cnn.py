"""Phase 4 : export du CNN de lecture en ONNX opset 12 (PC), pour ONNX Runtime (PC) et TensorRT (Jetson).

Même chaîne que celle vérifiée en phase 1 : modèle Keras -> SavedModel (entrée de forme FIXE, sinon la taille de
lot resterait variable) -> tf2onnx --opset 12 (TensorRT 8.2 ne lit pas les opsets récents). Deux tailles de lot,
à comparer :
  lecteur_lot1.onnx  : entrée (1, 32, 32, 1), un appel par vignette ;
  lecteur_lot16.onnx : entrée (16, 32, 32, 1), un appel pour toute la plaque : 99 % des plaques ont au plus 16
                       candidats ; les places vides sont remplies de zéros et leurs sorties ignorées.
Vérifications : opset 12 et version IR ; écart Keras / ONNX Runtime < 1e-4 sur toutes les vignettes d'origine
et même classe prédite pour toutes. Les latences se mesurent ensuite dans un processus propre (sans TensorFlow,
qui se disputerait les coeurs du processeur avec ONNX Runtime) :
    python -m outils.mesurer_latence modeles/lecteur_lot1.onnx modeles/lecteur_lot16.onnx --mesures 500
Copies dans sorties/verification/jetson/ pour le test anticipé sur Jetson (trtexec).

Utilisation, depuis la racine du projet :
    python -m entrainement.exporter_cnn binaire_16
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # masque les messages d'information de TensorFlow

import numpy as np
import onnxruntime as ort

from entrainement.donnees_kaggle import RACINE
from entrainement.extraire_caracteres import FICHIER_JEU
from entrainement.verifier_export_onnx import DOSSIER_JETSON, OPSET, decrire_onnx
from systeme.lecture import preparer_vignettes

DOSSIER_RUNS = RACINE / "sorties" / "runs" / "cnn"
DOSSIER_MODELES = RACINE / "modeles"
LOTS = (1, 16)


def exporter(modele, lot, dossier_travail):
    """Modèle Keras -> SavedModel à entrée (lot, 32, 32, 1) -> ONNX opset 12 ; renvoie le chemin du .onnx."""
    import tensorflow as tf
    savedmodel = dossier_travail / f"savedmodel_lot{lot}"
    shutil.rmtree(savedmodel, ignore_errors=True)
    signature = [tf.TensorSpec(shape=(lot, 32, 32, 1), dtype=tf.float32, name="vignettes")]
    modele.export(str(savedmodel), verbose=False, input_signature=signature)
    chemin = DOSSIER_MODELES / f"lecteur_lot{lot}.onnx"
    commande = [sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(savedmodel),
                "--opset", str(OPSET), "--output", str(chemin)]
    resultat = subprocess.run(commande, capture_output=True, text=True, errors="replace")
    if resultat.returncode != 0:
        print(resultat.stderr[-3000:])
        raise RuntimeError(f"tf2onnx a échoué (code {resultat.returncode})")
    return chemin


def verifier(chemin, sorties_keras, vignettes):
    """Mêmes vignettes dans ONNX Runtime, par paquets de la taille du lot (le dernier complété par des zéros)."""
    opset = decrire_onnx(chemin)
    session = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"])
    entree = session.get_inputs()[0]
    lot = entree.shape[0]
    morceaux = []
    for debut in range(0, len(vignettes), lot):
        morceau = vignettes[debut:debut + lot]
        complet = np.zeros((lot,) + vignettes.shape[1:], dtype=np.float32)
        complet[:len(morceau)] = morceau
        morceaux.append(session.run(None, {entree.name: complet})[0][:len(morceau)])
    sorties_onnx = np.concatenate(morceaux)
    ecart = float(np.abs(sorties_onnx - sorties_keras).max())
    memes = float(np.mean(sorties_onnx.argmax(axis=1) == sorties_keras.argmax(axis=1)))
    print(f"   Keras / ONNX Runtime sur {len(vignettes)} vignettes : écart max {ecart:.1e} (tolérance 1e-4), "
          f"même classe prédite : {memes * 100:.2f} %")
    return opset == OPSET and ecart < 1e-4 and memes == 1.0


def main():
    parseur = argparse.ArgumentParser(description="Export du CNN de lecture en ONNX opset 12.")
    parseur.add_argument("nom", help="essai dont on exporte le modèle final (sorties/runs/cnn/<nom>/final.keras)")
    arguments = parseur.parse_args()
    import keras
    dossier = DOSSIER_RUNS / arguments.nom
    modele = keras.models.load_model(dossier / "final.keras")
    with open(dossier / "resume.json", encoding="utf-8") as fichier:
        entree = json.load(fichier)["entree"]
    donnees = np.load(FICHIER_JEU)
    origine = donnees["version"] == 0
    vignettes = preparer_vignettes(donnees[entree][origine], entree)
    sorties_keras = modele.predict(vignettes, verbose=0, batch_size=256)

    DOSSIER_MODELES.mkdir(exist_ok=True)
    DOSSIER_JETSON.mkdir(parents=True, exist_ok=True)
    reussi = True
    for lot in LOTS:
        print(f"\nExport, lot de {lot} :")
        chemin = exporter(modele, lot, dossier)
        reussi &= verifier(chemin, sorties_keras, vignettes)
        shutil.copy2(chemin, DOSSIER_JETSON / chemin.name)
    print(f"\n{'Exports conformes' if reussi else 'ECHEC : voir ci-dessus'} ; copies dans {DOSSIER_JETSON}")
    print("Latences (processus propre) : python -m outils.mesurer_latence modeles/lecteur_lot1.onnx "
          "modeles/lecteur_lot16.onnx --mesures 500")


if __name__ == "__main__":
    main()
