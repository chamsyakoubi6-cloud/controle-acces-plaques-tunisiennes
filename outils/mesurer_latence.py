"""Mesure la latence d'inférence d'un ou plusieurs modèles (PC : .onnx ; Jetson : .engine, phase 6).

La mesure se fait dans un processus où seul le moteur d'inférence est chargé : PyTorch ou TensorFlow,
chargés dans le même processus, se disputeraient les coeurs du processeur et fausseraient le résultat
(mesuré en phase 2 : 80 ms au lieu de 33 ms pour le détecteur 640 sur le CPU du PC).

Utilisation, depuis la racine du projet :
    python -m outils.mesurer_latence modeles/detecteur.onnx
    python -m outils.mesurer_latence sorties/verification/jetson/detecteur_320.onnx sorties/verification/jetson/detecteur_640.onnx
"""

import argparse
import os
import time

import systeme  # réglages d'OpenCV, avant tout import de cv2 (par cohérence avec les autres scripts)
import numpy as np

from systeme.inference import creer_moteur


def mesurer(chemin_modele, nb_chauffe, nb_mesures):
    """Renvoie (forme d'entrée, médiane en ms, 90e percentile en ms) sur des entrées aléatoires.
    Les premiers passages (« chauffe ») sont exclus : ils incluent des allocations et optimisations initiales."""
    moteur = creer_moteur(chemin_modele)
    entree = np.random.rand(*moteur.forme_entree).astype(np.float32)
    for _ in range(nb_chauffe):
        moteur.executer(entree)
    durees = []
    for _ in range(nb_mesures):
        debut = time.perf_counter()
        moteur.executer(entree)
        durees.append(time.perf_counter() - debut)
    durees = np.array(durees) * 1000
    return moteur.forme_entree, float(np.median(durees)), float(np.percentile(durees, 90))


def main():
    parseur = argparse.ArgumentParser(description="Latence d'inférence de modèles .onnx (PC) ou .engine (Jetson).")
    parseur.add_argument("modeles", nargs="+", help="fichiers de modèles à mesurer")
    parseur.add_argument("--chauffe", type=int, default=10, help="passages ignorés au début (défaut : 10)")
    parseur.add_argument("--mesures", type=int, default=50, help="passages mesurés (défaut : 50)")
    arguments = parseur.parse_args()
    for chemin in arguments.modeles:
        forme, mediane, p90 = mesurer(chemin, arguments.chauffe, arguments.mesures)
        # Forme complète : (1, 3, 640, 640) pour le détecteur, (16, 32, 32, 1) pour le lecteur de caractères
        print("{} : entrée {} ; latence médiane {:.2f} ms ; 90 % des passages sous {:.2f} ms".format(
            os.path.basename(chemin), tuple(forme), mediane, p90))


if __name__ == "__main__":
    main()
