"""Étape 5 du pipeline : lecture des caractères par le CNN (phase 4).

Le CNN classe chaque vignette 32 x 32 (découpée par systeme/segmentation.py) en 11 classes : les chiffres
0 à 9 (indices 0 à 9) et « autre » (indice 10 : fragments du mot arabe, barres du cadre, vis, bruit).
La préparation des vignettes est définie ICI et importée par l'entraînement (entrainement/entrainer_cnn.py) :
une seule définition, donc aucun écart possible entre ce que le CNN a appris et ce qu'il reçoit en service.

Compatible Python 3.6, NumPy 1.19 (PC et Jetson).
"""

import numpy as np

from systeme.inference import creer_moteur

AUTRE = 10
NOMS_CLASSES = [str(k) for k in range(10)] + ["autre"]
ENTREES = ("gris", "binaire")
# Entrée retenue par la mesure (validation croisée par véhicule, hors pli, 2 785 vignettes) : la vignette grise.
# Deux fois moins d'erreurs que le masque binaire (24 contre 49 ; McNemar p = 0,00017), y compris sur fond bleu
# simulé, et beaucoup moins de barres du cadre lues comme des chiffres (6 contre 32). Voir docs/journal.md.
ENTREE = "gris"


def preparer_vignettes(vignettes, entree):
    """Vignettes 32 x 32 (uint8) -> tableau (N, 32, 32, 1) float32 pour le CNN.
      binaire : 0 ou 1 (seuil à mi-hauteur : les vignettes binaires valent 0 ou 255, mais une petite rotation
                d'augmentation crée des gris intermédiaires sur les bords des traits) ;
      gris    : chaque vignette étirée entre son minimum et son maximum, vers [0, 1] : le niveau du fond
                (noir ou bleu) et l'éclairage s'effacent, seul reste le contraste entre caractère et fond."""
    tableau = np.asarray(vignettes, dtype=np.float32)
    if entree == "binaire":
        tableau = (tableau > 127).astype(np.float32)
    else:
        minimum = tableau.min(axis=(1, 2), keepdims=True)
        maximum = tableau.max(axis=(1, 2), keepdims=True)
        tableau = (tableau - minimum) / np.maximum(maximum - minimum, 1.0)
    return tableau[:, :, :, np.newaxis]


class LecteurCaracteres:
    """Lit des vignettes avec le CNN exporté : ONNX Runtime sur PC (.onnx), TensorRT sur la Jetson (.engine,
    phase 6). La taille de lot est celle du modèle (entrée fixe) : les vignettes partent par paquets de cette
    taille, le dernier complété par des zéros dont les sorties sont ignorées."""

    def __init__(self, chemin_modele, entree=ENTREE):
        self.moteur = creer_moteur(chemin_modele)
        self.lot = self.moteur.forme_entree[0]
        self.entree = entree

    def lire(self, vignettes):
        """Vignettes 32 x 32 (uint8) -> (classes, confiances) : classe prédite et sa probabilité, pour chacune."""
        if len(vignettes) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        tableau = preparer_vignettes(vignettes, self.entree)
        probabilites = []
        for debut in range(0, len(tableau), self.lot):
            morceau = tableau[debut:debut + self.lot]
            complet = np.zeros((self.lot,) + tableau.shape[1:], dtype=np.float32)
            complet[:len(morceau)] = morceau
            probabilites.append(self.moteur.executer(complet)[:len(morceau)])
        probabilites = np.concatenate(probabilites)
        return probabilites.argmax(axis=1), probabilites.max(axis=1)
