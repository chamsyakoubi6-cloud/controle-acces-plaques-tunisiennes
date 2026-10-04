"""Mode debug : collecte les images intermédiaires de chaque étape du traitement classique.

Chaque fonction de traitement accepte un paramètre debug (None par défaut). S'il vaut un objet Debug,
elle y ajoute ses images intermédiaires : on peut ensuite les afficher (démonstration) ou les enregistrer
(captures pour le rapport ; seule possibilité sur la Jetson sans écran).

Compatible Python 3.6, NumPy 1.19, OpenCV 4.1.1 (PC et Jetson).
"""

import os

import cv2
import numpy as np


class Debug:
    """Liste ordonnée d'images nommées (le nom doit être en ASCII : cv2.putText ne connaît que l'ASCII)."""

    def __init__(self):
        self.etapes = []  # liste de (nom, image couleur)

    def ajouter(self, nom, image):
        if image.ndim == 2:  # image en niveaux de gris ou binaire : convertie pour pouvoir tout assembler
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.etapes.append((nom, image.copy()))

    def enregistrer(self, dossier, prefixe):
        """Une image PNG par étape : <prefixe>_01_decoupe.png, <prefixe>_02_..., dans l'ordre du traitement."""
        os.makedirs(dossier, exist_ok=True)
        for numero, (nom, image) in enumerate(self.etapes, start=1):
            cv2.imwrite(os.path.join(dossier, "{}_{:02d}_{}.png".format(prefixe, numero, nom)), image)

    def planche(self, largeur=450):
        """Toutes les étapes l'une sous l'autre, à la même largeur, chacune précédée de son nom."""
        morceaux = []
        for nom, image in self.etapes:
            hauteur = max(1, int(round(image.shape[0] * largeur / image.shape[1])))
            bandeau = np.zeros((18, largeur, 3), dtype=np.uint8)
            cv2.putText(bandeau, nom, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            morceaux.append(bandeau)
            morceaux.append(cv2.resize(image, (largeur, hauteur), interpolation=cv2.INTER_AREA))
        if not morceaux:
            return np.zeros((18, largeur, 3), dtype=np.uint8)
        return np.vstack(morceaux)
