"""Moteurs d'inférence : une même interface sur le PC (ONNX Runtime) et sur la Jetson (TensorRT).

Le reste du pipeline ne connaît que deux choses d'un moteur :
  - moteur.forme_entree : forme du tableau attendu, ex. (1, 3, 416, 416) ;
  - moteur.executer(tableau) : renvoie la sortie du modèle (tableau NumPy).
Chaque moteur importe sa bibliothèque à l'intérieur de sa classe : le PC n'a pas besoin de
TensorRT, et la Jetson n'a pas ONNX Runtime.
"""

import os


class MoteurOnnx:
    """Exécute un modèle .onnx avec ONNX Runtime, sur le processeur (PC)."""

    def __init__(self, chemin_modele):
        import onnxruntime  # importé ici seulement : la Jetson n'a pas ONNX Runtime

        self.session = onnxruntime.InferenceSession(chemin_modele, providers=["CPUExecutionProvider"])
        entree = self.session.get_inputs()[0]
        self.nom_entree = entree.name
        self.forme_entree = tuple(entree.shape)
        # Le pipeline suppose une entrée de taille fixe (export avec dynamic=False)
        if not all(isinstance(dimension, int) for dimension in self.forme_entree):
            raise ValueError("Entrée de taille variable {} : exporter le modèle avec une taille fixe".format(
                self.forme_entree))

    def executer(self, tableau):
        """tableau : float32 de forme self.forme_entree ; renvoie la première sortie du modèle."""
        return self.session.run(None, {self.nom_entree: tableau})[0]


def creer_moteur(chemin_modele):
    """Choisit le moteur d'après l'extension du fichier : .onnx (PC) ou .engine (Jetson)."""
    extension = os.path.splitext(chemin_modele)[1].lower()
    if extension == ".onnx":
        return MoteurOnnx(chemin_modele)
    if extension == ".engine":
        raise NotImplementedError("Moteur TensorRT (.engine) : prévu en phase 6 (portage sur la Jetson).")
    raise ValueError("Format de modèle inconnu : " + chemin_modele)
