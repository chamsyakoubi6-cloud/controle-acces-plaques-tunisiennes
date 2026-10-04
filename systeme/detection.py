"""Détection des plaques avec YOLOv8 : pré-traitement, inférence, décodage et NMS en NumPy.

Chaîne complète pour une image de la webcam (ex. 1280x720) :
  1. letterbox : réduction sans déformation à la taille d'entrée du modèle (ex. 416x416), bandes grises ;
  2. préparation : BGR -> RGB, HWC -> CHW, valeurs ramenées entre 0 et 1, dimension de lot ;
  3. inférence : moteur ONNX Runtime (PC) ou TensorRT (Jetson), sortie (1, 5, N) ;
  4. décodage : N boîtes candidates (centre, taille, score) -> coins, filtre par score ;
  5. NMS : suppression des boîtes redondantes ;
  6. retour aux coordonnées de l'image d'origine, pour découper ensuite la plaque en pleine résolution.
Les étapes 1 et 6 reproduisent exactement Ultralytics, avec qui le modèle a été entraîné
(vérifié par entrainement/verifier_detection_numpy.py).

Compatible Python 3.6, NumPy 1.19, OpenCV 4.1.1 (PC et Jetson).
"""

import time

import cv2
import numpy as np

from systeme.inference import creer_moteur

GRIS_BORDURE = 114  # couleur des bandes du letterbox : la même qu'à l'entraînement
# Seuils du système (un seul endroit : démo, chaîne, évaluation). Confiance 0,5 : choix de la phase 2 (cascade :
# priorité au rappel), conservé en phase 5 faute de vidéos réelles pour le réajuster (docs/journal.md).
SEUIL_CONFIANCE = 0.5
SEUIL_IOU = 0.45


def letterbox(image, taille):
    """Réduit l'image dans un carré taille x taille sans la déformer, en complétant par des bandes grises.

    Renvoie (image_carree, rapport, (decalage_x, decalage_y)). Pour revenir à l'image d'origine :
    retirer le décalage, puis diviser par le rapport.
    """
    hauteur, largeur = image.shape[:2]
    rapport = min(taille / hauteur, taille / largeur)
    nouvelle_largeur, nouvelle_hauteur = int(round(largeur * rapport)), int(round(hauteur * rapport))
    if (nouvelle_largeur, nouvelle_hauteur) != (largeur, hauteur):
        image = cv2.resize(image, (nouvelle_largeur, nouvelle_hauteur), interpolation=cv2.INTER_LINEAR)
    # Bandes réparties de part et d'autre. Quand la marge est impaire, le +/- 0,1 décide de quel côté
    # va le pixel en trop (même règle qu'Ultralytics).
    marge_x = (taille - nouvelle_largeur) / 2
    marge_y = (taille - nouvelle_hauteur) / 2
    haut, bas = int(round(marge_y - 0.1)), int(round(marge_y + 0.1))
    gauche, droite = int(round(marge_x - 0.1)), int(round(marge_x + 0.1))
    image = cv2.copyMakeBorder(image, haut, bas, gauche, droite, cv2.BORDER_CONSTANT,
                               value=(GRIS_BORDURE, GRIS_BORDURE, GRIS_BORDURE))
    return image, rapport, (gauche, haut)


def preparer_entree(image_bgr, taille):
    """Image de la caméra -> tableau (1, 3, taille, taille) en float32, le format attendu par le modèle."""
    image_carree, rapport, decalage = letterbox(image_bgr, taille)
    # BGR (ordre d'OpenCV) -> RGB (ordre vu à l'entraînement), puis HWC -> CHW (canaux en premier)
    tableau = image_carree[:, :, ::-1].transpose(2, 0, 1)
    # Tableau contigu en float32, valeurs ramenées de [0, 255] à [0, 1]
    tableau = np.ascontiguousarray(tableau, dtype=np.float32) / np.float32(255.0)
    return tableau[np.newaxis], rapport, decalage  # dimension de lot ajoutée : (1, 3, H, W)


def decoder_sorties(sortie, seuil_confiance):
    """Sortie brute de YOLOv8 (1, 4 + nb_classes, N) -> boîtes (x1, y1, x2, y2) et scores au-dessus du seuil.

    Pour chacune des N boîtes candidates, le modèle donne : centre x, centre y, largeur, hauteur
    (en pixels de l'image carrée), puis un score par classe, déjà compris entre 0 et 1.
    """
    candidats = sortie[0].T                   # (N, 4 + nb_classes) : une ligne par boîte candidate
    scores = candidats[:, 4:].max(axis=1)     # une seule classe ici : c'est directement son score
    retenus = scores > seuil_confiance
    centres_x, centres_y, largeurs, hauteurs = candidats[retenus, :4].T
    boites = np.stack([centres_x - largeurs / 2, centres_y - hauteurs / 2,
                       centres_x + largeurs / 2, centres_y + hauteurs / 2], axis=1)
    return boites, scores[retenus]


def iou(boite, boites):
    """Recouvrement (aire de l'intersection / aire de l'union) entre une boîte et un ensemble de boîtes."""
    x1 = np.maximum(boite[0], boites[:, 0])
    y1 = np.maximum(boite[1], boites[:, 1])
    x2 = np.minimum(boite[2], boites[:, 2])
    y2 = np.minimum(boite[3], boites[:, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aire = (boite[2] - boite[0]) * (boite[3] - boite[1])
    aires = (boites[:, 2] - boites[:, 0]) * (boites[:, 3] - boites[:, 1])
    return intersection / (aire + aires - intersection + 1e-9)


def nms(boites, scores, seuil_iou):
    """Suppression des non-maxima : parmi des boîtes qui se recouvrent, on ne garde que la plus sûre.

    On prend la boîte au meilleur score, on élimine celles qui la recouvrent au-delà du seuil,
    puis on recommence avec les boîtes restantes. Renvoie les indices des boîtes gardées.
    """
    ordre = np.argsort(-scores, kind="stable")  # du meilleur score au moins bon
    gardees = []
    while ordre.size > 0:
        meilleure = ordre[0]
        gardees.append(meilleure)
        recouvrements = iou(boites[meilleure], boites[ordre[1:]])
        ordre = ordre[1:][recouvrements <= seuil_iou]
    return np.array(gardees, dtype=np.int64)


def vers_image_origine(boites, rapport, decalage, forme_image):
    """Ramène des boîtes de l'image carrée (letterbox) dans l'image d'origine, bornées à ses limites."""
    boites = boites.copy()
    boites[:, [0, 2]] = (boites[:, [0, 2]] - decalage[0]) / rapport
    boites[:, [1, 3]] = (boites[:, [1, 3]] - decalage[1]) / rapport
    hauteur, largeur = forme_image[:2]
    boites[:, [0, 2]] = np.clip(boites[:, [0, 2]], 0, largeur)
    boites[:, [1, 3]] = np.clip(boites[:, [1, 3]], 0, hauteur)
    return boites


class DetecteurPlaques:
    """Détecteur complet : image de la caméra -> plaques trouvées, en coordonnées de cette image."""

    def __init__(self, chemin_modele, seuil_confiance=SEUIL_CONFIANCE, seuil_iou=SEUIL_IOU):
        self.moteur = creer_moteur(chemin_modele)
        _, _, hauteur, largeur = self.moteur.forme_entree
        if hauteur != largeur:
            raise ValueError("Entrée carrée attendue, reçu {}x{}".format(largeur, hauteur))
        self.taille = hauteur  # lue dans le modèle : aucune incohérence possible avec l'entraînement
        self.seuil_confiance = seuil_confiance
        self.seuil_iou = seuil_iou
        self.durees = {}       # durée de chaque étape de la dernière détection, en secondes

    def detecter(self, image_bgr):
        """Renvoie un tableau (K, 5) : x1, y1, x2, y2, score pour chaque plaque, du meilleur score au moins bon."""
        debut = time.perf_counter()
        entree, rapport, decalage = preparer_entree(image_bgr, self.taille)
        apres_pretraitement = time.perf_counter()
        sortie = self.moteur.executer(entree)
        apres_inference = time.perf_counter()
        boites, scores = decoder_sorties(sortie, self.seuil_confiance)
        gardees = nms(boites, scores, self.seuil_iou)
        boites = vers_image_origine(boites[gardees], rapport, decalage, image_bgr.shape)
        detections = np.hstack([boites, scores[gardees][:, np.newaxis]]).astype(np.float32)
        fin = time.perf_counter()
        self.durees = {"pretraitement": apres_pretraitement - debut,
                       "inference": apres_inference - apres_pretraitement,
                       "posttraitement": fin - apres_inference}
        return detections


def dessiner_detections(image, detections, couleur=(0, 255, 0)):
    """Encadre chaque plaque détectée et écrit son score (vue de contrôle et démonstration)."""
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), couleur, 2)
        cv2.putText(image, "plaque {:.2f}".format(score), (int(x1), max(15, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)
    return image
