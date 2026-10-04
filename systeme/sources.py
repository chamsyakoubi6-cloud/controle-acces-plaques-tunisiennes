"""Sources d'images du système : webcam, fichier vidéo ou photos (phases 2 et 5).

Toutes les sources fournissent la même chose : des couples (temps en secondes, image BGR). Le reste du système
ne sait pas d'où viennent les images.
  - webcam : temps réel, compté depuis l'ouverture ;
  - vidéo  : temps de la vidéo (horodatage de chaque image dans le fichier). Une mesure ne dépend donc ni de la
             vitesse du PC ni de sa charge : elle est reproductible ;
  - photos : chaque photo est une scène à part, suivie de 10 s sans image (le suivi ne relie jamais deux photos).
             Elle peut être répétée, comme une voiture arrêtée devant la barrière (démonstration).
Cadence simulée (vidéo) : on ne garde que les images qui tombent sur une grille de 1/cadence seconde, comme une
caméra qui ne livrerait que « cadence » images par seconde. Pourquoi : le délai de décision dépend du nombre
d'images traitées par seconde. Un téléphone filme à 30 images/s ; la chaîne en traite moins (15 à 20 sur le PC,
à mesurer sur la Jetson en phase 6). Les images écartées sont seulement décodées (grab), sans conversion.
Largeur maximale : une image plus large est réduite (ex. 1920 -> 1280 px, la résolution de la webcam), pour
évaluer sur des images comparables à celles que verra le système.

Compatible Python 3.6, OpenCV 4.1.1 (PC et Jetson).
"""

import math
import os
import sys
import time

import cv2

EXTENSIONS_PHOTOS = (".jpg", ".jpeg", ".png")
PAUSE_PHOTOS = 10.0                # secondes sans image après chaque photo : sa piste se ferme (1 s) et la
                                   # barrière a le temps de se refermer avant la suivante
IMAGES_PAR_SECONDE_DEFAUT = 30.0   # si le fichier vidéo n'indique pas sa cadence
CADENCE_PHOTOS = 10.0              # images par seconde d'une photo répétée


def ouvrir_webcam(index=0, largeur=1280, hauteur=720):
    """Ouvre la webcam avec le pilote adapté au système.

    Renvoie (capture, nom_pilote) ; capture.isOpened() indique si l'ouverture a réussi.
    La résolution demandée n'est pas garantie : la vraie taille se lit sur les images reçues.
    """
    # Pilote vidéo : Media Foundation (MSMF) sous Windows, V4L2 sous Linux (Jetson).
    # Sur la webcam HP du PC, DirectShow plafonne à 10 images/s alors que MSMF donne 30 images/s
    # en 1280x720 (mesuré en phase 1). L'ouverture rapide de MSMF dépend d'un réglage fait dans
    # systeme/__init__.py, avant l'import de cv2.
    if sys.platform.startswith("win"):
        pilote, nom_pilote = cv2.CAP_MSMF, "Media Foundation"
    else:
        pilote, nom_pilote = cv2.CAP_V4L2, "V4L2"
    capture = cv2.VideoCapture(index, pilote)
    if capture.isOpened():
        # MJPG = images compressées par la caméra elle-même. En format brut, le débit USB limite
        # souvent le 1280x720 à 5-10 images/s (utile surtout en V4L2 sur la Jetson).
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, largeur)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, hauteur)
    return capture, nom_pilote


def reduire(image, largeur_max):
    """Réduit l'image à largeur_max pixels de large, proportions gardées (None ou 0 : taille d'origine)."""
    if not largeur_max or image.shape[1] <= largeur_max:
        return image
    hauteur = int(round(image.shape[0] * largeur_max / image.shape[1]))
    # INTER_AREA : moyenne des pixels regroupés, sans crénelage (recommandé pour réduire)
    return cv2.resize(image, (largeur_max, hauteur), interpolation=cv2.INTER_AREA)


def images_webcam(index=0, largeur_max=None):
    """Images de la webcam, horodatées en temps réel depuis l'ouverture."""
    capture, nom_pilote = ouvrir_webcam(index)
    if not capture.isOpened():
        raise IOError("Webcam {} introuvable (pilote {})".format(index, nom_pilote))
    debut = time.monotonic()
    try:
        while True:
            lu, image = capture.read()
            if not lu:
                print("Lecture de la webcam impossible")
                break
            yield time.monotonic() - debut, reduire(image, largeur_max)
    finally:
        capture.release()


def images_video(chemin, cadence=None, largeur_max=None):
    """Images d'un fichier vidéo, horodatées par le temps de la vidéo ; cadence : images gardées par seconde
    (None : toutes)."""
    capture = cv2.VideoCapture(chemin)
    if not capture.isOpened():
        raise IOError("Vidéo illisible : " + chemin)
    images_par_seconde = capture.get(cv2.CAP_PROP_FPS)
    if not images_par_seconde > 0:   # 0 ou NaN : cadence inconnue
        images_par_seconde = IMAGES_PAR_SECONDE_DEFAUT
    prochain, precedent = 0.0, None
    try:
        while capture.grab():
            # Temps de l'image : son horodatage dans le fichier, exact même si le téléphone a filmé à cadence
            # variable (basse lumière). OpenCV 4.1.1 (Jetson) renvoie à la place numéro / images par seconde.
            # Horodatage absent ou qui recule : on avance d'une image à la cadence nominale.
            temps = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if precedent is not None and temps <= precedent:
                temps = precedent + 1.0 / images_par_seconde
            precedent = temps
            if cadence:
                if temps < prochain - 1e-6:
                    continue
                # Point suivant de la grille, calculé (pas cumulé) : aucune dérive d'arrondi sur une longue vidéo
                prochain = (math.floor(temps * cadence + 1e-6) + 1) / cadence
            lu, image = capture.retrieve()
            if lu:
                yield temps, reduire(image, largeur_max)
    finally:
        capture.release()


def repetitions_photo(duree):
    """Nombre de passages d'une photo pour la montrer « duree » secondes à CADENCE_PHOTOS (au moins un)."""
    return max(1, int(round(duree * CADENCE_PHOTOS)))


def images_photos(chemins, repetitions=1, largeur_max=None):
    """Chaque photo, répétée « repetitions » fois à CADENCE_PHOTOS images par seconde, puis PAUSE_PHOTOS secondes
    sans image. La pause suit la photo : même montrée longtemps, une photo ne chevauche jamais la suivante, et le
    temps ne recule jamais (le suivi et la barrière en ont besoin)."""
    duree = repetitions / CADENCE_PHOTOS
    for rang, chemin in enumerate(chemins):
        image = cv2.imread(chemin)
        if image is None:
            print("Photo illisible, ignorée :", chemin)
            continue
        image = reduire(image, largeur_max)
        debut = rang * (duree + PAUSE_PHOTOS)
        for repetition in range(repetitions):
            yield debut + repetition / CADENCE_PHOTOS, image


def ouvrir_source(source, cadence=None, largeur_max=None, repetitions=1):
    """Texte de la ligne de commande -> (images, description, en_direct).
    source : index de webcam (« 0 »), fichier vidéo, photo, ou dossier de photos (dans l'ordre alphabétique)."""
    if source.isdigit():
        return images_webcam(int(source), largeur_max), "webcam {}".format(source), True
    if os.path.isdir(source):
        chemins = sorted(os.path.join(source, nom) for nom in os.listdir(source)
                         if os.path.splitext(nom)[1].lower() in EXTENSIONS_PHOTOS)
        description = "{} photos de {}".format(len(chemins), source)
        return images_photos(chemins, repetitions, largeur_max), description, False
    if not os.path.isfile(source):
        raise IOError("Source introuvable : " + source)
    if os.path.splitext(source)[1].lower() in EXTENSIONS_PHOTOS:
        return images_photos([source], repetitions, largeur_max), "photo " + source, False
    description = "vidéo {}{}".format(source, ", {} images/s".format(cadence) if cadence else "")
    return images_video(source, cadence, largeur_max), description, False
