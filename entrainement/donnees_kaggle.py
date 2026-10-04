"""Outils communs pour le jeu Kaggle « Tunisian Licensed Plates » (annotations Pascal VOC).

Utilisé par inspecter_donnees.py et preparer_yolo.py (PC uniquement).
"""

import csv
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import numpy as np

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_KAGGLE = RACINE / "donnees" / "kaggle_brut" / "dataset"
FICHIER_CORRECTIONS = Path(__file__).resolve().parent / "corrections_kaggle.csv"
PARTIES = ("train", "test")

# Sens de rotation (vocabulaire du fichier de corrections) -> constante OpenCV
ROTATIONS_OPENCV = {"horaire": cv2.ROTATE_90_CLOCKWISE, "antihoraire": cv2.ROTATE_90_COUNTERCLOCKWISE}


# ---------------------------------------------------------------- Lecture

def lister_images(partie):
    """Chemins des images d'une partie du jeu ("train" ou "test"), triés par numéro.
    zfill complète le nom par des zéros ("22" -> "0000000022") pour trier 22 avant 142."""
    return sorted((DOSSIER_KAGGLE / partie).glob("*.jpg"), key=lambda chemin: chemin.stem.zfill(10))


def lire_cadres(chemin_image):
    """Lit l'annotation Pascal VOC (fichier .xml de LabelImg) associée à une image.
    Renvoie la liste des cadres [(xmin, ymin, xmax, ymax), ...] en pixels."""
    racine_xml = ET.parse(chemin_image.with_suffix(".xml")).getroot()
    cadres = []
    for objet in racine_xml.findall("object"):
        cadres.append(tuple(float(objet.findtext("bndbox/" + cle)) for cle in ("xmin", "ymin", "xmax", "ymax")))
    return cadres


def lire_corrections():
    """Rotations à appliquer aux images couchées, lues dans corrections_kaggle.csv.
    Renvoie un dictionnaire {("train", "570.jpg"): "horaire", ...}."""
    corrections = {}
    if FICHIER_CORRECTIONS.exists():
        with open(FICHIER_CORRECTIONS, encoding="utf-8", newline="") as fichier:
            for ligne in csv.DictReader(fichier):
                corrections[(ligne["partie"], ligne["image"])] = ligne["rotation"]
    return corrections


# ------------------------------------------------------- Doublons (dHash)

def empreinte_dhash(image_bgr):
    """Empreinte perceptuelle de 64 bits (« difference hash »).
    L'image est réduite à 9x8 pixels en niveaux de gris ; chaque bit indique si un pixel est
    plus clair que son voisin de droite. Deux photos presque identiques (recompressées,
    redimensionnées) ont presque les mêmes bits."""
    gris = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    vignette = cv2.resize(gris, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (vignette[:, 1:] > vignette[:, :-1]).flatten()  # 64 booléens


def distances_hamming(empreintes):
    """Matrice des distances entre empreintes : nombre de bits différents (0 = identiques)."""
    bits = np.array(empreintes)
    return (bits[:, np.newaxis, :] != bits[np.newaxis, :, :]).sum(axis=2)


def grouper(nb_elements, paires):
    """Regroupe de proche en proche les éléments reliés par des paires (i, j) :
    si A ressemble à B et B à C, alors A, B et C forment un seul groupe.
    Renvoie une liste de groupes (listes d'indices)."""
    parent = list(range(nb_elements))

    def representant(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, j in paires:
        parent[representant(i)] = representant(j)
    groupes = {}
    for i in range(nb_elements):
        groupes.setdefault(representant(i), []).append(i)
    return list(groupes.values())


# ------------------------------------------------------ Images couchées

def tourner_image_et_cadres(image, cadres, sens):
    """Tourne l'image d'un quart de tour ("horaire" ou "antihoraire") et recalcule ses cadres.
    Un quart de tour transforme un rectangle droit en un autre rectangle droit : le calcul est exact."""
    hauteur, largeur = image.shape[:2]
    image_tournee = cv2.rotate(image, ROTATIONS_OPENCV[sens])
    nouveaux_cadres = []
    for x0, y0, x1, y1 in cadres:
        if sens == "horaire":  # le point (x, y) va en (hauteur - y, x)
            nouveaux_cadres.append((hauteur - y1, x0, hauteur - y0, x1))
        else:                  # le point (x, y) va en (y, largeur - x)
            nouveaux_cadres.append((y0, largeur - x1, y1, largeur - x0))
    return image_tournee, nouveaux_cadres


# --------------------------------------------------------------- Figures

def dessiner_cadres(image, cadres, couleur=(0, 0, 255)):
    """Dessine des cadres (x0, y0, x1, y1) ; épaisseur adaptée à la taille de l'image."""
    epaisseur = max(2, image.shape[1] // 250)
    for x0, y0, x1, y1 in cadres:
        cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), couleur, epaisseur)
    return image


def texte_ascii(texte):
    """Retire les accents (« manquée » -> « manquee ») : la police de cv2.putText ne connaît que l'ASCII."""
    return unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode("ascii")


def faire_vignette(image, largeur, hauteur, texte=""):
    """Réduit l'image dans un cadre largeur x hauteur sans la déformer (fond noir), avec une légende."""
    texte = texte_ascii(texte)
    rapport = min(largeur / image.shape[1], hauteur / image.shape[0])
    reduite = cv2.resize(image, (max(1, int(image.shape[1] * rapport)), max(1, int(image.shape[0] * rapport))),
                         interpolation=cv2.INTER_AREA)
    vignette = np.zeros((hauteur, largeur, 3), dtype=np.uint8)
    y, x = (hauteur - reduite.shape[0]) // 2, (largeur - reduite.shape[1]) // 2
    vignette[y:y + reduite.shape[0], x:x + reduite.shape[1]] = reduite
    if texte:
        cv2.putText(vignette, texte, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(vignette, texte, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return vignette


def assembler_planche(vignettes, colonnes):
    """Assemble des vignettes de même taille en une grille (les cases vides restent noires)."""
    hauteur, largeur = vignettes[0].shape[:2]
    while len(vignettes) % colonnes:
        vignettes = vignettes + [np.zeros((hauteur, largeur, 3), dtype=np.uint8)]
    lignes = [np.hstack(vignettes[i:i + colonnes]) for i in range(0, len(vignettes), colonnes)]
    return np.vstack(lignes)
