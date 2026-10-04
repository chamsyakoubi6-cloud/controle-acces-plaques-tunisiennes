"""Étape 3b du pipeline : binarisation de la plaque redressée (450 x 100) : caractères en blanc, fond en noir.

  1. Niveaux de gris : « luminance » (conversion standard, qui pondère les canaux) ou « minimum »
     (min(B, V, R) : un caractère blanc est clair dans les trois canaux, un fond noir OU BLEU a au moins
     un canal faible, le rouge pour le bleu).
  2. Contraste : CLAHE, égalisation d'histogramme locale et limitée (ombres, nuit, reflets).
  3. Seuillage : « otsu » (un seuil global unique) ou « adaptatif » (chaque pixel est comparé à la
     moyenne pondérée de son voisinage, diminuée de la constante C : résiste aux éclairages inégaux).
     Caractères blancs sur fond noir : le seuillage direct les donne en blanc, au premier plan.
  4. Effacement du liseré : une ouverture avec de longs segments horizontaux et verticaux ne garde que
     les lignes longues (plus longues que tout caractère) ; on les retire de l'image.
  5. Nettoyage : ouverture morphologique, qui retire les points plus petits que le noyau.
Les valeurs par défaut ci-dessous sont celles retenues d'après les mesures (voir docs/journal.md).

Compatible Python 3.6, NumPy 1.19, OpenCV 4.1.1 (PC et Jetson).
"""

import cv2
import numpy as np

# Valeurs retenues d'après les mesures sur 122 plaques de réglage (taux de segmentation correcte) :
GRIS = "minimum"      # identique à « luminance » sur fond noir (46,7 %) ; bien meilleur sur fond bleu clair simulé
                      # (76 % contre 64 %)
CLAHE = True          # sans CLAHE : -3 points
CLAHE_LIMITE, CLAHE_GRILLE = 2.0, (8, 2)   # grille de 8 x 2 tuiles d'environ 56 x 50 px sur une plaque 450 x 100
SEUILLAGE = "adaptatif"  # Otsu : 33 % ; adaptatif : plateau de 41 à 45 % pour des blocs de 31 à 51 px et C de -20 à -5
TAILLE_BLOC = 31      # au centre du plateau (px, impair) ; un bloc de 15 px, plus petit qu'un caractère, s'effondre
CONSTANTE = -10       # négatif : un pixel doit dépasser la moyenne locale de 10 niveaux pour être « caractère »
NOYAU = 3             # traits de 7,6 px au moins (p10) : 3 x 3 nettoie sans les ronger ; 4 x 4 et 5 x 5 dégradent
LISERE = True         # effet non significatif : +1,6 point sur 122 plaques ; sur 548, stricte -0,7 et
                      # « aucun chiffre perdu » +0,4 : conservé, le retirer serait courir après le bruit
LONGUEUR_LIGNE_H, LONGUEUR_LIGNE_V = 0.40, 0.80  # lignes à effacer : fraction de la largeur / hauteur
# Seconde passe près des bords (itération ciblée, option « bords » de l'étape 3, mesurée : voir docs/journal.md) :
# lignes plus courtes, cherchées seulement dans les bandes du bord de la plaque (restes du liseré, coins arrondis)
LISERE_BORDS = False
BANDE_HAUT_BAS, BANDE_COTES = 0.15, 0.05          # bandes du bord : fraction de la hauteur / de la largeur
LONGUEUR_BORD_H, LONGUEUR_BORD_V = 0.15, 0.50     # lignes plus courtes : fraction de la largeur / hauteur


def niveaux_de_gris(image_bgr, methode=GRIS):
    if methode == "minimum":
        return image_bgr.min(axis=2)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def seuiller(gris, methode=SEUILLAGE, taille_bloc=TAILLE_BLOC, constante=CONSTANTE):
    if methode == "otsu":
        _, binaire = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binaire
    return cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, taille_bloc, constante)


def lignes(binaire, longueur_h, longueur_v):
    """Pixels des lignes horizontales et verticales d'au moins ces longueurs (ouverture par des segments)."""
    noyau_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, longueur_h), 1))
    noyau_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, longueur_v)))
    return cv2.morphologyEx(binaire, cv2.MORPH_OPEN, noyau_h), cv2.morphologyEx(binaire, cv2.MORPH_OPEN, noyau_v)


def effacer_lisere(binaire, bords=False):
    """Retire les longues lignes horizontales et verticales (liseré, bords du cadre). Avec bords, seconde passe :
    des lignes plus courtes, mais seulement dans les bandes du bord de la plaque, jamais au milieu, là où sont
    les chiffres."""
    hauteur, largeur = binaire.shape
    horizontales, verticales = lignes(binaire, int(LONGUEUR_LIGNE_H * largeur), int(LONGUEUR_LIGNE_V * hauteur))
    resultat = cv2.subtract(binaire, cv2.bitwise_or(horizontales, verticales))
    if bords:
        horizontales, verticales = lignes(resultat, int(LONGUEUR_BORD_H * largeur), int(LONGUEUR_BORD_V * hauteur))
        bande_h, bande_v = int(BANDE_HAUT_BAS * hauteur), int(BANDE_COTES * largeur)
        dans_bandes_h = np.zeros_like(binaire)
        dans_bandes_h[:bande_h] = 255
        dans_bandes_h[hauteur - bande_h:] = 255
        dans_bandes_v = np.zeros_like(binaire)
        dans_bandes_v[:, :bande_v] = 255
        dans_bandes_v[:, largeur - bande_v:] = 255
        restes = cv2.bitwise_or(cv2.bitwise_and(horizontales, dans_bandes_h), cv2.bitwise_and(verticales, dans_bandes_v))
        resultat = cv2.subtract(resultat, restes)
    return resultat


def binariser(plaque_bgr, debug=None, gris=GRIS, clahe=CLAHE, seuillage=SEUILLAGE, taille_bloc=TAILLE_BLOC,
              constante=CONSTANTE, noyau=NOYAU, lisere=LISERE, lisere_bords=LISERE_BORDS):
    """Plaque redressée (couleur) -> (image binaire, image grise utilisée pour le seuillage)."""
    image_grise = niveaux_de_gris(plaque_bgr, gris)
    if debug is not None:
        debug.ajouter("gris_" + gris, image_grise)
    if clahe:
        image_grise = cv2.createCLAHE(clipLimit=CLAHE_LIMITE, tileGridSize=CLAHE_GRILLE).apply(image_grise)
        if debug is not None:
            debug.ajouter("contraste_clahe", image_grise)
    binaire = seuiller(image_grise, seuillage, taille_bloc, constante)
    if debug is not None:
        debug.ajouter("seuil_" + seuillage, binaire)
    if lisere:
        binaire = effacer_lisere(binaire, lisere_bords)
        if debug is not None:
            debug.ajouter("sans_lisere", binaire)
    if noyau > 0:
        binaire = cv2.morphologyEx(binaire, cv2.MORPH_OPEN, np.ones((noyau, noyau), dtype=np.uint8))
        if debug is not None:
            debug.ajouter("nettoyage_{}x{}".format(noyau, noyau), binaire)
    return binaire, image_grise
