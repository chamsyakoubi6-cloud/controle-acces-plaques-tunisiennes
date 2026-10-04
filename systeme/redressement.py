"""Étape 3a du pipeline : redressement de la plaque (quatre coins, puis homographie).

Point de départ : le cadre de la plaque dans l'image en pleine résolution (détecté par YOLO, ou annoté).
  1. Découpe du cadre élargi d'une marge : le cadre YOLO peut être un peu serré, et les coins d'une
     plaque vue de biais dépassent souvent du cadre.
  2. Recherche des quatre coins, par quatre méthodes classiques proposées dans l'ordre (la cascade) :
       a. « lisere »     : contour du liseré blanc qui borde la plupart des plaques (seuil d'Otsu) ;
       b. « zone_noire » : contour de la zone noire de la plaque (seuil d'Otsu inversé) ;
       c. « caracteres » : rectangle orienté englobant les blobs clairs de taille « caractère »,
                           agrandi aux proportions d'une plaque (corrige la rotation, pas la perspective) ;
       d. « bords »      : contour formé par les bords de Canny (voiture de la teinte de la plaque ou du liseré) ;
     et, en dernier recours, « cadre » : le cadre lui-même, sans correction. Le choix entre ces propositions
     se fait dans systeme/traitement.py, d'après la rangée de caractères qu'elles donnent.
  3. Homographie des quatre coins vers un rectangle de 450 x 100 px, aux proportions de la plaque avant
     (450 x 100 mm), celle que la caméra de la barrière voit quand la voiture arrive.
Après redressement, toutes les plaques ont la même taille : les paramètres en pixels des étapes suivantes
(taille de bloc, noyaux, filtres) ont le même sens pour toutes les plaques.

Compatible Python 3.6, NumPy 1.19, OpenCV 4.1.1 (PC et Jetson).
"""

import cv2
import numpy as np

LARGEUR_PLAQUE, HAUTEUR_PLAQUE = 450, 100
RAPPORT_PLAQUE = LARGEUR_PLAQUE / HAUTEUR_PLAQUE
MARGE_X, MARGE_Y = 0.15, 0.30        # marge ajoutée au cadre, en fraction de sa largeur et de sa hauteur
TOLERANCES_POLYGONE = (0.02, 0.03, 0.05)  # approxPolyDP : écarts tolérés, en fraction du périmètre, essayés dans l'ordre
AIRE_MIN, AIRE_MAX = 0.25, 1.60      # aire du quadrilatère, en fraction de l'aire du cadre (une plaque inclinée
                                     # de 25 degrés n'occupe qu'environ 35 % de son cadre)
RAPPORT_MIN, RAPPORT_MAX = 1.5, 7.0  # forme de plaque : largeur / hauteur du quadrilatère
COTE_COURT_MIN = 0.25                # côtés courts d'au moins 25 % de la hauteur du cadre (écarte les « copeaux »)
COTES_OPPOSES_MAX = 2.0              # en perspective, deux côtés opposés d'une plaque restent de longueurs comparables
PROPORTION_CARACTERES = 0.65         # hauteur des caractères / hauteur de la plaque (méthode « caracteres »)
# Élargissement du quadrilatère avant l'homographie, en fraction de ses côtés : le contour trouvé est souvent
# le bord INTÉRIEUR du liseré, collé aux caractères ; sans marge, les chiffres touchent le bord de l'image
# redressée et la segmentation les écarte. Mesuré sur 122 plaques de réglage : 34 % de segmentations
# correctes sans marge, 45 % avec 2 % / 5 % ; au-delà de 10 % en vertical, le taux baisse (plaque trop petite).
MARGE_REDRESSEMENT_X, MARGE_REDRESSEMENT_Y = 0.02, 0.05
LARGEUR_VUE_COINS = 480              # px : largeur minimale de l'image de contrôle des coins (mode debug)


def decouper(image, boite):
    """Découpe le cadre (x1, y1, x2, y2) élargi de la marge.
    Renvoie (decoupe, origine, cadre) : origine = coin haut-gauche de la découpe dans l'image,
    cadre = le cadre exprimé dans les coordonnées de la découpe."""
    x1, y1, x2, y2 = [float(v) for v in boite[:4]]
    largeur, hauteur = x2 - x1, y2 - y1
    x0 = int(max(0, x1 - MARGE_X * largeur))
    y0 = int(max(0, y1 - MARGE_Y * hauteur))
    x_fin = int(min(image.shape[1], x2 + MARGE_X * largeur))
    y_fin = int(min(image.shape[0], y2 + MARGE_Y * hauteur))
    return image[y0:y_fin, x0:x_fin], (x0, y0), (x1 - x0, y1 - y0, x2 - x0, y2 - y0)


def ordonner_coins(points):
    """Range 4 points dans l'ordre haut-gauche, haut-droit, bas-droit, bas-gauche, même pour une plaque
    fortement inclinée (l'astuce x + y / y - x, plus courante, échoue au-delà de 45 degrés).
    1. Tri par angle autour du centre : l'axe y pointe vers le bas, donc un angle croissant parcourt les
       coins dans le sens des aiguilles d'une montre à l'écran.
    2. On fait tourner la liste jusqu'à ce que le premier côté soit un grand côté orienté vers la droite :
       c'est le bord supérieur de la plaque, parcouru de gauche à droite."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    centre = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))]
    for decalage in range(4):
        tournes = np.roll(points, -decalage, axis=0)
        cote_haut, cote_droit = tournes[1] - tournes[0], tournes[2] - tournes[1]
        if np.linalg.norm(cote_haut) >= np.linalg.norm(cote_droit) and cote_haut[0] > 0:
            return tournes
    return points  # cas dégénéré (plaque verticale) : ordre du tri par angle


def aire_du_cadre(cadre):
    x1, y1, x2, y2 = cadre
    return (x2 - x1) * (y2 - y1)


def quadrilatere(contour, cadre):
    """Simplifie un contour en polygone ; renvoie ses 4 coins ordonnés s'il ressemble à une plaque, sinon None.
    L'enveloppe convexe efface d'abord les « dents » du contour (caractères qui touchent le bord de la zone),
    puis l'approximation polygonale est essayée avec des tolérances croissantes jusqu'à obtenir 4 sommets."""
    enveloppe = cv2.convexHull(contour)
    perimetre = cv2.arcLength(enveloppe, True)
    polygone = None
    for tolerance in TOLERANCES_POLYGONE:
        polygone = cv2.approxPolyDP(enveloppe, tolerance * perimetre, True)
        if len(polygone) == 4:
            break
    if len(polygone) != 4:
        return None
    if not AIRE_MIN * aire_du_cadre(cadre) <= cv2.contourArea(polygone) <= AIRE_MAX * aire_du_cadre(cadre):
        return None
    _, (cote_1, cote_2), _ = cv2.minAreaRect(polygone)
    if not RAPPORT_MIN <= max(cote_1, cote_2) / max(min(cote_1, cote_2), 1.0) <= RAPPORT_MAX:
        return None
    # Le quadrilatère doit être CELUI de la plaque : il contient le centre du cadre détecté (YOLO centre bien
    # son cadre sur la plaque), et son propre centre est dans le cadre. Écarte, par exemple, le trottoir clair
    # sous la plaque, dont l'aire peut ressembler à celle du cadre.
    x1, y1, x2, y2 = cadre
    centre_cadre = ((x1 + x2) / 2, (y1 + y2) / 2)
    centre_polygone = polygone.reshape(4, 2).mean(axis=0)
    if cv2.pointPolygonTest(polygone, centre_cadre, False) < 0:
        return None
    if not (x1 <= centre_polygone[0] <= x2 and y1 <= centre_polygone[1] <= y2):
        return None
    # Géométrie d'une vraie plaque vue en perspective : côtés courts assez longs, côtés opposés comparables
    coins = ordonner_coins(polygone)
    haut, droite, bas, gauche = [float(np.linalg.norm(coins[(k + 1) % 4] - coins[k])) for k in range(4)]
    if min(droite, gauche) < COTE_COURT_MIN * (y2 - y1):
        return None
    if max(haut, bas) > COTES_OPPOSES_MAX * max(min(haut, bas), 1.0) or \
            max(droite, gauche) > COTES_OPPOSES_MAX * max(min(droite, gauche), 1.0):
        return None
    return coins


def meilleur_quadrilatere(binaire, cadre):
    """Parmi tous les contours, le quadrilatère « plaque » dont l'aire est la plus proche de celle du cadre."""
    contours, _ = cv2.findContours(binaire, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    trouves = [q for q in (quadrilatere(c, cadre) for c in contours) if q is not None]
    if not trouves:
        return None
    return min(trouves, key=lambda q: abs(cv2.contourArea(q) - aire_du_cadre(cadre)))


def coins_lisere(flou, cadre):
    """Méthode a : le liseré blanc (et les caractères) ressortent en blanc avec un seuil d'Otsu ;
    le bord intérieur ou extérieur du liseré forme un quadrilatère."""
    _, clair = cv2.threshold(flou, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return meilleur_quadrilatere(clair, cadre)


def coins_zone_noire(flou, cadre):
    """Méthode b : la zone noire de la plaque, avec un seuil inversé. Une fermeture morphologique bouche
    les trous laissés par les caractères blancs (noyau plus large que l'épaisseur d'un trait)."""
    _, sombre = cv2.threshold(flou, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    taille = max(3, int(0.2 * (cadre[3] - cadre[1])))
    sombre = cv2.morphologyEx(sombre, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (taille, taille)))
    return meilleur_quadrilatere(sombre, cadre)


def coins_bords(flou, cadre):
    """Méthode d : bords de Canny. Quand la carrosserie a la teinte de la plaque (voiture sombre) ou du
    liseré (voiture blanche), les seuils globaux confondent les zones ; mais le liseré entre la plaque
    noire et la carrosserie crée toujours des bords. Seuils de Canny réglés automatiquement autour de la
    médiane des gris (règle classique : 0,66 x et 1,33 x la médiane) ; une dilatation 3 x 3 referme les
    petites coupures du contour."""
    mediane = float(np.median(flou))
    bords = cv2.Canny(flou, 0.66 * mediane, 1.33 * mediane)
    bords = cv2.dilate(bords, np.ones((3, 3), dtype=np.uint8))
    return meilleur_quadrilatere(bords, cadre)


def coins_caracteres(flou, cadre):
    """Méthode c : rectangle orienté minimal englobant les blobs clairs de taille « caractère », agrandi
    aux proportions d'une plaque. Corrige la rotation, mais pas la perspective."""
    x1, y1, x2, y2 = cadre
    hauteur_cadre = y2 - y1
    _, clair = cv2.threshold(flou, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nb, etiquettes, stats, _ = cv2.connectedComponentsWithStats(clair, connectivity=8)
    retenus = [i for i in range(1, nb)
               if 0.3 * hauteur_cadre <= stats[i, cv2.CC_STAT_HEIGHT] <= 0.9 * hauteur_cadre
               and stats[i, cv2.CC_STAT_WIDTH] <= 1.5 * stats[i, cv2.CC_STAT_HEIGHT]]
    if len(retenus) < 3:
        return None
    ys, xs = np.nonzero(np.isin(etiquettes, retenus))
    coins = ordonner_coins(cv2.boxPoints(cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))))
    # Axes du rectangle calculés à partir des coins : indépendant de la convention d'angle de minAreaRect,
    # qui a changé entre OpenCV 4.1 (Jetson) et les versions récentes (PC)
    axe_largeur = (coins[1] - coins[0] + coins[2] - coins[3]) / 2
    axe_hauteur = (coins[3] - coins[0] + coins[2] - coins[1]) / 2
    longueur, epaisseur = np.linalg.norm(axe_largeur), np.linalg.norm(axe_hauteur)
    if epaisseur < 1 or longueur < epaisseur:  # les caractères doivent former une ligne horizontale
        return None
    hauteur = epaisseur / PROPORTION_CARACTERES
    largeur = max(longueur, RAPPORT_PLAQUE * hauteur)
    centre = coins.mean(axis=0)
    # Garde-fou : la plaque reconstruite doit rester plausible par rapport au cadre (centre dans le cadre,
    # aire du même ordre) ; sinon, les blobs retenus n'étaient pas des caractères.
    if not (x1 <= centre[0] <= x2 and y1 <= centre[1] <= y2
            and 0.3 * aire_du_cadre(cadre) <= largeur * hauteur <= 2.5 * aire_du_cadre(cadre)):
        return None
    u, v = axe_largeur / longueur, axe_hauteur / epaisseur
    return np.array([centre - u * largeur / 2 - v * hauteur / 2, centre + u * largeur / 2 - v * hauteur / 2,
                     centre + u * largeur / 2 + v * hauteur / 2, centre - u * largeur / 2 + v * hauteur / 2],
                    dtype=np.float32)


METHODES = (("lisere", coins_lisere), ("zone_noire", coins_zone_noire), ("caracteres", coins_caracteres),
            ("bords", coins_bords))


def propositions_coins(decoupe, cadre):
    """Propositions de coins dans l'ordre de la cascade : (nom, coins) pour chaque méthode qui trouve un
    quadrilatère plausible, puis le cadre lui-même (« cadre »). C'est un générateur : une méthode n'est calculée
    que si l'on demande la proposition suivante (la plupart des plaques s'arrêtent à la première)."""
    gris = cv2.cvtColor(decoupe, cv2.COLOR_BGR2GRAY)
    flou = cv2.GaussianBlur(gris, (5, 5), 0)  # le flou atténue le bruit avant les seuillages
    for nom, methode in METHODES:
        coins = methode(flou, cadre)
        if coins is not None:
            yield nom, coins
    x1, y1, x2, y2 = cadre
    yield "cadre", np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def dessiner_coins(decoupe, cadre, essais, nom_retenu):
    """Découpe avec le cadre (bleu), les quadrilatères essayés (gris) et celui retenu (vert). Une découpe étroite
    est d'abord agrandie à LARGEUR_VUE_COINS : sinon, une fois l'image affichée en grand (vue technique de
    l'interface, captures), le texte et les traits couvriraient la plaque."""
    echelle = max(1.0, LARGEUR_VUE_COINS / float(decoupe.shape[1]))
    vue = cv2.resize(decoupe, None, fx=echelle, fy=echelle, interpolation=cv2.INTER_CUBIC) if echelle > 1 else decoupe.copy()
    x1, y1, x2, y2 = (v * echelle for v in cadre)
    cv2.rectangle(vue, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
    for nom, coins in essais:
        couleur, epaisseur = ((0, 255, 0), 2) if nom == nom_retenu else ((150, 150, 150), 1)
        cv2.polylines(vue, [(coins * echelle).astype(np.int32)], True, couleur, epaisseur)
    cv2.putText(vue, " > ".join(nom for nom, _ in essais), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return vue


def trouver_coins(decoupe, cadre, debug=None):
    """Première proposition de la cascade ; renvoie (coins, nom de la méthode)."""
    nom, coins = next(propositions_coins(decoupe, cadre))
    if debug is not None:
        debug.ajouter("coins", dessiner_coins(decoupe, cadre, [(nom, coins)], nom))
    return coins, nom


def couverture_du_cadre(coins, cadre):
    """(part de la largeur du cadre couverte par le quadrilatère, part du quadrilatère qui sort du cadre), en
    largeur. Une plaque coupée couvre mal son cadre ; des coins pris sur la carrosserie ou le bord d'un écran
    en débordent largement. Le cadre YOLO, lui, est en général juste."""
    x1, _, x2, _ = cadre
    gauche, droite = float(coins[:, 0].min()), float(coins[:, 0].max())
    recouvrement = max(0.0, min(droite, x2) - max(gauche, x1))
    return recouvrement / (x2 - x1), (droite - gauche - recouvrement) / (x2 - x1)


def elargir(coins, marge_x, marge_y):
    """Écarte chaque coin du quadrilatère (haut-gauche, haut-droit, bas-droit, bas-gauche) d'une fraction de
    la longueur des côtés voisins. En suivant les côtés, l'élargissement respecte la perspective."""
    haut_gauche, haut_droit, bas_droit, bas_gauche = coins
    gauche, droite = bas_gauche - haut_gauche, bas_droit - haut_droit      # côtés verticaux, vers le bas
    haut, bas = haut_droit - haut_gauche, bas_droit - bas_gauche          # côtés horizontaux, vers la droite
    return np.array([haut_gauche - marge_y * gauche - marge_x * haut,
                     haut_droit - marge_y * droite + marge_x * haut,
                     bas_droit + marge_y * droite + marge_x * bas,
                     bas_gauche + marge_y * gauche - marge_x * bas], dtype=np.float32)


def redresser(decoupe, coins, marge_x=MARGE_REDRESSEMENT_X, marge_y=MARGE_REDRESSEMENT_Y):
    """Homographie : les quatre coins, élargis de la marge, vont sur les coins d'un rectangle de 450 x 100 px.
    Hors de la découpe, on complète en noir, la couleur du fond de la plaque."""
    destination = np.array([[0, 0], [LARGEUR_PLAQUE - 1, 0], [LARGEUR_PLAQUE - 1, HAUTEUR_PLAQUE - 1],
                            [0, HAUTEUR_PLAQUE - 1]], dtype=np.float32)
    matrice = cv2.getPerspectiveTransform(elargir(coins, marge_x, marge_y), destination)
    return cv2.warpPerspective(decoupe, matrice, (LARGEUR_PLAQUE, HAUTEUR_PLAQUE), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
