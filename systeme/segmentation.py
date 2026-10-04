"""Étape 4 du pipeline : segmentation des caractères de la plaque binarisée (450 x 100).

  1. Composantes connexes (8-connexité) : chaque blob blanc, avec son cadre et son aire. Les blobs qui
     touchent le bord de l'image sont écartés (restes du liseré ou du cadre).
  2. Candidats (filtres larges) : tout blob de taille plausible pour un caractère, y compris les fragments
     du mot arabe et les vis. Ce sont les entrées du CNN, qui les classera en chiffres ou « autre ».
  3. Chiffres probables (filtres stricts : hauteur, rapport largeur/hauteur, position verticale) : servent
     à l'évaluation de la segmentation et à l'étiquetage automatique des caractères (phase 4).
  4. Tri de gauche à droite ; séparation série / numéro au plus grand écart entre chiffres probables
     (c'est là que se trouve le mot arabe).
  5. Découpes normalisées 32 x 32 : réduites SANS déformation (le plus grand côté passe à 28 px), puis
     centrées. Étirer un « 1 » en carré le ferait ressembler à un autre chiffre.
Les bornes des filtres sont exprimées en fraction de la hauteur de la plaque (100 px après redressement).

Compatible Python 3.6, NumPy 1.19, OpenCV 4.1.1 (PC et Jetson).
"""

import cv2
import numpy as np

# Candidats : filtres larges
CANDIDAT_HAUTEUR_MIN, CANDIDAT_HAUTEUR_MAX = 0.20, 0.95   # en fraction de la hauteur de la plaque
CANDIDAT_LARGEUR_MAX = 0.40                               # en fraction de la largeur de la plaque
CANDIDAT_AIRE_MIN = 30                                    # pixels
# Chiffres probables : filtres stricts. Balayage sur 122 plaques de réglage : une hauteur minimale de 0,45
# écartait de vrais chiffres quand le quadrilatère est plus grand que la plaque (46,7 %) ; de 0,25 à 0,40 :
# 49,2 %. On retient 0,30, environ la moitié de la hauteur typique d'un chiffre (0,64). Le maximum n'a pas
# d'influence entre 0,80 et 0,95. (Attention au biais de sélection : mesurer la hauteur des chiffres sur les
# seules plaques réussies ne voit jamais les chiffres plus petits que le filtre.)
CHIFFRE_HAUTEUR_MIN, CHIFFRE_HAUTEUR_MAX = 0.30, 0.90     # en fraction de la hauteur de la plaque
CHIFFRE_RAPPORT_MIN, CHIFFRE_RAPPORT_MAX = 0.18, 0.90     # largeur / hauteur (le « 1 » est très fin)
CHIFFRE_CENTRE_MIN, CHIFFRE_CENTRE_MAX = 0.25, 0.75       # centre vertical, en fraction de la hauteur
CHIFFRE_MARGE_BORD = 0.01                                 # à moins de 1 % des bords gauche/droit : barre du cadre, vis
# Régularité : sur une plaque, tous les chiffres ont la même hauteur et sont alignés. Cela écarte les
# fragments du mot arabe (moins hauts) et les restes du cadre (plus hauts).
TOLERANCE_HAUTEUR = 0.15      # écart toléré à la hauteur médiane ; optimum mesuré (10 % : 46,7 % ; 15 % : 49,2 % ;
                              # 20 % : 37,7 % ; 25 % : 32,8 %) : trop strict, on perd des chiffres ; trop large,
                              # les fragments arabes passent
TOLERANCE_ALIGNEMENT = 0.20                               # écart toléré au centre médian, en fraction de la hauteur
# Découpage des blobs larges (caractères soudés) par projection verticale. Mesuré sur 122 plaques : sans
# découpage, 49,2 % (stricte) et 56,6 % (aucun chiffre perdu) ; avec, 50,8 % et 62,3 %.
SCISSION = True
SCISSION_PARTS_MAX = 2                                    # au plus 2 caractères soudés (3 : pas mieux)
SCISSION_FENETRE = 0.35                                   # creux cherché à +/- 35 % d'une largeur de chiffre
TAILLE_VIGNETTE, COTE_UTILE = 32, 28                      # découpe 32 x 32, caractère inscrit dans 28 px
# Chiffres soudés au liseré (itération ciblée, option « rangee » de l'étape 3, mesurée : voir docs/journal.md)
DETACHER = False
RANGEE_DEFAUT = (0.18, 0.82)    # rangée d'une plaque bien redressée (chiffres de 0,64 h, centrés), si < 3 chiffres
MARGE_RANGEE = 0.10             # marge au-dessus et au-dessous de la rangée, en fraction de sa hauteur
BANDE_COTES = 0.05              # bandes de côté (fraction de la largeur), où passent les traits verticaux du liseré


def composantes(binaire):
    """Blobs blancs de l'image : liste de dictionnaires x, y, l, h, aire (sans le fond, étiquette 0)."""
    nb, _, stats, _ = cv2.connectedComponentsWithStats(binaire, connectivity=8)
    return [{"x": int(stats[i, cv2.CC_STAT_LEFT]), "y": int(stats[i, cv2.CC_STAT_TOP]),
             "l": int(stats[i, cv2.CC_STAT_WIDTH]), "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
             "aire": int(stats[i, cv2.CC_STAT_AREA])} for i in range(1, nb)]


def est_candidat(blob, hauteur, largeur):
    touche_bord = (blob["x"] == 0 or blob["y"] == 0 or blob["x"] + blob["l"] >= largeur
                   or blob["y"] + blob["h"] >= hauteur)
    return (not touche_bord
            and CANDIDAT_HAUTEUR_MIN * hauteur <= blob["h"] <= CANDIDAT_HAUTEUR_MAX * hauteur
            and blob["l"] <= CANDIDAT_LARGEUR_MAX * largeur
            and blob["aire"] >= CANDIDAT_AIRE_MIN)


def est_chiffre_probable(blob, hauteur, largeur):
    centre = blob["y"] + blob["h"] / 2
    return (CHIFFRE_HAUTEUR_MIN * hauteur <= blob["h"] <= CHIFFRE_HAUTEUR_MAX * hauteur
            and CHIFFRE_RAPPORT_MIN <= blob["l"] / blob["h"] <= CHIFFRE_RAPPORT_MAX
            and CHIFFRE_CENTRE_MIN * hauteur <= centre <= CHIFFRE_CENTRE_MAX * hauteur
            and blob["x"] > CHIFFRE_MARGE_BORD * largeur
            and blob["x"] + blob["l"] < (1 - CHIFFRE_MARGE_BORD) * largeur)


def filtrer_regularite(blobs, hauteur):
    """Régularité : sur une plaque, tous les chiffres ont la même hauteur et sont alignés. Écarte les blobs dont
    la hauteur ou le centre vertical s'éloigne de la médiane (fragments du mot arabe, restes du cadre).
    Sans effet sous 3 blobs : la médiane n'y serait pas fiable. hauteur : hauteur de la plaque redressée."""
    if len(blobs) < 3:
        return list(blobs)
    hauteur_mediane = float(np.median([b["h"] for b in blobs]))
    centre_median = float(np.median([b["y"] + b["h"] / 2 for b in blobs]))
    return [b for b in blobs
            if abs(b["h"] - hauteur_mediane) <= TOLERANCE_HAUTEUR * hauteur_mediane
            and abs(b["y"] + b["h"] / 2 - centre_median) <= TOLERANCE_ALIGNEMENT * hauteur]


def choisir_chiffres(candidats, hauteur, largeur):
    """Chiffres probables : filtres stricts, puis régularité (même hauteur, centres alignés)."""
    return filtrer_regularite([b for b in candidats if est_chiffre_probable(b, hauteur, largeur)], hauteur)


def pente_rangee(candidats, hauteur, largeur):
    """Pente (dy/dx) de la droite des centres des candidats de la taille d'un chiffre (filtres stricts, avant
    la régularité) ; None s'il y en a moins de 3. Sur une plaque bien redressée, la rangée est horizontale."""
    stricts = [b for b in candidats if est_chiffre_probable(b, hauteur, largeur)]
    if len(stricts) < 3:
        return None
    x = np.array([b["x"] + b["l"] / 2.0 for b in stricts], dtype=np.float32)
    y = np.array([b["y"] + b["h"] / 2.0 for b in stricts], dtype=np.float32)
    return float(np.polyfit(x, y, 1)[0])


def scinder_blobs_larges(binaire, candidats, chiffres):
    """Caractères soudés : un blob qui a la hauteur d'un chiffre mais la largeur de deux ou trois chiffres est
    coupé aux colonnes les moins remplies (projection verticale : nombre de pixels blancs par colonne), là où
    les caractères ne se touchent que par un pont étroit. Renvoie la nouvelle liste de candidats."""
    if len(chiffres) < 2:
        return candidats
    hauteur_mediane = float(np.median([b["h"] for b in chiffres]))
    largeur_mediane = float(np.median([b["l"] for b in chiffres]))
    resultat = []
    for blob in candidats:
        nb_parts = int(round(blob["l"] / largeur_mediane))
        if (any(blob is c for c in chiffres) or nb_parts < 2 or nb_parts > SCISSION_PARTS_MAX
                or abs(blob["h"] - hauteur_mediane) > TOLERANCE_HAUTEUR * hauteur_mediane):
            resultat.append(blob)
            continue
        zone = binaire[blob["y"]:blob["y"] + blob["h"], blob["x"]:blob["x"] + blob["l"]]
        projection = (zone > 0).sum(axis=0)
        coupures = [0]
        for k in range(1, nb_parts):
            attendue = int(k * blob["l"] / nb_parts)                  # coupure attendue si les caractères sont égaux
            fenetre = max(1, int(SCISSION_FENETRE * largeur_mediane))  # on cherche le creux autour d'elle
            debut, fin = max(coupures[-1] + 1, attendue - fenetre), min(blob["l"] - 1, attendue + fenetre)
            coupures.append(debut + int(np.argmin(projection[debut:fin])) if fin > debut else attendue)
        coupures.append(blob["l"])
        for gauche, droite in zip(coupures[:-1], coupures[1:]):
            morceau = zone[:, gauche:droite]
            lignes = np.nonzero(morceau.any(axis=1))[0]
            if len(lignes) == 0:
                continue
            resultat.append({"x": blob["x"] + gauche, "y": blob["y"] + int(lignes[0]), "l": droite - gauche,
                             "h": int(lignes[-1] - lignes[0] + 1), "aire": int((morceau > 0).sum())})
    return sorted(resultat, key=lambda b: b["x"])


def separer_au_plus_grand_ecart(blobs):
    """Coupe une liste de blobs triés de gauche à droite au plus grand écart horizontal entre deux voisins.
    Renvoie (groupe de gauche, groupe de droite)."""
    if len(blobs) < 2:
        return list(blobs), []
    ecarts = [blobs[k + 1]["x"] - (blobs[k]["x"] + blobs[k]["l"]) for k in range(len(blobs) - 1)]
    coupure = int(np.argmax(ecarts)) + 1
    return list(blobs[:coupure]), list(blobs[coupure:])


def vignette(image, blob, fond):
    """Découpe normalisée 32 x 32 d'un blob : proportions conservées, centrée sur un fond uniforme."""
    marge = 2
    morceau = image[max(0, blob["y"] - marge):blob["y"] + blob["h"] + marge,
                    max(0, blob["x"] - marge):blob["x"] + blob["l"] + marge]
    rapport = COTE_UTILE / max(morceau.shape[:2])
    reduit = cv2.resize(morceau, (max(1, int(round(morceau.shape[1] * rapport))),
                                  max(1, int(round(morceau.shape[0] * rapport)))), interpolation=cv2.INTER_AREA)
    carre = np.full((TAILLE_VIGNETTE, TAILLE_VIGNETTE), fond, dtype=np.uint8)
    y0 = (TAILLE_VIGNETTE - reduit.shape[0]) // 2
    x0 = (TAILLE_VIGNETTE - reduit.shape[1]) // 2
    carre[y0:y0 + reduit.shape[0], x0:x0 + reduit.shape[1]] = reduit
    return carre


def decouper_vignettes(binaire, gris, blobs):
    """Découpes 32 x 32 des blobs, en gris et en binaire (deux listes, dans l'ordre des blobs).
    Seule définition de la découpe : le système et la préparation du jeu de caractères (phase 4) l'utilisent,
    ce qui garantit que le CNN voit en service exactement le même type de vignettes qu'à l'entraînement."""
    fond_gris = int(np.percentile(gris, 20))  # niveau du fond de la plaque, pour compléter les vignettes grises
    return [vignette(gris, b, fond_gris) for b in blobs], [vignette(binaire, b, 0) for b in blobs]


def rangee_des_chiffres(chiffres, hauteur):
    """(haut, bas) de la rangée des chiffres, marge comprise : médianes des chiffres probables s'il y en a au
    moins 3, sinon la rangée d'une plaque bien redressée."""
    if len(chiffres) >= 3:
        haut = float(np.median([b["y"] for b in chiffres]))
        bas = float(np.median([b["y"] + b["h"] for b in chiffres]))
    else:
        haut, bas = RANGEE_DEFAUT[0] * hauteur, RANGEE_DEFAUT[1] * hauteur
    marge = MARGE_RANGEE * (bas - haut)
    return int(max(0, haut - marge)), int(min(hauteur, bas + marge + 1))


def detacher_du_lisere(binaire, chiffres):
    """Copie de l'image binaire où les traits du liseré soudés aux chiffres sont effacés (None si rien n'a
    changé). Principe : un trait du liseré dépasse la rangée des chiffres, un chiffre jamais. Pour chaque grosse
    composante qui n'est pas un candidat et traverse la rangée :
      - dans les bandes de côté, les colonnes blanches de part en part de la rangée qui la débordent en haut ET en
        bas (trait vertical du liseré) sont effacées ;
      - ce qui dépasse la rangée en haut ou en bas (trait horizontal du liseré) est effacé."""
    hauteur, largeur = binaire.shape
    haut, bas = rangee_des_chiffres(chiffres, hauteur)
    nb, etiquettes, stats, _ = cv2.connectedComponentsWithStats(binaire, connectivity=8)
    cotes = np.zeros(largeur, dtype=bool)
    cotes[:int(BANDE_COTES * largeur)] = True
    cotes[largeur - int(BANDE_COTES * largeur):] = True
    nettoye, change = binaire.copy(), False
    for i in range(1, nb):
        blob = {"x": int(stats[i, cv2.CC_STAT_LEFT]), "y": int(stats[i, cv2.CC_STAT_TOP]),
                "l": int(stats[i, cv2.CC_STAT_WIDTH]), "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
                "aire": int(stats[i, cv2.CC_STAT_AREA])}
        traverse = min(blob["y"] + blob["h"], bas) - max(blob["y"], haut) >= 0.5 * (bas - haut)
        if est_candidat(blob, hauteur, largeur) or blob["aire"] < CANDIDAT_AIRE_MIN or not traverse:
            continue
        masque = etiquettes == i
        verticales = cotes & masque[haut:bas].all(axis=0) & masque[:haut].any(axis=0) & masque[bas:].any(axis=0)
        efface = masque.copy()
        efface[haut:bas] &= verticales[np.newaxis, :]   # dans la rangée, seules les colonnes du liseré s'effacent
        if efface.any():
            nettoye[efface] = 0
            change = True
    return nettoye if change else None


def segmenter_binaire(binaire):
    """Candidats et chiffres probables d'une image binaire (sans les découpes)."""
    hauteur, largeur = binaire.shape
    candidats = sorted((b for b in composantes(binaire) if est_candidat(b, hauteur, largeur)), key=lambda b: b["x"])
    chiffres = choisir_chiffres(candidats, hauteur, largeur)
    if SCISSION:
        candidats = scinder_blobs_larges(binaire, candidats, chiffres)
        chiffres = choisir_chiffres(candidats, hauteur, largeur)
    for blob in candidats:
        blob["chiffre"] = any(blob is c for c in chiffres)
    return candidats, chiffres


def segmenter(binaire, gris, debug=None, detacher=None):
    """Segmentation d'une plaque binarisée. Renvoie un dictionnaire :
      candidats : blobs candidats triés de gauche à droite (chacun avec « chiffre » : True/False) ;
      serie, numero : chiffres probables à gauche et à droite du plus grand écart ;
      vignettes_grises, vignettes_binaires : découpes 32 x 32 des candidats, dans le même ordre ;
      binaire : l'image binaire utilisée (celle d'entrée, ou sa copie détachée du liseré).
    detacher : seconde segmentation après detacher_du_lisere, gardée si elle donne plus de chiffres probables
    (par défaut DETACHER)."""
    detacher = DETACHER if detacher is None else detacher
    candidats, chiffres = segmenter_binaire(binaire)
    if detacher:
        nettoye = detacher_du_lisere(binaire, chiffres)
        if nettoye is not None:
            candidats_2, chiffres_2 = segmenter_binaire(nettoye)
            if len(chiffres_2) > len(chiffres):
                binaire, candidats, chiffres = nettoye, candidats_2, chiffres_2
    serie, numero = separer_au_plus_grand_ecart(chiffres)
    vignettes_grises, vignettes_binaires = decouper_vignettes(binaire, gris, candidats)
    resultat = {
        "candidats": candidats, "serie": serie, "numero": numero, "binaire": binaire,
        "vignettes_grises": vignettes_grises, "vignettes_binaires": vignettes_binaires,
    }
    if debug is not None:
        debug.ajouter("segmentation", dessiner_segmentation(binaire, resultat))
        if candidats:
            debug.ajouter("vignettes_32x32", np.hstack(resultat["vignettes_grises"]))
    return resultat


def dessiner_segmentation(binaire, resultat):
    """Candidats en orange, chiffres probables de la série en vert et du numéro en cyan."""
    vue = cv2.cvtColor(binaire, cv2.COLOR_GRAY2BGR)
    serie = [id(b) for b in resultat["serie"]]
    numero = [id(b) for b in resultat["numero"]]
    for blob in resultat["candidats"]:
        couleur = (0, 255, 0) if id(blob) in serie else (255, 255, 0) if id(blob) in numero else (0, 140, 255)
        cv2.rectangle(vue, (blob["x"], blob["y"]), (blob["x"] + blob["l"] - 1, blob["y"] + blob["h"] - 1), couleur, 2)
    return vue
