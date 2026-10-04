"""Mesures de la phase 5, sans image : listes simulées, pistes visées, rejeu de la décision, indicateurs et règle
de choix (PC et Jetson).

La perception (détection, traitement et lecture de chaque plaque, à 10 images/s) est faite une seule fois par
outils/evaluer.py et enregistrée. Ici, on ne fait que rejouer et compter. Le rejeu passe par le code du système
lui-même (ControleAcces de systeme/chaine.py : suivi et vote) : ce qui est mesuré est exactement ce qui tourne.
Définitions et règle de choix, fixées avant toute mesure : docs/journal.md, phase 5 (« Évaluation »).

Enregistrement d'une vidéo (produit par outils/evaluer.py ; une photo a la même forme, avec une seule image) :
    {"numero": "215 TU 4567", "fichier": "215TU4567.mp4", "largeur": 1280, "hauteur": 720,
     "images": [{"temps": 0.0, "detections": [[x1, y1, x2, y2, score], ...],
                 "lectures": [{"statut": "lue", "lecture": "215 TU 4567", "confiance": 0.98}, ...]}, ...]}
Une lecture par détection, dans le même ordre.

Compatible Python 3.6, NumPy 1.19 (PC et Jetson).
"""

import itertools
import math
import random

import numpy as np

from systeme.chaine import ControleAcces
from systeme.decision import format_valide
from systeme.suivi import Suivi
from systeme.vote import VARIANTES, Vote

# Grille du balayage (arrêt B) : 2 variantes x 4 seuils x 5 K x 3 W x 3 D = 360 réglages.
# Un réglage s'écrit (variante, seuil du détecteur, K, W, D).
SEUILS = (0.25, 0.35, 0.5, 0.65)
VOIX = (1, 2, 3, 4, 5)
FENETRES = (5, 10, 20)
ATTENTES = (1.0, 2.0, 3.0)
# Pistes visées (règle géométrique, indépendante des lectures)
FIN_VIDEO = 2.0        # secondes : la plus grande détection de cette fin de vidéo donne la plaque visée
ECART_VISEE = 0.15     # écart horizontal toléré à la plaque visée, en fraction de la largeur de l'image
CHIFFRES = "0123456789"
GRAINE_LISTES = 2026
# Règle de choix
ECART_VEHICULES = 1             # « à un véhicule près » (avec ~30 vidéos, un véhicule pèse 3,3 points)
PART_DECISIONS_RAPIDES = 0.90   # au moins 90 % des décisions...
DELAI_RAPIDE = 3.0              # ... prises en moins de 3 s
# Alerte « plaque sans voix » (systeme/vote.py) : (départ, X). Désactivée tant que X n'est pas mesuré.
SANS_ALERTE = (None, None)
CENTILE_PORTEE, CENTILE_PORTEE_REPLI = 90, 95   # X : décision de l'étudiant (95 % si 90 % perd une ouverture)
# Issues d'une lecture d'image : mêmes définitions qu'en phase 4 (entrainement/evaluer_lecture.py)
ISSUES = ("correcte", "rejetée", "substitution", "plus courte", "plus longue")


# --- Rejeu ---------------------------------------------------------------------------------------------------

def detections_au_seuil(image, seuil):
    """Détections d'une image enregistrée qui dépassent le seuil (comme le détecteur : score > seuil), avec leurs
    lectures. Enregistrées au seuil 0,25 ; un seuil plus haut donne le même résultat que s'il avait été appliqué
    au détecteur, car la NMS n'efface une boîte qu'au profit d'une boîte plus sûre."""
    gardees = [i for i, d in enumerate(image["detections"]) if d[4] > seuil]
    detections = np.array([image["detections"][i] for i in gardees], dtype=np.float32).reshape(-1, 5)
    return detections, [image["lectures"][i] for i in gardees]


def pistes_suivies(video, seuil, pas=1):
    """Le suivi du système seul (sans vote) : {id de piste: [(temps, cadre, lecture), ...]}.
    pas : 1 = toutes les images enregistrées (10 par seconde), 2 = une sur deux (5 par seconde).
    Le suivi ne dépend que des détections : les identifiants sont les mêmes que dans rejouer()."""
    suivi, pistes = Suivi(), {}
    for image in video["images"][::pas]:
        detections, lectures = detections_au_seuil(image, seuil)
        associations, _ = suivi.mettre_a_jour(image["temps"], detections)
        for piste, j in associations:
            pistes.setdefault(piste.id, []).append((image["temps"], detections[j], lectures[j]))
    return pistes


def classer_pistes(pistes, video):
    """Pistes visées (True) ou autres plaques (False), et position horizontale de la plaque visée.
    La plus grande détection des 2 dernières secondes de la vidéo donne la plaque visée (l'étudiant est arrêté
    devant elle) ; à défaut, le centre de l'image. Une piste est visée si sa position horizontale médiane en est
    à moins de 15 % de la largeur de l'image : les plaques voisines sont à environ 20 % à 10 m, puis plus loin."""
    fin = video["images"][-1]["temps"] if video["images"] else 0.0
    recentes = [c for detections in pistes.values() for t, c, _ in detections if t >= fin - FIN_VIDEO]
    x_visee = video["largeur"] / 2.0
    if recentes:
        plus_grande = max(recentes, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
        x_visee = float(plus_grande[0] + plus_grande[2]) / 2
    visees = {}
    for numero_piste, detections in pistes.items():
        x_median = np.median([(c[0] + c[2]) / 2.0 for _, c, _ in detections])
        visees[numero_piste] = bool(abs(x_median - x_visee) <= ECART_VISEE * video["largeur"])
    return visees, x_visee


def rejouer(video, vote, seuil, pas=1):
    """Rejoue le suivi et le vote du système sur une vidéo enregistrée, comme systeme/chaine.py en service :
    seules les pistes sans décision sont lues. Renvoie la liste des décisions (dictionnaires de systeme/vote.py)."""
    controle = ControleAcces(vote)
    decisions, temps = [], 0.0
    for image in video["images"][::pas]:
        temps = image["temps"]
        detections, lectures = detections_au_seuil(image, seuil)
        for piste, j in controle.associer(temps, detections, video["largeur"]):
            if controle.voter(temps, piste, lectures[j]) is not None:
                decisions.append(piste.decision)
        decisions.extend(p.decision for p in controle.finir_image(temps))
    decisions.extend(p.decision for p in controle.finir(temps))
    return decisions


# --- Listes simulées -----------------------------------------------------------------------------------------

def variantes_adverses(numero):
    """Toutes les lectures à une modification près d'un numéro « 215 TU 4567 » : un chiffre supprimé, remplacé par
    un autre ou ajouté, dans la série ou dans le numéro. Seules les variantes au format officiel sont gardées
    (série de 1 à 3 chiffres, numéro de 1 à 4, pas de zéro en tête). Aucune hypothèse sur les confusions du CNN :
    c'est le pire cas pour une erreur d'un seul chiffre (décision de l'étudiant, journal)."""
    groupes = numero.split(" TU ")
    variantes = set()
    for rang, texte in enumerate(groupes):
        modifies = set()
        for i in range(len(texte) + 1):
            for chiffre in CHIFFRES:
                modifies.add(texte[:i] + chiffre + texte[i:])              # chiffre ajouté
            if i < len(texte):
                modifies.add(texte[:i] + texte[i + 1:])                    # chiffre supprimé
                for chiffre in CHIFFRES:
                    modifies.add(texte[:i] + chiffre + texte[i + 1:])      # chiffre remplacé
        for modifie in modifies:
            nouveaux = list(groupes)
            nouveaux[rang] = modifie
            if format_valide(nouveaux[0], nouveaux[1]):
                variantes.add("{} TU {}".format(nouveaux[0], nouveaux[1]))
    return variantes - {numero}


def listes_croisees(numeros, graine=GRAINE_LISTES):
    """Les deux listes simulées : L1 = une moitié des véhicules tirée au sort, L2 = l'autre moitié. Chaque liste
    = ses autorisés + les variantes adverses des non autorisés, moins les vrais numéros des non autorisés (sinon,
    une lecture juste compterait comme une ouverture à tort). Renvoie [(nom, ensemble des numéros de la liste)]."""
    ordre = sorted(numeros)
    random.Random(graine).shuffle(ordre)
    moities = (set(ordre[:len(ordre) // 2]), set(ordre[len(ordre) // 2:]))
    listes = []
    for rang, autorises in enumerate(moities):
        non_autorises = set(numeros) - autorises
        adverses = set()
        for numero in non_autorises:
            adverses |= variantes_adverses(numero)
        listes.append(("L{}".format(rang + 1), autorises | (adverses - non_autorises)))
    return listes


# --- Indicateurs ---------------------------------------------------------------------------------------------

def issue_passage(decisions, visees, numero):
    """Issue d'un passage (une vidéo, une liste), d'après les décisions des pistes visées :
    « ouverture à tort » (sur un autre numéro que le vrai), sinon « ouverture », sinon « refus », sinon « non lu »."""
    des_visees = [d for d in decisions if visees.get(d["piste"], False)]
    ouvertures = [d["numero"] for d in des_visees if d["decision"] == "ouverture"]
    if any(n != numero for n in ouvertures):
        return "ouverture à tort"
    if ouvertures:
        return "ouverture"
    if any(d["decision"] == "refus" for d in des_visees):
        return "refus"
    return "non lu"


def fiabilite_images(videos, suivis):
    """Lectures acceptées justes / lectures acceptées, sur toutes les images des pistes visées.
    suivis : pour chaque vidéo, (pistes, visees). Renvoie (fiabilité, nombre de lectures acceptées)."""
    acceptees = justes = 0
    for video, (pistes, visees) in zip(videos, suivis):
        for numero_piste, detections in pistes.items():
            if not visees[numero_piste]:
                continue
            for _, _, lecture in detections:
                if lecture["statut"] == "lue":
                    acceptees += 1
                    justes += lecture["lecture"] == video["numero"]
    return (justes / acceptees if acceptees else 0.0), acceptees


def mesurer_reglage(videos, suivis, listes, reglage, fiabilite_des_images, pas=1, alerte=SANS_ALERTE):
    """Indicateurs d'un réglage (variante, seuil, K, W, D) sur toutes les vidéos et les deux listes.
    alerte : (départ, X) de l'alerte « plaque sans voix », toujours donnée explicitement : les mesures n'emploient
    jamais un X non mesuré (systeme/vote.py, PORTEE_MESUREE). Renvoie (indicateurs, passages)."""
    variante, seuil, voix, fenetre, attente = reglage
    depart_alerte, portee = alerte
    passages = []
    for nom_liste, autorises in listes:
        vote = Vote(autorises, variante, voix, fenetre, attente, depart_alerte=depart_alerte, portee=portee)
        for video, (_, visees) in zip(videos, suivis):
            decisions = rejouer(video, vote, seuil, pas)
            passages.append({"video": video, "liste": nom_liste, "autorise": video["numero"] in autorises,
                             "decisions": decisions, "visees": visees,
                             "issue": issue_passage(decisions, visees, video["numero"])})
    return indicateurs(passages, fiabilite_des_images), passages


def indicateurs(passages, fiabilite_des_images):
    """Indicateurs d'un ensemble de passages (définitions du journal)."""
    autorises = [p for p in passages if p["autorise"]]
    tranchees = justes = autres_ouvertures = 0
    delais, delais_ouvertures = [], []
    for passage in passages:
        vrai = passage["video"]["numero"]
        for d in passage["decisions"]:
            if not passage["visees"].get(d["piste"], False):
                autres_ouvertures += d["decision"] == "ouverture"
                continue
            if d["decision"] in ("ouverture", "refus"):
                tranchees += 1
                justes += d["numero"] == vrai
            if d["delai_s"] is not None:
                delais.append(d["delai_s"])
                if d["decision"] == "ouverture" and d["numero"] == vrai:
                    delais_ouvertures.append(d["delai_s"])
    nombre = max(len(autorises), 1)
    nb_ouvertures = sum(p["issue"] == "ouverture" for p in autorises)
    return {
        "nb_ouvertures": nb_ouvertures,
        "ouvertures": nb_ouvertures / nombre,
        "refus": sum(p["issue"] == "refus" for p in autorises) / nombre,
        "non_lus": sum(p["issue"] == "non lu" for p in autorises) / nombre,
        "a_tort": sum(p["issue"] == "ouverture à tort" for p in passages),
        "fiabilite_decisions": justes / tranchees if tranchees else 0.0,
        "fiabilite_images": fiabilite_des_images,
        "rapides": sum(x < DELAI_RAPIDE - 1e-6 for x in delais) / len(delais) if delais else 0.0,
        "delai_median": float(np.median(delais_ouvertures)) if delais_ouvertures else float("nan"),
        "delai_p90": float(np.percentile(delais_ouvertures, 90)) if delais_ouvertures else float("nan"),
        "autres_ouvertures": autres_ouvertures,
        "passages_autorises": len(autorises),
    }


def grille():
    """Les 360 réglages, dans un ordre fixe (utilisé pour départager les dernières égalités)."""
    return list(itertools.product(VARIANTES, SEUILS, VOIX, FENETRES, ATTENTES))


def balayer(videos, listes, pas=1, suivre=None, alerte=SANS_ALERTE):
    """Indicateurs des 360 réglages : {reglage: indicateurs}. Le suivi ne dépend que du seuil : pistes, classement
    et fiabilité des lectures image par image sont calculés une fois par seuil. suivre : fonction appelée après
    chaque seuil (progression). alerte : comme pour mesurer_reglage."""
    resultats = {}
    for seuil in SEUILS:
        suivis = []
        for video in videos:
            pistes = pistes_suivies(video, seuil, pas)
            suivis.append((pistes, classer_pistes(pistes, video)[0]))
        fiabilite_des_images = fiabilite_images(videos, suivis)[0]
        for reglage in grille():
            if reglage[1] == seuil:
                resultats[reglage] = mesurer_reglage(videos, suivis, listes, reglage, fiabilite_des_images, pas,
                                                     alerte)[0]
        if suivre is not None:
            suivre(seuil)
    return resultats


# --- Alerte « plaque sans voix » (arrêt B) -------------------------------------------------------------------

def largeurs_lecture(videos, suivis):
    """Largeur relative (cadre / image) de la plaque visée : à sa première lecture juste, pour chaque vidéo où elle
    finit lue ; et la plus grande atteinte, pour chaque vidéo (la plaque arrêtée devant la barrière).
    Renvoie (largeurs à la première lecture juste, largeurs maximales)."""
    premieres, maximums = [], []
    for video, (pistes, visees) in zip(videos, suivis):
        detections = sorted((d for n, ds in pistes.items() if visees[n] for d in ds), key=lambda d: d[0])
        if not detections:
            continue
        largeurs = [float(c[2] - c[0]) / video["largeur"] for _, c, _ in detections]
        maximums.append(max(largeurs))
        justes = [l for (_, _, lecture), l in zip(detections, largeurs)
                  if lecture["statut"] == "lue" and lecture["lecture"] == video["numero"]]
        if justes:
            premieres.append(justes[0])
    return premieres, maximums


def effet_alerte(videos, suivis, listes, reglage, fiabilite_des_images, alerte, pas=1):
    """Effet d'une alerte (départ, X) sur un réglage, par rapport au même réglage sans alerte : les deux garde-fous
    du journal. perdues : passages autorisés ouverts correctement sans l'alerte, plus avec ; jamais_lus : passages
    dont aucune piste visée n'a eu de voix ; transformes : parmi eux, ceux que l'alerte fait finir en « non lu » ;
    delai_median : délai de ces « non lu », compté depuis la première détection de la piste."""
    sans = mesurer_reglage(videos, suivis, listes, reglage, fiabilite_des_images, pas, SANS_ALERTE)[1]
    avec = mesurer_reglage(videos, suivis, listes, reglage, fiabilite_des_images, pas, alerte)[1]
    perdues = sum(1 for a, b in zip(sans, avec)
                  if a["autorise"] and a["issue"] == "ouverture" and b["issue"] != "ouverture")
    jamais_lus, delais = 0, []
    for a, b in zip(sans, avec):
        if any(a["visees"].get(d["piste"], False) and d["voix"] > 0 for d in a["decisions"]):
            continue
        jamais_lus += 1
        alertes = [d for d in b["decisions"] if b["visees"].get(d["piste"], False) and d["decision"] == "non lu"
                   and d["depuis_alerte_s"] is not None]
        if alertes:
            delais.append(min(d["depuis_detection_s"] for d in alertes))
    return {"perdues": perdues, "jamais_lus": jamais_lus, "transformes": len(delais),
            "delai_median": float(np.median(delais)) if delais else float("nan")}


def choisir_alerte(taille, arret):
    """Règle de l'étudiant, fixée avant la mesure : la taille, sauf si la variante « arrêt » fait mieux sur les deux
    garde-fous à la fois, c'est-à-dire aucune ouverture correcte perdue, et plus de véhicules jamais lus transformés
    en « non lu » (ou autant, plus vite) ; à égalité, la taille, plus simple et plus robuste."""
    plus = arret["transformes"] > taille["transformes"]
    aussi_vite = arret["transformes"] == taille["transformes"] and arret["delai_median"] < taille["delai_median"]
    return "arret" if arret["perdues"] == 0 and (plus or aussi_vite) else "taille"


# --- Règle de choix ------------------------------------------------------------------------------------------

def voisins(reglage):
    """Réglages voisins dans la grille : un cran de plus ou de moins sur un seul paramètre (seuil, K, W ou D), dans
    la même variante."""
    resultat = []
    for rang, axe in enumerate((SEUILS, VOIX, FENETRES, ATTENTES), start=1):
        position = axe.index(reglage[rang])
        for decalage in (-1, 1):
            if 0 <= position + decalage < len(axe):
                voisin = list(reglage)
                voisin[rang] = axe[position + decalage]
                resultat.append(tuple(voisin))
    return resultat


def bons_reglages(resultats, admissibles, niveau):
    """Bons réglages pour un niveau d'ouvertures correctes (nombre de véhicules) : admissibles, à un véhicule près
    de ce niveau, et au moins 90 % des décisions prises en moins de 3 s. Renvoie (bons, nombre de bons voisins de
    chacun)."""
    bons = set(r for r in admissibles
               if resultats[r]["nb_ouvertures"] >= niveau - ECART_VEHICULES
               and resultats[r]["rapides"] >= PART_DECISIONS_RAPIDES)
    return bons, {r: sum(v in bons for v in voisins(r)) for r in bons}


def choisir(resultats):
    """Règle de choix du journal. Renvoie (réglage retenu ou None, étapes) ; étapes : niveau d'ouvertures retenu
    (nombre de véhicules), nombre de réglages qui passent chaque étape, nombre de bons voisins de chaque bon
    réglage."""
    admissibles = [r for r, m in resultats.items()
                   if m["a_tort"] == 0 and m["fiabilite_decisions"] >= m["fiabilite_images"]]
    etapes = {"reglages": len(resultats), "admissibles": len(admissibles), "niveau": None, "bons": 0,
              "non isoles": 0, "K le plus grand": 0, "bons_voisins": {}}
    # On part du meilleur nombre d'ouvertures correctes. Si aucun bon réglage n'y est entouré d'au moins la moitié
    # de bons voisins (un pic isolé, pas un plateau), on descend au nombre suivant.
    non_isoles = []
    for niveau in sorted(set(resultats[r]["nb_ouvertures"] for r in admissibles), reverse=True):
        bons, bons_voisins = bons_reglages(resultats, admissibles, niveau)
        non_isoles = [r for r in bons if 2 * bons_voisins[r] >= len(voisins(r))]
        if non_isoles:
            etapes.update({"niveau": niveau, "bons": len(bons), "non isoles": len(non_isoles),
                           "bons_voisins": bons_voisins})
            break
    if not non_isoles:
        return None, etapes
    k_max = max(r[2] for r in non_isoles)                     # le plus sûr
    candidats = [r for r in non_isoles if r[2] == k_max]
    etapes["K le plus grand"] = len(candidats)
    ordre = {r: i for i, r in enumerate(grille())}
    # Centre du plateau (le plus de bons voisins), puis le plus d'ouvertures, puis la majorité, puis l'ordre fixe
    retenu = min(candidats, key=lambda r: (-bons_voisins[r], -resultats[r]["nb_ouvertures"], r[0] != "majorite",
                                           ordre[r]))
    return retenu, etapes


# --- Lectures d'une image (photos) ---------------------------------------------------------------------------

def issue_lecture(lecture, verite):
    """Issue d'une lecture d'image (mêmes définitions qu'en phase 4) ; lecture None : plaque non détectée."""
    if lecture is None or lecture["statut"] != "lue":
        return "rejetée"
    if lecture["lecture"] == verite:
        return "correcte"
    lus, vrais = sum(c.isdigit() for c in lecture["lecture"]), sum(c.isdigit() for c in verite)
    return "substitution" if lus == vrais else ("plus courte" if lus < vrais else "plus longue")


def lecture_visee(image, seuil):
    """Lecture de la plaque visée d'une photo : la plus grande détection au-dessus du seuil (None : aucune)."""
    detections, lectures = detections_au_seuil(image, seuil)
    if not len(detections):
        return None
    aires = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
    return lectures[int(aires.argmax())]


def mcnemar(gagnes, perdus):
    """Test de McNemar exact (bilatéral) : probabilité d'un écart au moins aussi grand entre plaques gagnées et
    perdues si les deux variantes se valaient (chaque plaque discordante : pile ou face)."""
    n = gagnes + perdus
    if n == 0:
        return 1.0
    queue = sum(math.factorial(n) // (math.factorial(i) * math.factorial(n - i))
                for i in range(min(gagnes, perdus) + 1)) / 2.0 ** n
    return min(1.0, 2 * queue)
