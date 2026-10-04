"""Traitement classique d'une plaque (phase 3) : redressement, binarisation, segmentation.

Aucun réseau de neurones ici : l'entrée est le cadre de la plaque dans l'image en pleine résolution
(détecté par YOLO dans le système, ou annoté pour l'évaluation).

Contrôle du redressement par la rangée de caractères : la cascade de systeme/redressement.py propose des coins
dans l'ordre ; chaque proposition est redressée, binarisée et segmentée, et l'on garde la PREMIÈRE dont la
rangée de chiffres est bonne ; si aucune ne l'est, celle qui donne le plus de chiffres probables (et non plus
simplement la première qui trouve des coins). Variantes du contrôle, comparées par la mesure :
  « aucun »     : la première proposition (comportement d'origine) ;
  « groupes »   : série de 2 à 3 chiffres probables et numéro de 2 à 4 ;
  « pente »     : groupes, et rangée horizontale (pente des candidats de la taille d'un chiffre) ;
  « geometrie » : pente, et quadrilatère qui couvre son cadre YOLO sans trop en déborder ;
  « complete »  : geometrie, et au moins 6 chiffres probables (98 % des plaques de réglage en ont 6 ou 7) :
                  une rangée plausible mais incomplète n'arrête plus la cascade, la proposition suivante peut
                  retrouver le chiffre perdu.

Compatible Python 3.6, NumPy 1.19, OpenCV 4.1.1 (PC et Jetson).
"""

import time

import numpy as np

from systeme.binarisation import binariser
from systeme.decision import NUMERO_MAX, NUMERO_MIN, SERIE_MAX, SERIE_MIN
from systeme.redressement import couverture_du_cadre, decouper, dessiner_coins, propositions_coins, redresser
from systeme.segmentation import pente_rangee, segmenter

CONTROLES = ("aucun", "groupes", "pente", "geometrie", "complete")
# Variante retenue par l'étudiant (réglage, hors pli, 527 plaques) : « complete », 65,5 % de lectures correctes
# contre 58,1 % pour la cascade d'origine, gain présent dans les 5 plis, fiabilité des lectures acceptées
# inchangée (84,8 % contre 85,2 %). Conçue après avoir vu les premières mesures : son gain n'a pas pu être confirmé
# sur des données vierges, la phase 5 s'étant terminée sans vidéos réelles ; conservée (décision de l'étudiant).
CONTROLE = "complete"
# Seuils du diagnostic : ce que les plaques bien lues ne dépassent presque jamais (1er ou 99e centile, réglage)
PENTE_MAX = 0.059
COUVERTURE_MIN, DEBORDEMENT_MAX = 0.83, 0.69
CHIFFRES_COMPLETE = 6        # variante « complete » : longueurs saisies au réglage, 3+4, 2+4 et 3+3 = 98 %


def rangee_bonne(segmentation, coins, cadre, controle):
    """La rangée de caractères obtenue avec ces coins est-elle celle d'une plaque bien redressée ?"""
    if controle == "aucun":
        return True
    bonne = (SERIE_MIN <= len(segmentation["serie"]) <= SERIE_MAX
             and NUMERO_MIN <= len(segmentation["numero"]) <= NUMERO_MAX)
    if bonne and controle in ("pente", "geometrie", "complete"):
        hauteur, largeur = segmentation["hauteur"], segmentation["largeur"]
        pente = pente_rangee(segmentation["candidats"], hauteur, largeur)
        bonne = pente is None or abs(pente) <= PENTE_MAX
    if bonne and controle in ("geometrie", "complete"):
        couverture, debordement = couverture_du_cadre(coins, cadre)
        bonne = couverture >= COUVERTURE_MIN and debordement <= DEBORDEMENT_MAX
    if bonne and controle == "complete":
        bonne = len(segmentation["serie"]) + len(segmentation["numero"]) >= CHIFFRES_COMPLETE
    return bonne


def traiter_plaque(image, boite, debug=None, reglages_binarisation=None, controle=None, detacher=None):
    """Image complète + cadre (x1, y1, x2, y2) -> dictionnaire du résultat de la segmentation, complété par :
    methode_coins (méthode retenue), essais (méthodes essayées, dans l'ordre), coins (dans l'image complète),
    plaque (redressée, couleur), gris, binaire (celle qu'a utilisée la segmentation), durees (secondes par étape,
    essais compris). reglages_binarisation : paramètres de binariser(), pour les comparaisons ; controle :
    variante du contrôle du redressement (par défaut CONTROLE) ; detacher : séparation des chiffres soudés au
    liseré (par défaut celle de systeme/segmentation.py)."""
    controle = controle or CONTROLE
    reglages = reglages_binarisation or {}
    durees = {"redressement": 0.0, "binarisation": 0.0, "segmentation": 0.0}
    debut = time.perf_counter()
    decoupe, origine, cadre = decouper(image, boite)
    essais, retenu, meilleur = [], None, None
    for nom, coins in propositions_coins(decoupe, cadre):
        plaque = redresser(decoupe, coins)
        apres_redressement = time.perf_counter()
        binaire, gris = binariser(plaque, **reglages)
        apres_binarisation = time.perf_counter()
        segmentation = segmenter(binaire, gris, detacher=detacher)
        fin = time.perf_counter()
        durees["redressement"] += apres_redressement - debut
        durees["binarisation"] += apres_binarisation - apres_redressement
        durees["segmentation"] += fin - apres_binarisation
        debut = fin
        essais.append((nom, coins))
        segmentation.update({"hauteur": binaire.shape[0], "largeur": binaire.shape[1]})
        essai = {"nom": nom, "coins": coins, "plaque": plaque, "binaire": binaire, "gris": gris,
                 "segmentation": segmentation}
        if rangee_bonne(segmentation, coins, cadre, controle):
            retenu = essai
            break
        nombre = len(segmentation["serie"]) + len(segmentation["numero"])
        if meilleur is None or nombre > meilleur[0]:   # à égalité, la première dans l'ordre de la cascade
            meilleur = (nombre, essai)
    if retenu is None:
        retenu = meilleur[1]
    resultat = retenu["segmentation"]
    if debug is not None:
        # Images intermédiaires de la proposition retenue seulement (les essais rejetés ne sont pas détaillés)
        debug.ajouter("decoupe", decoupe)
        debug.ajouter("coins", dessiner_coins(decoupe, cadre, essais, retenu["nom"]))
        debug.ajouter("redressee", retenu["plaque"])
        binaire, gris = binariser(retenu["plaque"], debug, **reglages)
        segmenter(binaire, gris, debug, detacher=detacher)
    resultat.update({
        "methode_coins": retenu["nom"], "essais": [nom for nom, _ in essais],
        "coins": retenu["coins"] + np.array(origine, dtype=np.float32),
        "plaque": retenu["plaque"], "gris": retenu["gris"], "durees": durees,
    })
    return resultat
