"""Étape 6 du pipeline, pour une image : assemblage du numéro à partir des lectures du CNN (phase 4).

  1. Les candidats lus « autre » par le CNN sont écartés.
  2. Contrôle de régularité de la phase 3 (option, utile si le CNN a lu TOUS les candidats) : un fragment du
     mot arabe lu à tort comme un chiffre n'a pas la hauteur ni l'alignement des vrais chiffres.
  3. Tri de gauche à droite et coupure au plus grand écart : série à gauche, numéro à droite (le mot arabe,
     au milieu, crée le plus grand écart une fois ses fragments écartés).
  4. Contrôle du format officiel : série de 1 à 3 chiffres, numéro de 1 à 4, pas de zéro en tête.
  5. Contrôle de vraisemblance : série et numéro d'au moins 2 chiffres. Une lecture à laquelle la segmentation a
     fait perdre presque tous ses chiffres (« 1 TU 1 ») respecte le format officiel ; aucune des 547 plaques de
     réglage saisies n'a de groupe d'un seul chiffre, et un numéro d'un chiffre n'a qu'environ 0,1 % de chances
     (numéros de 1 à 9999). Mesuré hors pli (option A, seuil 0,6) : les lectures plus courtes tombent de 20,8 %
     à 9,5 % des plaques, sans perdre une seule lecture correcte.
  6. Seuil de confiance : la confiance de la plaque est celle de son chiffre le moins sûr (le maillon le plus
     faible) ; en dessous du seuil, la lecture est rejetée plutôt que de risquer une erreur.
Pour la phase 5 : une lecture « improbable » (groupe d'un chiffre) qui correspond EXACTEMENT à une plaque de la
liste des autorisés doit être acceptée, sinon une voiture autorisée au numéro d'un seul chiffre serait refusée à
chaque passage ; c'est pourquoi le texte assemblé est conservé dans « lecture », même rejeté.
Limite connue : un chiffre perdu par la segmentation donne une lecture plus courte de format vraisemblable
(« 215 TU 456 ») ; la phase 5 s'en protège par le vote sur plusieurs images et parce que la barrière ne s'ouvre
que sur une correspondance exacte avec la liste.

Compatible Python 3.6, NumPy 1.19 (PC et Jetson).
"""

from systeme.lecture import AUTRE, ENTREE
from systeme.segmentation import filtrer_regularite, separer_au_plus_grand_ecart

SERIE_MAX, NUMERO_MAX = 3, 4      # format officiel
SERIE_MIN, NUMERO_MIN = 2, 2      # vraisemblance (voir le point 5 ci-dessus)
# Seuil retenu par la règle fixée d'avance (le plus petit qui ramène les substitutions sous 1 % des plaques,
# mesuré hors pli sur 529 plaques de réglage) : 0,6.
SEUIL = 0.6


def format_valide(serie, numero):
    """Format officiel : série de 1 à 3 chiffres, numéro de 1 à 4, sans zéro en tête (aucune des 690 plaques
    saisies n'en a)."""
    return (1 <= len(serie) <= SERIE_MAX and 1 <= len(numero) <= NUMERO_MAX
            and serie[0] != "0" and numero[0] != "0")


def format_vraisemblable(serie, numero):
    return len(serie) >= SERIE_MIN and len(numero) >= NUMERO_MIN


def lire_plaque(candidats, classes, confiances, hauteur_plaque, seuil=SEUIL, regularite=False):
    """Assemble le numéro d'une plaque.
    candidats : blobs de la segmentation ; classes, confiances : classe prédite par le CNN pour chacun, et sa
    probabilité ; hauteur_plaque : hauteur de la plaque redressée (contrôle d'alignement) ; seuil : confiance
    minimale pour accepter la lecture. Renvoie un dictionnaire :
      texte     : « 215 TU 4567 » si la lecture est acceptée, sinon None ;
      statut    : « lue », « doute » (sous le seuil), « format » (hors format officiel), « improbable »
                  (groupe d'un chiffre) ou « aucun chiffre » ;
      lecture   : le texte assemblé, même rejeté (affichage, diagnostic, comparaison à la liste en phase 5) ;
      confiance : probabilité du chiffre le moins sûr ; chiffres : blobs retenus, de gauche à droite."""
    lecture_de = {id(b): (int(c), float(p)) for b, c, p in zip(candidats, classes, confiances) if int(c) != AUTRE}
    blobs = [b for b in candidats if id(b) in lecture_de]
    if regularite:
        blobs = filtrer_regularite(blobs, hauteur_plaque)
    if not blobs:
        return {"texte": None, "statut": "aucun chiffre", "lecture": "", "confiance": 0.0, "chiffres": []}
    blobs = sorted(blobs, key=lambda b: b["x"])
    gauche, droite = separer_au_plus_grand_ecart(blobs)
    serie = "".join(str(lecture_de[id(b)][0]) for b in gauche)
    numero = "".join(str(lecture_de[id(b)][0]) for b in droite)
    confiance = min(lecture_de[id(b)][1] for b in blobs)
    lecture = serie + " TU " + numero
    if not format_valide(serie, numero):
        statut = "format"
    elif not format_vraisemblable(serie, numero):
        statut = "improbable"
    elif confiance < seuil:
        statut = "doute"
    else:
        statut = "lue"
    return {"texte": lecture if statut == "lue" else None, "statut": statut, "lecture": lecture,
            "confiance": confiance, "chiffres": blobs}


def lire_numero(resultat, lecteur, seuil=SEUIL):
    """Lecture d'une plaque traitée (résultat de systeme/traitement.py), telle que le système la fait.
    Option A, retenue par la mesure : le CNN ne lit que les chiffres probables de la phase 3 et en écarte les
    intrus (barres du cadre, fragments arabes). Renvoie le dictionnaire de lire_plaque, complété par
    « classes » et « confiances » (sorties du CNN pour chaque chiffre probable, dans l'ordre des candidats)."""
    probables = [b for b in resultat["candidats"] if b["chiffre"]]
    tous = resultat["vignettes_grises"] if ENTREE == "gris" else resultat["vignettes_binaires"]
    vignettes = [v for b, v in zip(resultat["candidats"], tous) if b["chiffre"]]
    classes, confiances = lecteur.lire(vignettes)
    decision = lire_plaque(probables, classes, confiances, resultat["binaire"].shape[0], seuil)
    decision.update({"classes": classes, "confiances": confiances, "probables": probables})
    return decision
