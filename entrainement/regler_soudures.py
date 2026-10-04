"""Phase 3, itération ciblée, étape 3 : séparer les chiffres soudés aux restes du liseré (réglage, hors pli) (PC).

Options comparées par-dessus le contrôle du redressement retenu (« complete ») :
  reference : sans séparation ;
  bords     : seconde passe d'effacement des lignes, plus courtes, dans les bandes du bord seulement
              (systeme/binarisation.py, lisere_bords) ;
  rangee    : détachement des traits du liseré qui dépassent la rangée des chiffres (systeme/segmentation.py,
              detacher), gardé seulement s'il donne plus de chiffres probables.
Mêmes mesures que regler_redressement.py : lectures correctes, rejetées, substitutions et plus courtes,
fiabilité des lectures acceptées, gains et pertes par rapport à la référence, gain dans chaque pli, temps.
Règle fixée avant la mesure (docs/journal.md) : l'option qui donne le plus de lectures correctes, à condition
que les substitutions restent sous 1 % et que la fiabilité des lectures acceptées ne baisse pas de plus de
1 point par rapport à la référence ; à 0,5 point près, la plus simple (bords, puis rangee).

Utilisation, depuis la racine du projet :
    python -m entrainement.regler_soudures gris_8
"""

import argparse
import csv

import numpy as np

from entrainement.evaluer_lecture import ISSUES, LecteurPlis
from entrainement.evaluer_traitement import MODELE_DETECTION, lire_plaques
from entrainement.extraire_caracteres import FICHIER_PLIS, NB_PLIS
from entrainement.regler_redressement import evaluer_variante, fiabilite, preparer
from systeme.detection import DetecteurPlaques

OPTIONS = (("reference", {}), ("bords", {"reglages_binarisation": {"lisere_bords": True}}),
           ("rangee", {"detacher": True}))
OBJECTIF_SUBSTITUTIONS = 0.01
BAISSE_FIABILITE_MAX = 0.01      # 1 point, environ l'incertitude de mesure (± 1,8 point sur 400 lectures acceptées)
ECART_SIMPLICITE = 0.005


def main():
    parseur = argparse.ArgumentParser(description="Chiffres soudés au liseré : comparaison des options.")
    parseur.add_argument("nom", help="essai de validation croisée dont on lit les modèles (ex. gris_8)")
    arguments = parseur.parse_args()
    with open(FICHIER_PLIS, encoding="utf-8", newline="") as fichier:
        plis = {ligne["image"]: int(ligne["pli"]) for ligne in csv.DictReader(fichier)}
    plaques = [p for p in lire_plaques("reglage")[0] if f"{p['chemin'].parent.name}/{p['image']}" in plis]
    lecteur = LecteurPlis(arguments.nom)
    entrees = preparer(plaques, plis, DetecteurPlaques(str(MODELE_DETECTION)))
    print(f"Réglage, hors pli ({arguments.nom}) : {len(entrees)} plaques, contrôle du redressement « complete »")

    resultats = {nom: evaluer_variante(entrees, lecteur, **reglages) for nom, reglages in OPTIONS}
    parts = {n: {i: r[0].count(i) / len(entrees) for i in ISSUES} for n, r in resultats.items()}
    fiabilites = {n: fiabilite(r[0]) for n, r in resultats.items()}
    reference = [i == "correcte" for i in resultats["reference"][0]]
    print("\noption      " + "  ".join(f"{i:>12s}" for i in ISSUES) + "   fiabilité  gains  pertes  traitement (ms)")
    for nom, (issues, _, durees) in resultats.items():
        justes = [i == "correcte" for i in issues]
        gains = sum(1 for a, b in zip(reference, justes) if b and not a)
        pertes = sum(1 for a, b in zip(reference, justes) if a and not b)
        print(f"{nom:10s}  " + "  ".join(f"{parts[nom][i] * 100:10.1f} %" for i in ISSUES)
              + f"   {fiabilites[nom] * 100:7.1f} %  {gains:5d}  {pertes:6d}  médiane {np.median(durees) * 1000:.1f}, "
              f"p95 {np.percentile(durees, 95) * 1000:.1f}")
    print("\nLectures correctes par pli (%) :")
    for nom, (issues, _, _) in resultats.items():
        print(f"   {nom:10s}: " + "  ".join(
            f"pli {k} {np.mean([i == 'correcte' for i, e in zip(issues, entrees) if e['pli'] == k]) * 100:5.1f}"
            for k in range(NB_PLIS)))

    admissibles = [n for n, _ in OPTIONS if n != "reference"
                   and parts[n]["substitution"] < OBJECTIF_SUBSTITUTIONS
                   and fiabilites[n] >= fiabilites["reference"] - BAISSE_FIABILITE_MAX
                   and parts[n]["correcte"] > parts["reference"]["correcte"]]
    if not admissibles:
        print("\nAucune option ne respecte les conditions avec un gain : on garde la référence")
        return
    meilleure = max(parts[n]["correcte"] for n in admissibles)
    retenue = next(n for n in admissibles if parts[n]["correcte"] >= meilleure - ECART_SIMPLICITE)
    print(f"\nOptions admissibles : {admissibles} ; option retenue par la règle : {retenue} "
          f"({parts[retenue]['correcte'] * 100:.1f} % de lectures correctes contre "
          f"{parts['reference']['correcte'] * 100:.1f} %, fiabilité {fiabilites[retenue] * 100:.1f} % contre "
          f"{fiabilites['reference'] * 100:.1f} %)")


if __name__ == "__main__":
    main()
