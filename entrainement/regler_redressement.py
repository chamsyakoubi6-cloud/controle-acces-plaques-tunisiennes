"""Phase 3, itération ciblée : contrôle du redressement par la rangée de caractères (réglage, hors pli) (PC).

Compare les variantes du contrôle de systeme/traitement.py (aucun, groupes, pente, geometrie) en lecture de bout
en bout, hors pli : CNN du pli qui n'a jamais vu le véhicule, option A, seuil 0,6, règle de vraisemblance.
Pour chaque variante : lectures correctes, rejetées, substitutions et plus courtes ; gains ET pertes par
rapport à « aucun » (la cascade d'origine) ; gain dans chaque pli (un gain réel se retrouve dans tous) ;
nombre de méthodes essayées et temps du traitement. Chaque plaque n'est détectée qu'une fois : son cadre YOLO et
une découpe plus large que celle du traitement restent en mémoire, si bien que toutes les variantes voient
exactement la même entrée.
Règle fixée d'avance : la variante qui donne le plus de lectures correctes, à condition que les substitutions
restent sous 1 % et que les lectures plus courtes n'augmentent pas ; à 0,5 point près, la plus simple (dans
l'ordre : groupes, pente, geometrie).

Utilisation, depuis la racine du projet :
    python -m entrainement.regler_redressement gris_8
"""

import argparse
import csv
import time

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import numpy as np

from entrainement.diagnostiquer_traitement import lire_option_a
from entrainement.evaluer_lecture import ISSUES, LecteurPlis, issue
from entrainement.evaluer_traitement import MODELE_DETECTION, cadre_vrai, lire_plaques
from entrainement.extraire_caracteres import FICHIER_PLIS, NB_PLIS
from systeme.detection import DetecteurPlaques, iou
from systeme.traitement import CONTROLES, traiter_plaque

MARGE_MEMOIRE_X, MARGE_MEMOIRE_Y = 0.30, 0.60   # plus large que la découpe du traitement (0,15 / 0,30)
OBJECTIF_SUBSTITUTIONS = 0.01
ECART_SIMPLICITE = 0.005


def preparer(plaques, plis, detecteur):
    """Cadre YOLO de chaque plaque (celui qui recouvre la vraie plaque, IoU >= 0,5) et découpe large autour."""
    entrees = []
    for plaque in plaques:
        image = cv2.imread(str(plaque["chemin"]))
        vrai = cadre_vrai(plaque["chemin"], image.shape[1], image.shape[0])
        detections = detecteur.detecter(image)
        recouvrements = iou(np.array(vrai), detections[:, :4]) if len(detections) else np.zeros(0)
        nom = f"{plaque['chemin'].parent.name}/{plaque['image']}"
        entree = {"nom": nom, "verite": f"{plaque['serie']} TU {plaque['numero']}", "pli": plis[nom], "decoupe": None}
        if len(recouvrements) and recouvrements.max() >= 0.5:
            x1, y1, x2, y2 = (float(v) for v in detections[int(recouvrements.argmax()), :4])
            x0 = int(max(0, x1 - MARGE_MEMOIRE_X * (x2 - x1)))
            y0 = int(max(0, y1 - MARGE_MEMOIRE_Y * (y2 - y1)))
            x_fin = int(min(image.shape[1], x2 + MARGE_MEMOIRE_X * (x2 - x1)))
            y_fin = int(min(image.shape[0], y2 + MARGE_MEMOIRE_Y * (y2 - y1)))
            entree.update({"decoupe": image[y0:y_fin, x0:x_fin].copy(), "boite": (x1 - x0, y1 - y0, x2 - x0, y2 - y0)})
        entrees.append(entree)
    return entrees


def evaluer_variante(entrees, lecteur, **reglages):
    """Issue de chaque plaque, méthodes essayées et durée du traitement (secondes). reglages : paramètres de
    traiter_plaque (controle, detacher, reglages_binarisation)."""
    issues, essais, durees = [], [], []
    for entree in entrees:
        if entree["decoupe"] is None:
            issues.append("rejetée")
            continue
        debut = time.perf_counter()
        resultat = traiter_plaque(entree["decoupe"], entree["boite"], **reglages)
        durees.append(time.perf_counter() - debut)
        essais.append(len(resultat["essais"]))
        issues.append(issue(lire_option_a(resultat, lecteur, entree["pli"]), entree["verite"]))
    return issues, essais, durees


def fiabilite(issues):
    """Part de lectures justes parmi les lectures acceptées (correctes ou fausses)."""
    acceptees = [i for i in issues if i != "rejetée"]
    return acceptees.count("correcte") / max(len(acceptees), 1)


def main():
    parseur = argparse.ArgumentParser(description="Contrôle du redressement : comparaison des variantes.")
    parseur.add_argument("nom", help="essai de validation croisée dont on lit les modèles (ex. gris_8)")
    arguments = parseur.parse_args()
    with open(FICHIER_PLIS, encoding="utf-8", newline="") as fichier:
        plis = {ligne["image"]: int(ligne["pli"]) for ligne in csv.DictReader(fichier)}
    plaques = [p for p in lire_plaques("reglage")[0] if f"{p['chemin'].parent.name}/{p['image']}" in plis]
    lecteur = LecteurPlis(arguments.nom)
    entrees = preparer(plaques, plis, DetecteurPlaques(str(MODELE_DETECTION)))
    print(f"Réglage, hors pli ({arguments.nom}) : {len(entrees)} plaques ; détectées : "
          f"{sum(e['decoupe'] is not None for e in entrees)}")

    resultats = {controle: evaluer_variante(entrees, lecteur, controle=controle) for controle in CONTROLES}
    parts = {c: {i: r[0].count(i) / len(entrees) for i in ISSUES} for c, r in resultats.items()}
    reference = [i == "correcte" for i in resultats["aucun"][0]]
    print("\nvariante     " + "  ".join(f"{i:>12s}" for i in ISSUES)
          + "   fiabilité  gains  pertes  méthodes  traitement (ms)")
    for controle, (issues, essais, durees) in resultats.items():
        justes = [i == "correcte" for i in issues]
        gains = sum(1 for a, b in zip(reference, justes) if b and not a)
        pertes = sum(1 for a, b in zip(reference, justes) if a and not b)
        print(f"{controle:10s}  " + "  ".join(f"{parts[controle][i] * 100:10.1f} %" for i in ISSUES)
              + f"   {fiabilite(issues) * 100:7.1f} %  {gains:5d}  {pertes:6d}  {np.mean(essais):8.2f}  "
              f"médiane {np.median(durees) * 1000:.1f}, "
              f"p95 {np.percentile(durees, 95) * 1000:.1f}")
    print("\nLectures correctes par pli (%) :")
    for controle, (issues, _, _) in resultats.items():
        texte = "  ".join(f"pli {k} {np.mean([i == 'correcte' for i, e in zip(issues, entrees) if e['pli'] == k]) * 100:5.1f}"
                          for k in range(NB_PLIS))
        print(f"   {controle:10s}: {texte}")

    # Règle fixée d'avance
    admissibles = [c for c in CONTROLES if c != "aucun"
                   and parts[c]["substitution"] < OBJECTIF_SUBSTITUTIONS
                   and parts[c]["plus courte"] <= parts["aucun"]["plus courte"]]
    if not admissibles:
        print("\nAucune variante ne respecte les conditions : on garde « aucun » (à discuter avant tout changement)")
        return
    meilleure = max(parts[c]["correcte"] for c in admissibles)
    retenue = next(c for c in admissibles if parts[c]["correcte"] >= meilleure - ECART_SIMPLICITE)
    print(f"\nVariantes admissibles : {admissibles} ; variante retenue par la règle : {retenue} "
          f"({parts[retenue]['correcte'] * 100:.1f} % de lectures correctes, contre "
          f"{parts['aucun']['correcte'] * 100:.1f} % sans contrôle)")


if __name__ == "__main__":
    main()
