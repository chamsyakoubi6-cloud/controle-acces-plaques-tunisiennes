"""Phase 3, étape 5 : évaluation du traitement classique (redressement, binarisation, segmentation) (PC).

Par défaut, on part des CADRES VRAIS (annotations) pour mesurer le traitement classique seul, sans les
erreurs du détecteur. Avec --cadres yolo, on part des cadres détectés par YOLO (modeles/detecteur.onnx),
comme en fonctionnement réel : chaîne détection + traitement.
Une plaque est « correctement segmentée » si l'on trouve le bon nombre de chiffres probables de chaque côté
du plus grand écart (série et numéro), comparé à la vérité terrain saisie.
Les réglages se choisissent sur les plaques de RÉGLAGE (train + val) ; le TEST ne sert qu'une fois, à la fin.

Sorties : bilan chiffré, taux par méthode de redressement et par catégorie de difficulté (mesurée), planches
des échecs et d'un échantillon de plaques redressées dans docs/figures/phase3/ (non versionnées).

Utilisation, depuis la racine du projet :
    python -m entrainement.evaluer_traitement                 plaques de réglage saisies
    python -m entrainement.evaluer_traitement --partie test   évaluation finale (une seule fois)
    python -m entrainement.evaluer_traitement --partie test --cadres yolo    chaîne détection + traitement
"""

import argparse
import csv
import random
from pathlib import Path

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import numpy as np

from entrainement.donnees_kaggle import RACINE, assembler_planche, faire_vignette
from systeme.detection import DetecteurPlaques, iou
from systeme.segmentation import dessiner_segmentation
from systeme.traitement import traiter_plaque

MODELE_DETECTION = RACINE / "modeles" / "detecteur.onnx"
DOSSIER_YOLO = RACINE / "donnees" / "yolo"
FICHIER_VERITE = RACINE / "donnees" / "verite" / "numeros.csv"
DOSSIER_FIGURES = RACINE / "docs" / "figures" / "phase3"
# Catégories de difficulté, mesurées sur chaque plaque (seuils expliqués dans categories())
PETITE_PX = 30              # hauteur du cadre dans l'image d'origine
DE_BIAIS_RAPPORT = 2.5      # largeur / hauteur du cadre (4,5 pour une plaque vue de face)
SOMBRE_LUMINANCE = 70       # luminance moyenne de l'image (0-255)
REFLET_PART = 0.05          # part de pixels saturés (>= 250) sur la plaque redressée
CONTRASTE_MIN = 80          # écart entre les 5e et 95e centiles de gris de la plaque redressée
FLOU_PERCENTILE = 15        # « floue » : parmi les 15 % de plaques les moins nettes (variance du laplacien)


def lire_plaques(partie):
    """Plaques saisies d'une partie ("reglage" = train + val, ou "test"), avec leur cadre vrai."""
    if not FICHIER_VERITE.exists():
        return [], {}
    with open(FICHIER_VERITE, encoding="utf-8", newline="") as fichier:
        lignes = list(csv.DictReader(fichier))
    parties = ("test",) if partie == "test" else ("train", "val")
    lignes = [ligne for ligne in lignes if ligne["partie"] in parties]
    statuts = {}
    for ligne in lignes:
        statuts[ligne["statut"]] = statuts.get(ligne["statut"], 0) + 1
    plaques = []
    for ligne in lignes:
        if ligne["statut"] != "numero":
            continue
        chemin = DOSSIER_YOLO / "images" / ligne["partie"] / ligne["image"]
        serie, _, numero = ligne["numero"].split()
        plaques.append({"image": ligne["image"], "chemin": chemin, "serie": serie, "numero": numero})
    return plaques, statuts


def cadre_vrai(chemin, largeur, hauteur):
    ligne = (DOSSIER_YOLO / "labels" / chemin.parent.name / (chemin.stem + ".txt")).read_text(encoding="utf-8").split()
    _, xc, yc, l, h = (float(v) for v in ligne[:5])
    return (xc - l / 2) * largeur, (yc - h / 2) * hauteur, (xc + l / 2) * largeur, (yc + h / 2) * hauteur


def mesurer(plaque, reglages_binarisation=None, detecteur=None):
    """Traite une plaque et mesure ses indicateurs de difficulté ; renvoie un dictionnaire de résultats.
    Avec un détecteur : cadre détecté par YOLO recouvrant la vraie plaque (IoU >= 0,5), sinon plaque
    « non détectée » (comptée comme un échec, sans traitement)."""
    image = cv2.imread(str(plaque["chemin"]))
    cadre = cadre_vrai(plaque["chemin"], image.shape[1], image.shape[0])
    if detecteur is not None:
        detections = detecteur.detecter(image)
        recouvrements = iou(np.array(cadre), detections[:, :4]) if len(detections) else np.zeros(0)
        if not len(recouvrements) or recouvrements.max() < 0.5:
            return {"plaque": plaque, "detectee": False, "correcte": False, "sans_perte": False}
        cadre = tuple(float(v) for v in detections[int(recouvrements.argmax()), :4])
    resultat = traiter_plaque(image, cadre, reglages_binarisation=reglages_binarisation)
    gris_plaque = cv2.cvtColor(resultat["plaque"], cv2.COLOR_BGR2GRAY)
    return {
        "plaque": plaque, "detectee": True, "image": image, "cadre": cadre, "resultat": resultat,
        "correcte": (len(resultat["serie"]) == len(plaque["serie"]) and len(resultat["numero"]) == len(plaque["numero"])),
        # Mesure tolérante : aucun chiffre perdu de chaque côté ; les candidats en trop (barres du cadre,
        # fragments arabes) sont admis, car c'est le CNN (phase 4) qui les classera « autre ».
        "sans_perte": (len(resultat["serie"]) >= len(plaque["serie"]) and len(resultat["numero"]) >= len(plaque["numero"])),
        "hauteur_px": cadre[3] - cadre[1],
        "rapport_cadre": (cadre[2] - cadre[0]) / max(cadre[3] - cadre[1], 1),
        "luminance": float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()),
        "reflet": float(np.mean(gris_plaque >= 250)),
        "contraste": float(np.percentile(gris_plaque, 95) - np.percentile(gris_plaque, 5)),
        "nettete": float(cv2.Laplacian(gris_plaque, cv2.CV_64F).var()),  # faible = plaque floue
    }


def evaluer(plaques, reglages_binarisation=None, detecteur=None):
    """Traite toutes les plaques ; renvoie (liste des mesures, taux de segmentation correcte)."""
    mesures = [mesurer(p, reglages_binarisation, detecteur) for p in plaques]
    taux = sum(m["correcte"] for m in mesures) / max(len(mesures), 1)
    return mesures, taux


def categories(mesure, seuil_flou):
    """Catégories de difficulté d'une plaque, d'après ses indicateurs mesurés (une plaque peut en cumuler)."""
    etiquettes = []
    if mesure["hauteur_px"] < PETITE_PX:
        etiquettes.append("petite")
    if mesure["rapport_cadre"] < DE_BIAIS_RAPPORT:
        etiquettes.append("de biais")
    if mesure["luminance"] < SOMBRE_LUMINANCE:
        etiquettes.append("sombre")
    if mesure["reflet"] > REFLET_PART:
        etiquettes.append("reflet")
    if mesure["contraste"] < CONTRASTE_MIN:
        etiquettes.append("faible contraste")
    if mesure["nettete"] < seuil_flou:
        etiquettes.append("floue")
    return etiquettes


def panneau_echec(mesure, etiquettes):
    """Vignette d'un échec : découpe avec les coins trouvés, puis segmentation, avec attendu / obtenu."""
    resultat, plaque = mesure["resultat"], mesure["plaque"]
    vue = mesure["image"].copy()
    cv2.polylines(vue, [resultat["coins"].astype(np.int32)], True, (0, 255, 0), max(2, vue.shape[1] // 400))
    x1, y1, x2, y2 = mesure["cadre"]
    marge_x, marge_y = 0.3 * (x2 - x1), 0.6 * (y2 - y1)
    zone = vue[max(0, int(y1 - marge_y)):int(y2 + marge_y), max(0, int(x1 - marge_x)):int(x2 + marge_x)]
    haut = faire_vignette(zone, 300, 110)
    bas = cv2.resize(dessiner_segmentation(resultat["binaire"], resultat), (300, 67))
    texte = (f"{plaque['image']} attendu {len(plaque['serie'])}/{len(plaque['numero'])} obtenu "
             f"{len(resultat['serie'])}/{len(resultat['numero'])} {resultat['methode_coins']} {','.join(etiquettes)}")
    bandeau = faire_vignette(np.zeros((18, 300, 3), dtype=np.uint8), 300, 18, texte[:52])
    return np.vstack([bandeau, haut, bas])


def main():
    parseur = argparse.ArgumentParser(description="Évaluation du traitement classique des plaques.")
    parseur.add_argument("--partie", choices=("reglage", "test"), default="reglage")
    parseur.add_argument("--cadres", choices=("vrais", "yolo"), default="vrais",
                         help="cadres annotés (traitement seul) ou détectés par YOLO (détection + traitement)")
    arguments = parseur.parse_args()

    plaques, statuts = lire_plaques(arguments.partie)
    print(f"Partie {arguments.partie}, cadres {arguments.cadres} : statuts saisis {statuts} ; "
          f"plaques évaluées : {len(plaques)}")
    if not plaques:
        print("Aucune plaque saisie : lancer d'abord python -m entrainement.saisir_numeros")
        return
    detecteur = DetecteurPlaques(str(MODELE_DETECTION)) if arguments.cadres == "yolo" else None
    mesures, taux = evaluer(plaques, detecteur=detecteur)
    if detecteur is not None:
        nb_detectees = sum(m["detectee"] for m in mesures)
        print(f"\nPlaques détectées par YOLO (IoU >= 0,5 avec la vraie plaque) : {nb_detectees}/{len(mesures)}")
    print(f"\nSegmentation correcte, mesure stricte (exactement le bon nombre de chiffres de chaque côté) : "
          f"{sum(m['correcte'] for m in mesures)}/{len(mesures)} = {taux * 100:.1f} %")
    sans_perte = sum(m["sans_perte"] for m in mesures)
    print(f"Aucun chiffre perdu, mesure tolérante (candidats en trop admis, le CNN les écartera) : "
          f"{sans_perte}/{len(mesures)} = {sans_perte / len(mesures) * 100:.1f} %")

    traitees = [m for m in mesures if m["detectee"]]  # les ventilations ne portent que sur les plaques traitées
    print("\nPar méthode de redressement :")
    for methode in ("lisere", "zone_noire", "caracteres", "bords", "cadre"):
        lot = [m for m in traitees if m["resultat"]["methode_coins"] == methode]
        if lot:
            print(f"   {methode:11s}: {len(lot):4d} plaques ({len(lot) / len(traitees) * 100:4.1f} %), "
                  f"correctes {sum(m['correcte'] for m in lot) / len(lot) * 100:5.1f} %")

    seuil_flou = float(np.percentile([m["nettete"] for m in traitees], FLOU_PERCENTILE))
    print(f"\nPar catégorie de difficulté (seuil de netteté pour « floue » : {seuil_flou:.0f}) :")
    par_categorie = {}
    for mesure in traitees:
        mesure["categories"] = categories(mesure, seuil_flou)
        for etiquette in mesure["categories"] or ["aucune difficulté"]:
            par_categorie.setdefault(etiquette, []).append(mesure["correcte"])
    for etiquette, resultats in sorted(par_categorie.items(), key=lambda e: -len(e[1])):
        print(f"   {etiquette:18s}: {len(resultats):4d} plaques, correctes {np.mean(resultats) * 100:5.1f} %")

    durees = {etape: np.median([m["resultat"]["durees"][etape] for m in traitees]) * 1000
              for etape in ("redressement", "binarisation", "segmentation")}
    print("\nDurées médianes (ms) :", {etape: round(valeur, 1) for etape, valeur in durees.items()})

    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    suffixe = arguments.partie + ("_yolo" if arguments.cadres == "yolo" else "")
    echecs = [m for m in traitees if not m["correcte"]]
    if echecs:
        cv2.imwrite(str(DOSSIER_FIGURES / f"echecs_{suffixe}.jpg"),
                    assembler_planche([panneau_echec(m, m["categories"]) for m in echecs[:60]], 5))
    echantillon = random.Random(0).sample(traitees, min(60, len(traitees)))
    cv2.imwrite(str(DOSSIER_FIGURES / f"redressement_{suffixe}.jpg"),
                assembler_planche([faire_vignette(m["resultat"]["plaque"], 225, 50, m["plaque"]["image"])
                                   for m in echantillon], 6))
    print(f"\nFigures : {DOSSIER_FIGURES} (echecs_{suffixe}.jpg, redressement_{suffixe}.jpg)")


if __name__ == "__main__":
    main()
