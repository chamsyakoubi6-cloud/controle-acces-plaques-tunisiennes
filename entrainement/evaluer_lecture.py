"""Phase 4 : lecture de la plaque entière, de bout en bout (PC).

Chaîne complète, comme en service : détection YOLO (le cadre retenu doit recouvrir la vraie plaque : IoU
>= 0,5) -> traitement de la phase 3 -> CNN sur les vignettes -> assemblage du numéro (systeme/decision.py)
-> comparaison au numéro saisi. Issues possibles pour chaque plaque :
  correcte ;
  rejetée : doute (sous le seuil de confiance), format (hors format officiel), improbable (groupe d'un seul
      chiffre), aucun chiffre, ou plaque non détectée ;
  fausse, classée en : substitution (même nombre de chiffres, au moins un confondu : le rôle du seuil),
      plus courte (chiffre perdu : la phase 5 s'en protège), plus longue (intrus accepté).

Réglage (décisions), chaque image lue par le modèle du pli qui n'a jamais vu son véhicule (hors pli). Deux
façons de choisir les vignettes lues par le CNN, comparées par la mesure :
  A : les chiffres probables de la phase 3 seulement (le CNN écarte les intrus) ;
  B : tous les candidats, puis le contrôle de régularité de la phase 3 sur ceux lus comme chiffres.
Règles fixées d'avance : seuil = le plus petit qui ramène les substitutions sous 1 % des plaques ; B seulement
s'il apporte au moins 0,5 point de lectures correctes (sinon A, plus simple). Retenu : A, seuil 0,6.

Test, une seule fois, choix figés : la fonction même du système (systeme.decision.lire_numero, option A) avec le
modèle final exporté en ONNX (systeme.lecture.LecteurCaracteres). Résultats sur les 142 plaques, sur les 124 de
véhicules jamais vus (ni par YOLO ni par le CNN) et par source d'image ; exactitude du CNN seul sur les plaques
du test strictement segmentées.

Utilisation, depuis la racine du projet :
    python -m entrainement.evaluer_lecture gris_8                                      réglage, hors pli
    python -m entrainement.evaluer_lecture gris_8 --test --seuil 0.6 --modele modeles/lecteur.onnx
"""

import argparse
import csv
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # masque les messages d'information de TensorFlow

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import matplotlib

matplotlib.use("Agg")  # figures enregistrées dans des fichiers, sans fenêtre
import matplotlib.pyplot as plt
import numpy as np

from entrainement.donnees_kaggle import RACINE, assembler_planche, faire_vignette
from entrainement.evaluer_traitement import MODELE_DETECTION, cadre_vrai, lire_plaques
from entrainement.extraire_caracteres import DOSSIER_FIGURES, FICHIER_PLIS, NB_PLIS, etiqueter, vehicule
from systeme.decision import lire_numero, lire_plaque
from systeme.detection import DetecteurPlaques, iou
from systeme.lecture import ENTREE, LecteurCaracteres, preparer_vignettes
from systeme.traitement import traiter_plaque

DOSSIER_RUNS = RACINE / "sorties" / "runs" / "cnn"
SEUILS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)
OBJECTIF_SUBSTITUTIONS = 0.01    # substitutions : moins de 1 % des plaques
ECART_OPTIONS = 0.005            # B doit apporter au moins 0,5 point de lectures correctes
ISSUES = ("correcte", "rejetée", "substitution", "plus courte", "plus longue")
MOTIFS = ("doute", "format", "improbable", "aucun chiffre", "non détectée")
# Couleurs des issues : palette catégorielle de référence, dans l'ordre fixe
COULEURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
ENCRE, ENCRE_SECONDAIRE, GRILLE = "#0b0b0b", "#52514e", "#e2e1dc"
NON_DETECTEE = {"texte": None, "statut": "non détectée", "lecture": "", "confiance": 0.0, "chiffres": []}


class LecteurPlis:
    """Lecture hors pli : chaque plaque est lue par le modèle du pli qui n'a jamais vu son véhicule."""

    def __init__(self, nom):
        import keras  # import local : TensorFlow n'est chargé que pour la lecture du réglage
        self.modeles = [keras.models.load_model(DOSSIER_RUNS / nom / f"pli{k}.keras") for k in range(NB_PLIS)]
        with open(DOSSIER_RUNS / nom / "resume.json", encoding="utf-8") as fichier:
            self.entree = json.load(fichier)["entree"]

    def lire(self, resultat, pli):
        """Classe et confiance de chaque candidat de la plaque."""
        vignettes = resultat["vignettes_grises"] if self.entree == "gris" else resultat["vignettes_binaires"]
        probabilites = np.asarray(self.modeles[pli](preparer_vignettes(vignettes, self.entree), training=False))
        return probabilites.argmax(axis=1), probabilites.max(axis=1)


def detecter_et_traiter(plaque, detecteur):
    """Image, détection du cadre qui recouvre la vraie plaque (IoU >= 0,5) et traitement de la phase 3 ;
    renvoie (enregistrement, résultat du traitement ou None si la plaque n'est pas détectée)."""
    image = cv2.imread(str(plaque["chemin"]))
    cadre = cadre_vrai(plaque["chemin"], image.shape[1], image.shape[0])
    detections = detecteur.detecter(image)
    recouvrements = iou(np.array(cadre), detections[:, :4]) if len(detections) else np.zeros(0)
    enregistrement = {"nom": f"{plaque['chemin'].parent.name}/{plaque['image']}", "vehicule": vehicule(plaque),
                      "verite": f"{plaque['serie']} TU {plaque['numero']}", "cote_image": max(image.shape[:2]),
                      "detectee": False}
    if not len(recouvrements) or recouvrements.max() < 0.5:
        return enregistrement, None
    boite = tuple(float(v) for v in detections[int(recouvrements.argmax()), :4])
    resultat = traiter_plaque(image, boite)
    enregistrement.update({"detectee": True, "plaque": resultat["plaque"], "boite": boite})
    return enregistrement, resultat


def issue(decision, verite):
    if decision["statut"] != "lue":
        return "rejetée"
    if decision["texte"] == verite:
        return "correcte"
    lus, vrais = sum(c.isdigit() for c in decision["texte"]), sum(c.isdigit() for c in verite)
    return "substitution" if lus == vrais else ("plus courte" if lus < vrais else "plus longue")


def afficher_issues(nom_lot, paires):
    """paires : (enregistrement, décision). Affiche la part de chaque issue et les motifs de rejet."""
    issues = [issue(d, e["verite"]) for e, d in paires]
    motifs = [d["statut"] for e, d in paires if d["statut"] != "lue"]
    print(f"   {nom_lot:30s}" + "  ".join(f"{issues.count(i) / len(paires) * 100:10.1f} %" for i in ISSUES)
          + "   | rejets : " + ", ".join(f"{m} {motifs.count(m)}" for m in MOTIFS if motifs.count(m)))


def planche_fausses(paires, chemin):
    """Lectures fausses acceptées : plaque redressée, lecture et vérité."""
    cases = []
    for e, decision in paires:
        genre = issue(decision, e["verite"])
        if genre in ("substitution", "plus courte", "plus longue"):
            texte = f"{e['nom']} lu {decision['texte']} vrai {e['verite']} ({genre})"
            cases.append(np.vstack([faire_vignette(np.zeros((18, 450, 3), dtype=np.uint8), 450, 18, texte[:62]),
                                    e["plaque"]]))
    if cases:
        cv2.imwrite(str(chemin), assembler_planche(cases[:60], 2))
    return len(cases)


# ------------------------------------------------------------------ Réglage (décisions)

def lire_reglage(nom):
    """Détection, traitement et lecture hors pli de toutes les plaques de réglage gardées. Le CNN lit TOUS les
    candidats une fois ; les décisions se rejouent ensuite pour chaque option et chaque seuil."""
    with open(FICHIER_PLIS, encoding="utf-8", newline="") as fichier:
        plis = {ligne["image"]: int(ligne["pli"]) for ligne in csv.DictReader(fichier)}
    plaques = [p for p in lire_plaques("reglage")[0] if f"{p['chemin'].parent.name}/{p['image']}" in plis]
    lecteur, detecteur = LecteurPlis(nom), DetecteurPlaques(str(MODELE_DETECTION))
    enregistrements = []
    for plaque in plaques:
        e, resultat = detecter_et_traiter(plaque, detecteur)
        if resultat is not None:
            classes, confiances = lecteur.lire(resultat, plis[e["nom"]]) if resultat["candidats"] else ([], [])
            e.update({"candidats": resultat["candidats"], "classes": np.asarray(classes),
                      "confiances": np.asarray(confiances), "hauteur": resultat["binaire"].shape[0]})
        enregistrements.append(e)
    return enregistrements


def decider(e, option, seuil):
    if not e["detectee"]:
        return NON_DETECTEE
    garder = [k for k, b in enumerate(e["candidats"]) if option == "B" or b["chiffre"]]
    return lire_plaque([e["candidats"][k] for k in garder], e["classes"][garder], e["confiances"][garder],
                       e["hauteur"], seuil, regularite=(option == "B"))


def balayer(enregistrements, option):
    """Part de chaque issue pour chaque seuil : tableau (nombre de seuils, nombre d'issues)."""
    parts = np.zeros((len(SEUILS), len(ISSUES)))
    for i, seuil in enumerate(SEUILS):
        for e in enregistrements:
            parts[i, ISSUES.index(issue(decider(e, option, seuil), e["verite"]))] += 1
    return parts / len(enregistrements)


def choisir_seuil(parts):
    """Indice du plus petit seuil qui ramène les substitutions sous l'objectif (le dernier à défaut)."""
    for i in range(len(SEUILS)):
        if parts[i, ISSUES.index("substitution")] < OBJECTIF_SUBSTITUTIONS:
            return i
    return len(SEUILS) - 1


def figure_seuils(balayages, choix, chemin, titre):
    figure, axes = plt.subplots(1, len(balayages), figsize=(6.4 * len(balayages), 4.6), sharey=True, squeeze=False)
    for axe, (option, parts) in zip(axes[0], balayages.items()):
        positions = np.arange(len(SEUILS))
        for k, (nom_issue, couleur) in enumerate(zip(ISSUES, COULEURS)):
            axe.plot(positions, parts[:, k] * 100, color=couleur, linewidth=2, marker="o", markersize=4, label=nom_issue)
        axe.axvline(choix[option], color=ENCRE_SECONDAIRE, linewidth=1)
        axe.set_xticks(positions, [f"{s:g}" for s in SEUILS], rotation=45, fontsize=8)
        axe.set_title(f"option {option} (trait vertical : seuil retenu, {SEUILS[choix[option]]:g})", fontsize=10,
                      color=ENCRE, loc="left")
        axe.set_xlabel("seuil de confiance", color=ENCRE_SECONDAIRE)
        axe.grid(color=GRILLE, linewidth=0.8)
        for cote in ("top", "right"):
            axe.spines[cote].set_visible(False)
    axes[0][0].set_ylabel("plaques (%)", color=ENCRE_SECONDAIRE)
    poignees, libelles = axes[0][0].get_legend_handles_labels()
    figure.legend(poignees, libelles, loc="lower center", ncol=len(ISSUES), frameon=False, fontsize=9)
    figure.suptitle(titre, fontsize=10, color=ENCRE)
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    figure.savefig(chemin, dpi=110)
    plt.close(figure)


def evaluer_reglage(nom):
    enregistrements = lire_reglage(nom)
    print(f"Réglage, hors pli ({nom}) : {len(enregistrements)} plaques (véhicules du test exclus) ; détectées "
          f"(IoU >= 0,5) : {sum(e['detectee'] for e in enregistrements)}")
    balayages, choix = {}, {}
    for option in ("A", "B"):
        balayages[option] = balayer(enregistrements, option)
        choix[option] = choisir_seuil(balayages[option])
        print(f"\nOption {option} : part des plaques (%) selon le seuil de confiance")
        print("   seuil   " + "  ".join(f"{nom_issue:>12s}" for nom_issue in ISSUES))
        for i, seuil in enumerate(SEUILS):
            repere = "  <- seuil retenu" if i == choix[option] else ""
            print(f"   {seuil:<6g}  " + "  ".join(f"{v * 100:12.1f}" for v in balayages[option][i]) + repere)
    correctes = {option: balayages[option][choix[option], 0] for option in ("A", "B")}
    retenue = "B" if correctes["B"] - correctes["A"] >= ECART_OPTIONS else "A"
    print(f"\nLectures correctes au seuil retenu : A {correctes['A'] * 100:.1f} % (seuil {SEUILS[choix['A']]:g}), "
          f"B {correctes['B'] * 100:.1f} % (seuil {SEUILS[choix['B']]:g}) -> option retenue : {retenue}")
    print("\nAu seuil retenu de chaque option :       " + "  ".join(f"{i:>12s}" for i in ISSUES))
    for option in ("A", "B"):
        seuil = SEUILS[choix[option]]
        afficher_issues(f"option {option}, seuil {seuil:g}", [(e, decider(e, option, seuil)) for e in enregistrements])

    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    figure_seuils(balayages, choix, DOSSIER_FIGURES / f"seuils_{nom}.png",
                  f"Lecture de bout en bout, réglage hors pli ({nom}, {len(enregistrements)} plaques)")
    seuil = SEUILS[choix[retenue]]
    nombre = planche_fausses([(e, decider(e, retenue, seuil)) for e in enregistrements],
                             DOSSIER_FIGURES / f"fausses_{nom}.jpg")
    print(f"Figures : seuils_{nom}.png, fausses_{nom}.jpg ({nombre} lectures fausses acceptées)")


# ------------------------------------------------------------------ Test (une seule fois)

def evaluer_test(seuil, chemin_modele):
    plaques = lire_plaques("test")[0]
    vehicules_reglage = {vehicule(p) for p in lire_plaques("reglage")[0]}
    lecteur, detecteur = LecteurCaracteres(chemin_modele), DetecteurPlaques(str(MODELE_DETECTION))
    print(f"TEST (une seule fois) : {len(plaques)} plaques ; lire_numero (option A), seuil {seuil:g}, "
          f"modèle {chemin_modele}, entrée {ENTREE}")
    paires, vignettes_strictes, etiquettes_strictes = [], [], []
    for plaque in plaques:
        e, resultat = detecter_et_traiter(plaque, detecteur)
        decision = NON_DETECTEE if resultat is None else lire_numero(resultat, lecteur, seuil)
        paires.append((e, decision))
        # CNN seul : sur une plaque strictement segmentée, l'étiquette de chaque vignette se déduit du numéro
        # saisi, comme pour le jeu de caractères ; on lit alors TOUS ses candidats.
        etiquettes = etiqueter(resultat, plaque) if resultat is not None else None
        if etiquettes is not None:
            vignettes_strictes += resultat["vignettes_grises"] if ENTREE == "gris" else resultat["vignettes_binaires"]
            etiquettes_strictes += etiquettes
    lots = (("toutes", lambda e: True), ("véhicules jamais vus", lambda e: e["vehicule"] not in vehicules_reglage),
            ("images web <= 700 px", lambda e: e["cote_image"] <= 700),
            ("images > 700 px", lambda e: e["cote_image"] > 700))
    print("\n   plaques                       " + "  ".join(f"{i:>12s}" for i in ISSUES))
    for nom_lot, garder in lots:
        lot = [(e, d) for e, d in paires if garder(e)]
        afficher_issues(f"{nom_lot} ({len(lot)})", lot)
    jamais_vus = [e for e, d in paires if e["vehicule"] not in vehicules_reglage]
    print(f"\nDétection (IoU >= 0,5) : toutes {sum(e['detectee'] for e, d in paires)}/{len(paires)} ; véhicules "
          f"jamais vus {sum(e['detectee'] for e in jamais_vus)}/{len(jamais_vus)}")
    classes, _ = lecteur.lire(vignettes_strictes)
    etiquettes_strictes = np.array(etiquettes_strictes)
    print(f"CNN seul, plaques du test strictement segmentées ({sum(1 for _ in etiquettes_strictes)} vignettes) : "
          f"exactitude {np.mean(classes == etiquettes_strictes) * 100:.2f} % "
          f"({int(np.sum(classes != etiquettes_strictes))} erreurs)")
    nombre = planche_fausses(paires, DOSSIER_FIGURES / "fausses_test.jpg")
    print(f"Figure : fausses_test.jpg ({nombre} lectures fausses acceptées)")


def main():
    parseur = argparse.ArgumentParser(description="Lecture de la plaque entière, de bout en bout.")
    parseur.add_argument("nom", help="essai de validation croisée (dossier de sorties/runs/cnn/)")
    parseur.add_argument("--test", action="store_true", help="évaluation finale sur le test (une seule fois)")
    parseur.add_argument("--seuil", type=float, help="test : seuil de confiance figé")
    parseur.add_argument("--modele", help="test : modèle final exporté (.onnx)")
    arguments = parseur.parse_args()
    if arguments.test:
        if arguments.seuil is None or arguments.modele is None:
            parseur.error("--test exige --seuil et --modele : les choix sont figés avant le test")
        evaluer_test(arguments.seuil, arguments.modele)
    else:
        evaluer_reglage(arguments.nom)


if __name__ == "__main__":
    main()
