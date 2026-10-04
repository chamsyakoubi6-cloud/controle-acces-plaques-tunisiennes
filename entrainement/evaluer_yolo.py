"""Phase 2, étape 4 : évaluation des modèles de détection et choix de la taille d'entrée (PC).

1. Validation des trois modèles (320, 416, 640) : mAP50, mAP50-95, précision, rappel.
   Règle fixée AVANT les entraînements : on retient la plus petite taille dont le mAP50-95
   est à moins de 0,02 du meilleur.
2. Seuil de confiance : courbes F1 / précision / rappel en fonction de la confiance, sur la
   VALIDATION (jamais sur le test), pour proposer un seuil de fonctionnement.
3. Test, une seule fois, sur le modèle retenu : métriques finales et courbes pour le rapport.
4. Échecs sur le test au seuil de fonctionnement (confiance 0,5, IoU de NMS 0,45) : plaques
   manquées, cadres imprécis (IoU < 0,5) et fausses détections, dessinés sur une planche.
Le modèle retenu est copié dans modeles/detecteur.pt.

Utilisation, depuis la racine du projet :
    python -m entrainement.evaluer_yolo
Figures : docs/figures/phase2/ (non versionnées : elles montrent des plaques).
"""

import os

os.environ["YOLO_AUTOINSTALL"] = "False"  # Ultralytics ne doit rien installer tout seul

import csv
import random
import shutil
from pathlib import Path

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import matplotlib

matplotlib.use("Agg")  # figures enregistrées dans des fichiers, sans fenêtre
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from entrainement.donnees_kaggle import RACINE, assembler_planche, faire_vignette
from systeme.detection import iou

TAILLES = (320, 416, 640)
ECART_TOLERE = 0.02            # règle de choix de la taille, fixée avant les entraînements
CONFIANCE, IOU_NMS = 0.5, 0.45  # seuils de fonctionnement de départ
IOU_CORRECTE = 0.5             # une détection est juste si son IoU avec le vrai cadre atteint 0,5
FICHIER_DONNEES = RACINE / "donnees" / "yolo" / "plaques.yaml"
DOSSIER_RUNS = RACINE / "sorties" / "runs" / "detection"
DOSSIER_FIGURES = RACINE / "docs" / "figures" / "phase2"


def poids(taille):
    return DOSSIER_RUNS / f"yolov8n_{taille}" / "weights" / "best.pt"


def valider(taille, partie, graphiques=False):
    """Métriques d'Ultralytics sur une partie du jeu. Réglages standard du calcul du mAP :
    confiance 0,001 (toute la courbe précision-rappel) et IoU de NMS 0,7."""
    return YOLO(str(poids(taille))).val(data=str(FICHIER_DONNEES), split=partie, imgsz=taille, batch=16,
                                        device=0, plots=graphiques, verbose=False, project=str(DOSSIER_RUNS),
                                        name=f"eval_{partie}_{taille}", exist_ok=True)


# ------------------------------------------------ 1. Choix de la taille

def choisir_taille():
    print("\n1. Validation des trois modèles")
    metriques = {}
    for taille in TAILLES:
        boites = valider(taille, "val").box
        metriques[taille] = boites
        print(f"   entrée {taille} : mAP50 {boites.map50:.3f}   mAP50-95 {boites.map:.3f}   "
              f"précision {boites.mp:.3f}   rappel {boites.mr:.3f}")
    meilleur = max(boites.map for boites in metriques.values())
    retenue = min(t for t, boites in metriques.items() if boites.map >= meilleur - ECART_TOLERE)
    print(f"   règle : plus petite taille à moins de {ECART_TOLERE} du meilleur mAP50-95 ({meilleur:.3f})"
          f" -> taille retenue : {retenue}")
    return retenue


def figure_comparaison():
    """Évolution du mAP50-95 et de la perte de localisation en validation, pour les trois tailles."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for taille in TAILLES:
        with open(DOSSIER_RUNS / f"yolov8n_{taille}" / "results.csv", encoding="utf-8", newline="") as fichier:
            lignes = list(csv.DictReader(fichier))
        epoques = [int(ligne["epoch"]) for ligne in lignes]
        axes[0].plot(epoques, [float(ligne["metrics/mAP50-95(B)"]) for ligne in lignes], label=f"entrée {taille}")
        axes[1].plot(epoques, [float(ligne["val/box_loss"]) for ligne in lignes], label=f"entrée {taille}")
    axes[0].set(title="mAP50-95 en validation", xlabel="époque", ylabel="mAP50-95")
    axes[1].set(title="Perte de localisation (box) en validation", xlabel="époque", ylabel="perte")
    for axe in axes:
        axe.legend()
        axe.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(DOSSIER_FIGURES / "comparaison_tailles.png", dpi=120)
    plt.close(figure)


# ---------------------------------------------- 2. Seuil de confiance

def etudier_seuil(taille):
    """Courbes F1, précision et rappel en fonction du seuil de confiance, sur la validation."""
    print("\n2. Seuil de confiance (validation)")
    boites = valider(taille, "val", graphiques=True).box
    confiances = boites.px                       # 1000 seuils de 0 à 1
    f1, precision, rappel = boites.f1_curve[0], boites.p_curve[0], boites.r_curve[0]
    i_max = int(f1.argmax())
    i_depart = int(np.abs(confiances - CONFIANCE).argmin())
    plateau = confiances[f1 >= f1[i_max] - 0.01]  # seuils dont le F1 est à moins de 0,01 du maximum
    for nom, i in (("F1 maximal", i_max), (f"seuil de départ {CONFIANCE}", i_depart)):
        print(f"   {nom:22s}: confiance {confiances[i]:.3f}  F1 {f1[i]:.3f}  "
              f"précision {precision[i]:.3f}  rappel {rappel[i]:.3f}")
    print(f"   F1 à moins de 0,01 du maximum pour une confiance entre {plateau.min():.2f} et {plateau.max():.2f}")
    print("   seuil -> précision / rappel / F1")
    for seuil in (0.25, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        i = int(np.abs(confiances - seuil).argmin())
        print(f"   {seuil:.2f} -> {precision[i]:.3f} / {rappel[i]:.3f} / {f1[i]:.3f}")

    figure, axe = plt.subplots(figsize=(8, 5))
    axe.plot(confiances, f1, label="F1", linewidth=2)
    axe.plot(confiances, precision, label="précision")
    axe.plot(confiances, rappel, label="rappel")
    axe.axvspan(plateau.min(), plateau.max(), color="tab:green", alpha=0.15, label="F1 à moins de 0,01 du max")
    axe.axvline(confiances[i_max], color="tab:green", linestyle=":", label=f"F1 max ({confiances[i_max]:.2f})")
    axe.axvline(CONFIANCE, color="tab:red", linestyle="--", label=f"seuil de départ ({CONFIANCE})")
    axe.set(title=f"Choix du seuil de confiance - validation, entrée {taille}", xlabel="seuil de confiance",
            ylabel="valeur", xlim=(0, 1), ylim=(0, 1.02))
    axe.legend(loc="lower left")
    axe.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(DOSSIER_FIGURES / "seuil_confiance_validation.png", dpi=120)
    plt.close(figure)


# ------------------------------------------------------------ 3. Test

def evaluer_test(taille):
    print("\n3. Test (une seule fois, modèle retenu)")
    boites = valider(taille, "test", graphiques=True).box
    print(f"   mAP50 {boites.map50:.3f}   mAP50-95 {boites.map:.3f}   précision {boites.mp:.3f}   rappel {boites.mr:.3f}")
    # Courbes d'Ultralytics sur le test (précision-rappel, F1, matrice de confusion) pour le rapport
    dossier = DOSSIER_FIGURES / "test"
    dossier.mkdir(parents=True, exist_ok=True)
    for figure in (DOSSIER_RUNS / f"eval_test_{taille}").glob("*.png"):
        shutil.copy2(figure, dossier / figure.name)


# ---------------------------------------------------------- 4. Échecs

def lire_etiquettes(chemin_image, largeur, hauteur):
    """Cadres vrais d'une image, lus dans son fichier YOLO et remis en pixels (x1, y1, x2, y2)."""
    chemin = chemin_image.parent.parent.parent / "labels" / chemin_image.parent.name / (chemin_image.stem + ".txt")
    cadres = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        _, xc, yc, l, h = (float(v) for v in ligne.split())
        cadres.append(((xc - l / 2) * largeur, (yc - h / 2) * hauteur, (xc + l / 2) * largeur, (yc + h / 2) * hauteur))
    return np.array(cadres, dtype=np.float32)


def analyser_echecs(taille):
    """Prédictions au seuil de fonctionnement sur le test, comparées aux vrais cadres.

    rect=False : letterbox CARRÉ, comme notre modèle ONNX à entrée fixe. Sans cela, Ultralytics prend
    un letterbox rectangulaire pour une image seule (marge minimale) et les scores diffèrent
    (ex. test/140.jpg : 0,79 en rectangulaire, 0,39 en carré)."""
    print(f"\n4. Échecs sur le test (confiance {CONFIANCE}, IoU de NMS {IOU_NMS})")
    images = sorted((RACINE / "donnees" / "yolo" / "images" / "test").glob("*.jpg"), key=lambda c: c.stem.zfill(10))
    modele = YOLO(str(poids(taille)))
    nb_plaques = nb_trouvees = nb_fausses = 0
    ious_justes, echecs, reussites, ratees = [], [], [], []
    for resultat in modele.predict([str(c) for c in images], imgsz=taille, conf=CONFIANCE, iou=IOU_NMS,
                                   rect=False, device=0, verbose=False, stream=True):
        chemin = Path(resultat.path)
        image = resultat.orig_img.copy()
        vrais = lire_etiquettes(chemin, image.shape[1], image.shape[0])
        predits = resultat.boxes.xyxy.cpu().numpy()
        scores = resultat.boxes.conf.cpu().numpy()
        problemes = []
        predits_justes = set()
        for vrai in vrais:
            nb_plaques += 1
            recouvrements = iou(vrai, predits) if len(predits) else np.zeros(0)
            if len(recouvrements) and recouvrements.max() >= IOU_CORRECTE:
                nb_trouvees += 1
                ious_justes.append(recouvrements.max())
                predits_justes.add(int(recouvrements.argmax()))
            elif len(recouvrements):
                problemes.append(f"imprécise IoU {recouvrements.max():.2f}")
                ratees.append((chemin, vrai))
            else:
                problemes.append("manquée")
                ratees.append((chemin, vrai))
        fausses = [k for k in range(len(predits)) if k not in predits_justes and
                   (len(vrais) == 0 or iou(predits[k], vrais).max() < IOU_CORRECTE)]
        nb_fausses += len(fausses)
        if fausses:
            problemes.append(f"{len(fausses)} fausse(s) {', '.join(f'{scores[k]:.2f}' for k in fausses)}")

        # Vrai cadre en vert, cadres prédits en rouge avec leur score
        epaisseur = max(2, image.shape[1] // 250)
        for x1, y1, x2, y2 in vrais:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), epaisseur)
        for (x1, y1, x2, y2), score in zip(predits, scores):
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), epaisseur)
            cv2.putText(image, f"{score:.2f}", (int(x1), max(20, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        image.shape[1] / 1000, (0, 0, 255), epaisseur)
        if problemes:
            echecs.append(faire_vignette(image, 320, 240, f"{chemin.name}: {'; '.join(problemes)}"))
            print(f"   {chemin.name:9s} {'; '.join(problemes)}")
        else:
            reussites.append(faire_vignette(image, 320, 240, f"{chemin.name}"))

    print(f"   taux de détection : {nb_trouvees}/{nb_plaques} = {nb_trouvees / nb_plaques * 100:.1f} % ; "
          f"fausses détections : {nb_fausses} ; IoU moyenne des détections justes : {np.mean(ious_justes):.3f}")

    # Plaques ratées : de peu (score juste sous le seuil) ou de beaucoup ? On relance avec un seuil très bas.
    for chemin, vrai in ratees:
        resultat = modele.predict(str(chemin), imgsz=taille, conf=0.001, iou=IOU_NMS, rect=False, device=0,
                                  verbose=False)[0]
        boites, scores = resultat.boxes.xyxy.cpu().numpy(), resultat.boxes.conf.cpu().numpy()
        recouvrent = iou(vrai, boites) >= IOU_CORRECTE if len(boites) else np.zeros(0, dtype=bool)
        meilleur = f"{scores[recouvrent].max():.3f}" if recouvrent.any() else "aucune boîte, même à 0,001"
        print(f"   {chemin.name} ratée : meilleur score d'une boîte sur la vraie plaque = {meilleur}")
    if echecs:
        cv2.imwrite(str(DOSSIER_FIGURES / "echecs_test.jpg"), assembler_planche(echecs, 4))
    cv2.imwrite(str(DOSSIER_FIGURES / "exemples_detections_test.jpg"),
                assembler_planche(random.Random(0).sample(reussites, min(16, len(reussites))), 4))


def main():
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    retenue = choisir_taille()
    figure_comparaison()
    etudier_seuil(retenue)
    evaluer_test(retenue)
    analyser_echecs(retenue)
    # Le modèle retenu devient le détecteur du projet
    shutil.copy2(poids(retenue), RACINE / "modeles" / "detecteur.pt")
    shutil.copy2(DOSSIER_RUNS / f"yolov8n_{retenue}" / "results.png", DOSSIER_FIGURES / f"entrainement_{retenue}.png")
    print(f"\nModèle retenu (entrée {retenue}) copié dans modeles/detecteur.pt ; figures dans {DOSSIER_FIGURES}")


if __name__ == "__main__":
    main()
