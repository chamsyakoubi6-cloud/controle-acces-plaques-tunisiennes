"""Phase 3, étape 6 : réglage des paramètres du traitement classique, justifiés par des mesures (PC).

Les mesures se font sur les plaques de RÉGLAGE saisies (train + val), jamais sur le test. Figures dans
docs/figures/phase3/ (non versionnées : elles montrent des plaques).
  --balayage : taux de segmentation correcte selon le seuillage (Otsu ou adaptatif), la taille de bloc
               et la constante C, avec ou sans CLAHE, le noyau de nettoyage, l'effacement du liseré et le
               canal gris. Chaque plaque n'est redressée qu'une fois ; seule la binarisation varie.
  --mesures  : distributions mesurées sur les chiffres des plaques bien segmentées (hauteur, largeur /
               hauteur, centre vertical), comparées aux autres candidats, et épaisseur des traits
               (transformée de distance) : justifient les filtres et la taille des noyaux.
  --bleu     : fond bleu simulé (plaques de location, absentes du jeu) : canal gris « luminance »
               contre « minimum ». Les vraies photos de location servent ensuite de vérification seule.
  --etapes   : figures étape par étape pour les plaques nommées (ex. --etapes 525,295,159).

Utilisation, depuis la racine du projet :
    python -m entrainement.regler_traitement --balayage --mesures --bleu
"""

import argparse

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import matplotlib

matplotlib.use("Agg")  # figures enregistrées dans des fichiers, sans fenêtre
import matplotlib.pyplot as plt
import numpy as np

from entrainement.donnees_kaggle import assembler_planche, faire_vignette
from entrainement.evaluer_traitement import DOSSIER_FIGURES, DOSSIER_YOLO, cadre_vrai, lire_plaques
from systeme import binarisation
from systeme.binarisation import binariser
from systeme.debug import Debug
from systeme.redressement import decouper, redresser, trouver_coins
from systeme.segmentation import segmenter
from systeme.traitement import traiter_plaque

TAILLES_BLOC = (15, 21, 31, 41, 51, 61, 81)
CONSTANTES = (-20, -15, -10, -5, 0)
BLEUS = {"bleu fonce": (150, 60, 10), "bleu clair": (200, 120, 40)}  # couleurs BGR du fond simulé


def redresser_toutes(plaques):
    """Redressement unique de chaque plaque de réglage (il ne dépend pas des réglages de binarisation)."""
    redressees = []
    for plaque in plaques:
        image = cv2.imread(str(plaque["chemin"]))
        decoupe, _, cadre = decouper(image, cadre_vrai(plaque["chemin"], image.shape[1], image.shape[0]))
        coins, _ = trouver_coins(decoupe, cadre)
        redressees.append((plaque, redresser(decoupe, coins)))
    return redressees


def correcte(plaque, segmentation):
    return len(segmentation["serie"]) == len(plaque["serie"]) and len(segmentation["numero"]) == len(plaque["numero"])


def taux(redressees, **reglages):
    """Taux de segmentation correcte pour des réglages de binarisation donnés."""
    bonnes = 0
    for plaque, image in redressees:
        binaire, gris = binariser(image, **reglages)
        bonnes += correcte(plaque, segmenter(binaire, gris))
    return bonnes / len(redressees)


# ---------------------------------------------------------------- Balayage

def balayage(redressees):
    print("\n=== Balayage des réglages de binarisation (plaques de réglage) ===")
    reference = {"gris": binarisation.GRIS, "clahe": binarisation.CLAHE, "noyau": binarisation.NOYAU,
                 "lisere": binarisation.LISERE}
    grille = np.zeros((len(CONSTANTES), len(TAILLES_BLOC)))
    for i, constante in enumerate(CONSTANTES):
        for j, taille in enumerate(TAILLES_BLOC):
            grille[i, j] = taux(redressees, seuillage="adaptatif", taille_bloc=taille, constante=constante, **reference)
    otsu = taux(redressees, seuillage="otsu", **reference)
    i, j = np.unravel_index(int(grille.argmax()), grille.shape)
    meilleur = {"seuillage": "adaptatif", "taille_bloc": TAILLES_BLOC[j], "constante": CONSTANTES[i]}
    print(f"   Otsu : {otsu * 100:.1f} % ; meilleur adaptatif : bloc {TAILLES_BLOC[j]}, C = {CONSTANTES[i]} -> "
          f"{grille[i, j] * 100:.1f} %")
    print("   adaptatif, taux (%) : lignes = C, colonnes = taille de bloc", TAILLES_BLOC)
    for constante, ligne in zip(CONSTANTES, grille):
        print(f"      C = {constante:3d} : " + "  ".join(f"{v * 100:5.1f}" for v in ligne))

    # Variantes, autour du meilleur seuillage : un seul facteur change à la fois
    variantes = {
        "réglage de référence": {},
        "sans CLAHE": {"clahe": False},
        "sans effacement du liseré": {"lisere": False},
        "sans nettoyage (noyau 0)": {"noyau": 0},
        "nettoyage 3x3": {"noyau": 3},
        "canal gris « minimum »": {"gris": "minimum"},
    }
    resultats_variantes = {}
    for nom, changement in variantes.items():
        reglages = dict(reference, **meilleur)
        reglages.update(changement)
        resultats_variantes[nom] = taux(redressees, **reglages)
        print(f"   {nom:28s}: {resultats_variantes[nom] * 100:5.1f} %")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for constante, ligne in zip(CONSTANTES, grille):
        axes[0].plot(TAILLES_BLOC, ligne * 100, marker="o", label=f"adaptatif, C = {constante}")
    axes[0].axhline(otsu * 100, color="black", linestyle="--", label="Otsu (seuil global)")
    axes[0].set(title="Segmentation correcte selon le seuillage", xlabel="taille de bloc (px)",
                ylabel="plaques correctement segmentées (%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    noms = list(resultats_variantes)
    axes[1].barh(noms, [resultats_variantes[n] * 100 for n in noms], color="tab:blue")
    axes[1].set(title="Effet de chaque étape (un facteur à la fois)", xlabel="plaques correctement segmentées (%)")
    axes[1].invert_yaxis()
    figure.tight_layout()
    figure.savefig(DOSSIER_FIGURES / "balayage_binarisation.png", dpi=120)
    plt.close(figure)
    return meilleur


# ----------------------------------------------------------------- Mesures

def mesures(redressees):
    """Géométrie des chiffres (plaques bien segmentées) et des autres candidats ; épaisseur des traits."""
    print("\n=== Mesures sur les chiffres des plaques bien segmentées ===")
    chiffres, autres, epaisseurs = [], [], []
    for plaque, image in redressees:
        binaire, gris = binariser(image)
        segmentation = segmenter(binaire, gris)
        if not correcte(plaque, segmentation):
            continue
        hauteur = binaire.shape[0]
        for blob in segmentation["candidats"]:
            valeurs = (blob["h"] / hauteur, blob["l"] / blob["h"], (blob["y"] + blob["h"] / 2) / hauteur)
            (chiffres if blob["chiffre"] else autres).append(valeurs)
            if blob["chiffre"]:
                masque = binaire[blob["y"]:blob["y"] + blob["h"], blob["x"]:blob["x"] + blob["l"]]
                # Transformée de distance : distance de chaque pixel blanc au noir le plus proche ;
                # sur l'axe d'un trait, elle vaut la demi-épaisseur du trait.
                distance = cv2.distanceTransform(masque, cv2.DIST_L2, 3)
                epaisseurs.append(2 * float(np.percentile(distance[distance > 0], 90)))
    chiffres, autres = np.array(chiffres), np.array(autres)
    noms = ("hauteur / hauteur de la plaque", "largeur / hauteur", "centre vertical / hauteur de la plaque")
    for k, nom in enumerate(noms):
        print(f"   {nom:40s} chiffres : p1 {np.percentile(chiffres[:, k], 1):.2f}, médiane "
              f"{np.median(chiffres[:, k]):.2f}, p99 {np.percentile(chiffres[:, k], 99):.2f}")
    print(f"   épaisseur des traits (px) : p10 {np.percentile(epaisseurs, 10):.1f}, médiane {np.median(epaisseurs):.1f}, "
          f"p90 {np.percentile(epaisseurs, 90):.1f} ({len(chiffres)} chiffres, {len(autres)} autres candidats)")

    figure, axes = plt.subplots(1, 4, figsize=(17, 4))
    for k, nom in enumerate(noms):
        axes[k].hist(chiffres[:, k], bins=30, alpha=0.7, label="chiffres")
        if len(autres):
            axes[k].hist(autres[:, k], bins=30, alpha=0.6, label="autres candidats (arabe, vis...)")
        axes[k].set(title=nom, ylabel="nombre")
        axes[k].legend(fontsize=8)
    axes[3].hist(epaisseurs, bins=20, color="tab:green")
    axes[3].set(title="Épaisseur des traits des chiffres (px)", ylabel="nombre")
    figure.tight_layout()
    figure.savefig(DOSSIER_FIGURES / "mesures_chiffres.png", dpi=120)
    plt.close(figure)


# ------------------------------------------------------------- Fond bleu

def simuler_fond_bleu(image, couleur, masque=None):
    """Recolore en bleu le fond d'une plaque redressée (les caractères gardent leurs pixels), en conservant les
    variations d'éclairage du fond. masque : pixels des caractères (calculé si absent : binarisation dilatée)."""
    if masque is None:
        masque = cv2.dilate(binariser(image)[0], np.ones((3, 3), dtype=np.uint8))
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    eclairage = gris / max(float(gris[masque == 0].mean()), 1.0)  # variations d'éclairage du fond
    bleue = image.copy()
    fond = masque == 0
    bleue[fond] = np.clip(np.array(couleur, dtype=np.float32) * eclairage[fond][:, np.newaxis], 0, 255).astype(np.uint8)
    return bleue


def fond_bleu(redressees):
    """Recolore en bleu le fond des plaques bien segmentées (les caractères gardent leurs pixels), en
    conservant les variations d'éclairage du fond, puis compare les deux canaux gris."""
    print("\n=== Fond bleu simulé (plaques bien segmentées sur fond noir) ===")
    retenues = []
    for plaque, image in redressees:
        binaire, gris = binariser(image)
        if correcte(plaque, segmenter(binaire, gris)):
            retenues.append((plaque, image, cv2.dilate(binaire, np.ones((3, 3), dtype=np.uint8))))
    exemples = []
    for nom_bleu, couleur in BLEUS.items():
        for canal in ("luminance", "minimum"):
            bonnes = 0
            for k, (plaque, image, masque) in enumerate(retenues):
                bleue = simuler_fond_bleu(image, couleur, masque)
                binaire, gris_utilise = binariser(bleue, gris=canal)
                bonnes += correcte(plaque, segmenter(binaire, gris_utilise))
                if k < 3 and canal == "luminance":
                    exemples.append(faire_vignette(bleue, 450, 100, nom_bleu))
            print(f"   {nom_bleu:10s} - canal {canal:9s}: {bonnes}/{len(retenues)} = {bonnes / len(retenues) * 100:5.1f} %")
    cv2.imwrite(str(DOSSIER_FIGURES / "fond_bleu_simule.jpg"), assembler_planche(exemples, 3))


# ----------------------------------------------------------------- Étapes

def etapes(noms):
    for nom in noms:
        chemins = list(DOSSIER_YOLO.glob(f"images/*/{nom}.jpg"))
        if not chemins:
            print(f"   {nom}.jpg introuvable")
            continue
        image = cv2.imread(str(chemins[0]))
        debug = Debug()
        traiter_plaque(image, cadre_vrai(chemins[0], image.shape[1], image.shape[0]), debug)
        cv2.imwrite(str(DOSSIER_FIGURES / f"etapes_{nom}.png"), debug.planche(450))
        print(f"   figure étape par étape : etapes_{nom}.png")


def main():
    parseur = argparse.ArgumentParser(description="Réglage du traitement classique, justifié par des mesures.")
    parseur.add_argument("--balayage", action="store_true")
    parseur.add_argument("--mesures", action="store_true")
    parseur.add_argument("--bleu", action="store_true")
    parseur.add_argument("--etapes", default="", help="noms d'images séparés par des virgules (ex. 525,295)")
    arguments = parseur.parse_args()
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)

    if arguments.etapes:
        etapes([nom.strip() for nom in arguments.etapes.split(",") if nom.strip()])
    if arguments.balayage or arguments.mesures or arguments.bleu:
        plaques, _ = lire_plaques("reglage")
        if not plaques:
            print("Aucune plaque de réglage saisie : lancer d'abord python -m entrainement.saisir_numeros")
            return
        print(f"{len(plaques)} plaques de réglage saisies")
        redressees = redresser_toutes(plaques)
        if arguments.balayage:
            balayage(redressees)
        if arguments.mesures:
            mesures(redressees)
        if arguments.bleu:
            fond_bleu(redressees)


if __name__ == "__main__":
    main()
