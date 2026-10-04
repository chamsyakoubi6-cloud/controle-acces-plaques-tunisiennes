"""Vérification et correction de la vérité terrain saisie (donnees/verite/numeros.csv) (PC).

  --voisines   : fautes de saisie probables, repérées SANS aucun modèle. Une même voiture est souvent
                 photographiée plusieurs fois de suite : deux images voisines (numéros de fichier proches)
                 dont les numéros saisis diffèrent d'un chiffre, ou par l'inversion de deux chiffres voisins,
                 sont suspectes. Planche des paires : docs/figures/phase4/saisie_voisines.jpg.
  --planche P  : planches de relecture de toute une partie (reglage ou test) : chaque plaque agrandie avec
                 son numéro saisi, à relire à l'oeil. Aucun modèle : le test reste intact.
  --corriger image=valeur ... : corrections décidées à l'oeil sur l'image ; valeur = « 215 TU 4567 »,
                 « illisible » ou « autre_format ». Copie datée dans OneDrive AVANT la modification (pour
                 pouvoir revenir en arrière), puis réécriture du fichier et nouvelle copie datée.

Le fichier reste hors de git (numéros de vraies plaques) : le journal ne cite que les noms d'images.

Utilisation, depuis la racine du projet :
    python -m entrainement.verifier_saisie --voisines
    python -m entrainement.verifier_saisie --planche test
    python -m entrainement.verifier_saisie --corriger train/184.jpg="215 TU 4567" val/483.jpg=illisible
"""

import argparse
from datetime import datetime
from pathlib import Path

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import numpy as np

from entrainement.donnees_kaggle import RACINE, assembler_planche, faire_vignette
from entrainement.evaluer_traitement import DOSSIER_YOLO, cadre_vrai
from entrainement.saisir_numeros import lire_saisies, normaliser, reecrire_saisies, sauvegarder

DOSSIER_FIGURES = RACINE / "docs" / "figures" / "phase4"
ECART_IMAGES = 3      # photos consécutives : numéros de fichier distants d'au plus 3
DISTANCE_MAX = 2      # au plus 2 opérations : un chiffre changé (1), deux chiffres voisins inversés (1)...
PAR_PLANCHE = 48      # plaques par planche de relecture (4 colonnes x 12 lignes)


def distance_edition(a, b):
    """Nombre minimal de suppressions, insertions, substitutions ou inversions de deux caractères voisins
    pour passer du texte a au texte b (distance de Damerau-Levenshtein, version simple)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def paires_suspectes(saisies):
    """Paires d'images voisines d'une même partie (réglage ou test) aux numéros proches mais différents."""
    paires = []
    for parties in (("train", "val"), ("test",)):
        lignes = sorted(((int(Path(l["image"]).stem), l) for l in saisies.values()
                         if l["partie"] in parties and l["statut"] == "numero"), key=lambda e: e[0])
        for k, (n1, l1) in enumerate(lignes):
            for n2, l2 in lignes[k + 1:]:
                if n2 - n1 > ECART_IMAGES:
                    break
                a, b = l1["numero"].replace(" TU ", "|"), l2["numero"].replace(" TU ", "|")
                if a != b and distance_edition(a, b) <= DISTANCE_MAX:
                    paires.append((l1, l2))
    return paires


def case(ligne, largeur=420, hauteur=120):
    """Plaque agrandie (autour du cadre vrai) avec son nom d'image et sa saisie."""
    chemin = DOSSIER_YOLO / "images" / ligne["partie"] / ligne["image"]
    image = cv2.imread(str(chemin))
    x1, y1, x2, y2 = cadre_vrai(chemin, image.shape[1], image.shape[0])
    marge_x, marge_y = 0.15 * (x2 - x1), 0.40 * (y2 - y1)
    zone = image[max(0, int(y1 - marge_y)):int(y2 + marge_y), max(0, int(x1 - marge_x)):int(x2 + marge_x)]
    rapport = min(largeur / zone.shape[1], hauteur / zone.shape[0])
    if rapport > 1:                              # agrandissement lissé, plus lisible que des pixels carrés
        zone = cv2.resize(zone, None, fx=rapport, fy=rapport, interpolation=cv2.INTER_CUBIC)
    valeur = ligne["numero"] if ligne["statut"] == "numero" else ligne["statut"]
    bandeau = faire_vignette(np.zeros((22, largeur, 3), dtype=np.uint8), largeur, 22,
                             f"{ligne['partie']}/{ligne['image']} : {valeur}")
    return np.vstack([bandeau, faire_vignette(zone, largeur, hauteur)])


def voisines():
    paires = paires_suspectes(lire_saisies())
    print(f"Paires d'images voisines aux numéros proches mais différents : {len(paires)}")
    for l1, l2 in paires:
        print(f"   {l1['partie']}/{l1['image']} {l1['numero']:14s} <-> {l2['partie']}/{l2['image']} {l2['numero']}")
    if paires:
        DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
        cases = [c for l1, l2 in paires for c in (case(l1), case(l2))]
        cv2.imwrite(str(DOSSIER_FIGURES / "saisie_voisines.jpg"), assembler_planche(cases, 4))
        print(f"Planche : {DOSSIER_FIGURES / 'saisie_voisines.jpg'} (une paire par demi-ligne)")


def planches(partie):
    parties = ("train", "val") if partie == "reglage" else ("test",)
    lignes = sorted((l for l in lire_saisies().values() if l["partie"] in parties),
                    key=lambda l: (l["partie"], int(Path(l["image"]).stem)))
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(lignes), PAR_PLANCHE):
        chemin = DOSSIER_FIGURES / f"relecture_{partie}_{k // PAR_PLANCHE + 1}.jpg"
        cv2.imwrite(str(chemin), assembler_planche([case(l, 320, 100) for l in lignes[k:k + PAR_PLANCHE]], 4))
        print(f"   {chemin.name} : {lignes[k]['partie']}/{lignes[k]['image']} ... "
              f"{lignes[min(k + PAR_PLANCHE, len(lignes)) - 1]['partie']}/{lignes[min(k + PAR_PLANCHE, len(lignes)) - 1]['image']}")


def corriger(demandes):
    saisies = lire_saisies()
    corrections = []
    for demande in demandes:                     # tout est vérifié avant de modifier quoi que ce soit
        image, _, valeur = demande.partition("=")
        image = image.split("/")[-1]
        if image not in saisies:
            raise SystemExit(f"Image inconnue dans la saisie : {image}")
        if valeur in ("illisible", "autre_format"):
            corrections.append((image, valeur, ""))
        else:
            numero = normaliser(valeur.replace("TU", " "))
            if numero is None:
                raise SystemExit(f"Format invalide pour {image} : {valeur!r} (attendu : 215 TU 4567)")
            corrections.append((image, "numero", numero))
    copie = sauvegarder("_avant_correction")
    if copie is None:
        raise SystemExit("Copie de sauvegarde impossible (OneDrive introuvable) : rien n'est modifié")
    print(f"Copie avant correction : {copie}")
    for image, statut, numero in corrections:
        ligne = saisies[image]
        ancien = ligne["numero"] if ligne["statut"] == "numero" else ligne["statut"]
        print(f"   {ligne['partie']}/{image} : {ancien} -> {numero if statut == 'numero' else statut}")
        ligne.update({"statut": statut, "numero": numero, "date": f"{datetime.now():%Y-%m-%d %H:%M:%S}"})
    reecrire_saisies(saisies)
    print(f"{len(corrections)} correction(s) enregistrée(s) ; copie après correction : "
          f"{sauvegarder('_apres_correction')}")


def main():
    parseur = argparse.ArgumentParser(description="Vérification et correction de la vérité terrain saisie.")
    parseur.add_argument("--voisines", action="store_true", help="fautes probables entre images voisines")
    parseur.add_argument("--planche", choices=("reglage", "test"), help="planches de relecture d'une partie")
    parseur.add_argument("--corriger", nargs="+", metavar="IMAGE=VALEUR", help="corrections décidées à l'oeil")
    arguments = parseur.parse_args()
    if arguments.corriger:
        corriger(arguments.corriger)
    if arguments.voisines:
        voisines()
    if arguments.planche:
        planches(arguments.planche)


if __name__ == "__main__":
    main()
