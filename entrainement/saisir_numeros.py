"""Phase 3 : saisie de la vérité terrain, le numéro de chaque plaque du jeu Kaggle (PC uniquement).

Une fenêtre montre la plaque agrandie ; on tape le numéro directement dans cette fenêtre.
  - Chiffres : rangée du haut en AZERTY sans Maj (& é " ' ( - è _ ç à), avec Maj, ou pavé numérique.
    Attention : le « - » du pavé numérique donne le même code que la touche 6 en AZERTY, il compte comme 6.
  - Espace entre la série et le numéro : « 215 4567 » est enregistré « 215 TU 4567 ».
  - Entrée : valider ; Retour arrière : effacer un chiffre.
  - i : plaque illisible (même pour un humain) ; f : autre format (deux lignes, ancien modèle...).
  - z : revenir à la plaque précédente (de cette séance) ; Echap : quitter (tout est déjà enregistré).
Ordre de présentation (fixe) : les plaques de réglage (train + val) dans un ordre tiré au hasard, les
100 premières permettant de régler la phase 3 dès la première séance ; le test en dernier.

Fichier : donnees/verite/numeros.csv, hors de git (numéros de vraies plaques). À chaque fin de séance,
une copie datée est enregistrée dans OneDrive (dossier plaques_sauvegarde).

Utilisation, depuis la racine du projet :
    python -m entrainement.saisir_numeros
"""

import csv
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import numpy as np

from entrainement.donnees_kaggle import RACINE

DOSSIER_YOLO = RACINE / "donnees" / "yolo"
FICHIER_VERITE = RACINE / "donnees" / "verite" / "numeros.csv"
DOSSIER_SAUVEGARDE = "plaques_sauvegarde"  # sous-dossier de OneDrive
COLONNES = ["image", "partie", "statut", "numero", "date"]
GRAINE = 42
FENETRE = "Saisie des numeros"

# AZERTY, rangée du haut sans Maj -> chiffre. Ce sont les codes renvoyés par cv2.waitKey sous Windows
# (é, è, ç et à ont les codes 233, 232, 231 et 224).
AZERTY = {ord("&"): "1", 233: "2", ord('"'): "3", ord("'"): "4", ord("("): "5",
          ord("-"): "6", 232: "7", ord("_"): "8", 231: "9", 224: "0"}
TOUCHE_ENTREE, TOUCHE_ECHAP, TOUCHE_EFFACER, TOUCHE_ESPACE = 13, 27, 8, 32


# ------------------------------------------------------------ Données

def ordre_de_saisie():
    """Liste fixe de (partie, chemin) : le réglage (train + val) mélangé avec une graine fixe, puis le test."""
    def lister(partie):
        return sorted((DOSSIER_YOLO / "images" / partie).glob("*.jpg"), key=lambda c: c.stem.zfill(10))

    reglage = lister("train") + lister("val")
    test = lister("test")
    aleatoire = random.Random(GRAINE)
    aleatoire.shuffle(reglage)
    aleatoire.shuffle(test)
    return [(chemin.parent.name, chemin) for chemin in reglage + test]


def lire_saisies():
    """Saisies déjà enregistrées : {nom de l'image: ligne}. Permet de reprendre d'une séance à l'autre."""
    if not FICHIER_VERITE.exists():
        return {}
    with open(FICHIER_VERITE, encoding="utf-8", newline="") as fichier:
        return {ligne["image"]: ligne for ligne in csv.DictReader(fichier)}


def ajouter_saisie(ligne):
    """Ajoute une ligne au fichier aussitôt : rien n'est perdu si la séance s'interrompt."""
    nouveau = not FICHIER_VERITE.exists()
    with open(FICHIER_VERITE, "a", encoding="utf-8", newline="") as fichier:
        tableau = csv.DictWriter(fichier, fieldnames=COLONNES)
        if nouveau:
            tableau.writeheader()
        tableau.writerow(ligne)


def reecrire_saisies(saisies):
    """Réécrit tout le fichier (après un retour en arrière)."""
    with open(FICHIER_VERITE, "w", encoding="utf-8", newline="") as fichier:
        tableau = csv.DictWriter(fichier, fieldnames=COLONNES)
        tableau.writeheader()
        tableau.writerows(saisies.values())


def sauvegarder(suffixe=""):
    """Copie datée du fichier dans OneDrive (fin de séance) ; renvoie le chemin de la copie, ou None.
    suffixe : distingue deux copies faites dans la même seconde (avant / après une correction), sinon la
    seconde écraserait la première."""
    onedrive = os.environ.get("OneDrive")
    if not FICHIER_VERITE.exists() or not onedrive or not Path(onedrive).is_dir():
        return None
    dossier = Path(onedrive) / DOSSIER_SAUVEGARDE
    dossier.mkdir(exist_ok=True)
    copie = dossier / f"numeros_{datetime.now():%Y%m%d_%H%M%S}{suffixe}.csv"
    shutil.copy2(FICHIER_VERITE, copie)
    return copie


def normaliser(texte):
    """« 215 4567 » -> « 215 TU 4567 » ; None si le format n'est pas 1 à 3 chiffres, espace, 1 à 4 chiffres."""
    morceaux = texte.split()
    if len(morceaux) != 2:
        return None
    serie, numero = morceaux
    if serie.isdigit() and numero.isdigit() and 1 <= len(serie) <= 3 and 1 <= len(numero) <= 4:
        return f"{serie} TU {numero}"
    return None


def interpreter_touche(code, texte):
    """Applique une touche à la saisie en cours ; renvoie (texte, action).
    action : None (on continue), "valider", "illisible", "autre_format", "retour" ou "quitter"."""
    caractere = chr(code) if 32 <= code < 127 else ""
    if code == TOUCHE_ENTREE:
        return texte, "valider"
    if code == TOUCHE_ECHAP:
        return texte, "quitter"
    if code == TOUCHE_EFFACER:
        return texte[:-1], None
    if code == TOUCHE_ESPACE:
        if texte and " " not in texte:  # un seul espace, entre la série et le numéro
            texte += " "
        return texte, None
    chiffre = AZERTY.get(code) or (caractere if caractere.isdigit() else "")
    if chiffre:
        return (texte + chiffre if len(texte) < 8 else texte), None
    actions = {"i": "illisible", "f": "autre_format", "z": "retour"}
    return texte, actions.get(caractere.lower())


# ------------------------------------------------------------ Affichage

def cadre_vrai(chemin, largeur, hauteur):
    """Premier cadre annoté de l'image, remis en pixels."""
    ligne = (DOSSIER_YOLO / "labels" / chemin.parent.name / (chemin.stem + ".txt")).read_text(encoding="utf-8").split()
    _, xc, yc, l, h = (float(v) for v in ligne[:5])
    return (xc - l / 2) * largeur, (yc - h / 2) * hauteur, (xc + l / 2) * largeur, (yc + h / 2) * hauteur


def image_de_saisie(chemin, texte, message, progression):
    """Plaque agrandie en haut, image entière en bas à gauche, textes en bas à droite (ASCII : cv2.putText)."""
    image = cv2.imread(str(chemin))
    hauteur, largeur = image.shape[:2]
    x0, y0, x1, y1 = cadre_vrai(chemin, largeur, hauteur)
    marge_x, marge_y = 0.15 * (x1 - x0), 0.40 * (y1 - y0)
    decoupe = image[max(0, int(y0 - marge_y)):int(y1 + marge_y), max(0, int(x0 - marge_x)):int(x1 + marge_x)]
    rapport = min(1000 / decoupe.shape[1], 380 / decoupe.shape[0])
    plaque = cv2.resize(decoupe, (int(decoupe.shape[1] * rapport), int(decoupe.shape[0] * rapport)),
                        interpolation=cv2.INTER_CUBIC)

    ecran = np.zeros((680, 1000, 3), dtype=np.uint8)
    ecran[:plaque.shape[0], (1000 - plaque.shape[1]) // 2:(1000 - plaque.shape[1]) // 2 + plaque.shape[1]] = plaque
    apercu = image.copy()
    cv2.rectangle(apercu, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), max(2, largeur // 200))
    rapport = min(420 / largeur, 280 / hauteur)
    apercu = cv2.resize(apercu, (int(largeur * rapport), int(hauteur * rapport)), interpolation=cv2.INTER_AREA)
    ecran[680 - apercu.shape[0]:, :apercu.shape[1]] = apercu

    def ecrire(ligne_texte, y, taille=0.6, couleur=(255, 255, 255)):
        cv2.putText(ecran, ligne_texte, (440, y), cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, 1, cv2.LINE_AA)

    ecrire(progression, 420)
    ecrire("Numero : " + texte + "_", 470, 1.1, (0, 255, 255))
    ecrire(message, 510, 0.6, (0, 0, 255))
    ecrire("serie ESPACE numero, puis Entree (ex. 215 4567)", 560, 0.5, (200, 200, 200))
    ecrire("i : illisible   f : autre format (2 lignes...)", 590, 0.5, (200, 200, 200))
    ecrire("z : plaque precedente   Echap : quitter", 620, 0.5, (200, 200, 200))
    ecrire(chemin.parent.name + "/" + chemin.name, 660, 0.45, (150, 150, 150))
    return ecran


def saisir_une_plaque(chemin, progression):
    """Boucle de saisie d'une plaque ; renvoie (action, numero)."""
    texte, message = "", ""
    while True:
        cv2.imshow(FENETRE, image_de_saisie(chemin, texte, message, progression))
        code = cv2.waitKey(50)
        if code == -1:
            # Fenêtre fermée avec la croix : on quitte comme avec Echap
            if cv2.getWindowProperty(FENETRE, cv2.WND_PROP_VISIBLE) < 1:
                return "quitter", ""
            continue
        texte, action = interpreter_touche(code, texte)
        if action == "valider":
            numero = normaliser(texte)
            if numero:
                return "numero", numero
            message = "Format invalide : 1 a 3 chiffres, espace, 1 a 4 chiffres"
        elif action:
            return action, ""
        else:
            message = ""


def main():
    FICHIER_VERITE.parent.mkdir(parents=True, exist_ok=True)
    ordre = ordre_de_saisie()
    saisies = lire_saisies()
    a_faire = [(partie, chemin) for partie, chemin in ordre if chemin.name not in saisies]
    historique = []  # noms saisis pendant cette séance, pour revenir en arrière
    indice = 0
    cv2.namedWindow(FENETRE, cv2.WINDOW_AUTOSIZE)
    while indice < len(a_faire):
        partie, chemin = a_faire[indice]
        nb_reglage = sum(1 for ligne in saisies.values() if ligne["partie"] != "test")
        progression = (f"{len(saisies) + 1}/{len(ordre)} ({'test' if partie == 'test' else 'reglage'})"
                       f" | reglage saisi : {nb_reglage} | seance : {len(historique)}")
        action, numero = saisir_une_plaque(chemin, progression)
        if action == "quitter":
            break
        if action == "retour":
            if historique:
                del saisies[historique.pop()]
                reecrire_saisies(saisies)
                indice -= 1
            continue
        statut = "numero" if action == "numero" else action
        ligne = {"image": chemin.name, "partie": partie, "statut": statut, "numero": numero,
                 "date": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
        saisies[chemin.name] = ligne
        ajouter_saisie(ligne)
        historique.append(chemin.name)
        indice += 1
    cv2.destroyAllWindows()

    statuts = {}
    for ligne in saisies.values():
        statuts[ligne["statut"]] = statuts.get(ligne["statut"], 0) + 1
    print(f"Séance : {len(historique)} plaque(s) saisie(s) ; total : {len(saisies)}/{len(ordre)} ; {statuts}")
    copie = sauvegarder()
    print(f"Sauvegarde : {copie}" if copie else "Pas de sauvegarde (OneDrive introuvable ou aucune saisie)")


if __name__ == "__main__":
    main()
