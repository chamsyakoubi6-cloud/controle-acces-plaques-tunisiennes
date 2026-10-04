"""Données de validation de la phase 5 : noms des fichiers, inventaire et découpage par véhicule (PC et Jetson).

Le nom de chaque fichier donne le vrai numéro (voir validation/README.md) :
    215TU4567.mp4    215TU4567_1.jpg    215TU4567_bleu_2.jpg    215TU4567 (2).jpeg
Après le numéro viennent des étiquettes, séparées par « _ ». Deux sont reconnues : « bleu » (plaque de location,
fond bleu) et « arriere » (plaque arrière). Les autres (« 1 », « nuit »...) servent seulement à distinguer les
prises. « (2) » est la variante que Windows ajoute à un nom déjà pris.

Le découpage se fait PAR VÉHICULE (même numéro = même voiture), jamais par fichier : toutes les prises d'une
voiture vont dans la même partie. Sinon, le test contiendrait des voitures déjà vues pendant les réglages : il
mesurerait la mémoire du système plutôt que sa capacité à lire une voiture nouvelle (même leçon qu'en phase 4).
  developpement : toutes les décisions de la phase 5 (vote, seuils, options du traitement) ;
  test          : le système retenu, évalué une seule fois, à la fin.
Le tirage au sort est stratifié par fond (noir, bleu), pour que chaque partie reçoive autant de plaques bleues,
qui sont rares. Dans chaque fond, les véhicules nouveaux sont pris dans un ordre tiré au sort. Chacun va dans la
partie qui a le moins de véhicules tirés de ce fond ; à égalité, la partie est tirée au sort.
Le tirage est reproductible (graine fixe) et stable : un véhicule déjà attribué ne change jamais de partie, et les
véhicules filmés plus tard sont tirés à leur tour. Le découpage est enregistré dans validation/decoupage.csv,
hors de git comme tout le dossier.

Utilisation, depuis la racine du projet :
    python -m outils.validation               inventaire et contrôles (noms, fichiers lisibles, parties)
    python -m outils.validation --decouper    tire au sort la partie des véhicules nouveaux
    python -m outils.validation --decouper --forcer-developpement "vus avant le protocole"
                                              met tous les véhicules nouveaux en développement (une seule fois,
                                              pour les véhicules photographiés avant le protocole)

Compatible Python 3.6, OpenCV 4.1.1 (PC et Jetson).
"""

import argparse
import csv
import os
import random
import re
import sys
import time

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2

from systeme.decision import format_valide

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_VALIDATION = os.path.join(RACINE, "validation")
DOSSIERS = (os.path.join(DOSSIER_VALIDATION, "photos"), os.path.join(DOSSIER_VALIDATION, "videos"))
FICHIER_DECOUPAGE = os.path.join(DOSSIER_VALIDATION, "decoupage.csv")
NATURES = {".jpg": "photo", ".jpeg": "photo", ".png": "photo", ".mp4": "video", ".mov": "video"}
# Série, « TU », numéro, étiquettes facultatives, variante « (2) » de Windows facultative
MOTIF_NOM = re.compile(r"^(\d{1,3})TU(\d{1,4})((?:_[A-Za-z0-9]+)*)(?: \(\d+\))?$", re.IGNORECASE)
FICHIERS_SYSTEME = ("desktop.ini", "thumbs.db")
PARTIES = ("developpement", "test")
COLONNES = ("numero", "partie", "fond", "raison", "date")
RAISON_TIRAGE = "tirage"
GRAINE = 2026
# Contrôles des vidéos (protocole : paysage, 30 images/s, 10 à 15 s)
IMAGES_PAR_SECONDE_MIN = 20
DUREE_MIN = 4.0


def lire_nom(nom_fichier):
    """Nom de fichier -> dictionnaire (numero « 215 TU 4567 », nature « photo » ou « video », etiquettes, bleu,
    arriere), ou None si le nom ne suit pas la convention (y compris un numéro hors du format officiel)."""
    racine, extension = os.path.splitext(nom_fichier)
    nature = NATURES.get(extension.lower())
    trouve = MOTIF_NOM.match(racine)
    if nature is None or trouve is None:
        return None
    serie, numero = trouve.group(1), trouve.group(2)
    if not format_valide(serie, numero):
        return None
    etiquettes = [e.lower() for e in trouve.group(3).split("_") if e]
    return {"numero": serie + " TU " + numero, "nature": nature, "etiquettes": etiquettes,
            "bleu": "bleu" in etiquettes, "arriere": "arriere" in etiquettes}


def lire_inventaire():
    """Fichiers de validation/photos et validation/videos, regroupés par véhicule.
    Renvoie (vehicules, invalides) : vehicules[numero] = {"photos": [chemins], "videos": [chemins],
    "fond": « noir » ou « bleu », "fonds_incoherents": bool, "arriere": bool} ; invalides : noms hors convention."""
    vehicules, invalides = {}, []
    for dossier in DOSSIERS:
        if not os.path.isdir(dossier):
            continue
        for nom in sorted(os.listdir(dossier)):
            chemin = os.path.join(dossier, nom)
            if not os.path.isfile(chemin) or nom.startswith(".") or nom.lower() in FICHIERS_SYSTEME:
                continue
            info = lire_nom(nom)
            if info is None:
                invalides.append(os.path.join(os.path.basename(dossier), nom))
                continue
            vehicule = vehicules.setdefault(info["numero"], {"photos": [], "videos": [], "fonds": set(),
                                                             "arriere": False})
            vehicule[info["nature"] + "s"].append(chemin)
            vehicule["fonds"].add("bleu" if info["bleu"] else "noir")
            vehicule["arriere"] = vehicule["arriere"] or info["arriere"]
    for vehicule in vehicules.values():
        # Une seule prise étiquetée « bleu » suffit : l'étiquette a pu être oubliée sur les autres
        vehicule["fond"] = "bleu" if "bleu" in vehicule["fonds"] else "noir"
        vehicule["fonds_incoherents"] = len(vehicule.pop("fonds")) > 1
    return vehicules, invalides


def lire_decoupage():
    """validation/decoupage.csv -> {numero: ligne} (vide si le fichier n'existe pas encore)."""
    if not os.path.exists(FICHIER_DECOUPAGE):
        return {}
    with open(FICHIER_DECOUPAGE, encoding="utf-8", newline="") as fichier:
        return {ligne["numero"]: ligne for ligne in csv.DictReader(fichier)}


def ecrire_decoupage(decoupage):
    with open(FICHIER_DECOUPAGE, "w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=COLONNES)
        ecrivain.writeheader()
        for numero in sorted(decoupage):
            ecrivain.writerow({c: decoupage[numero][c] for c in COLONNES})


def tirer_parties(nouveaux, fonds, decoupage):
    """Tire au sort la partie de chaque véhicule nouveau (voir l'en-tête) ; complète decoupage sur place.
    Seuls les véhicules tirés au sort comptent pour l'équilibre : ceux mis d'office en développement (vus avant
    le protocole) n'ont pas de vidéo, et les deux parties doivent avoir autant de véhicules filmés."""
    hasard = random.Random(GRAINE)
    ordre = sorted(nouveaux)
    hasard.shuffle(ordre)
    for numero in ordre:
        fond = fonds[numero]
        tires = {partie: sum(1 for v in decoupage.values()
                             if v["raison"] == RAISON_TIRAGE and v["fond"] == fond and v["partie"] == partie)
                 for partie in PARTIES}
        if tires["developpement"] == tires["test"]:
            partie = hasard.choice(PARTIES)
        else:
            partie = min(PARTIES, key=lambda p: tires[p])
        decoupage[numero] = {"numero": numero, "partie": partie, "fond": fond, "raison": RAISON_TIRAGE,
                             "date": time.strftime("%Y-%m-%d")}


def controler_video(chemin):
    """(problemes, rotation, description) d'une vidéo : problèmes (liste de textes, vide si tout va bien), rotation
    enregistrée par le téléphone (degrés) et description courte.
    Rotation : un iPhone tenu en paysage « à l'envers » enregistre 180 degrés. OpenCV 4.11 (PC, où se fait
    l'évaluation) l'applique ; OpenCV 4.1.1 (Jetson) ne la connaît pas : simple remarque."""
    capture = cv2.VideoCapture(chemin)
    if not capture.isOpened():
        return ["illisible (format ou codec ?)"], 0, ""
    images_par_seconde = capture.get(cv2.CAP_PROP_FPS)
    nombre = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    propriete_rotation = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    rotation = capture.get(propriete_rotation) if propriete_rotation is not None else 0
    lu, image = capture.read()
    capture.release()
    if not lu:
        return ["aucune image décodée (codec ?)"], rotation, ""
    hauteur, largeur = image.shape[:2]
    duree = nombre / images_par_seconde if images_par_seconde > 0 else 0.0
    problemes = []
    if hauteur > largeur:
        problemes.append("filmée en portrait (le protocole demande le paysage)")
    if images_par_seconde < IMAGES_PAR_SECONDE_MIN:
        problemes.append("{:.0f} images/s seulement".format(images_par_seconde))
    if duree < DUREE_MIN:
        problemes.append("trop courte ({:.1f} s)".format(duree))
    return problemes, rotation, "{}x{}, {:.0f} images/s, {:.1f} s".format(largeur, hauteur, images_par_seconde, duree)


def controler_photo(chemin):
    """(problemes, portrait) d'une photo. Une photo en portrait reste utilisable : simple remarque."""
    image = cv2.imread(chemin)
    if image is None:
        return ["illisible (format ? HEIC non pris en charge : choisir JPEG)"], False
    return [], image.shape[0] > image.shape[1]


def afficher_inventaire(vehicules, invalides, decoupage):
    nb_photos = sum(len(v["photos"]) for v in vehicules.values())
    nb_videos = sum(len(v["videos"]) for v in vehicules.values())
    print("Inventaire de validation/ : {} véhicules, {} vidéos, {} photos".format(len(vehicules), nb_videos, nb_photos))
    print("\npartie          véhicules  dont bleus  avec vidéo  vidéos  photos")
    groupes = [(p, [n for n in vehicules if n in decoupage and decoupage[n]["partie"] == p]) for p in PARTIES]
    groupes.append(("sans partie", [n for n in vehicules if n not in decoupage]))
    for nom, numeros in groupes:
        print("{:14s}  {:9d}  {:10d}  {:10d}  {:6d}  {:6d}".format(
            nom, len(numeros), sum(vehicules[n]["fond"] == "bleu" for n in numeros),
            sum(len(vehicules[n]["videos"]) > 0 for n in numeros), sum(len(vehicules[n]["videos"]) for n in numeros),
            sum(len(vehicules[n]["photos"]) for n in numeros)))
    if groupes[-1][1]:
        print("  -> attribuer une partie : python -m outils.validation --decouper")
    sans_fichier = sorted(n for n in decoupage if n not in vehicules)
    if sans_fichier:
        print("\nVéhicules du découpage sans aucun fichier (numéro corrigé ?) : {}".format(", ".join(sans_fichier)))
    if invalides:
        print("\nNoms hors convention (ignorés) :")
        for nom in invalides:
            print("   ", nom)
    for numero in sorted(vehicules):
        vehicule = vehicules[numero]
        if vehicule["fonds_incoherents"]:
            print("Attention : {} a des prises avec et sans l'étiquette « bleu »".format(numero))
        if numero in decoupage and decoupage[numero]["fond"] != vehicule["fond"]:
            print("Attention : {} est « {} » dans le découpage mais « {} » d'après les noms".format(
                numero, decoupage[numero]["fond"], vehicule["fond"]))


def controler_fichiers(vehicules):
    """Ouvre chaque fichier : un format illisible (HEIC, HEVC non décodé...) se découvre ici, pas pendant la mesure."""
    nb_problemes = nb_portraits = nb_tournees = 0
    descriptions = []
    for numero in sorted(vehicules):
        for chemin in vehicules[numero]["videos"]:
            problemes, rotation, description = controler_video(chemin)
            descriptions.append(description)
            nb_tournees += rotation != 0
            for probleme in problemes:
                print("   {} : {}".format(os.path.basename(chemin), probleme))
            nb_problemes += len(problemes)
        for chemin in vehicules[numero]["photos"]:
            problemes, portrait = controler_photo(chemin)
            for probleme in problemes:
                print("   {} : {}".format(os.path.basename(chemin), probleme))
            nb_problemes += len(problemes)
            nb_portraits += portrait
    print("Contrôle des fichiers : {} problème(s)".format(nb_problemes))
    if nb_portraits:
        print("Remarque : {} photo(s) en portrait (le protocole demande le paysage)".format(nb_portraits))
    if nb_tournees:
        print("Remarque : {} vidéo(s) avec une rotation enregistrée, appliquée sur le PC (pas par l'OpenCV 4.1.1 "
              "de la Jetson)".format(nb_tournees))
    if descriptions:
        print("Vidéos : " + " ; ".join(sorted(set(d for d in descriptions if d))[:10]))
    return nb_problemes


def main():
    parseur = argparse.ArgumentParser(description="Données de validation : inventaire et découpage par véhicule.")
    parseur.add_argument("--decouper", action="store_true", help="attribuer une partie aux véhicules nouveaux")
    parseur.add_argument("--forcer-developpement", metavar="RAISON",
                         help="avec --decouper : tous les véhicules nouveaux en développement, pour cette raison")
    arguments = parseur.parse_args()
    if arguments.forcer_developpement and not arguments.decouper:
        parseur.error("--forcer-developpement s'utilise avec --decouper")

    vehicules, invalides = lire_inventaire()
    decoupage = lire_decoupage()
    if arguments.decouper:
        nouveaux = sorted(n for n in vehicules if n not in decoupage)
        if not nouveaux:
            print("Aucun véhicule nouveau : découpage inchangé")
        elif arguments.forcer_developpement:
            for numero in nouveaux:
                decoupage[numero] = {"numero": numero, "partie": "developpement", "fond": vehicules[numero]["fond"],
                                     "raison": arguments.forcer_developpement, "date": time.strftime("%Y-%m-%d")}
            ecrire_decoupage(decoupage)
            print("{} véhicule(s) nouveau(x) mis en développement ({})".format(len(nouveaux),
                                                                            arguments.forcer_developpement))
        else:
            tirer_parties(nouveaux, {n: vehicules[n]["fond"] for n in nouveaux}, decoupage)
            ecrire_decoupage(decoupage)
            print("Tirage au sort de {} véhicule(s) nouveau(x) : {} en développement, {} en test".format(
                len(nouveaux), sum(decoupage[n]["partie"] == "developpement" for n in nouveaux),
                sum(decoupage[n]["partie"] == "test" for n in nouveaux)))
        print()
    afficher_inventaire(vehicules, invalides, decoupage)
    print()
    nb_problemes = controler_fichiers(vehicules)
    sys.exit(1 if invalides or nb_problemes else 0)


if __name__ == "__main__":
    main()
