"""Démonstration du système, enrichie à chaque phase (PC et Jetson).

  Phase 2 : les plaques encadrées sur l'image (détection YOLO + NumPy).
  Phase 3 : à droite, la plaque redressée, binarisée et segmentée, avec ses découpes 32 x 32.
  Phase 4 : le numéro lu par le CNN sous le cadre (vert : lecture acceptée ; orange : rejetée, avec le motif) ;
            sous chaque découpe lue, la classe prédite et sa confiance.
  Phase 5 : le système complet (systeme/chaine.py). Chaque plaque suivie a sa piste : vote en cours (lecture en
            tête, voix sur K, chronomètre depuis la première voix), puis sa décision. En bas : l'état de la
            barrière (simulée sur PC) et les dernières décisions, inscrites au journal des passages.

Source : la webcam (défaut), une vidéo ou des photos. Une vidéo est lue au temps de la vidéo, à 10 images
traitées par seconde comme l'évaluation : la démo y prend les mêmes décisions que outils/evaluer.py. Chaque
photo est montrée comme une voiture arrêtée devant la caméra, 1 s par défaut (--duree-photo, en temps de la source :
la démo défile aussi vite que le PC traite les images ; l'interface web, elle, suit le rythme réel).

Utilisation, depuis la racine du projet :
    python -m outils.demo                                   système complet, webcam
    python -m outils.demo --source validation/videos/215TU4567.mp4
    python -m outils.demo --source validation/photos
    python -m outils.demo --sans-vote                       lecture image par image (présentation de la phase 4)
    python -m outils.demo --sans-lecture                    détection + traitement (phase 3)
    python -m outils.demo --detection-seule                 détection seulement (phase 2)
    python -m outils.demo --sans-affichage --duree 20       sans écran (Jetson en SSH)
Touches, dans la fenêtre : q pour quitter ; espace pour la pause ; s pour enregistrer une capture dans
sorties/captures/ (la fenêtre et les étapes du traitement de la plaque détaillée, pour le rapport).
Liste des autorisés : config/autorises.csv (hors de git) ; à défaut, l'exemple fictif config/autorises.exemple.csv.
Journal des passages : sorties/passages/ (hors de git).
"""

import argparse
import collections
import os
import sys
import time

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import numpy as np

from systeme.action import Barriere, JournalPassages, charger_autorises
from systeme.chaine import COULEURS_DECISIONS, Chaine, ControleAcces
from systeme.debug import Debug
from systeme.decision import SEUIL, lire_numero
from systeme.detection import SEUIL_CONFIANCE, SEUIL_IOU, DetecteurPlaques, dessiner_detections
from systeme.lecture import AUTRE, LecteurCaracteres
from systeme.segmentation import dessiner_segmentation
from systeme.sources import ouvrir_source, repetitions_photo
from systeme.traitement import traiter_plaque
from systeme.vote import Vote, decrire_alerte

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELE_PAR_DEFAUT = os.path.join(RACINE, "modeles", "detecteur.onnx")
LECTEUR_PAR_DEFAUT = os.path.join(RACINE, "modeles", "lecteur.onnx")
LISTE_PAR_DEFAUT = os.path.join(RACINE, "config", "autorises.csv")
LISTE_EXEMPLE = os.path.join(RACINE, "config", "autorises.exemple.csv")
DOSSIER_CAPTURES = os.path.join(RACINE, "sorties", "captures")
DOSSIER_PASSAGES = os.path.join(RACINE, "sorties", "passages")
TITRE = "Plaques (q : quitter, espace : pause, s : capture)"
LARGEUR_VUE, HAUTEUR_VUE = 960, 540   # image de la source, réduite sans déformation (1280 x 720 -> 960 x 540)
LARGEUR_PANNEAU = 470                 # colonne de droite : plaques de 450 px de large + marges
HAUTEUR_BANDEAU = 120                 # bandeau du bas (phase 5) : barrière et dernières décisions
LARGEUR_MAX = 1280                    # images plus larges réduites à la résolution de la webcam (comme l'évaluation)
ETAPES_DETECTION = ("pretraitement", "inference", "posttraitement")
ETAPES_TRAITEMENT = ("redressement", "binarisation", "segmentation")
BLANC, VERT, ORANGE, JAUNE, ROUGE_FONCE = (255, 255, 255), (0, 220, 0), (0, 140, 255), (0, 230, 230), (40, 40, 150)


# --- Dessin --------------------------------------------------------------------------------------------------

def ecrire(image, texte, position, taille=0.55, couleur=BLANC):
    """Texte bordé de noir (ASCII seulement : cv2.putText ne connaît pas les accents)."""
    cv2.putText(image, texte, position, cv2.FONT_HERSHEY_SIMPLEX, taille, (0, 0, 0), 4)
    cv2.putText(image, texte, position, cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, 1)


def ajuster(image, largeur, hauteur):
    """Image réduite pour tenir dans largeur x hauteur sans déformation (bandes noires autour), et la
    transformation (échelle, décalage x, décalage y) qui y amène les coordonnées de l'image d'origine."""
    echelle = min(largeur / float(image.shape[1]), hauteur / float(image.shape[0]))
    taille = (min(largeur, int(round(image.shape[1] * echelle))), min(hauteur, int(round(image.shape[0] * echelle))))
    reduite = cv2.resize(image, taille, interpolation=cv2.INTER_AREA)
    vue = np.zeros((hauteur, largeur, 3), dtype=np.uint8)
    dx, dy = (largeur - taille[0]) // 2, (hauteur - taille[1]) // 2
    vue[dy:dy + taille[1], dx:dx + taille[0]] = reduite
    return vue, (echelle, dx, dy)


def vers_vue(cadre, transformation):
    """Cadre (x1, y1, x2, y2) de l'image d'origine -> coordonnées entières dans la vue réduite."""
    echelle, dx, dy = transformation
    return (int(cadre[0] * echelle + dx), int(cadre[1] * echelle + dy),
            int(cadre[2] * echelle + dx), int(cadre[3] * echelle + dy))


def texte_lecture(decision):
    """(texte, couleur) d'une lecture : numéro en vert si acceptée, sinon « ? », la lecture et le motif en orange."""
    if decision["statut"] == "lue":
        return decision["texte"], VERT
    return "? {} ({})".format(decision["lecture"] or "-", decision["statut"]), ORANGE


def panneau_traitement(resultat, decision=None, duree_lecture=0.0, entete=None):
    """Colonne de droite : plaque redressée, binarisée, segmentée, découpes 32 x 32 (avec la lecture du CNN
    sous chaque chiffre probable), lecture assemblée et durées ; entete : ligne de titre facultative."""
    colonne = np.zeros((HAUTEUR_VUE, LARGEUR_PANNEAU, 3), dtype=np.uint8)
    if resultat is None:
        ecrire(colonne, "aucune plaque lue", (10, 30))
        return colonne
    y = 8
    if entete:
        ecrire(colonne, entete, (10, y + 14), 0.55, JAUNE)
        y += 22
    vues = [("redressee", resultat["plaque"]),
            ("binarisee", cv2.cvtColor(resultat["binaire"], cv2.COLOR_GRAY2BGR)),
            ("segmentee (vert serie, cyan numero, orange autre)", dessiner_segmentation(resultat["binaire"], resultat))]
    if decision is not None:
        del vues[1]   # la vue segmentée montre déjà l'image binaire : place libérée pour la lecture
    for titre, image in vues:
        ecrire(colonne, titre, (10, y + 14))
        y += 20
        colonne[y:y + image.shape[0], 10:10 + image.shape[1]] = image
        y += image.shape[0] + 8
    # Lecture du CNN pour chaque chiffre probable (option A : seuls eux sont lus)
    lu = {}
    if decision is not None:
        for blob, classe, confiance in zip(decision["probables"], decision["classes"], decision["confiances"]):
            lu[id(blob)] = (int(classe), float(confiance))
    ecrire(colonne, "decoupes 32x32" + (" : classe lue et confiance" if decision else " (entree du CNN)"), (10, y + 14))
    y += 22
    x = 10
    for vignette, blob in zip(resultat["vignettes_grises"], resultat["candidats"]):
        if x + 34 > LARGEUR_PANNEAU:
            break
        colonne[y:y + 32, x:x + 32] = cv2.cvtColor(vignette, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(colonne, (x - 1, y - 1), (x + 32, y + 32), (0, 255, 0) if blob["chiffre"] else (0, 140, 255), 1)
        if id(blob) in lu:
            classe, confiance = lu[id(blob)]
            ecrire(colonne, "a" if classe == AUTRE else str(classe), (x + 10, y + 50), 0.55)
            ecrire(colonne, "{:.2f}".format(confiance), (x + 1, y + 66), 0.35)
        x += 36
    y += 88 if decision is not None else 50
    if decision is not None:
        texte, couleur = texte_lecture(decision)
        ecrire(colonne, texte, (10, y + 4), 0.8, couleur)
        y += 34
    ecrire(colonne, "coins : {} | chiffres : serie {} , numero {}".format(
        resultat["methode_coins"], len(resultat["serie"]), len(resultat["numero"])), (10, y))
    ms = [1000 * resultat["durees"][etape] for etape in ETAPES_TRAITEMENT]
    texte = "redressement {:.1f} | binarisation {:.1f} | segm. {:.1f}".format(*ms)
    if decision is not None:
        texte += " | CNN {:.1f}".format(1000 * duree_lecture)
    ecrire(colonne, texte + " (ms)", (10, y + 22), 0.42)
    return colonne


# --- Phase 5 : pistes, vote, barrière ------------------------------------------------------------------------

def dessiner_pistes(vue, transformation, chaine, temps):
    """Cadre de chaque piste vue sur l'image, avec son numéro de piste ; son état, en haut à gauche (une ligne par
    piste, pour que les textes ne se chevauchent pas quand les plaques sont proches) : décision prise ; sinon vote
    en cours (lecture en tête, voix sur K, chronomètre depuis la première voix) ; sinon dernière lecture, qui ne
    vote pas."""
    vote = chaine.controle.vote
    ligne = 0
    for piste in chaine.controle.suivi.pistes:
        if not piste.vue:
            continue
        x1, y1, x2, y2 = vers_vue(piste.boite, transformation)
        if piste.decision is not None:
            couleur = COULEURS_DECISIONS[piste.decision["decision"]]
            texte = "{} {}".format(piste.decision["decision"].upper(), piste.decision["numero"])
        elif piste.voix:
            couleur = JAUNE
            resume = vote.decrire(piste, temps)
            texte = "{} : {}/{} voix, {:.1f}/{:.0f} s".format(resume["meneur"], resume["voix"], resume["voix_requises"],
                                                             temps - piste.premiere_voix, vote.attente_max)
        else:
            couleur = BLANC
            lecture = piste.derniere_lecture
            texte = "?" if lecture is None else "? {} ({})".format(lecture["lecture"] or "-", lecture["statut"])
            if piste.debut_alerte is not None:   # plaque sans voix à portée de lecture : « non lu » à l'échéance
                texte += " ; alerte {:.1f}/{:.0f} s".format(temps - piste.debut_alerte, vote.attente_alerte)
        cv2.rectangle(vue, (x1, y1), (x2, y2), couleur, 2)
        ecrire(vue, "piste {}".format(piste.id), (x1, max(15, y1 - 6)), 0.45, couleur)
        ecrire(vue, "piste {} : {}".format(piste.id, texte), (10, 52 + 24 * ligne), 0.6, couleur)
        ligne += 1


def bandeau_systeme(chaine, temps, dernieres, largeur):
    """Bandeau du bas : état de la barrière, réglage du vote et dernières décisions (la plus récente en haut)."""
    bandeau = np.zeros((HAUTEUR_BANDEAU, largeur, 3), dtype=np.uint8)
    barriere = chaine.controle.barriere
    etat = barriere.resume(temps)
    ouverte = etat["etat"] == "ouverte"
    cv2.rectangle(bandeau, (10, 10), (330, HAUTEUR_BANDEAU - 10), VERT if ouverte else ROUGE_FONCE, -1)
    ecrire(bandeau, "BARRIERE {}".format("OUVERTE" if ouverte else "FERMEE"), (25, 55), 0.9)
    ecrire(bandeau, "depuis {:.1f} s (commande {})".format(etat["depuis_s"], barriere.commande.nom), (25, 88), 0.5)
    vote = chaine.controle.vote
    ecrire(bandeau, "vote : {}, K = {}, W = {}, attente max {:.0f} s ; liste : {} plaques".format(
        vote.variante, vote.voix_min, vote.fenetre, vote.attente_max, len(vote.autorises)), (350, 25), 0.5)
    for rang, d in enumerate(reversed(dernieres)):
        delai = "" if d["delai_s"] is None else ", {:.1f} s".format(d["delai_s"])
        ecrire(bandeau, "{}  piste {}  {}  {}  ({} voix{})".format(d["heure"][11:], d["piste"], d["decision"].upper(),
                                                                d["numero"] or "-", d["voix"], delai),
               (350, 50 + 20 * rang), 0.5, COULEURS_DECISIONS[d["decision"]])
    return bandeau


def fenetre_systeme(chaine, image, temps, dernieres):
    """Fenêtre de la phase 5 : l'image et ses pistes, la plaque détaillée à droite, le bandeau en bas."""
    vue, transformation = ajuster(image, LARGEUR_VUE, HAUTEUR_VUE)
    dessiner_pistes(vue, transformation, chaine, temps)
    latences = chaine.latences
    traitement = sum(latences.get(etape, 0.0) for etape in ETAPES_TRAITEMENT)
    ecrire(vue, "{:.0f} images/s | detection {:.0f} ms/image | traitement {:.0f} ms et lecture {:.1f} ms/plaque".format(
        chaine.cadence, 1000 * latences.get("detection", 0.0), 1000 * traitement, 1000 * latences.get("lecture", 0.0)),
        (10, 25))
    plaque = chaine.plaque
    if plaque is None:
        panneau = panneau_traitement(None)
    else:
        age = temps - plaque["temps"]
        entete = "plaque detaillee : piste {}{}".format(plaque["piste"], "" if age < 0.05 else
                                                         " (lue il y a {:.1f} s)".format(age))
        panneau = panneau_traitement(plaque["resultat"], plaque["lecture"], latences.get("lecture", 0.0), entete)
    haut = np.hstack([vue, panneau])
    return np.vstack([haut, bandeau_systeme(chaine, temps, dernieres, haut.shape[1])])


# --- Boucles -------------------------------------------------------------------------------------------------

def montrer(fenetre):
    """Affiche la fenêtre ; renvoie la touche pressée. Espace : pause jusqu'à la touche suivante."""
    cv2.imshow(TITRE, fenetre)
    touche = cv2.waitKey(1) & 0xFF
    if touche == ord(" "):
        touche = cv2.waitKey(0) & 0xFF
    return touche


def enregistrer_capture(fenetre, image, cadre):
    """La fenêtre, et toutes les étapes du traitement de la plaque détaillée (images pour le rapport)."""
    dossier = os.path.join(DOSSIER_CAPTURES, time.strftime("capture_%Y%m%d_%H%M%S"))
    os.makedirs(dossier, exist_ok=True)
    cv2.imwrite(os.path.join(dossier, "fenetre.png"), fenetre)
    if cadre is not None:
        debug = Debug()
        traiter_plaque(image, cadre, debug)
        debug.enregistrer(dossier, "etape")
    print("Capture enregistrée :", dossier)


def noter(decision, dernieres, decisions):
    """Une décision vient de tomber : bandeau, bilan et console."""
    dernieres.append(decision)
    decisions.append(decision)
    delai = "" if decision["delai_s"] is None else ", délai {:.1f} s".format(decision["delai_s"])
    print("{}  piste {} : {} {} ({} voix{})".format(decision["heure"], decision["piste"], decision["decision"],
                                                  decision["numero"] or "-", decision["voix"], delai))


def boucle_systeme(arguments, images, chaine):
    """Phase 5 : le système complet sur chaque image de la source. Renvoie (nombre d'images, décisions, durée)."""
    dernieres, decisions = collections.deque(maxlen=4), []
    nb_images, debut, temps, image, fenetre, arret = 0, time.time(), 0.0, None, None, False
    for temps, image in images:
        for piste in chaine.traiter(temps, image):
            noter(piste.decision, dernieres, decisions)
        nb_images += 1
        if arguments.sans_affichage:
            if nb_images % 30 == 0:
                print("{} images, {:.1f} images/s, pistes : {}, barrière {}".format(
                    nb_images, chaine.cadence, len(chaine.controle.suivi.pistes), chaine.controle.barriere.etat))
        else:
            fenetre = fenetre_systeme(chaine, image, temps, dernieres)
            touche = montrer(fenetre)
            if touche == ord("s") and chaine.plaque is not None:
                enregistrer_capture(fenetre, chaine.plaque["image"], chaine.plaque["cadre"])
            arret = touche == ord("q")
        if arret or (arguments.duree and time.time() - debut >= arguments.duree):
            arret = True
            break
    for piste in chaine.finir(temps):
        noter(piste.decision, dernieres, decisions)
    duree = time.time() - debut
    if fenetre is not None and not arret:
        # Fin de la vidéo ou des photos : le dernier état reste affiché jusqu'à une touche
        cv2.imshow(TITRE, fenetre_systeme(chaine, image, temps, dernieres))
        cv2.waitKey(0)
    return nb_images, decisions, duree


def boucle_phases(arguments, images, detecteur, lecteur):
    """Phases 2 à 4, pour les présenter une à une : la plaque la plus sûre de chaque image est détectée, traitée et
    lue, image par image, sans suivi ni vote."""
    nb_images = nb_avec_plaque = nb_traitees = nb_lues = 0
    cumuls = dict.fromkeys(ETAPES_DETECTION + ETAPES_TRAITEMENT + ("lecture",), 0.0)
    cadence = 0.0
    debut = precedent = time.time()
    for _, image in images:
        detections = detecteur.detecter(image)
        for etape in ETAPES_DETECTION:
            cumuls[etape] += detecteur.durees[etape]
        # Traitement de la plaque la plus sûre (les détections sont triées par score décroissant),
        # AVANT de dessiner les cadres sur l'image : sinon le dessin se retrouverait dans la découpe.
        resultat = decision = None
        duree_lecture = 0.0
        if len(detections) and not arguments.detection_seule:
            resultat = traiter_plaque(image, detections[0])
            nb_traitees += 1
            for etape in ETAPES_TRAITEMENT:
                cumuls[etape] += resultat["durees"][etape]
            if lecteur is not None:
                debut_lecture = time.perf_counter()
                decision = lire_numero(resultat, lecteur, arguments.seuil)
                duree_lecture = time.perf_counter() - debut_lecture
                cumuls["lecture"] += duree_lecture
                nb_lues += decision["statut"] == "lue"
        nb_images += 1
        nb_avec_plaque += len(detections) > 0
        maintenant = time.time()
        instantanee = 1.0 / max(maintenant - precedent, 1e-6)
        cadence = instantanee if cadence == 0 else 0.9 * cadence + 0.1 * instantanee  # moyenne glissante
        precedent = maintenant

        if arguments.sans_affichage:
            if nb_images % 30 == 0:
                texte = "{} images, {:.1f} images/s, plaques : {}".format(nb_images, cadence, len(detections))
                if resultat is not None:
                    texte += " ; chiffres serie {}, numero {}".format(len(resultat["serie"]), len(resultat["numero"]))
                if decision is not None:
                    texte += " ; lecture : " + texte_lecture(decision)[0]
                print(texte)
        else:
            vue, transformation = ajuster(dessiner_detections(image.copy(), detections), LARGEUR_VUE, HAUTEUR_VUE)
            ms = [1000 * detecteur.durees[etape] for etape in ETAPES_DETECTION]
            ecrire(vue, "{:.0f} images/s | detection : pre {:.1f} ms, inference {:.1f} ms, post {:.1f} ms".format(
                cadence, *ms), (10, 25))
            if decision is not None:
                # Numéro lu, sous le cadre de la plaque (le score de détection est écrit au-dessus)
                x1, _, _, y2 = vers_vue(detections[0], transformation)
                texte, couleur = texte_lecture(decision)
                ecrire(vue, texte, (x1, min(HAUTEUR_VUE - 8, y2 + 28)), 0.9, couleur)
            fenetre = vue if arguments.detection_seule else np.hstack(
                [vue, panneau_traitement(resultat, decision, duree_lecture)])
            touche = montrer(fenetre)
            if touche == ord("q"):
                break
            if touche == ord("s"):
                enregistrer_capture(fenetre, image, detections[0] if resultat is not None else None)
        if arguments.duree and maintenant - debut >= arguments.duree:
            break

    if nb_images:
        duree_totale = time.time() - debut
        print("Bilan : {} images en {:.1f} s ({:.1f} images/s) ; plaque détectée sur {:.0f} % des images".format(
            nb_images, duree_totale, nb_images / duree_totale, 100.0 * nb_avec_plaque / nb_images))
        print("Détection, temps moyens : pré-traitement {:.1f} ms, inférence {:.1f} ms, post-traitement {:.1f} ms".format(
            *[1000 * cumuls[etape] / nb_images for etape in ETAPES_DETECTION]))
        if nb_traitees:
            print("Traitement classique, temps moyens : redressement {:.1f} ms, binarisation {:.1f} ms, "
                  "segmentation {:.1f} ms".format(*[1000 * cumuls[etape] / nb_traitees for etape in ETAPES_TRAITEMENT]))
        if nb_traitees and lecteur is not None:
            print("Lecture (CNN + assemblage), temps moyen {:.2f} ms ; lectures acceptées : {} sur {} plaques "
                  "traitées".format(1000 * cumuls["lecture"] / nb_traitees, nb_lues, nb_traitees))


def charger_liste(chemin):
    """Liste des autorisés ; à défaut de config/autorises.csv, l'exemple fictif (aucune vraie plaque n'ouvre)."""
    if chemin == LISTE_PAR_DEFAUT and not os.path.exists(chemin):
        print("config/autorises.csv absent : liste d'exemple fictive (aucune vraie plaque ne sera autorisée)")
        chemin = LISTE_EXEMPLE
    return charger_autorises(chemin)


def main():
    parseur = argparse.ArgumentParser(description="Démonstration : le système complet, ou une phase à la fois.")
    parseur.add_argument("--source", default="0",
                         help="index de webcam (défaut : 0), fichier vidéo, photo ou dossier de photos")
    parseur.add_argument("--cadence", type=float, default=10.0,
                         help="vidéo : images traitées par seconde de vidéo (défaut : 10, comme l'évaluation ; 0 : toutes)")
    parseur.add_argument("--duree-photo", type=float, default=1.0,
                         help="photos : durée d'affichage de chaque photo, en secondes (défaut : 1)")
    parseur.add_argument("--modele", default=MODELE_PAR_DEFAUT, help="détecteur : .onnx (PC) ou .engine (Jetson)")
    parseur.add_argument("--lecteur", default=LECTEUR_PAR_DEFAUT, help="CNN de lecture : .onnx (PC) ou .engine")
    parseur.add_argument("--autorises", default=LISTE_PAR_DEFAUT, help="liste des autorisés (défaut : config/autorises.csv)")
    parseur.add_argument("--journal", default=DOSSIER_PASSAGES, help="dossier du journal des passages (défaut : sorties/passages)")
    parseur.add_argument("--confiance", type=float, default=SEUIL_CONFIANCE,
                         help="seuil de confiance du détecteur (défaut : {})".format(SEUIL_CONFIANCE))
    parseur.add_argument("--iou", type=float, default=SEUIL_IOU, help="seuil d'IoU de la NMS (défaut : {})".format(SEUIL_IOU))
    parseur.add_argument("--seuil", type=float, default=SEUIL, help="seuil de confiance de la lecture (défaut : {})"
                         .format(SEUIL))
    parseur.add_argument("--sans-vote", action="store_true", help="lecture image par image, sans suivi ni vote (phase 4)")
    parseur.add_argument("--sans-lecture", action="store_true", help="détection et traitement, sans CNN (phase 3)")
    parseur.add_argument("--detection-seule", action="store_true", help="n'afficher que la détection (phase 2)")
    parseur.add_argument("--sans-affichage", action="store_true", help="pas de fenêtre (Jetson sans écran)")
    parseur.add_argument("--duree", type=float, default=0, help="arrêt après ce nombre de secondes (0 = jamais)")
    arguments = parseur.parse_args()
    systeme_complet = not (arguments.detection_seule or arguments.sans_lecture or arguments.sans_vote)
    lecture_active = not (arguments.detection_seule or arguments.sans_lecture)

    detecteur = DetecteurPlaques(arguments.modele, arguments.confiance, arguments.iou)
    lecteur = LecteurCaracteres(arguments.lecteur) if lecture_active else None
    try:
        images, description, _ = ouvrir_source(arguments.source, arguments.cadence or None, LARGEUR_MAX,
                                               repetitions_photo(arguments.duree_photo))
        mode = ("système complet" if systeme_complet else "détection seule" if arguments.detection_seule
                else "sans lecture" if arguments.sans_lecture else "lecture image par image")
        print("Source : {} ; détecteur {} (entrée {}x{}, confiance {}, IoU {}) ; {}".format(
            description, os.path.basename(arguments.modele), detecteur.taille, detecteur.taille, arguments.confiance,
            arguments.iou, mode))
        if not systeme_complet:
            boucle_phases(arguments, images, detecteur, lecteur)
            return
        autorises = charger_liste(arguments.autorises)
        vote = Vote(autorises)
        chaine = Chaine(detecteur, lecteur, ControleAcces(vote, Barriere()), JournalPassages(arguments.journal),
                        arguments.seuil)
        print("Vote : {}, K = {}, W = {}, attente maximale {:.0f} s ; liste : {} plaques ; lecture : seuil {}".format(
            vote.variante, vote.voix_min, vote.fenetre, vote.attente_max, len(autorises), arguments.seuil))
        print(decrire_alerte(vote))
        nb_images, decisions, duree = boucle_systeme(arguments, images, chaine)
        if nb_images:
            types = [d["decision"] for d in decisions]
            print("Bilan : {} images en {:.1f} s ({:.1f} images/s) ; décisions : {} ouverture(s), {} refus, {} non lu ; "
                  "journal des passages : {}".format(nb_images, duree, nb_images / max(duree, 1e-6),
                                                     types.count("ouverture"), types.count("refus"),
                                                     types.count("non lu"), arguments.journal))
            print("Latences moyennes (glissantes) : détection {:.1f} ms par image ; redressement {:.1f}, binarisation "
                  "{:.1f}, segmentation {:.1f}, lecture {:.1f} ms par plaque".format(
                      *[1000 * chaine.latences.get(e, 0.0) for e in ("detection",) + ETAPES_TRAITEMENT + ("lecture",)]))
    except (IOError, ValueError) as erreur:
        sys.exit(str(erreur))
    finally:
        if not arguments.sans_affichage:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
