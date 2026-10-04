"""Évaluation du système complet sur les photos et vidéos de l'étudiant (phase 5) (PC ; compatible Python 3.6).

Deux temps, pour que tous les réglages du vote soient mesurés sur exactement les mêmes lectures :
  1. Perception, lente, mise en cache dans sorties/evaluation/<variante>/ :
     - chaque vidéo est lue à 10 images/s (temps de la vidéo) et réduite à 1280 px de large ; sur chaque image,
       le détecteur au seuil 0,25, puis le traitement et le CNN sur CHAQUE plaque détectée ;
     - chaque photo, de même, sur son image unique.
     Le cache est refait si un modèle a changé (date du fichier), ou avec --refaire.
  2. Mesures, rapides (outils/mesures.py) : le suivi et le vote du système sont rejoués sur ces lectures.
Variantes du traitement (perception refaite pour chacune) : complete (le système), aucun (sans contrôle du
redressement), bords et rangee (séparation des soudures au liseré, par-dessus complete).
Définitions, indicateurs et règles : docs/journal.md, phase 5.

Utilisation, depuis la racine du projet :
    python -m outils.evaluer                        développement, réglage du système : bilan par vidéo, par photo
    python -m outils.evaluer --planches             + planches de contrôle des pistes visées (une par vidéo)
    python -m outils.evaluer --balayage             arrêt B : les 360 réglages et la règle de choix
    python -m outils.evaluer --comparer aucun       photos : complete contre aucun (ou : bords,rangee)
    python -m outils.evaluer --variante aucun       bilan avec une autre variante du traitement
    python -m outils.evaluer --portee               arrêt B, après le balayage : X de l'alerte et son départ
    python -m outils.evaluer --test                 test final, UNE SEULE FOIS, réglage du système
La phase 5 s'est terminée sans vidéos réelles (décision de l'étudiant) : cet outil est gardé pour un usage futur,
avec la procédure de l'arrêt B décrite au journal (perspective).
Alerte « plaque sans voix » (systeme/vote.py) : tant que X n'est pas mesuré (PORTEE_MESUREE faux : valeur choisie
par raisonnement), les mesures tournent SANS l'alerte et le test final refuse de tourner. --portee mesure X ; une fois
X inscrit dans systeme/vote.py avec PORTEE_MESUREE vrai, toutes les mesures prennent l'alerte du système.
Résultats détaillés (numéros réels : hors de git) : sorties/evaluation/.
"""

import argparse
import csv
import json
import os
import sys
import time

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import numpy as np

from outils import mesures
from outils.validation import lire_decoupage, lire_inventaire
from systeme.decision import lire_numero
from systeme.detection import SEUIL_CONFIANCE, DetecteurPlaques
from systeme.lecture import LecteurCaracteres
from systeme.sources import images_photos, images_video
from systeme.traitement import traiter_plaque
from systeme.vote import (ATTENTE_MAX, DEPART_ALERTE, FENETRE, PORTEE_LECTURE, PORTEE_MESUREE, VARIANTE,
                          VOIX_MIN)

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELE_DETECTION = os.path.join(RACINE, "modeles", "detecteur.onnx")
MODELE_LECTURE = os.path.join(RACINE, "modeles", "lecteur.onnx")
DOSSIER_EVALUATION = os.path.join(RACINE, "sorties", "evaluation")
CADENCE = 10.0                       # images traitées par seconde (simulée, temps de la vidéo)
LARGEUR_MAX = 1280                   # résolution de la webcam
SEUIL_PERCEPTION = min(mesures.SEUILS)
VARIANTES_TRAITEMENT = {"complete": {}, "aucun": {"controle": "aucun"},
                        "bords": {"reglages_binarisation": {"lisere_bords": True}}, "rangee": {"detacher": True}}
OPTIONS_SOUDURE = ("bords", "rangee")          # de la plus simple à la moins simple
REGLAGE_SYSTEME = (VARIANTE, SEUIL_CONFIANCE, VOIX_MIN, FENETRE, ATTENTE_MAX)
# Les mesures n'emploient qu'un X mesuré : sans lui, elles tournent sans l'alerte
ALERTE_DES_MESURES = (DEPART_ALERTE, PORTEE_LECTURE) if PORTEE_MESUREE else mesures.SANS_ALERTE
# Règles de l'arrêt B sur les photos (journal)
BAISSE_FIABILITE_MAX = 0.01
OBJECTIF_SUBSTITUTIONS = 0.01
ECART_SIMPLICITE = 0.005
VERT, ORANGE, BLANC = (0, 200, 0), (0, 140, 255), (255, 255, 255)


# --- Perception (avec cache) ---------------------------------------------------------------------------------

def lire_plaques(image, detections, lecteur, reglages):
    """Traitement et lecture de chaque plaque détectée : une lecture par détection, dans le même ordre."""
    lectures = []
    for cadre in detections:
        lecture = lire_numero(traiter_plaque(image, cadre, **reglages), lecteur)
        lectures.append({"statut": lecture["statut"], "lecture": lecture["lecture"],
                         "confiance": round(float(lecture["confiance"]), 4)})
    return lectures


def percevoir(chemin, numero, nature, detecteur, lecteur, reglages):
    """Perception d'une vidéo (10 images/s) ou d'une photo, au format de outils/mesures.py."""
    if nature == "video":
        source = images_video(chemin, CADENCE, LARGEUR_MAX)
    else:
        source = images_photos([chemin], 1, LARGEUR_MAX)
    images, hauteur, largeur = [], 0, 0
    for temps, image in source:
        hauteur, largeur = image.shape[:2]
        detections = detecteur.detecter(image)
        images.append({"temps": round(temps, 4),
                       "detections": [[round(float(v), 1) for v in d[:4]] + [round(float(d[4]), 4)] for d in detections],
                       "lectures": lire_plaques(image, detections, lecteur, reglages)})
    return {"numero": numero, "fichier": os.path.basename(chemin), "nature": nature, "largeur": largeur,
            "hauteur": hauteur, "images": images}


class Perception:
    """Perception des fichiers d'une variante du traitement, avec cache. Les modèles ne sont chargés que si un
    fichier doit vraiment être lu."""

    def __init__(self, variante, refaire=False):
        self.dossier = os.path.join(DOSSIER_EVALUATION, variante)
        self.reglages = VARIANTES_TRAITEMENT[variante]
        self.refaire = refaire
        # Date de modification des modèles : un cache fait avec d'autres modèles est refait
        self.empreinte = {"detecteur": os.path.getmtime(MODELE_DETECTION), "lecteur": os.path.getmtime(MODELE_LECTURE)}
        self.detecteur = self.lecteur = None

    def charger(self, chemin, numero, nature):
        cache = os.path.join(self.dossier, nature + "s", os.path.basename(chemin) + ".json")
        if os.path.exists(cache) and not self.refaire:
            with open(cache, encoding="utf-8") as fichier:
                enregistrement = json.load(fichier)
            if enregistrement.get("modeles") == self.empreinte:
                return enregistrement
        if self.detecteur is None:
            self.detecteur = DetecteurPlaques(MODELE_DETECTION, SEUIL_PERCEPTION)
            self.lecteur = LecteurCaracteres(MODELE_LECTURE)
        debut = time.time()
        enregistrement = percevoir(chemin, numero, nature, self.detecteur, self.lecteur, self.reglages)
        enregistrement["modeles"] = self.empreinte
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w", encoding="utf-8") as fichier:
            json.dump(enregistrement, fichier)
        print("   perception : {} ({} image(s), {:.0f} s)".format(os.path.basename(chemin), len(enregistrement["images"]),
                                                                time.time() - debut))
        return enregistrement


def fichiers_de_la_partie(partie):
    """(numéros des véhicules de la partie, [(chemin, numéro)] des vidéos, idem des photos)."""
    vehicules, _ = lire_inventaire()
    decoupage = lire_decoupage()
    sans_partie = sorted(n for n in vehicules if n not in decoupage)
    if sans_partie:
        print("Attention : {} véhicule(s) sans partie, ignoré(s) (python -m outils.validation --decouper)".format(
            len(sans_partie)))
    numeros = sorted(n for n in vehicules if n in decoupage and decoupage[n]["partie"] == partie)
    videos = [(chemin, n) for n in numeros for chemin in vehicules[n]["videos"]]
    photos = [(chemin, n) for n in numeros for chemin in vehicules[n]["photos"]]
    return numeros, videos, photos


# --- Bilan d'un réglage --------------------------------------------------------------------------------------

def pourcent(valeur):
    return "{:.1f} %".format(100.0 * valeur)


def suivre_videos(videos, seuil, pas=1):
    """Pistes de chaque vidéo et leur classement (visées ou autres plaques) : [(pistes, visees)]."""
    suivis = []
    for video in videos:
        pistes = mesures.pistes_suivies(video, seuil, pas)
        suivis.append((pistes, mesures.classer_pistes(pistes, video)[0]))
    return suivis


def diagnostic_video(video, pistes, visees):
    """Ce que voit et lit le système sur la plaque visée d'une vidéo (aide au contrôle du protocole)."""
    instants, largeurs, issues, premiere_juste = set(), [], [], None
    for numero_piste, detections in pistes.items():
        if not visees[numero_piste]:
            continue
        for temps, cadre, lecture in detections:
            instants.add(temps)
            part = float(cadre[2] - cadre[0]) / video["largeur"]
            largeurs.append(part)
            issue = mesures.issue_lecture(lecture, video["numero"])
            issues.append(issue)
            if issue == "correcte" and (premiere_juste is None or temps < premiere_juste[0]):
                premiere_juste = (temps, part)
    return {"detectee": len(instants) / max(len(video["images"]), 1), "largeur_max": max(largeurs, default=0.0),
            "justes": issues.count("correcte"), "fausses": sum(i not in ("correcte", "rejetée") for i in issues),
            "rejetees": issues.count("rejetée"), "premiere_juste": premiere_juste,
            "visees": sum(visees.values()), "autres": len(visees) - sum(visees.values())}


def texte_decisions(passage):
    """Décisions des pistes visées d'un passage, en clair."""
    morceaux = []
    for d in passage["decisions"]:
        if passage["visees"].get(d["piste"], False):
            delai = "" if d["delai_s"] is None else ", {:.1f} s".format(d["delai_s"])
            morceaux.append("{} {} ({} voix{})".format(d["decision"], d["numero"] or "-", d["voix"], delai))
    return " ; ".join(morceaux) or "aucune décision"


def bilan_videos(videos, numeros, reglage, dossier, partie):
    """Bilan d'un réglage : une ligne par vidéo, puis les indicateurs (les deux listes croisées)."""
    listes = mesures.listes_croisees(numeros)
    suivis = suivre_videos(videos, reglage[1])
    fiabilite, nb_acceptees = mesures.fiabilite_images(videos, suivis)
    resume, passages = mesures.mesurer_reglage(videos, suivis, listes, reglage, fiabilite, alerte=ALERTE_DES_MESURES)
    print("\nVidéos ({} ; réglage {}) :".format(len(videos), texte_reglage(reglage)))
    print("   fichier                    visée détectée  largeur max  lectures justes/fausses/rejetées  "
          "1re juste (largeur)  pistes (visées/autres)")
    for video, (pistes, visees) in zip(videos, suivis):
        d = diagnostic_video(video, pistes, visees)
        premiere = "-" if d["premiere_juste"] is None else "{:.1f} s ({})".format(
            d["premiere_juste"][0], pourcent(d["premiere_juste"][1]))
        print("   {:26s} {:>14s}  {:>11s}  {:>12d} / {:d} / {:d}  {:>19s}  {:d} / {:d}".format(
            video["fichier"], pourcent(d["detectee"]), pourcent(d["largeur_max"]), d["justes"], d["fausses"],
            d["rejetees"], premiere, d["visees"], d["autres"]))
        for passage in passages:
            if passage["video"] is video:
                print("      {} ({}) : {:17s} {}".format(passage["liste"], "autorisé" if passage["autorise"] else
                                                       "non autorisé", passage["issue"], texte_decisions(passage)))
    afficher_indicateurs(resume, nb_acceptees)
    ecrire_passages(passages, os.path.join(dossier, "passages_{}.csv".format(partie)))
    return resume, passages


def afficher_indicateurs(m, nb_acceptees):
    print("\nIndicateurs (définitions : docs/journal.md) :")
    print("   ouvertures correctes : {} sur {} passages autorisés ({}) ; refus {} ; non lus {}".format(
        m["nb_ouvertures"], m["passages_autorises"], pourcent(m["ouvertures"]), pourcent(m["refus"]),
        pourcent(m["non_lus"])))
    print("   ouvertures à tort : {} (doit rester nul) ; ouvertures sur une autre plaque : {} (à vérifier sur les "
          "planches)".format(m["a_tort"], m["autres_ouvertures"]))
    print("   fiabilité des décisions : {} ; des lectures image par image : {} ({} lectures acceptées)".format(
        pourcent(m["fiabilite_decisions"]), pourcent(m["fiabilite_images"]), nb_acceptees))
    print("   décisions en moins de 3 s : {} ; délai des ouvertures correctes : médiane {:.1f} s, 90e centile "
          "{:.1f} s".format(pourcent(m["rapides"]), m["delai_median"], m["delai_p90"]))


def ecrire_passages(passages, chemin):
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.writer(fichier)
        ecrivain.writerow(["fichier", "numero", "liste", "autorise", "issue", "decisions"])
        for p in passages:
            ecrivain.writerow([p["video"]["fichier"], p["video"]["numero"], p["liste"], int(p["autorise"]),
                               p["issue"], texte_decisions(p)])


def texte_reglage(reglage):
    variante, seuil, voix, fenetre, attente = reglage
    return "{}, seuil {}, K = {}, W = {}, D = {:.0f} s".format(variante, seuil, voix, fenetre, attente)


def bilan_photos(photos, seuil):
    """Lecture de la plaque visée de chaque photo (la plus grande détection) : issues et fiabilité."""
    issues = [mesures.issue_lecture(mesures.lecture_visee(p["images"][0], seuil), p["numero"]) for p in photos]
    acceptees = [i for i in issues if i != "rejetée"]
    print("\nPhotos ({}, plaque visée = la plus grande détection, seuil {}) :".format(len(photos), seuil))
    print("   " + " ; ".join("{} {}".format(i, issues.count(i)) for i in mesures.ISSUES)
          + " ; fiabilité des lectures acceptées : {}".format(
              pourcent(issues.count("correcte") / len(acceptees)) if acceptees else "-"))
    return issues


# --- Planches de contrôle des pistes visées ------------------------------------------------------------------

def ecrire(image, texte, position, couleur=BLANC, taille=0.5):
    cv2.putText(image, texte, position, cv2.FONT_HERSHEY_SIMPLEX, taille, (0, 0, 0), 3)
    cv2.putText(image, texte, position, cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, 1)


def planche_video(chemin, video, seuil, sortie):
    """Dernière image (bande de la plaque visée, dernier cadre de chaque piste), puis, pour chaque piste, l'image
    où son cadre est le plus grand : de quoi vérifier à l'oeil le classement visée / autre plaque."""
    pistes = mesures.pistes_suivies(video, seuil)
    visees, x_visee = mesures.classer_pistes(pistes, video)
    voulues = {}
    for numero_piste, detections in pistes.items():
        temps, cadre, lecture = max(detections, key=lambda d: float((d[1][2] - d[1][0]) * (d[1][3] - d[1][1])))
        voulues.setdefault(round(temps, 4), []).append((numero_piste, cadre, lecture))
    fin = round(video["images"][-1]["temps"], 4) if video["images"] else None
    cases, vue_fin = [], None
    for temps, image in images_video(chemin, CADENCE, LARGEUR_MAX):
        for numero_piste, cadre, lecture in voulues.get(round(temps, 4), []):
            cases.append(case_piste(image, numero_piste, cadre, lecture, visees[numero_piste], pistes, x_visee,
                                    video["largeur"]))
        if round(temps, 4) == fin:
            vue_fin = vue_finale(image, pistes, visees, x_visee)
    if vue_fin is None:
        return
    lignes = [vue_fin]
    for debut in range(0, len(cases), 3):
        rangee = cases[debut:debut + 3] + [np.zeros_like(cases[0])] * (3 - len(cases[debut:debut + 3]))
        lignes.append(np.hstack(rangee))
    os.makedirs(os.path.dirname(sortie), exist_ok=True)
    cv2.imwrite(sortie, np.vstack(lignes), [cv2.IMWRITE_JPEG_QUALITY, 85])


def vue_finale(image, pistes, visees, x_visee):
    vue = image.copy()
    largeur = vue.shape[1]
    bande = vue.copy()
    x1, x2 = int(x_visee - mesures.ECART_VISEE * largeur), int(x_visee + mesures.ECART_VISEE * largeur)
    cv2.rectangle(bande, (max(x1, 0), 0), (min(x2, largeur - 1), vue.shape[0] - 1), VERT, -1)
    vue = cv2.addWeighted(bande, 0.15, vue, 0.85, 0)
    for numero_piste, detections in pistes.items():
        cadre = detections[-1][1]
        couleur = VERT if visees[numero_piste] else ORANGE
        cv2.rectangle(vue, (int(cadre[0]), int(cadre[1])), (int(cadre[2]), int(cadre[3])), couleur, 2)
        ecrire(vue, "piste {}".format(numero_piste), (int(cadre[0]), max(15, int(cadre[1]) - 6)), couleur)
    ecrire(vue, "bande verte : plaque visee +/- {:.0f} % de la largeur".format(100 * mesures.ECART_VISEE), (10, 25))
    return cv2.resize(vue, (1200, int(round(vue.shape[0] * 1200.0 / largeur))), interpolation=cv2.INTER_AREA)


def case_piste(image, numero_piste, cadre, lecture, visee, pistes, x_visee, largeur_image):
    """Découpe autour du plus grand cadre d'une piste, avec son classement et sa lecture (400 x 200)."""
    x1, y1, x2, y2 = (float(v) for v in cadre[:4])
    marge_x, marge_y = 0.6 * (x2 - x1), 1.5 * (y2 - y1)
    decoupe = image[int(max(0, y1 - marge_y)):int(min(image.shape[0], y2 + marge_y)),
                    int(max(0, x1 - marge_x)):int(min(image.shape[1], x2 + marge_x))]
    case = np.zeros((200, 400, 3), dtype=np.uint8)
    if decoupe.size:
        echelle = min(400.0 / decoupe.shape[1], 150.0 / decoupe.shape[0])
        petite = cv2.resize(decoupe, (max(1, int(decoupe.shape[1] * echelle)), max(1, int(decoupe.shape[0] * echelle))))
        case[50:50 + petite.shape[0], :petite.shape[1]] = petite
    x_median = np.median([(c[0] + c[2]) / 2.0 for _, c, _ in pistes[numero_piste]])
    ecrire(case, "piste {} : {} ({} images, ecart {:.0f} %)".format(
        numero_piste, "VISEE" if visee else "autre", len(pistes[numero_piste]),
        100.0 * abs(x_median - x_visee) / largeur_image), (5, 18), VERT if visee else ORANGE)
    ecrire(case, "{} {}".format(lecture["statut"], lecture["lecture"]), (5, 40))
    return case


# --- Balayage (arrêt B) --------------------------------------------------------------------------------------

def balayage(videos, numeros, dossier, partie):
    listes = mesures.listes_croisees(numeros)
    debut = time.time()
    print("\nBalayage des {} réglages sur {} vidéos et 2 listes...".format(len(mesures.grille()), len(videos)))
    resultats = mesures.balayer(videos, listes, alerte=ALERTE_DES_MESURES,
                                suivre=lambda s: print("   seuil {} : fait ({:.0f} s)".format(
        s, time.time() - debut)))
    retenu, etapes = mesures.choisir(resultats)
    ecrire_balayage(resultats, etapes, os.path.join(dossier, "balayage_{}.csv".format(partie)))
    print("\nRègle de choix : {} réglages ; admissibles {} ; niveau d'ouvertures {} ; bons {} ; non isolés {} ; "
          "au K le plus grand {}".format(etapes["reglages"], etapes["admissibles"],
                                       "-" if etapes["niveau"] is None else "{} véhicules".format(etapes["niveau"]),
                                       etapes["bons"], etapes["non isoles"], etapes["K le plus grand"]))
    if retenu is None:
        print("Aucun réglage ne respecte la règle : à discuter avec l'étudiant avant tout changement.")
        return None
    print("Réglage retenu : {} ({} bons voisins sur {})".format(
        texte_reglage(retenu), etapes["bons_voisins"][retenu], len(mesures.voisins(retenu))))
    afficher_indicateurs(resultats[retenu], mesures.fiabilite_images(videos, suivre_videos(videos, retenu[1]))[1])
    print("\nVoisinage du réglage retenu (ouvertures correctes, ouvertures à tort, décisions en moins de 3 s) :")
    for voisin in mesures.voisins(retenu):
        v = resultats[voisin]
        print("   {:46s} {:>8s}  {:d}  {:>8s}{}".format(texte_reglage(voisin), pourcent(v["ouvertures"]), v["a_tort"],
                                                       pourcent(v["rapides"]),
                                                       "  bon" if voisin in etapes["bons_voisins"] else ""))
    # Sensibilité à la cadence : 5 images par seconde (une image enregistrée sur deux)
    suivis = suivre_videos(videos, retenu[1], pas=2)
    fiabilite = mesures.fiabilite_images(videos, suivis)[0]
    m5 = mesures.mesurer_reglage(videos, suivis, listes, retenu, fiabilite, pas=2, alerte=ALERTE_DES_MESURES)[0]
    print("\nÀ 5 images/s : ouvertures correctes {} ; ouvertures à tort {} ; décisions en moins de 3 s {} ; "
          "délai médian {:.1f} s".format(pourcent(m5["ouvertures"]), m5["a_tort"], pourcent(m5["rapides"]),
                                         m5["delai_median"]))
    return retenu


def ecrire_balayage(resultats, etapes, chemin):
    colonnes = ["ouvertures", "refus", "non_lus", "a_tort", "fiabilite_decisions", "fiabilite_images", "rapides",
                "delai_median", "delai_p90", "autres_ouvertures"]
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.writer(fichier)
        ecrivain.writerow(["variante", "seuil", "K", "W", "D"] + colonnes + ["bon", "bons_voisins"])
        for reglage in mesures.grille():
            m = resultats[reglage]
            bon = reglage in etapes["bons_voisins"]
            ecrivain.writerow(list(reglage) + [round(m[c], 4) if isinstance(m[c], float) else m[c] for c in colonnes]
                              + [int(bon), etapes["bons_voisins"].get(reglage, "")])


# --- Alerte « plaque sans voix » (arrêt B) -------------------------------------------------------------------

def texte_effet(effet):
    delai = "-" if effet["transformes"] == 0 else "{:.1f} s".format(effet["delai_median"])
    return "{} ouverture(s) correcte(s) perdue(s) ; {} passage(s) sans voix sur la plaque visée, dont {} finissent en " \
           "« non lu » (délai médian depuis la détection : {})".format(
               effet["perdues"], effet["jamais_lus"], effet["transformes"], delai)


def mesurer_portee(videos, numeros):
    """Arrêt B, après le choix du réglage du vote (règles du journal, fixées avant la mesure) : X = largeur de la
    plaque à sa première lecture juste, au 90e centile (95e si le 90e fait perdre une ouverture correcte), puis le
    départ de l'alerte, « taille » ou « arret ». N'emploie que des X mesurés, jamais la valeur choisie par
    raisonnement."""
    listes = mesures.listes_croisees(numeros)
    suivis = suivre_videos(videos, REGLAGE_SYSTEME[1])
    fiabilite = mesures.fiabilite_images(videos, suivis)[0]
    premieres, maximums = mesures.largeurs_lecture(videos, suivis)
    print("\nAlerte « plaque sans voix », réglage du système ({}) :".format(texte_reglage(REGLAGE_SYSTEME)))
    if not premieres:
        print("   aucune plaque visée lue : X ne peut pas être mesuré")
        return
    print("   première lecture juste : {} vidéos sur {} ; largeur de la plaque alors : médiane {}".format(
        len(premieres), len(maximums), pourcent(float(np.median(premieres)))))
    taille = portee = None
    for centile in (mesures.CENTILE_PORTEE, mesures.CENTILE_PORTEE_REPLI):
        portee = float(np.percentile(premieres, centile))
        atteinte = sum(m >= portee for m in maximums) / float(len(maximums))
        taille = mesures.effet_alerte(videos, suivis, listes, REGLAGE_SYSTEME, fiabilite, ("taille", portee))
        print("   « taille », X au {}e centile = {} (atteint par la plaque arrêtée dans {} des vidéos) : {}".format(
            centile, pourcent(portee), pourcent(atteinte), texte_effet(taille)))
        if taille["perdues"] == 0:
            break
    if taille["perdues"] > 0:
        print("   Règle non respectée : même au 95e centile, X fait perdre une ouverture correcte. À discuter avec "
              "l'étudiant avant tout changement.")
        return
    arret = mesures.effet_alerte(videos, suivis, listes, REGLAGE_SYSTEME, fiabilite, ("arret", None))
    print("   « arret » (variante) : {}".format(texte_effet(arret)))
    retenu = mesures.choisir_alerte(taille, arret)
    print("Départ retenu par la règle : « {} ». À inscrire dans systeme/vote.py : DEPART_ALERTE = \"{}\", "
          "PORTEE_LECTURE = {:.3f}, PORTEE_MESUREE = True".format(retenu, retenu, portee))


# --- Comparaison des variantes du traitement (photos) --------------------------------------------------------

def comparer(photos_par_variante, variantes, seuil):
    """Photos : complete contre chaque variante, appariées ; puis la règle de l'arrêt B (journal)."""
    issues = {v: [mesures.issue_lecture(mesures.lecture_visee(p["images"][0], seuil), p["numero"])
                  for p in photos_par_variante[v]] for v in ["complete"] + variantes}
    n = len(issues["complete"])
    parts = {v: {i: issues[v].count(i) / float(n) for i in mesures.ISSUES} for v in issues}
    fiabilites = {}
    for v in issues:
        acceptees = [i for i in issues[v] if i != "rejetée"]
        fiabilites[v] = issues[v].count("correcte") / float(len(acceptees)) if acceptees else 0.0
    print("\nPhotos ({}), comparaison appariée :".format(n))
    print("   variante   " + "  ".join("{:>12s}".format(i) for i in mesures.ISSUES) + "   fiabilité  gagnées  perdues  p")
    for v in issues:
        gagnees = sum(1 for a, b in zip(issues["complete"], issues[v]) if b == "correcte" and a != "correcte")
        perdues = sum(1 for a, b in zip(issues["complete"], issues[v]) if a == "correcte" and b != "correcte")
        print("   {:10s} ".format(v) + "  ".join("{:>12s}".format(pourcent(parts[v][i])) for i in mesures.ISSUES)
              + "   {:>9s}  {:7d}  {:7d}  {:.3f}".format(pourcent(fiabilites[v]), gagnees, perdues,
                                                         mesures.mcnemar(gagnees, perdues)))
    if "aucun" in variantes:
        confirmee = (parts["complete"]["correcte"] >= parts["aucun"]["correcte"]
                     and fiabilites["complete"] >= fiabilites["aucun"] - BAISSE_FIABILITE_MAX)
        print("Règle « complete » : {}".format("confirmée" if confirmee else "NON confirmée : discussion avec l'étudiant"))
    options = [v for v in OPTIONS_SOUDURE if v in variantes]
    if options:
        admissibles = [v for v in options if parts[v]["substitution"] < OBJECTIF_SUBSTITUTIONS
                       and fiabilites[v] >= fiabilites["complete"] - BAISSE_FIABILITE_MAX
                       and parts[v]["correcte"] > parts["complete"]["correcte"]]
        if not admissibles:
            print("Règle des soudures : aucune option admissible avec un gain, on garde complete sans séparation")
        else:
            meilleure = max(parts[v]["correcte"] for v in admissibles)
            retenue = next(v for v in admissibles if parts[v]["correcte"] >= meilleure - ECART_SIMPLICITE)
            print("Règle des soudures : option retenue {} (admissibles : {})".format(retenue, ", ".join(admissibles)))


# --- Programme principal -------------------------------------------------------------------------------------

def main():
    parseur = argparse.ArgumentParser(description="Évaluation du système complet (phase 5).")
    parseur.add_argument("--variante", default="complete", choices=sorted(VARIANTES_TRAITEMENT),
                         help="variante du traitement (défaut : complete, le système)")
    parseur.add_argument("--planches", action="store_true", help="planches de contrôle des pistes visées")
    parseur.add_argument("--balayage", action="store_true", help="arrêt B : les 360 réglages et la règle de choix")
    parseur.add_argument("--comparer", help="photos : complete contre ces variantes (ex. aucun, ou bords,rangee)")
    parseur.add_argument("--portee", action="store_true",
                         help="arrêt B, après le balayage : mesure X de l'alerte et choisit son départ")
    parseur.add_argument("--test", action="store_true", help="test final : UNE SEULE FOIS, réglage du système")
    parseur.add_argument("--refaire", action="store_true", help="refaire la perception même si elle est en cache")
    arguments = parseur.parse_args()
    if arguments.test and not PORTEE_MESUREE:
        sys.exit("Test final refusé : X de l'alerte n'a pas été mesuré (valeur choisie par raisonnement). Le mesurer "
                 "(python -m outils.evaluer --portee), l'inscrire dans systeme/vote.py avec PORTEE_MESUREE = True, "
                 "puis relancer.")
    if arguments.test and (arguments.balayage or arguments.portee):
        parseur.error("ni balayage ni mesure de X sur le test : les réglages se choisissent sur le développement")
    partie = "test" if arguments.test else "developpement"
    if arguments.test:
        print("=== TEST FINAL : une seule fois, avec le réglage du système. Aucune décision ne doit en découler. ===")
    numeros, fichiers_videos, fichiers_photos = fichiers_de_la_partie(partie)
    print("Alerte « plaque sans voix » : {}".format(
        "départ « {} », X = {} (mesuré)".format(DEPART_ALERTE, PORTEE_LECTURE) if PORTEE_MESUREE else
        "désactivée dans les mesures (X non mesuré : valeur choisie par raisonnement)"))
    print("Partie {} : {} véhicules, {} vidéos, {} photos ; variante du traitement : {}".format(
        partie, len(numeros), len(fichiers_videos), len(fichiers_photos), arguments.variante))
    if not numeros:
        sys.exit("Aucun véhicule dans cette partie (python -m outils.validation)")
    dossier = os.path.join(DOSSIER_EVALUATION, arguments.variante)
    perception = Perception(arguments.variante, arguments.refaire)
    videos = [perception.charger(chemin, numero, "video") for chemin, numero in fichiers_videos]
    photos = [perception.charger(chemin, numero, "photo") for chemin, numero in fichiers_photos]

    if videos:
        bilan_videos(videos, numeros, REGLAGE_SYSTEME, dossier, partie)
    if photos:
        bilan_photos(photos, REGLAGE_SYSTEME[1])
    if arguments.planches:
        for (chemin, _), video in zip(fichiers_videos, videos):
            planche_video(chemin, video, REGLAGE_SYSTEME[1],
                          os.path.join(DOSSIER_EVALUATION, "planches", partie, video["fichier"] + ".jpg"))
        print("\nPlanches : {}".format(os.path.join(DOSSIER_EVALUATION, "planches", partie)))
    if arguments.portee and videos:
        mesurer_portee(videos, numeros)
    if arguments.balayage and videos:
        balayage(videos, numeros, dossier, partie)
    if arguments.comparer and photos:
        variantes = [v.strip() for v in arguments.comparer.split(",") if v.strip()]
        inconnues = [v for v in variantes if v not in VARIANTES_TRAITEMENT or v == "complete"]
        if inconnues:
            sys.exit("Variantes inconnues : {}".format(", ".join(inconnues)))
        photos_par_variante = {"complete": photos if arguments.variante == "complete" else
                               [Perception("complete", arguments.refaire).charger(c, n, "photo")
                                for c, n in fichiers_photos]}
        for variante in variantes:
            autre = Perception(variante, arguments.refaire)
            photos_par_variante[variante] = [autre.charger(c, n, "photo") for c, n in fichiers_photos]
        comparer(photos_par_variante, variantes, REGLAGE_SYSTEME[1])


if __name__ == "__main__":
    main()
