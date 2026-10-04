"""Vérification de la logique du système sur des scénarios synthétiques (phase 5, PC et Jetson).

Chaque règle du suivi, du vote, de la barrière, des sources et des mesures de l'évaluation est rejouée sur un
scénario construit à la main, dont on connaît la bonne réponse : cadres et lectures inventés, numéros fictifs,
sans modèle ni image réelle. Le script vérifie que le code fait ce que dit docs/journal.md (« Définitions du
système » et « Évaluation »). Il ne règle rien.

Utilisation, depuis la racine du projet :
    python -m outils.verifier_logique
Code de sortie 0 si tout est conforme.

Compatible Python 3.6, NumPy 1.19, OpenCV 4.1.1 (PC et Jetson).
"""

import os
import shutil
import sys
import tempfile

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import numpy as np

import re

from outils import mesures
from outils.interface import Instantane, creer_application, ecran_disponible, passage_api
from systeme.action import Barriere, JournalPassages, charger_autorises, normaliser_numero
from systeme.chaine import Chaine, ControleAcces
from systeme.sources import ouvrir_source, repetitions_photo
from systeme.suivi import Suivi
from systeme.vote import PORTEE_MESUREE, Vote

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CADENCE = 10.0                 # images par seconde ; temps = rang / CADENCE (valeur exacte au plus près)
A, COURTE, B, C = "215 TU 4567", "215 TU 456", "180 TU 1234", "199 TU 777"
UN_CHIFFRE = "21 TU 5"         # plaque autorisée au numéro d'un seul chiffre
AUTORISES = {A, UN_CHIFFRE}
resultats = []


def verifier(nom, condition, detail=""):
    resultats.append(bool(condition))
    print("{:7s} {}{}".format("OK" if condition else "ECHEC", nom, "" if condition else "  ({})".format(detail)))


def lecture(statut, texte="", confiance=0.9):
    """Lecture d'une image, comme la renvoie systeme/decision.py (seuls les champs utiles au vote)."""
    return {"statut": statut, "lecture": texte, "confiance": confiance}


def cadre(x, y, largeur, hauteur, score=0.9):
    return np.array([x, y, x + largeur, y + hauteur, score], dtype=np.float32)


def rejouer(lectures, vote, fin=None):
    """Une seule plaque immobile, une lecture par image (None : plaque non détectée). Renvoie la décision de la
    piste et le temps où elle est tombée (ou (None, None))."""
    controle = ControleAcces(vote)
    for rang, texte_lecture in enumerate(lectures):
        temps = rang / CADENCE
        detections = np.zeros((0, 5), dtype=np.float32) if texte_lecture is None else cadre(500, 400, 150, 33)[None]
        for piste, _ in controle.associer(temps, detections):
            if controle.voter(temps, piste, texte_lecture) is not None:
                return piste.decision, temps
        for piste in controle.finir_image(temps):
            return piste.decision, temps
    decidees = controle.finir(fin if fin is not None else len(lectures) / CADENCE)
    return (decidees[0].decision, decidees[0].decision["temps"]) if decidees else (None, None)


def scenarios_suivi():
    print("\n--- Suivi ---")
    suivi = Suivi()
    for rang in range(30):   # une plaque qui approche : son cadre grandit de 3 % par image
        echelle = 1.03 ** rang
        suivi.mettre_a_jour(rang / CADENCE, cadre(600 - 75 * echelle, 400, 150 * echelle, 33 * echelle)[None])
    verifier("une plaque qui approche donne une seule piste", suivi.nb_pistes == 1, suivi.nb_pistes)

    suivi = Suivi()
    vide = np.zeros((0, 5), dtype=np.float32)
    temps = [0.0, 0.1, 0.2, 1.0, 1.1]          # 0,9 s sans détection (de 0,2 à 1,1 s) : toléré
    for t in temps:
        suivi.mettre_a_jour(t, cadre(500, 400, 150, 33)[None] if t != 1.0 else vide)
    verifier("0,9 s sans détection : la piste continue", suivi.nb_pistes == 1, suivi.nb_pistes)
    fermees = []
    for t in (1.5, 2.0, 2.2):                  # plus de détection depuis 1,1 s : fermée à 2,2 s (1,1 s > 1 s)
        fermees += suivi.mettre_a_jour(t, vide)[1]
    associations, _ = suivi.mettre_a_jour(2.3, cadre(500, 400, 150, 33)[None])
    verifier("trou de plus de 1 s : piste fermée, puis nouvelle piste",
             len(fermees) == 1 and fermees[0].id == 1 and associations[0][0].id == 2,
             [p.id for p in fermees])

    suivi = Suivi()
    for rang in range(10):
        associations, _ = suivi.mettre_a_jour(rang / CADENCE, np.stack([cadre(100 + rang, 400, 150, 33),
                                                                    cadre(900 - rang, 380, 140, 31)]))
    verifier("deux plaques éloignées : deux pistes qui gardent leur identité",
             suivi.nb_pistes == 2 and sorted(p.id for p, _ in associations) == [1, 2], suivi.nb_pistes)


def scenarios_vote():
    print("\n--- Vote, variante majorité (K = 3, W = 10, D = 3 s) ---")
    majorite = Vote(AUTORISES, "majorite", 3, 10, 3.0)
    decision, temps = rejouer([lecture("lue", A)] * 5, majorite)
    verifier("3 lectures concordantes d'une plaque autorisée : ouverture à la 3e",
             decision["decision"] == "ouverture" and decision["numero"] == A and abs(temps - 0.2) < 1e-9
             and abs(decision["delai_s"] - 0.2) < 1e-9, decision)
    decision, temps = rejouer([lecture("lue", B)] * 5, majorite)
    verifier("3 lectures concordantes d'une plaque non autorisée : refus",
             decision["decision"] == "refus" and decision["numero"] == B and not decision["autorise"], decision)
    suite = [lecture("lue", t) for t in (A, COURTE, A, COURTE, A)] + [lecture("lue", A)] * 3
    decision, temps = rejouer(suite, majorite)
    verifier("A, courte, A, courte, A : A décide à la 5e voix (3 voix sur 5, plus de la moitié)",
             decision["numero"] == A and abs(temps - 0.4) < 1e-9, (decision, temps))
    decision, temps = rejouer([lecture("lue", t) for t in (A, B, C) * 20], majorite)
    verifier("trois lectures qui alternent : jamais de majorité, « non lu » à l'échéance (3 s après la 1re voix)",
             decision["decision"] == "non lu" and abs(temps - 3.0) < 1e-9, (decision, temps))
    decision, temps = rejouer([lecture("aucun chiffre")] * 10 + [lecture("lue", A)] * 3, majorite)
    verifier("le chronomètre part à la première voix, pas à la première détection",
             decision["decision"] == "ouverture" and abs(decision["delai_s"] - 0.2) < 1e-9
             and abs(decision["depuis_detection_s"] - 1.2) < 1e-9, decision)
    decision, temps = rejouer([lecture("doute", A, 0.4)] * 50 + [None] * 15, majorite)
    verifier("lectures « doute » seulement : aucune voix ni échéance, « non lu » quand la plaque disparaît",
             decision["decision"] == "non lu" and decision["delai_s"] is None and abs(temps - 6.0) < 1e-9,
             (decision, temps))
    decision, _ = rejouer([lecture("aucun chiffre")] * 20 + [None] * 15, majorite)
    verifier("aucun chiffre lu puis disparition : rien n'est noté (fausse détection probable)", decision is None,
             decision)
    decision, _ = rejouer([lecture("format", "1234 TU 5")] * 20 + [None] * 15, majorite)
    verifier("lectures hors format puis disparition : rien n'est noté", decision is None, decision)
    decision, _ = rejouer([lecture("improbable", UN_CHIFFRE)] * 3, majorite)
    verifier("« improbable » égale à une plaque autorisée : elle vote, ouverture",
             decision is not None and decision["decision"] == "ouverture" and decision["numero"] == UN_CHIFFRE,
             decision)
    decision, _ = rejouer([lecture("improbable", "21 TU 6")] * 3 + [None] * 15, majorite)
    verifier("« improbable » absente de la liste : elle ne vote pas (« non lu » à la disparition)",
             decision["decision"] == "non lu" and decision["voix"] == 0, decision)
    decision, _ = rejouer([lecture("doute", A, 0.5)] * 3 + [lecture("lue", A)] * 3, majorite)
    verifier("les lectures « doute » ne votent pas", decision["voix"] == 3 and decision["decision"] == "ouverture",
             decision)
    fenetre = Vote(AUTORISES, "majorite", 3, 5, 3.0)
    suite = [lecture("lue", t) for t in (A, A, B, C, B, C, B)]
    decision, temps = rejouer(suite, fenetre)
    verifier("fenêtre W = 5 : les 2 voix A sorties de la fenêtre, B décide (3 voix sur 5)",
             decision["numero"] == B and decision["decision"] == "refus" and abs(temps - 0.6) < 1e-9,
             (decision, temps))
    decision, temps = rejouer(suite, majorite)
    verifier("même suite, W = 10 : B n'a que 3 voix sur 7, pas de décision avant la disparition",
             decision["decision"] == "non lu", (decision, temps))
    un = Vote(AUTORISES, "majorite", 1, 10, 3.0)
    decision, temps = rejouer([lecture("lue", COURTE)] + [lecture("lue", A)] * 3, un)
    verifier("K = 1 : la première lecture acceptée décide (ici la lecture courte : refus)",
             decision["decision"] == "refus" and decision["numero"] == COURTE and temps == 0.0, decision)

    print("\n--- Vote, variante liste d'abord (K = 3, W = 10, D = 3 s) ---")
    liste = Vote(AUTORISES, "liste", 3, 10, 3.0)
    suite = [lecture("lue", COURTE)] * 3 + [lecture("lue", A)] * 3
    decision, temps = rejouer(suite, majorite)
    verifier("courte x3 puis A x3, majorité : refus dès la 3e lecture courte",
             decision["decision"] == "refus" and abs(temps - 0.2) < 1e-9, (decision, temps))
    decision, temps = rejouer(suite, liste)
    verifier("même suite, liste d'abord : ouverture quand A, autorisée, atteint 3 voix (sans majorité)",
             decision["decision"] == "ouverture" and decision["numero"] == A and abs(temps - 0.5) < 1e-9,
             (decision, temps))
    decision, temps = rejouer([lecture("lue", B)] * 40, liste)
    verifier("plaque non autorisée, liste d'abord : refus seulement à l'échéance",
             decision["decision"] == "refus" and abs(temps - 3.0) < 1e-9, (decision, temps))
    decision, temps = rejouer([lecture("lue", t) for t in (B, C) * 20], liste)
    verifier("liste d'abord, échéance sans majorité : « non lu »", decision["decision"] == "non lu", decision)
    courte = Vote(AUTORISES, "majorite", 3, 10, 1.0)
    decision, temps = rejouer([lecture("lue", t) for t in (A, B, C) * 20], courte)
    verifier("échéance D = 1 s : « non lu » 1 s après la première voix",
             decision["decision"] == "non lu" and abs(temps - 1.0) < 1e-9, (decision, temps))


def scenarios_barriere():
    print("\n--- Barrière (ouverture d'au moins 5 s, fermeture 3 s après le départ de la plaque) ---")
    barriere = Barriere()
    barriere.ouvrir(0.0)
    fermeture = None
    for rang in range(1, 200):
        temps = rang / CADENCE
        if temps <= 10.0 + 1e-9:
            barriere.signaler_presence(temps)
        barriere.mettre_a_jour(temps)
        if barriere.etat == "fermee":
            fermeture = temps
            break
    verifier("plaque vue jusqu'à 10 s : fermeture à 13 s", fermeture is not None and abs(fermeture - 13.0) < 1e-6,
             fermeture)
    barriere = Barriere()
    barriere.ouvrir(0.0)
    fermeture = None
    for rang in range(1, 200):
        temps = rang / CADENCE
        if temps <= 1.0 + 1e-9:
            barriere.signaler_presence(temps)
        barriere.mettre_a_jour(temps)
        if barriere.etat == "fermee":
            fermeture = temps
            break
    verifier("plaque partie à 1 s : fermeture à 5 s (ouverture minimale)",
             fermeture is not None and abs(fermeture - 5.0) < 1e-6, fermeture)
    barriere = Barriere()
    barriere.ouvrir(0.0)
    barriere.ouvrir(4.0)     # une seconde voiture autorisée pendant l'ouverture
    barriere.mettre_a_jour(8.9)
    reste_ouverte = barriere.etat == "ouverte"
    barriere.mettre_a_jour(9.0)
    verifier("seconde ouverture à 4 s : les délais repartent (fermeture à 9 s)",
             reste_ouverte and barriere.etat == "fermee", barriere.etat)


def scenario_complet():
    print("\n--- Passage complet (suivi, vote, barrière) ---")
    controle = ControleAcces(Vote(AUTORISES, "majorite", 3, 10, 3.0), Barriere())
    ouverture = fermeture = None
    lues = 0
    for rang in range(120):
        temps = rang / CADENCE
        if rang < 50:      # la voiture approche, puis s'arrête (cadre fixe)
            echelle = 1.03 ** min(rang, 35)
            detections = cadre(600 - 75 * echelle, 380, 150 * echelle, 33 * echelle)[None]
        else:              # elle est passée : plus de plaque dans l'image
            detections = np.zeros((0, 5), dtype=np.float32)
        for piste, _ in controle.associer(temps, detections):
            lues += 1
            texte = lecture("aucun chiffre") if rang < 20 else lecture("lue", A)
            if controle.voter(temps, piste, texte) is not None:
                ouverture = temps
        controle.finir_image(temps)
        if ouverture is not None and fermeture is None and controle.barriere.etat == "fermee":
            fermeture = temps
    verifier("ouverture à la 3e lecture acceptée (2,2 s)", ouverture is not None and abs(ouverture - 2.2) < 1e-9,
             ouverture)
    verifier("une piste décidée n'est plus lue", lues == 23, lues)
    verifier("fermeture 3 s après la dernière détection (4,9 s + 3 s = 7,9 s)",
             fermeture is not None and abs(fermeture - 7.9) < 1e-9, fermeture)


def scenarios_liste():
    print("\n--- Liste des autorisés ---")
    verifier("normalisation des numéros", normaliser_numero("215TU4567") == A
             and normaliser_numero(" 215 tu 4567 ") == A and normaliser_numero("0215 TU 1") is None
             and normaliser_numero("215 TU 04567") is None and normaliser_numero("215 4567") is None)
    exemple = charger_autorises(os.path.join(RACINE, "config", "autorises.exemple.csv"))
    verifier("lecture de config/autorises.exemple.csv (3 plaques fictives)", len(exemple) == 3, exemple)
    dossier = tempfile.mkdtemp()
    try:
        chemin = os.path.join(dossier, "autorises.csv")
        with open(chemin, "w", encoding="utf-8") as fichier:
            fichier.write("numero\n215 TU 4567\n215 TU O567\n")
        try:
            charger_autorises(chemin)
            erreur = False
        except ValueError:
            erreur = True
        verifier("une ligne illisible arrête le chargement", erreur)
    finally:
        shutil.rmtree(dossier)


def scenarios_sources():
    print("\n--- Sources ---")
    dossier = tempfile.mkdtemp()
    try:
        chemin = os.path.join(dossier, "essai.avi")
        ecrivain = cv2.VideoWriter(chemin, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (1920, 1080))
        for rang in range(90):     # 3 s à 30 images/s
            image = np.full((1080, 1920, 3), rang, dtype=np.uint8)
            ecrivain.write(image)
        ecrivain.release()
        images, _, _ = ouvrir_source(chemin, cadence=10, largeur_max=1280)
        temps_images = [(t, image.shape) for t, image in images]
        verifier("vidéo de 3 s à 30 images/s, cadence 10 : 30 images, une tous les 0,1 s, réduites à 1280 px",
                 len(temps_images) == 30 and all(abs(t - k * 0.1) < 1e-6 for k, (t, _) in enumerate(temps_images))
                 and temps_images[0][1] == (720, 1280, 3), (len(temps_images), temps_images[:3]))
        images, _, _ = ouvrir_source(chemin)
        verifier("sans cadence : toutes les images", len(list(images)) == 90)
        for nom in ("a.jpg", "b.jpg"):
            cv2.imwrite(os.path.join(dossier, nom), np.zeros((300, 400, 3), dtype=np.uint8))
        images, _, _ = ouvrir_source(dossier, repetitions=3)
        temps_photos = [round(t, 3) for t, _ in images]
        verifier("dossier de 2 photos répétées 3 fois : deux scènes, 10 s sans image après la première",
                 temps_photos == [0.0, 0.1, 0.2, 10.3, 10.4, 10.5], temps_photos)
        verifier("--duree-photo : 5 s = 50 passages à 10 images/s ; jamais moins d'un",
                 repetitions_photo(5) == 50 and repetitions_photo(0.01) == 1)
        images, _, _ = ouvrir_source(dossier, repetitions=repetitions_photo(12))
        temps_photos = [t for t, _ in images]
        verifier("photo montrée 12 s : la suivante commence 10 s après sa fin, le temps ne recule jamais",
                 abs(temps_photos[120] - 22.0) < 1e-9 and all(b > a for a, b in zip(temps_photos, temps_photos[1:])),
                 temps_photos[118:122])
    finally:
        shutil.rmtree(dossier)


def video_enregistree(numero, lectures_visee, lectures_voisine=None, n=60, score=0.8):
    """Vidéo enregistrée (format de outils/mesures.py) : plaque visée au centre qui grandit ; voisine éventuelle
    à droite (x ~ 1000), qui disparaît à mi-parcours."""
    images = []
    for k in range(n):
        largeur = 60 + 3 * k
        detections = [[640 - largeur / 2.0, 400, 640 + largeur / 2.0, 400 + largeur / 4.5, score]]
        lectures = [lectures_visee[min(k, len(lectures_visee) - 1)]]
        if lectures_voisine is not None and k < n // 2:
            detections.append([1000, 380, 1060, 393, 0.7])
            lectures.append(lectures_voisine[min(k, len(lectures_voisine) - 1)])
        images.append({"temps": k / CADENCE, "detections": detections, "lectures": lectures})
    return {"numero": numero, "fichier": numero.replace(" ", "") + ".mp4", "largeur": 1280, "hauteur": 720,
            "images": images}


def resultats_construits(ouvertures, rapides=lambda r: 1.0):
    """Indicateurs inventés pour les 360 réglages : tous admissibles ; ouvertures correctes (sur 30 véhicules) et
    rapidité données."""
    return {r: {"a_tort": 0, "fiabilite_decisions": 0.9, "fiabilite_images": 0.8, "nb_ouvertures": ouvertures(r),
                "ouvertures": ouvertures(r) / 30.0, "rapides": rapides(r)} for r in mesures.grille()}


def scenarios_mesures():
    print("\n--- Mesures de l'évaluation (outils/mesures.py) ---")
    variantes = mesures.variantes_adverses(A)
    verifier("variantes adverses à une modification près : 68 pour un numéro 3 + 4 (7 chiffres supprimés, 61 "
             "remplacés, aucun ajout possible au format)",
             len(variantes) == 68 and {"21 TU 4567", "215 TU 456", "215 TU 4537", "915 TU 4567"} <= variantes
             and A not in variantes, len(variantes))
    courtes = mesures.variantes_adverses("21 TU 456")
    verifier("variantes adverses : chiffres ajoutés là où le format le permet (111 pour un numéro 2 + 3)",
             len(courtes) == 111 and {"214 TU 456", "21 TU 4567", "1 TU 456"} <= courtes, len(courtes))
    verifier("variantes adverses : jamais de zéro en tête",
             all(mesures.format_valide(*v.split(" TU ")) for v in mesures.variantes_adverses("801 TU 8000")))
    vehicules = [A, B, C, "21 TU 456"]
    (_, l1), (_, l2) = mesures.listes_croisees(vehicules)
    verifier("listes croisées : chaque véhicule autorisé dans exactement une liste",
             all((v in l1) != (v in l2) for v in vehicules))
    sans_a = l1 if A not in l1 else l2
    verifier("listes croisées : variantes du non autorisé présentes, son vrai numéro absent",
             COURTE in sans_a and A not in sans_a)
    video = video_enregistree(A, [lecture("aucun chiffre")] * 10 + [lecture("lue", A)] * 50, [lecture("lue", B)] * 30)
    pistes = mesures.pistes_suivies(video, 0.5)
    visees, x_visee = mesures.classer_pistes(pistes, video)
    verifier("pistes visées : la plaque centrale oui, la voisine (30 % plus loin) non",
             visees == {1: True, 2: False} and abs(x_visee - 640) < 1, (visees, x_visee))
    decisions = mesures.rejouer(video, Vote({A, B}, "majorite", 3, 10, 3.0), 0.5)
    verifier("rejeu : la voisine autorisée ouvre, sans compter pour le véhicule visé",
             mesures.issue_passage(decisions, visees, A) == "ouverture"
             and sum(d["decision"] == "ouverture" for d in decisions) == 2, decisions)
    courte = video_enregistree(A, [lecture("lue", COURTE)] * 60)
    verifier("issue : lecture courte présente dans la liste -> ouverture à tort",
             mesures.issue_passage(mesures.rejouer(courte, Vote({COURTE}), 0.5), {1: True}, A) == "ouverture à tort")
    verifier("issue : lecture courte absente de la liste -> refus",
             mesures.issue_passage(mesures.rejouer(courte, Vote({A}), 0.5), {1: True}, A) == "refus")
    plateau = lambda r: 24 if r[2] <= 4 else 21
    retenu, _ = mesures.choisir(resultats_construits(plateau))
    verifier("règle : plateau pour K <= 4 -> K = 4, au centre du plateau",
             retenu == ("majorite", 0.35, 4, 10, 2.0), retenu)
    pic = lambda r: 29 if r == ("liste", 0.65, 5, 20, 3.0) else plateau(r)
    retenu, etapes = mesures.choisir(resultats_construits(pic))
    verifier("règle : un pic isolé (29 véhicules ouverts) est écarté, on descend au plateau (24)",
             retenu == ("majorite", 0.35, 4, 10, 2.0) and etapes["niveau"] == 24, (retenu, etapes["niveau"]))
    retenu, _ = mesures.choisir(resultats_construits(lambda r: plateau(r) - (r[2] == 4 and r[3] == 20)))
    verifier("règle : à un véhicule près, K = 4 reste bon même avec un véhicule de moins",
             retenu is not None and retenu[2] == 4, retenu)
    retenu, _ = mesures.choisir(resultats_construits(plateau, rapides=lambda r: 0.85 if r[2] == 4 else 1.0))
    verifier("règle : K = 4 trop lent (85 % des décisions en moins de 3 s) -> K = 3",
             retenu == ("majorite", 0.35, 3, 10, 2.0), retenu)
    a_tort = resultats_construits(plateau)
    for reglage in a_tort:
        a_tort[reglage]["a_tort"] = int(reglage[2] == 4 and reglage[0] == "majorite")
    retenu, _ = mesures.choisir(a_tort)
    verifier("règle : une ouverture à tort exclut le réglage (majorité K = 4 -> liste d'abord K = 4)",
             retenu == ("liste", 0.35, 4, 10, 2.0), retenu)
    verifier("McNemar exact : 10 plaques gagnées contre 0 -> p = 0,002",
             abs(mesures.mcnemar(10, 0) - 0.001953125) < 1e-12 and mesures.mcnemar(0, 0) == 1.0)


def rejouer_cadres(images, vote, largeur=1280):
    """Une plaque, avec son cadre et sa lecture à chaque image (cadre None : non détectée), et la largeur de l'image
    (l'alerte en a besoin). Renvoie la décision de la piste et son temps, ou (None, None)."""
    controle = ControleAcces(vote)
    temps = 0.0
    for rang, (boite, texte_lecture) in enumerate(images):
        temps = rang / CADENCE
        detections = np.zeros((0, 5), dtype=np.float32) if boite is None else boite[None]
        for piste, _ in controle.associer(temps, detections, largeur):
            if controle.voter(temps, piste, texte_lecture) is not None:
                return piste.decision, temps
        for piste in controle.finir_image(temps):
            return piste.decision, temps
    decidees = controle.finir(temps + 1.0 / CADENCE)
    return (decidees[0].decision, decidees[0].decision["temps"]) if decidees else (None, None)


def scenarios_alerte():
    print("\n--- Alerte « plaque sans voix » (X = 10 % de 1280 px = 128 px ; « non lu » 3 s après le départ) ---")
    taille = Vote(AUTORISES, "majorite", 3, 10, 3.0, depart_alerte="taille", portee=0.10)
    grande, petite = cadre(500, 400, 200, 44), cadre(500, 400, 100, 22)
    rien = lecture("aucun chiffre")
    decision, temps = rejouer_cadres([(grande, rien)] * 60, taille)
    verifier("plaque à portée de lecture, jamais lue : « non lu » 3 s après, sans voix",
             decision is not None and decision["decision"] == "non lu" and decision["voix"] == 0
             and abs(temps - 3.0) < 1e-9 and abs(decision["depuis_alerte_s"] - 3.0) < 1e-9, (decision, temps))
    suite = [(grande, rien)] * 29 + [(grande, lecture("lue", t)) for t in (A, B, C) * 30]   # jamais de majorité
    decision, temps = rejouer_cadres(suite, taille)
    verifier("première voix à 2,9 s : la piste garde l'échéance du vote (« non lu » à 5,9 s, avec ses voix)",
             decision["decision"] == "non lu" and decision["voix"] > 0 and abs(temps - 5.9) < 1e-9, (decision, temps))
    decision, temps = rejouer_cadres([(grande, rien)] * 20 + [(grande, lecture("lue", A))] * 10, taille)
    verifier("plaque lisible à portée : l'alerte ne gêne pas l'ouverture", decision["decision"] == "ouverture"
             and abs(temps - 2.2) < 1e-9, (decision, temps))
    decision, _ = rejouer_cadres([(petite, rien)] * 60 + [(None, rien)] * 15, taille)
    verifier("plaque restée sous X, jamais lue, puis partie : rien n'est noté (comme avant)", decision is None, decision)
    decision, temps = rejouer_cadres([(grande, rien)] * 15 + [(None, rien)] * 15, taille)
    verifier("plaque venue à portée puis repartie avant 3 s : « non lu » à la fermeture de la piste",
             decision is not None and decision["decision"] == "non lu" and decision["depuis_alerte_s"] is not None,
             (decision, temps))
    arret = Vote(AUTORISES, "majorite", 3, 10, 3.0, depart_alerte="arret")
    # Approche nette (le cadre grandit de 6 % par image) jusqu'à 1,9 s, puis voiture arrêtée sur ce dernier cadre
    approche = [(cadre(600 - 50 * 1.06 ** k, 400, 100 * 1.06 ** k, 22 * 1.06 ** k), rien) for k in range(20)]
    decision, temps = rejouer_cadres(approche + [(approche[-1][0], rien)] * 60, arret)
    verifier("« arret » : arrêtée à 1,9 s, alerte 1 s plus tard (2,9 s), « non lu » à 5,9 s",
             decision is not None and decision["decision"] == "non lu" and abs(temps - 5.9) < 1e-9
             and abs(decision["depuis_alerte_s"] - 3.0) < 1e-9, (decision, temps))
    hasard = np.random.RandomState(0)
    tremble = [(cadre(500 + hasard.uniform(-3, 3), 400, 200 + hasard.uniform(-3, 3), 44), rien) for _ in range(60)]
    decision, temps = rejouer_cadres(tremble, arret)
    verifier("« arret » : un cadre qui tremble de quelques pixels compte comme arrêté (« non lu » à 4 s)",
             decision is not None and abs(temps - 4.0) < 1e-9, (decision, temps))
    derive = [(cadre(300 + 4.0 * k, 400, 200, 44), rien) for k in range(60)]     # 40 px/s = 3 % de 1280 px
    decision, _ = rejouer_cadres(derive + [(None, rien)] * 15, arret)
    verifier("« arret » : une voiture qui avance lentement (3 % de la largeur par seconde) n'est pas arrêtée",
             decision is None, decision)
    sans = Vote(AUTORISES, "majorite", 3, 10, 3.0, depart_alerte=None)
    decision, _ = rejouer_cadres([(grande, rien)] * 60, sans)
    verifier("alerte désactivée : comportement d'avant la règle (aucune décision)", decision is None, decision)
    # Évaluation : jamais un X non mesuré
    from outils import evaluer
    if not PORTEE_MESUREE:
        sys.argv, refus = ["evaluer", "--test"], ""
        try:
            evaluer.main()
        except SystemExit as arret_programme:
            refus = str(arret_programme)
        verifier("X non mesuré : les mesures tournent sans l'alerte, le test final refuse de tourner",
                 evaluer.ALERTE_DES_MESURES == mesures.SANS_ALERTE and "refusé" in refus, refus)
    effet = lambda perdues, transformes, delai: {"perdues": perdues, "transformes": transformes, "delai_median": delai}
    choix = (mesures.choisir_alerte(effet(0, 3, 9.0), effet(0, 4, 9.0)),
             mesures.choisir_alerte(effet(0, 3, 9.0), effet(1, 5, 5.0)),
             mesures.choisir_alerte(effet(0, 3, 9.0), effet(0, 3, 9.0)),
             mesures.choisir_alerte(effet(0, 3, 9.0), effet(0, 3, 7.0)),
             mesures.choisir_alerte(effet(0, 3, 9.0), effet(0, 2, 5.0)))
    verifier("choix du départ : « arret » seulement s'il fait mieux sur les deux garde-fous ; à égalité, la taille",
             choix == ("arret", "taille", "taille", "arret", "taille"), choix)


def scenarios_interface():
    print("\n--- Interface web (outils/interface.py) : format des réponses, fichiers servis ---")
    ligne = {"heure": "2026-10-03 16:13:56", "numero": A, "decision": "ouverture", "autorise": "1", "voix": "3",
             "delai_s": "0.4", "capture": "20261003_161356_p6_215TU4567.jpg"}
    verifier("journal : texte du CSV -> nombres, booléen, heure HH:MM:SS, adresse de la capture",
             passage_api(ligne) == {"heure": "16:13:56", "date": "2026-10-03", "numero": A, "decision": "ouverture",
                                    "autorise": True, "voix": 3, "delai_s": 0.4,
                                    "capture": "/capture/20261003_161356_p6_215TU4567.jpg"}, passage_api(ligne))
    verifier("journal : une ligne incomplète (écriture en cours) est ignorée",
             passage_api({"heure": "2026-10-03 16:13:56", "numero": A, "voix": None}) is None)
    # État de la chaîne (sans modèle : etat() n'utilise que le suivi, le vote et la barrière)
    controle = ControleAcces(Vote({A}, "majorite", 3, 10, 3.0), Barriere())
    chaine = Chaine(None, None, controle)
    petite, grande = cadre(100, 400, 100, 22), cadre(500, 400, 200, 44)
    for piste, j in controle.associer(0.0, np.stack([petite, grande])):
        if j == 1:
            controle.voter(0.0, piste, lecture("lue", A))
    etat = chaine.etat(0.0)
    verifier("état : heure HH:MM:SS, temps de traitement dans l'ordre du pipeline",
             re.match(r"^\d\d:\d\d:\d\d$", etat["heure"]) is not None
             and list(etat["latences_ms"]) == ["detection", "redressement", "binarisation", "segmentation", "lecture"],
             etat["heure"])
    verifier("état : plus grande plaque d'abord, statut « en_cours », confiance vide sans voix",
             [p["id"] for p in etat["pistes"]] == [2, 1] and etat["pistes"][0]["statut"] == "en_cours"
             and etat["pistes"][0]["voix"] == 1 and etat["pistes"][1]["confiance"] is None, etat["pistes"])
    dossier = tempfile.mkdtemp()
    try:
        journal = JournalPassages(dossier)
        journal.enregistrer({"piste": 1, "decision": "refus", "numero": B, "autorise": False, "voix": 3,
                             "delai_s": 0.5, "heure": "2026-10-03 16:20:00"})
        client = creer_application(Instantane(etat), journal).test_client()
        reponse = client.get("/etat")
        ordre = re.findall(r'"(detection|redressement|binarisation|segmentation|lecture)"', reponse.get_data(as_text=True))
        verifier("/etat : JSON, temps de traitement non triés par ordre alphabétique",
                 reponse.status_code == 200 and reponse.is_json
                 and ordre == ["detection", "redressement", "binarisation", "segmentation", "lecture"], ordre)
        passages = client.get("/journal?n=5").get_json()
        verifier("/journal : liste au format de la page", len(passages) == 1 and passages[0]["numero"] == B
                 and passages[0]["autorise"] is False and passages[0]["heure"] == "16:20:00", passages)
        codes = {adresse: client.get(adresse).status_code
                 for adresse in ("/", "/support.js", "/vendor/react.production.min.js", "/secret.txt", "/etape/inconnue",
                                 "/etape/redressee", "/capture/..%2F..%2Fconfig%2Fautorises.csv", "/../config/autorises.csv")}
        verifier("fichiers : la page et ses scripts servis ; tout le reste refusé (404)",
                 codes == {"/": 200, "/support.js": 200, "/vendor/react.production.min.js": 200, "/secret.txt": 404,
                           "/etape/inconnue": 404, "/etape/redressee": 404,
                           "/capture/..%2F..%2Fconfig%2Fautorises.csv": 404, "/../config/autorises.csv": 404}, codes)
    finally:
        shutil.rmtree(dossier)
    choix = (ecran_disponible("PC", "win32", {}), ecran_disponible("Jetson", "linux", {"DISPLAY": ":0"}),
             ecran_disponible("PC", "linux", {}), ecran_disponible("PC", "linux", {"DISPLAY": ":0"}),
             ecran_disponible("PC", "linux", {"WAYLAND_DISPLAY": "wayland-0"}))
    verifier("navigateur ouvert par défaut : oui sur le PC Windows ; non sur la Jetson ; sous Linux, seulement avec un "
             "affichage graphique", choix == (True, False, False, True, True), choix)


def main():
    scenarios_suivi()
    scenarios_vote()
    scenarios_barriere()
    scenario_complet()
    scenarios_liste()
    scenarios_sources()
    scenarios_mesures()
    scenarios_alerte()
    scenarios_interface()
    print("\nBilan : {} vérifications, {} échec(s)".format(len(resultats), resultats.count(False)))
    sys.exit(0 if all(resultats) else 1)


if __name__ == "__main__":
    main()
