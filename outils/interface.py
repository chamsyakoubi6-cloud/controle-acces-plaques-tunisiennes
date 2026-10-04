"""Interface web locale du système (phase 5) : la page de supervision, servie par Flask (PC et Jetson).

La page est la maquette de Claude Design (interface/maquette/Supervision parking.dc.html), branchée sans changer son
apparence : elle interroge elle-même les adresses ci-dessous. Le système (systeme/chaine.py) tourne dans un fil
d'exécution à part. Après chaque image, il dépose un « instantané » protégé par un verrou : état, image annotée,
plaque détaillée. Flask ne fait que le lire.
  GET /                    la page
  GET /video               flux MJPEG (multipart/x-mixed-replace) : l'image annotée, une partie par image traitée
  GET /etat                état en direct (JSON) : heure, matériel, images/s, latences, barrière, pistes, décision
  GET /journal?n=50        derniers passages (JSON), le plus récent d'abord ; capture de chacun sur /capture/...
  GET /etape/<nom>         image (PNG) d'une étape du traitement de la plaque détaillée : decoupe, coins,
                           redressee, binaire, segmentation, vignettes
  GET /capture/<fichier>   capture d'un passage (journal des passages)
Sécurité :
  - écoute sur 127.0.0.1 par défaut : la page n'est visible que depuis cette machine. --ecoute (prévu pour la
    phase 6) permet de l'ouvrir depuis le PC relié à la Jetson : à n'utiliser que sur un câble ou un réseau local de
    confiance, car la page montre la vidéo et des numéros de plaque ;
  - jamais le mode debug de Flask : son débogueur permet d'exécuter du code à distance ;
  - seuls des fichiers nommés sont servis (la page, ses scripts, des captures au nom vérifié) : aucun chemin « ../ ».

Utilisation, depuis la racine du projet ; la page s'ouvre dans le navigateur dès que le serveur est prêt :
    python -m outils.interface                                      webcam
    python -m outils.interface --source validation/videos/215TU4567.mp4
    python -m outils.interface --source validation/photos --boucle  photos, en boucle (démonstration)
    python -m outils.interface --source validation/photos --duree-photo 5 --boucle
                                                                    photos montrées 5 s chacune (soutenance)
    python -m outils.interface --sans-navigateur                    sans ouvrir le navigateur
Le navigateur ne s'ouvre pas de lui-même sur la Jetson, ni sous Linux sans affichage graphique (SSH) : ouvrir
alors l'adresse affichée depuis un navigateur.
Une vidéo ou des photos sont lues au rythme réel (temps de la vidéo), comme une caméra.

Compatible Python 3.6, Flask 2.0.3 (PC et Jetson).
"""

import argparse
import logging
import os
import re
import sys
import threading
import time
import webbrowser

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from werkzeug.serving import make_server

from outils.demo import (DOSSIER_PASSAGES, LARGEUR_MAX, LECTEUR_PAR_DEFAUT, LISTE_PAR_DEFAUT, MODELE_PAR_DEFAUT,
                         charger_liste)
from systeme.action import Barriere, JournalPassages
from systeme.chaine import COULEURS_DECISIONS, MATERIEL, Chaine, ControleAcces
from systeme.debug import Debug
from systeme.decision import SEUIL
from systeme.detection import SEUIL_CONFIANCE, SEUIL_IOU, DetecteurPlaques
from systeme.lecture import LecteurCaracteres
from systeme.sources import ouvrir_source, reduire, repetitions_photo
from systeme.traitement import traiter_plaque
from systeme.vote import Vote, decrire_alerte

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_PAGE = os.path.join(RACINE, "interface", "maquette")
PAGE = "Supervision parking.dc.html"
SCRIPTS_PAGE = ("support.js", "vendor/react.production.min.js", "vendor/react-dom.production.min.js")
ETAPES = ("decoupe", "coins", "redressee", "binaire", "segmentation", "vignettes")
MOTIF_CAPTURE = re.compile(r"^[A-Za-z0-9_]+\.jpg$")   # noms écrits par systeme/action.py
LARGEUR_VIDEO = 960            # image du flux /video (la page la met à l'échelle)
QUALITE_JPEG = 80
PAUSE_ENTRE_BOUCLES = 5.0      # secondes de temps du système entre deux passages de la source (--boucle)
PAS_APRES_FIN = 0.2            # secondes : la barrière continue de vivre après la fin de la source
BLANC, JAUNE = (255, 255, 255), (0, 230, 230)


# --- Ce que dépose le fil du système --------------------------------------------------------------------------

class Instantane:
    """Dernier état du système, déposé après chaque image et lu par Flask. Condition = verrou + possibilité
    d'attendre la prochaine image (flux vidéo)."""

    def __init__(self, etat):
        self.condition = threading.Condition()
        self.etat = etat              # réponse de /etat
        self.jpeg = None              # image annotée, encodée en JPEG (/video)
        self.numero_image = 0         # compteur : une nouvelle image est arrivée
        self.plaque = None            # plaque détaillée (systeme/chaine.py) : /etape
        self.etapes = None            # (clé de la plaque, images des étapes) : calculées une fois par plaque


def annoter(image, chaine, temps):
    """Image du flux /video : réduite à 960 px de large, chaque piste vue encadrée avec son numéro et sa lecture en
    tête (jaune, voix sur K), son alerte si sa plaque n'a aucune voix, ou sa décision (couleurs de systeme/chaine.py).
    Le détail est sur la page."""
    vue = reduire(image, LARGEUR_VIDEO).copy()   # copie : l'image d'origine sert encore (captures, étapes)
    echelle = vue.shape[1] / float(image.shape[1])
    vote = chaine.controle.vote
    for piste in chaine.controle.suivi.pistes:
        if not piste.vue:
            continue
        x1, y1, x2, y2 = (int(v * echelle) for v in piste.boite[:4])
        if piste.decision is not None:
            couleur = COULEURS_DECISIONS[piste.decision["decision"]]
            texte = "{} {}".format(piste.decision["decision"].upper(), piste.decision["numero"])
        elif piste.voix:
            couleur = JAUNE
            meneur, voix, _ = vote.classement(piste)[0]
            texte = "{} {}/{}".format(meneur, voix, vote.voix_min)
        elif piste.debut_alerte is not None:      # plaque sans voix à portée de lecture : « non lu » à l'échéance
            couleur = BLANC
            texte = "alerte {:.1f}/{:.0f} s".format(temps - piste.debut_alerte, vote.attente_alerte)
        else:
            couleur, texte = BLANC, "lecture"
        cv2.rectangle(vue, (x1, y1), (x2, y2), couleur, 2)
        position = (x1, max(16, y1 - 7))
        cv2.putText(vue, "{} : {}".format(piste.id, texte), position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(vue, "{} : {}".format(piste.id, texte), position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, couleur, 1)
    return vue


def deposer(instantane, chaine, temps, image):
    """Après chaque image : état, image annotée et plaque détaillée, sous verrou ; réveille le flux vidéo."""
    _, jpeg = cv2.imencode(".jpg", annoter(image, chaine, temps), [cv2.IMWRITE_JPEG_QUALITY, QUALITE_JPEG])
    etat = chaine.etat(temps)
    with instantane.condition:
        instantane.etat = etat
        instantane.jpeg = jpeg.tobytes()
        instantane.plaque = chaine.plaque
        instantane.numero_image += 1
        instantane.condition.notify_all()


def patienter(chaine, instantane, temps, duree):
    """Attend « duree » secondes de temps réel sans image (None : indéfiniment), en faisant vivre la barrière :
    elle se referme comme devant une vraie voie (écart entre deux photos, pause entre deux passages en boucle, fin
    de la source). temps : temps du système au début de l'attente."""
    debut = time.monotonic()
    while True:
        ecoule = time.monotonic() - debut
        if duree is not None and ecoule >= duree:
            return
        if chaine.controle.barriere is not None:
            chaine.controle.barriere.mettre_a_jour(temps + ecoule)
        etat = chaine.etat(temps + ecoule)
        with instantane.condition:
            instantane.etat = etat
        time.sleep(PAS_APRES_FIN if duree is None else min(PAS_APRES_FIN, duree - ecoule))


def faire_tourner(chaine, ouvrir, boucle, instantane):
    """Fil du système. Une vidéo ou des photos sont lues au rythme réel (temps de la source), comme une caméra.
    Avec --boucle, la source recommence après une pause, et le temps du système continue d'avancer : le suivi et la
    barrière ne supportent pas un temps qui recule."""
    decalage, temps = 0.0, 0.0
    try:
        while True:
            images, _, en_direct = ouvrir()
            debut_reel, debut_source = time.monotonic(), None
            for temps_source, image in images:
                if not en_direct:
                    if debut_source is None:
                        debut_source = temps_source
                    attente = (temps_source - debut_source) - (time.monotonic() - debut_reel)
                    if attente > 0:
                        patienter(chaine, instantane, temps, attente)
                temps = decalage + temps_source
                chaine.traiter(temps, image)
                deposer(instantane, chaine, temps, image)
            chaine.finir(temps)
            if not boucle:
                break
            patienter(chaine, instantane, temps, PAUSE_ENTRE_BOUCLES)
            decalage = temps + PAUSE_ENTRE_BOUCLES
    except (IOError, ValueError) as erreur:
        print("Arrêt de la source :", erreur)
    chaine.cadence = 0.0          # plus aucune image traitée
    patienter(chaine, instantane, temps, None)


# --- Réponses de Flask ----------------------------------------------------------------------------------------

def flux_mjpeg(instantane):
    """Une partie JPEG par image traitée (multipart/x-mixed-replace : le navigateur remplace l'image à chaque
    partie). Sans nouvelle image (fin de la source), la dernière est renvoyée chaque seconde."""
    vue = -1
    while True:
        with instantane.condition:
            instantane.condition.wait_for(lambda: instantane.numero_image != vue, timeout=1.0)
            jpeg, vue = instantane.jpeg, instantane.numero_image
        if jpeg is not None:
            yield (b"--image\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode("ascii")
                   + b"\r\n\r\n" + jpeg + b"\r\n")


def images_etapes(instantane):
    """Images des étapes de la plaque détaillée : le traitement est refait en mode debug une seule fois par
    nouvelle lecture (la page redemande les six images chaque seconde). binaire : l'image binaire réellement
    utilisée par la segmentation."""
    with instantane.condition:
        plaque, cache = instantane.plaque, instantane.etapes
    if plaque is None:
        return {}
    cle = (plaque["piste"], plaque["temps"])
    if cache is not None and cache[0] == cle:
        return cache[1]
    debug = Debug()
    resultat = traiter_plaque(plaque["image"], plaque["cadre"], debug)
    images = dict(debug.etapes)
    etapes = {"decoupe": images.get("decoupe"), "coins": images.get("coins"), "redressee": images.get("redressee"),
              "binaire": resultat["binaire"], "segmentation": images.get("segmentation"),
              "vignettes": images.get("vignettes_32x32")}
    with instantane.condition:
        instantane.etapes = (cle, etapes)
    return etapes


def passage_api(ligne):
    """Ligne du journal des passages (CSV, texte) -> passage au format de /journal (nombres, booléens, heure
    HH:MM:SS comme sur la page, date à part). None si la ligne est incomplète (écriture en cours)."""
    try:
        heure = ligne["heure"] or ""
        delai = ligne["delai_s"] or ""
        return {"heure": heure[-8:], "date": heure[:10], "numero": ligne["numero"] or "",
                "decision": ligne["decision"] or "", "autorise": ligne["autorise"] == "1",
                "voix": int(ligne["voix"] or 0), "delai_s": float(delai) if delai else None,
                "capture": "/capture/" + ligne["capture"] if ligne["capture"] else ""}
    except (KeyError, TypeError, ValueError):
        return None


def creer_application(instantane, journal):
    application = Flask(__name__, static_folder=None)
    # Flask 2.0 trie les clés des réponses JSON : la page afficherait les temps de traitement par ordre alphabétique
    # au lieu de l'ordre du pipeline (detection, redressement, binarisation, segmentation, lecture).
    application.config["JSON_SORT_KEYS"] = False

    @application.route("/")
    def page():
        return send_from_directory(DOSSIER_PAGE, PAGE, mimetype="text/html")

    @application.route("/<path:fichier>")
    def script_page(fichier):
        if fichier not in SCRIPTS_PAGE:
            abort(404)
        # Type donné explicitement : sous Windows, le registre peut associer .js à text/plain
        return send_from_directory(DOSSIER_PAGE, fichier, mimetype="application/javascript")

    @application.route("/etat")
    def etat():
        with instantane.condition:
            return jsonify(instantane.etat)

    @application.route("/journal")
    def journal_passages():
        nombre = max(1, min(request.args.get("n", default=50, type=int), 500))
        passages = [passage_api(ligne) for ligne in journal.derniers(nombre)]
        return jsonify([p for p in passages if p is not None])

    @application.route("/video")
    def video():
        return Response(flux_mjpeg(instantane), mimetype="multipart/x-mixed-replace; boundary=image")

    @application.route("/etape/<nom>")
    def etape(nom):
        image = images_etapes(instantane).get(nom) if nom in ETAPES else None
        if image is None:
            abort(404)
        _, png = cv2.imencode(".png", image)
        return Response(png.tobytes(), mimetype="image/png", headers={"Cache-Control": "no-store"})

    @application.route("/capture/<fichier>")
    def capture(fichier):
        if not MOTIF_CAPTURE.match(fichier):
            abort(404)
        return send_from_directory(journal.dossier_captures, fichier, mimetype="image/jpeg")

    return application


def ecran_disponible(materiel=MATERIEL, plateforme=sys.platform, environnement=os.environ):
    """Peut-on ouvrir un navigateur sur cette machine ? Pas sur la Jetson : elle tourne sans écran, et sa page
    s'ouvre depuis le PC (phase 6). Pas non plus sous Linux sans affichage graphique (DISPLAY ou WAYLAND_DISPLAY
    absents, par exemple en SSH) : le module webbrowser y lancerait un navigateur en mode texte, qui bloquerait la
    console. Paramètres : pour les essais (outils/verifier_logique.py)."""
    if materiel == "Jetson":
        return False
    if plateforme.startswith("linux"):
        return bool(environnement.get("DISPLAY") or environnement.get("WAYLAND_DISPLAY"))
    return True


def main():
    parseur = argparse.ArgumentParser(description="Interface web locale du système (page de supervision).")
    parseur.add_argument("--source", default="0",
                         help="index de webcam (défaut : 0), fichier vidéo, photo ou dossier de photos")
    parseur.add_argument("--cadence", type=float, default=10.0,
                         help="vidéo : images traitées par seconde de vidéo (défaut : 10, comme l'évaluation)")
    parseur.add_argument("--duree-photo", type=float, default=1.0,
                         help="photos : durée d'affichage de chaque photo, en secondes (défaut : 1 ; par exemple 5 "
                              "pour laisser à l'échéance de 3 s le temps de tomber pendant une démonstration)")
    parseur.add_argument("--boucle", action="store_true", help="recommencer la vidéo ou les photos à la fin")
    parseur.add_argument("--modele", default=MODELE_PAR_DEFAUT, help="détecteur : .onnx (PC) ou .engine (Jetson)")
    parseur.add_argument("--lecteur", default=LECTEUR_PAR_DEFAUT, help="CNN de lecture : .onnx (PC) ou .engine")
    parseur.add_argument("--autorises", default=LISTE_PAR_DEFAUT, help="liste des autorisés (défaut : config/autorises.csv)")
    parseur.add_argument("--journal", default=DOSSIER_PASSAGES, help="dossier du journal des passages")
    parseur.add_argument("--seuil", type=float, default=SEUIL, help="seuil de confiance de la lecture")
    parseur.add_argument("--ecoute", default="127.0.0.1",
                         help="adresse d'écoute (défaut : 127.0.0.1, cette machine seulement)")
    parseur.add_argument("--port", type=int, default=5000, help="port (défaut : 5000)")
    parseur.add_argument("--sans-navigateur", action="store_true",
                         help="ne pas ouvrir le navigateur au démarrage (jamais ouvert sur la Jetson ni sans écran)")
    arguments = parseur.parse_args()
    if arguments.ecoute != "127.0.0.1":
        print("Attention : la page sera visible depuis le réseau, à l'adresse {}. Elle montre la vidéo et des numéros "
              "de plaque : seulement sur un câble ou un réseau local de confiance.".format(arguments.ecoute))

    try:
        autorises = charger_liste(arguments.autorises)
    except (IOError, ValueError) as erreur:
        sys.exit(str(erreur))
    detecteur = DetecteurPlaques(arguments.modele, SEUIL_CONFIANCE, SEUIL_IOU)
    journal = JournalPassages(arguments.journal)
    chaine = Chaine(detecteur, LecteurCaracteres(arguments.lecteur), ControleAcces(Vote(autorises), Barriere()),
                    journal, arguments.seuil)
    instantane = Instantane(chaine.etat(0.0))

    def ouvrir():
        return ouvrir_source(arguments.source, arguments.cadence or None, LARGEUR_MAX,
                             repetitions_photo(arguments.duree_photo))

    try:
        description = ouvrir()[1]      # vérifie tout de suite qu'une vidéo ou un dossier existe
    except IOError as erreur:
        sys.exit(str(erreur))
    # Le serveur écoute dès sa création (make_server au lieu de app.run, qui bloquerait) : on sait alors qu'il est
    # prêt. Port déjà pris : arrêt ici, avant d'allumer la caméra ou d'ouvrir le navigateur. Jamais de mode debug.
    try:
        serveur = make_server(arguments.ecoute, arguments.port, creer_application(instantane, journal), threaded=True)
    except OSError as erreur:
        sys.exit("Port {} indisponible ({}) : l'interface tourne-t-elle déjà ? Sinon, essayer --port 5051".format(
            arguments.port, erreur))
    fil = threading.Thread(target=faire_tourner, args=(chaine, ouvrir, arguments.boucle, instantane), daemon=True)
    fil.start()
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # sinon une ligne par requête (3 par seconde)
    # Écoute sur toutes les interfaces (0.0.0.0) : le navigateur de cette machine passe par 127.0.0.1
    adresse = "127.0.0.1" if arguments.ecoute in ("0.0.0.0", "::") else arguments.ecoute
    url = "http://{}:{}/".format(adresse, arguments.port)
    print("Source : {} ; liste : {} plaques".format(description, len(autorises)))
    print(decrire_alerte(chaine.controle.vote))
    print("Interface : {}  (Ctrl+C pour arrêter)".format(url))
    if not arguments.sans_navigateur:
        if ecran_disponible():
            # Le serveur écoute déjà : la demande du navigateur attend dans sa file, puis serve_forever y répond
            webbrowser.open(url)
        else:
            print("Navigateur non ouvert ({}) : ouvrir l'adresse ci-dessus depuis un navigateur".format(
                "Jetson" if MATERIEL == "Jetson" else "pas d'affichage graphique"))
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        print("Arrêt de l'interface")


if __name__ == "__main__":
    main()
