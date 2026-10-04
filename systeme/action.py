"""Étape 7 du pipeline : liste des autorisés, barrière et journal des passages (phase 5).

Liste des autorisés : config/autorises.csv (hors de git), une plaque par ligne dans la colonne « numero », au
format « 215 TU 4567 » (espaces facultatifs). Un exemple fictif est versionné : config/autorises.exemple.csv.
Barrière : la logique (quand ouvrir, quand fermer) est la même sur le PC et sur la Jetson ; seule la commande
change, simulée sur le PC (l'état est affiché), servo réel en phase 6.
Fermeture (décision de l'étudiant) : 3 s après que la plaque qui a ouvert a quitté l'image, et jamais moins de
5 s après l'ouverture. Limite, pour le rapport : une vraie installation exige un capteur de présence (boucle au
sol) pour ne jamais refermer sur une voiture, car la caméra perd la plaque avant dès que la voiture s'engage
sous la barrière.
Journal des passages : sorties/passages/journal.csv (heure, numéro, décision, autorisé, voix, délai, capture)
et l'image de chaque décision dans sorties/passages/captures/. Tout reste en local : sorties/ est hors de git.

Compatible Python 3.6, OpenCV 4.1.1 (PC et Jetson).
"""

import csv
import os
import re
import time

import cv2

from systeme.decision import format_valide

OUVERTURE_MIN = 5.0             # secondes
FERMETURE_APRES_DEPART = 3.0    # secondes sans voir la plaque qui a ouvert
COLONNES_JOURNAL = ("heure", "numero", "decision", "autorise", "voix", "delai_s", "capture")
MOTIF_NUMERO = re.compile(r"^\s*(\d{1,3})\s*TU\s*(\d{1,4})\s*$", re.IGNORECASE)


def normaliser_numero(texte):
    """« 215 TU 4567 », « 215TU4567 », « 215 tu 4567 » -> « 215 TU 4567 ». None si le texte n'a pas le format
    officiel (série de 1 à 3 chiffres, numéro de 1 à 4, sans zéro en tête)."""
    trouve = MOTIF_NUMERO.match(texte)
    if trouve is None or not format_valide(trouve.group(1), trouve.group(2)):
        return None
    return "{} TU {}".format(trouve.group(1), trouve.group(2))


def charger_autorises(chemin):
    """Liste des autorisés -> ensemble de numéros normalisés. Une ligne illisible arrête le chargement
    (ValueError) : mieux vaut corriger le fichier que de faire tourner la barrière sur une liste incomplète."""
    autorises = set()
    with open(chemin, encoding="utf-8", newline="") as fichier:
        lecteur = csv.DictReader(fichier)
        if "numero" not in (lecteur.fieldnames or []):
            raise ValueError("{} : colonne « numero » absente".format(chemin))
        for ligne in lecteur:
            numero = normaliser_numero(ligne["numero"] or "")
            if numero is None:
                raise ValueError("{}, ligne {} : numéro illisible « {} » (attendu : 215 TU 4567)".format(
                    chemin, lecteur.line_num, ligne["numero"]))
            autorises.add(numero)
    return autorises


class CommandeSimulee:
    """Commande simulée (PC) : ne pilote rien. La démonstration et l'interface web affichent l'état."""

    nom = "simulee"

    def ouvrir(self):
        pass

    def fermer(self):
        pass


class CommandeServo:
    """Commande du servo réel (phase 6, Jetson). Le moyen de commande reste à choisir : signal PWM d'une broche
    de la Jetson, carte PCA9685 en I2C, ou microcontrôleur relié en série."""

    nom = "servo"

    def __init__(self):
        raise NotImplementedError("Commande du servo : prévue en phase 6")


class Barriere:
    """Ouverture sur décision, fermeture automatique (voir l'en-tête). Les temps sont ceux de la source."""

    def __init__(self, commande=None, ouverture_min=OUVERTURE_MIN, fermeture_apres_depart=FERMETURE_APRES_DEPART):
        self.commande = commande or CommandeSimulee()
        self.ouverture_min = ouverture_min
        self.fermeture_apres_depart = fermeture_apres_depart
        self.commande.fermer()     # état connu au démarrage : fermée
        self.etat = "fermee"
        self.depuis = 0.0          # temps du dernier changement d'état
        self.ouverte_a = None      # temps de la dernière décision d'ouverture
        self.derniere_presence = None

    def ouvrir(self, temps):
        """Décision d'ouverture : ouvre, ou garde ouverte (une autre voiture autorisée), et relance les délais."""
        if self.etat == "fermee":
            self.commande.ouvrir()
            self.etat, self.depuis = "ouverte", temps
        self.ouverte_a = temps
        self.derniere_presence = temps

    def signaler_presence(self, temps):
        """La plaque d'une piste qui a ouvert la barrière est encore détectée."""
        self.derniere_presence = temps

    def mettre_a_jour(self, temps):
        """Ferme si l'ouverture dure depuis assez longtemps ET si la plaque est partie depuis assez longtemps."""
        if (self.etat == "ouverte" and temps - self.ouverte_a >= self.ouverture_min
                and temps - self.derniere_presence >= self.fermeture_apres_depart):
            self.commande.fermer()
            self.etat, self.depuis = "fermee", temps

    def resume(self, temps):
        return {"etat": self.etat, "depuis_s": round(temps - self.depuis, 1)}


class JournalPassages:
    """Journal des passages (CSV) et image de chaque décision, dans le dossier donné."""

    def __init__(self, dossier):
        self.dossier_captures = os.path.join(dossier, "captures")
        os.makedirs(self.dossier_captures, exist_ok=True)
        self.chemin = os.path.join(dossier, "journal.csv")
        if not os.path.exists(self.chemin):
            with open(self.chemin, "w", encoding="utf-8", newline="") as fichier:
                csv.DictWriter(fichier, fieldnames=COLONNES_JOURNAL).writeheader()

    def enregistrer(self, decision, image=None):
        """Ajoute une décision (dictionnaire de systeme/vote.py) ; image : capture à enregistrer (ou None).
        Renvoie la ligne écrite."""
        heure = decision.get("heure") or time.strftime("%Y-%m-%d %H:%M:%S")
        capture = ""
        if image is not None:
            # Nom ASCII sans espace : heure de la décision (« 2026-10-02 16:28:28 » -> « 20261002_162828 »),
            # piste et numéro (deux décisions dans la même seconde restent distinctes)
            horodatage = heure.replace("-", "").replace(":", "").replace(" ", "_")
            capture = "{}_p{}_{}.jpg".format(horodatage, decision["piste"],
                                             decision["numero"].replace(" ", "") or "inconnu")
            cv2.imwrite(os.path.join(self.dossier_captures, capture), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        delai = decision["delai_s"]
        ligne = {"heure": heure, "numero": decision["numero"], "decision": decision["decision"],
                 "autorise": int(decision["autorise"]), "voix": decision["voix"],
                 "delai_s": "" if delai is None else "{:.1f}".format(delai), "capture": capture}
        with open(self.chemin, "a", encoding="utf-8", newline="") as fichier:
            csv.DictWriter(fichier, fieldnames=COLONNES_JOURNAL).writerow(ligne)
        return ligne

    def derniers(self, nombre=50):
        """Les « nombre » dernières lignes du journal, la plus récente d'abord."""
        with open(self.chemin, encoding="utf-8", newline="") as fichier:
            lignes = list(csv.DictReader(fichier))
        return lignes[::-1][:nombre]
