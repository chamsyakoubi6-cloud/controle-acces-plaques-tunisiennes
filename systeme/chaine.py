"""Le système complet, image par image (phase 5).

Deux niveaux, pour que ce qui est mesuré soit exactement ce qui tourne :
  ControleAcces : suivi, vote et barrière, c'est-à-dire tout ce qui DÉCIDE, sans aucune image. L'évaluation le
                  rejoue sur des lectures enregistrées, une fois par réglage du vote.
  Chaine        : la boucle complète sur une image. Détection ; pour chaque plaque à lire, traitement classique et
                  lecture du CNN ; puis ControleAcces, journal des passages et état affiché. Elle sert la
                  démonstration (outils/demo.py) et l'interface web.
Seules les pistes sans décision sont lues. Une fois la décision prise, la plaque est seulement suivie (pour la
fermeture de la barrière), ce qui économise le calcul sur la Jetson.
Latences affichées : moyennes glissantes ; la détection par image, les autres étapes par plaque lue.

Compatible Python 3.6, NumPy 1.19, OpenCV 4.1.1 (PC et Jetson).
"""

import platform
import time

import cv2

from systeme.decision import SEUIL, lire_numero
from systeme.sources import reduire
from systeme.suivi import Suivi
from systeme.traitement import traiter_plaque

MATERIEL = "Jetson" if platform.machine() == "aarch64" else "PC"
ETAPES = ("detection", "redressement", "binarisation", "segmentation", "lecture")
LISSAGE = 0.1             # moyennes glissantes : poids de la nouvelle mesure
LARGEUR_CAPTURE = 960     # image enregistrée au journal (réduite : environ 100 Ko en JPEG)
COULEURS_DECISIONS = {"ouverture": (0, 200, 0), "refus": (0, 0, 230), "non lu": (0, 140, 255)}   # BGR


def heure():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def aire(cadre):
    return float((cadre[2] - cadre[0]) * (cadre[3] - cadre[1]))


class ControleAcces:
    """Suivi des plaques, vote et barrière (facultative) : la décision, sans image.
    Chaque méthode renvoie les pistes décidées à l'instant (leur décision est dans piste.decision)."""

    def __init__(self, vote, barriere=None, suivi=None):
        self.vote = vote
        self.barriere = barriere
        self.suivi = suivi or Suivi()
        self.fermees = []         # pistes fermées par la dernière association

    def associer(self, temps, detections, largeur_image=None):
        """Rattache les détections de l'image aux pistes. Renvoie les plaques à lire : liste de (piste, indice de
        la détection) pour les pistes sans décision. Signale à la barrière les plaques qui l'ont ouverte.
        largeur_image : pour l'alerte « plaque sans voix » (taille de la plaque, arrêt de la voiture) ; None :
        pas d'alerte."""
        associations, self.fermees = self.suivi.mettre_a_jour(temps, detections)
        a_lire = []
        for piste, indice in associations:
            self.vote.observer(piste, temps, largeur_image)
            if piste.decision is None:
                a_lire.append((piste, indice))
            elif piste.decision["decision"] == "ouverture" and self.barriere is not None:
                self.barriere.signaler_presence(temps)
        return a_lire

    def voter(self, temps, piste, lecture):
        """Lecture d'une plaque à lire (dictionnaire de systeme/decision.py)."""
        return self.appliquer(piste, self.vote.ajouter(piste, temps, lecture), temps)

    def finir_image(self, temps):
        """Après les lectures de l'image : échéances des pistes actives, pistes fermées, barrière."""
        decidees = [self.appliquer(p, self.vote.examiner(p, temps), temps) for p in self.suivi.pistes]
        decidees += [self.appliquer(p, self.vote.fermer(p, temps), temps) for p in self.fermees]
        self.fermees = []
        if self.barriere is not None:
            self.barriere.mettre_a_jour(temps)
        return [p for p in decidees if p is not None]

    def finir(self, temps):
        """Fin de la source : ferme toutes les pistes (« non lu » pour celles qui le méritent)."""
        decidees = [self.appliquer(p, self.vote.fermer(p, temps), temps) for p in self.suivi.tout_fermer()]
        return [p for p in decidees if p is not None]

    def appliquer(self, piste, decision, temps):
        """Renvoie la piste si une décision vient de tomber (et ouvre la barrière si c'est une ouverture)."""
        if decision is None:
            return None
        if decision["decision"] == "ouverture" and self.barriere is not None:
            self.barriere.ouvrir(temps)
        return piste


class Chaine:
    """Le système complet sur chaque image : détection, lecture des plaques, décision, action, état affiché."""

    def __init__(self, detecteur, lecteur, controle, journal=None, seuil_lecture=SEUIL):
        self.detecteur = detecteur
        self.lecteur = lecteur
        self.controle = controle
        self.journal = journal
        self.seuil_lecture = seuil_lecture
        self.latences = {}               # secondes, moyennes glissantes
        self.cadence = 0.0               # images traitées par seconde (horloge réelle), moyenne glissante
        self.fin_precedente = None
        self.detections = None           # détections de la dernière image (affichage)
        self.lectures_pistes = {}        # id de piste active -> sa dernière lecture : temps, piste, image, cadre,
                                         # resultat (traitement), lecture (CNN) ; pour la capture et l'affichage
        self.plaque = None               # plaque détaillée à l'affichage : dernière lecture de la plus grande
                                         # plaque visible (la voiture la plus proche), ou None
        self.derniere_decision = None

    def mesurer(self, etape, duree):
        ancienne = self.latences.get(etape)
        self.latences[etape] = duree if ancienne is None else (1 - LISSAGE) * ancienne + LISSAGE * duree

    def traiter(self, temps, image):
        """Traite une image de la source ; renvoie les pistes décidées sur cette image."""
        self.detections = self.detecteur.detecter(image)
        self.mesurer("detection", sum(self.detecteur.durees.values()))
        decidees = []
        for piste, indice in self.controle.associer(temps, self.detections, image.shape[1]):
            cadre = self.detections[indice]
            resultat = traiter_plaque(image, cadre)
            for etape in ("redressement", "binarisation", "segmentation"):
                self.mesurer(etape, resultat["durees"][etape])
            debut = time.perf_counter()
            lecture = lire_numero(resultat, self.lecteur, self.seuil_lecture)
            self.mesurer("lecture", time.perf_counter() - debut)
            # Image et cadre gardés : capture du journal ; l'interface pourra refaire le traitement en mode debug
            self.lectures_pistes[piste.id] = {"temps": temps, "piste": piste.id, "image": image, "cadre": cadre,
                                              "resultat": resultat, "lecture": lecture}
            if self.controle.voter(temps, piste, lecture) is not None:
                decidees.append(piste)
        decidees += self.controle.finir_image(temps)
        for piste in decidees:
            self.noter(piste)
        # Lectures des pistes fermées oubliées (après noter : la capture d'une piste qui se ferme en a besoin).
        # Plaque détaillée : la plus grande plaque visible, avec sa dernière lecture, même si sa décision est prise
        # et qu'elle n'est plus lue.
        actives = set(p.id for p in self.controle.suivi.pistes)
        self.lectures_pistes = {n: l for n, l in self.lectures_pistes.items() if n in actives}
        visibles = [p for p in self.controle.suivi.pistes if p.vue and p.id in self.lectures_pistes]
        self.plaque = self.lectures_pistes[max(visibles, key=lambda p: aire(p.boite)).id] if visibles else None
        fin = time.perf_counter()
        if self.fin_precedente is not None:
            instantanee = 1.0 / max(fin - self.fin_precedente, 1e-6)
            self.cadence = instantanee if self.cadence == 0 else (1 - LISSAGE) * self.cadence + LISSAGE * instantanee
        self.fin_precedente = fin
        return decidees

    def finir(self, temps):
        """Fin de la source : décisions des pistes encore ouvertes."""
        decidees = self.controle.finir(temps)
        for piste in decidees:
            self.noter(piste)
        self.lectures_pistes = {}   # la plaque détaillée reste affichée sur l'écran final
        return decidees

    def capture(self, piste):
        """Image de la dernière lecture de la piste, cadre de la plaque dessiné, réduite ; None si inconnue."""
        if piste.id not in self.lectures_pistes:
            return None
        cadre = self.lectures_pistes[piste.id]["cadre"]
        vue = self.lectures_pistes[piste.id]["image"].copy()
        couleur = COULEURS_DECISIONS.get(piste.decision["decision"], (255, 255, 255))
        cv2.rectangle(vue, (int(cadre[0]), int(cadre[1])), (int(cadre[2]), int(cadre[3])), couleur, 3)
        return reduire(vue, LARGEUR_CAPTURE)

    def noter(self, piste):
        """Décision d'une piste : heure, journal des passages (avec la capture), dernière décision affichée."""
        decision = piste.decision
        decision["heure"] = heure()
        if self.journal is not None:
            decision["capture"] = self.journal.enregistrer(decision, self.capture(piste))["capture"]
        self.derniere_decision = decision

    def etat(self, temps):
        """État du système pour l'affichage (contrat de /etat de l'interface web). Heures au format HH:MM:SS,
        celui qu'affiche la maquette (le journal des passages garde la date complète). Pistes : les plaques vues sur
        l'image d'abord, de la plus grande à la plus petite (la maquette détaille la première : la voiture la plus
        proche de la barrière)."""
        derniere = None
        if self.derniere_decision is not None:
            d = self.derniere_decision
            derniere = {"heure": d["heure"][-8:], "numero": d["numero"], "decision": d["decision"],
                        "autorise": d["autorise"],
                        "delai_s": None if d["delai_s"] is None else round(d["delai_s"], 1)}
        barriere = self.controle.barriere
        pistes = sorted(self.controle.suivi.pistes, key=lambda p: (p.vue, aire(p.boite)), reverse=True)
        return {"heure": heure()[-8:], "materiel": MATERIEL, "images_par_seconde": round(self.cadence, 1),
                "latences_ms": {etape: round(1000 * self.latences.get(etape, 0.0), 1) for etape in ETAPES},
                "barriere": None if barriere is None else barriere.resume(temps),
                "pistes": [self.controle.vote.decrire(p, temps) for p in pistes],
                "derniere_decision": derniere}
