"""Suivi des plaques d'une image à l'autre (phase 5) : chaque plaque détectée est rattachée à une « piste ».

Association par recouvrement : une détection prolonge la piste dont le dernier cadre la recouvre le plus
(IoU >= 0,3). Les paires sont prises de la plus recouvrante à la moins recouvrante (appariement glouton), et une
piste ou une détection ne sert qu'une fois. Une détection qui ne recouvre aucune piste en ouvre une nouvelle ; une
piste sans détection pendant 1 s est fermée.
Pourquoi si simple : la caméra est fixe, et une voiture qui approche de la barrière bouge peu d'une image à la
suivante. À 10 images/s, à 3 m et 1 m/s, son cadre grandit d'environ 3 % entre deux images : les cadres se
recouvrent à plus de 90 %. Un filtre de Kalman (prédiction du mouvement) serait superflu ici, et plus difficile à
justifier. La tolérance de 1 s couvre quelques images où le détecteur manque la plaque (reflet, flou).
Aucune image ici, seulement des cadres et des temps : l'évaluation peut rejouer le suivi sur des détections
enregistrées.

Compatible Python 3.6, NumPy 1.19 (PC et Jetson).
"""

from systeme.detection import iou

SEUIL_IOU = 0.3
DUREE_PERTE = 1.0     # secondes sans détection avant de fermer une piste


class Piste:
    """Une plaque suivie d'image en image, avec tout ce que le vote (systeme/vote.py) en retient."""

    def __init__(self, numero, temps, boite):
        self.id = numero
        self.debut = temps            # première détection
        self.derniere_vue = temps     # dernière détection
        self.boite = boite            # dernier cadre : x1, y1, x2, y2, score
        self.vue = True               # détectée sur l'image en cours ?
        # Rempli par le vote
        self.voix = []                # (temps, texte, confiance) de chaque lecture qui vote
        self.premiere_voix = None     # temps de la première voix : départ du chronomètre
        self.forme_de_plaque = False  # au moins une lecture en forme de plaque (lue, doute ou improbable)
        self.derniere_lecture = None  # dernière lecture (dictionnaire de systeme/decision.py)
        self.decision = None          # décision prise pour cette piste (une seule)
        self.debut_alerte = None      # départ de l'alerte « plaque sans voix » (à portée de lecture, ou arrêtée)
        self.historique = []          # (temps, cadre) de la dernière seconde : variante « arrêt » de l'alerte


class Suivi:
    """Pistes actives et rattachement des détections de chaque image."""

    def __init__(self, seuil_iou=SEUIL_IOU, duree_perte=DUREE_PERTE):
        self.seuil_iou = seuil_iou
        self.duree_perte = duree_perte
        self.pistes = []
        self.nb_pistes = 0            # compteur : identifiant de la prochaine piste

    def mettre_a_jour(self, temps, detections):
        """detections : tableau (K, 5) de l'image (x1, y1, x2, y2, score).
        Renvoie (associations, fermees) : liste de (piste, indice de la détection) pour chaque détection de
        l'image (pistes prolongées ou nouvelles), et liste des pistes fermées faute de détection."""
        paires = []
        for i, piste in enumerate(self.pistes):
            if len(detections) == 0:
                break
            recouvrements = iou(piste.boite[:4], detections[:, :4])
            paires.extend((float(r), i, j) for j, r in enumerate(recouvrements) if r >= self.seuil_iou)
        paires.sort(key=lambda paire: paire[0], reverse=True)
        for piste in self.pistes:
            piste.vue = False
        associations, pistes_prises, detections_prises = [], set(), set()
        for _, i, j in paires:
            if i in pistes_prises or j in detections_prises:
                continue
            pistes_prises.add(i)
            detections_prises.add(j)
            piste = self.pistes[i]
            piste.boite, piste.derniere_vue, piste.vue = detections[j], temps, True
            associations.append((piste, j))
        for j in range(len(detections)):
            if j not in detections_prises:
                self.nb_pistes += 1
                piste = Piste(self.nb_pistes, temps, detections[j])
                self.pistes.append(piste)
                associations.append((piste, j))
        fermees = [p for p in self.pistes if temps - p.derniere_vue > self.duree_perte]
        self.pistes = [p for p in self.pistes if temps - p.derniere_vue <= self.duree_perte]
        return associations, fermees

    def tout_fermer(self):
        """Fin de la source (vidéo terminée) : ferme toutes les pistes et les renvoie."""
        fermees, self.pistes = self.pistes, []
        return fermees
