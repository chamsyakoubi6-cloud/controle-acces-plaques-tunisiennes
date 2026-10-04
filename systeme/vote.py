"""Étape 6 du pipeline, sur plusieurs images : vote et décision pour chaque piste (phase 5).

Une lecture isolée peut se tromper (un chiffre perdu, une confusion), mais la même erreur se répète rarement sur
plusieurs images. Chaque piste (systeme/suivi.py) accumule donc des voix :
  - une lecture acceptée (statut « lue », systeme/decision.py) ;
  - une lecture « improbable » (groupe d'un seul chiffre) égale EXACTEMENT à une plaque autorisée. C'est la
    décision de l'étudiant en phase 4 : sinon, une voiture autorisée au numéro d'un seul chiffre serait refusée
    à chaque passage.
Les rejets « doute », « format » et « aucun chiffre » ne votent jamais. Seules comptent les W dernières voix de
la piste (la fenêtre). Deux variantes, comparées par la mesure en phase 5 :
  « majorite » : la lecture en tête décide dès qu'elle a au moins K voix ET plus de la moitié des voix de la
                 fenêtre : ouverture si elle est dans la liste des autorisés, refus sinon ;
  « liste »    : ouverture dès qu'une plaque autorisée a K voix dans la fenêtre, même sans majorité. Une lecture
                 plus courte (un chiffre perdu) en tête ne fait donc pas refuser une voiture autorisée dont la
                 lecture complète arrive ensuite. Le refus attend l'échéance, et exige alors une lecture en tête
                 avec K voix et la majorité ; sinon, la piste est « non lue ».
Chronomètre : il part à la PREMIÈRE VOIX de la piste, pas à sa première détection. La plaque est détectée de
loin, bien avant d'être lisible : compter l'approche comme de l'attente ferait déclarer « non lue » une voiture
qui approche encore. Sans décision à l'échéance (attente maximale), la piste est « non lue ».
Plaque illisible : une piste sans aucune voix n'a pas de chronomètre de vote. L'alerte prend le relais : « non lu »
3 s après que sa plaque est à portée de lecture (ou, variante, que la voiture s'est arrêtée), pour alerter le
gardien (voir DEPART_ALERTE). Une piste qui disparaît sans décision est aussi « non lue », si elle a eu une lecture
en forme de plaque ou si son alerte était partie ; sinon, rien n'est noté (fausse détection probable, ou plaque
restée trop loin).
Une seule décision par piste. La barrière ne s'ouvre que sur une correspondance EXACTE avec la liste.
Règles fixées avant toute mesure : voir docs/journal.md (phase 5, « Définitions du système »).

Compatible Python 3.6 (PC et Jetson).
"""

VARIANTES = ("majorite", "liste")
# Valeurs par défaut choisies par raisonnement, NON MESURÉES : la phase 5 s'est terminée sans évaluation sur vidéos
# réelles (décision de l'étudiant, docs/journal.md).
#   majorité : la variante la plus simple ; « liste d'abord » ouvrirait sans majorité, un risque qu'aucune mesure
#              ne justifie ici ;
#   K = 3    : une lecture isolée peut être fausse (chiffre perdu, confusion), la même erreur trois fois bien plus
#              rarement ; à 10 images/s, trois voix ne coûtent que 0,3 s ;
#   W = 10   : à peu près la dernière seconde de lectures ; la plaque se rapproche, les lectures récentes sont les
#              meilleures ;
#   D = 3 s  : l'attente maximale décidée par l'étudiant.
VARIANTE = "majorite"
VOIX_MIN = 3          # K
FENETRE = 10          # W : nombre de voix les plus récentes prises en compte
ATTENTE_MAX = 3.0     # D : secondes entre la première voix et l'échéance (attente maximale de l'étudiant : 3 s)
STATUTS_PLAQUE = ("lue", "doute", "improbable")   # lectures qui ont la forme d'une plaque (format officiel)

# Alerte « plaque sans voix » (décision de l'étudiant, journal) : une voiture arrêtée devant la barrière avec une
# plaque illisible doit aboutir à « non lu », pour alerter le gardien. Une piste qui n'a encore AUCUNE voix est
# déclarée « non lue » ATTENTE_ALERTE secondes après le départ de l'alerte ; une piste qui a des voix garde
# l'échéance du vote (les plaques lisibles ne changent pas de traitement). Départ de l'alerte :
#   « taille » (retenu) : la plaque est à portée de lecture, son cadre occupe au moins PORTEE_LECTURE de la largeur
#                         de l'image. Sur l'approche, la plaque est détectée de loin bien avant d'être lisible :
#                         partir de la détection déclencherait trop tôt ;
#   « arret » (variante codée, non retenue : plus fragile, le cadre détecté tremble et une voiture qui attend loin
#              dans une file déclencherait l'alerte) : la voiture est arrêtée, voir arretee() ;
#   None : pas d'alerte (comportement d'avant la règle).
DEPARTS_ALERTE = ("taille", "arret")
DEPART_ALERTE = "taille"
ATTENTE_ALERTE = 3.0     # T : l'attente maximale décidée par l'étudiant
# X, choisi par raisonnement, NON MESURÉ : entre le bas de la plage où le détecteur voit bien les plaques (8 % de la
# largeur de l'image) et la taille de la plaque au point d'arrêt (environ 17 % avec un téléphone à 2 m).
# PORTEE_MESUREE : faux tant que X n'a pas été mesuré sur des vidéos réelles. Les outils d'évaluation n'emploient pas
# un X non mesuré dans leurs mesures ; python -m outils.evaluer --portee le mesurerait (perspective, docs/journal.md).
PORTEE_LECTURE = 0.10
PORTEE_MESUREE = False
# Variante « arrêt » : voiture arrêtée = sur la dernière seconde, l'aire du cadre varie de moins de 10 % et son centre
# bouge de moins de 2 % de la largeur de l'image (seuils confirmés par l'étudiant).
DUREE_ARRET = 1.0
VARIATION_AIRE_MAX = 0.10
DEPLACEMENT_MAX = 0.02


def arretee(piste, temps, largeur_image):
    """La voiture de cette piste est-elle arrêtée ? Il faut au moins une seconde d'observation ; sur cette dernière
    seconde, l'aire du cadre varie de moins de 10 % et son centre bouge de moins de 2 % de la largeur de l'image."""
    recents = [(t, c) for t, c in piste.historique if t >= temps - DUREE_ARRET - 1e-6]
    if not recents or temps - recents[0][0] < DUREE_ARRET - 1e-6:
        return False
    aires = [(c[2] - c[0]) * (c[3] - c[1]) for _, c in recents]
    centres_x = [(c[0] + c[2]) / 2.0 for _, c in recents]
    centres_y = [(c[1] + c[3]) / 2.0 for _, c in recents]
    deplacement = max(max(centres_x) - min(centres_x), max(centres_y) - min(centres_y))
    return ((max(aires) - min(aires)) / max(aires) < VARIATION_AIRE_MAX
            and deplacement < DEPLACEMENT_MAX * largeur_image)


def decrire_alerte(vote):
    """Réglage de l'alerte d'un vote, en clair (console de la démo et de l'interface)."""
    if vote.depart_alerte is None:
        return "Alerte « plaque sans voix » désactivée"
    portee = ", X = {:.3f}".format(vote.portee) if vote.depart_alerte == "taille" else ""
    return "Alerte « plaque sans voix » : départ « {} »{}, « non lu » après {:.0f} s".format(
        vote.depart_alerte, portee, vote.attente_alerte)


class Vote:
    """Règle de décision, appliquée à chaque piste. autorises : ensemble de numéros « 215 TU 4567 ».
    depart_alerte, portee, attente_alerte : alerte « plaque sans voix » (voir plus haut) ; depart_alerte None :
    pas d'alerte. Les valeurs par défaut sont celles du système, choisies par raisonnement (voir plus haut)."""

    def __init__(self, autorises, variante=VARIANTE, voix_min=VOIX_MIN, fenetre=FENETRE, attente_max=ATTENTE_MAX,
                 depart_alerte=DEPART_ALERTE, portee=PORTEE_LECTURE, attente_alerte=ATTENTE_ALERTE):
        if variante not in VARIANTES:
            raise ValueError("Variante de vote inconnue : {} (choix : {})".format(variante, ", ".join(VARIANTES)))
        if depart_alerte is not None and depart_alerte not in DEPARTS_ALERTE:
            raise ValueError("Départ d'alerte inconnu : {} (choix : {})".format(depart_alerte, ", ".join(DEPARTS_ALERTE)))
        self.autorises = set(autorises)
        self.variante = variante
        self.voix_min = voix_min
        self.fenetre = fenetre
        self.attente_max = attente_max
        self.depart_alerte = depart_alerte
        self.portee = portee
        self.attente_alerte = attente_alerte

    def observer(self, piste, temps, largeur_image):
        """Cadre de la piste sur cette image (piste.boite) : fait partir son alerte si la plaque est à portée de
        lecture (« taille ») ou si la voiture est arrêtée (« arret »). Sans largeur d'image connue ou sans alerte,
        ne fait rien."""
        if self.depart_alerte is None or largeur_image is None or piste.decision is not None:
            return
        if self.depart_alerte == "arret":
            # Dernière seconde de cadres seulement : la mémoire reste bornée
            piste.historique = [(t, c) for t, c in piste.historique if t >= temps - DUREE_ARRET - 1e-6]
            piste.historique.append((temps, piste.boite))
        if piste.debut_alerte is not None:
            return
        if self.depart_alerte == "taille":
            remplie = piste.boite[2] - piste.boite[0] >= self.portee * largeur_image
        else:
            remplie = arretee(piste, temps, largeur_image)
        if remplie:
            piste.debut_alerte = temps

    def est_une_voix(self, lecture):
        return lecture["statut"] == "lue" or (lecture["statut"] == "improbable"
                                              and lecture["lecture"] in self.autorises)

    def classement(self, piste):
        """Lectures de la fenêtre, de la plus votée à la moins votée : liste de (texte, voix, confiance moyenne).
        À égalité de voix, la lecture votée le plus récemment passe devant."""
        comptes = {}
        for rang, (_, texte, confiance) in enumerate(piste.voix[-self.fenetre:]):
            voix, somme, _ = comptes.get(texte, (0, 0.0, 0))
            comptes[texte] = (voix + 1, somme + confiance, rang)
        ordre = sorted(comptes.items(), key=lambda element: (element[1][0], element[1][2]), reverse=True)
        return [(texte, voix, somme / voix) for texte, (voix, somme, _) in ordre]

    def ajouter(self, piste, temps, lecture):
        """Lecture d'une image (dictionnaire de systeme/decision.py) pour cette piste. Renvoie la décision si
        elle tombe maintenant, sinon None."""
        piste.derniere_lecture = lecture
        if lecture["statut"] in STATUTS_PLAQUE:
            piste.forme_de_plaque = True
        if piste.decision is None and self.est_une_voix(lecture):
            piste.voix.append((temps, lecture["lecture"], float(lecture["confiance"])))
            if piste.premiere_voix is None:
                piste.premiere_voix = temps
        return self.examiner(piste, temps)

    def examiner(self, piste, temps):
        """Applique la règle de décision. Appelée à chaque image pour chaque piste, même sans nouvelle lecture :
        l'échéance peut tomber. Renvoie la décision si elle tombe maintenant, sinon None."""
        if piste.decision is not None:
            return None
        if piste.premiere_voix is None:
            # Plaque sans voix : seule l'alerte peut tomber, ATTENTE_ALERTE secondes après son départ
            if piste.debut_alerte is not None and temps - piste.debut_alerte >= self.attente_alerte - 1e-9:
                return self.decider(piste, temps, "non lu", "", 0)
            return None
        classement = self.classement(piste)
        texte, voix, _ = classement[0]
        majorite = voix >= self.voix_min and 2 * voix > sum(v for _, v, _ in classement)
        if self.variante == "liste":
            for autre_texte, autres_voix, _ in classement:
                if autres_voix >= self.voix_min and autre_texte in self.autorises:
                    return self.decider(piste, temps, "ouverture", autre_texte, autres_voix)
        elif majorite:
            return self.decider(piste, temps, "ouverture" if texte in self.autorises else "refus", texte, voix)
        if temps - piste.premiere_voix >= self.attente_max - 1e-9:
            if self.variante == "liste" and majorite:
                return self.decider(piste, temps, "refus", texte, voix)
            return self.decider(piste, temps, "non lu", texte, voix)
        return None

    def fermer(self, piste, temps):
        """La piste disparaît. Renvoie la décision « non lu » si elle n'a pas été décidée et qu'elle a eu au moins une
        lecture en forme de plaque, ou que son alerte était partie (plaque venue à portée de lecture, ou voiture
        arrêtée, puis repartie sans être lue) ; sinon None."""
        if piste.decision is not None or not (piste.forme_de_plaque or piste.debut_alerte is not None):
            return None
        texte, voix = "", 0
        if piste.voix:
            texte, voix, _ = self.classement(piste)[0]
        return self.decider(piste, temps, "non lu", texte, voix)

    def decider(self, piste, temps, decision, texte, voix):
        """Enregistre la décision de la piste. numero : lecture décidée (pour « non lu » : la lecture en tête, à
        titre d'information, ou "") ; autorise : ce numéro est-il dans la liste ; delai_s : depuis la première
        voix (None sans voix) ; depuis_detection_s : depuis la première détection (information) ; depuis_alerte_s :
        depuis le départ de l'alerte (None si elle n'est pas partie)."""
        piste.decision = {
            "piste": piste.id, "temps": temps, "decision": decision, "numero": texte,
            "autorise": texte in self.autorises, "voix": voix,
            "delai_s": None if piste.premiere_voix is None else temps - piste.premiere_voix,
            "depuis_detection_s": temps - piste.debut,
            "depuis_alerte_s": None if piste.debut_alerte is None else temps - piste.debut_alerte,
        }
        return piste.decision

    def decrire(self, piste, temps):
        """Résumé d'une piste pour l'affichage (démonstration, interface web). statut : « en_cours » tant que la
        piste n'est pas décidée (le nom qu'attend la maquette de l'interface), sinon la décision. confiance : None
        tant que la piste n'a aucune voix (la page affiche alors un tiret, pas « 0 % »)."""
        meneur, voix, confiance = "", 0, None
        if piste.voix:
            meneur, voix, moyenne = self.classement(piste)[0]
            confiance = round(moyenne, 3)
        statut = "en_cours" if piste.decision is None else piste.decision["decision"]
        return {"id": piste.id, "meneur": meneur, "confiance": confiance, "voix": voix,
                "voix_requises": self.voix_min, "statut": statut, "age_s": round(temps - piste.debut, 1)}
