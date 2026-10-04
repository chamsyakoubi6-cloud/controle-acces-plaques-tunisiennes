"""Phase 4 : entraînement du CNN de lecture des caractères (TensorFlow / Keras, sur CPU) (PC).

Validation croisée à 5 plis par véhicule (plis fixés par extraire_caracteres.py) : pour chaque pli, un modèle
apprend sur les 4 autres plis (vignettes d'origine et versions dégradées) et est jugé sur les vignettes
d'ORIGINE du pli, dont il n'a vu ni le véhicule ni les versions dégradées. Les prédictions « hors pli » sont
enregistrées : chaque vignette du jeu est jugée par un modèle qui n'a jamais vu son véhicule.
Avec --final : un seul modèle, sur tous les plis, entraîné exactement de la même façon.

Architecture (« largeur » = nombre de filtres du 1er bloc, doublé à chaque bloc) :
  3 x [convolution 3x3 -> BatchNorm -> ReLU -> max-pooling 2]   (32 -> 16 -> 8 -> 4 px)
  Flatten -> Dropout 0,3 -> Dense 64 -> Dense 11 (softmax : probabilités des 11 classes)
Apprentissage : Adam, entropie croisée pondérée par classe (déséquilibre des classes), 30 époques fixes et
calendrier fixe du taux d'apprentissage (baisse de 5 % par époque). Rien ne dépend de la validation, qui est
seulement mesurée (courbes) : les modèles des plis et le modèle final suivent exactement la même trajectoire.
Augmentations à chaque époque, en plus des versions dégradées : petite rotation, décalage, échelle, épaisseur
des traits ; jamais de miroir ni de demi-tour (un 6 retourné devient un 9).

Sorties : sorties/runs/cnn/<nom>/ (modèles .keras, hors_pli.npz, resume.json) et
          docs/figures/phase4/courbes_<nom>.png (courbes d'apprentissage de chaque pli)

Utilisation, depuis la racine du projet :
    python -m entrainement.entrainer_cnn --entree binaire --largeur 16           validation croisée
    python -m entrainement.entrainer_cnn --entree binaire --largeur 16 --final   modèle final
    python -m entrainement.entrainer_cnn --entree gris --largeur 32 --reprendre  reprise après interruption
"""

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # masque les messages d'information de TensorFlow

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import keras
import matplotlib

matplotlib.use("Agg")  # figures enregistrées dans des fichiers, sans fenêtre
import matplotlib.pyplot as plt
import numpy as np
from keras import layers

from entrainement.donnees_kaggle import RACINE
from entrainement.extraire_caracteres import DOSSIER_FIGURES, FICHIER_JEU, NB_PLIS
from systeme.lecture import ENTREES, NOMS_CLASSES, preparer_vignettes

DOSSIER_RUNS = RACINE / "sorties" / "runs" / "cnn"
TAILLE_LOT = 64
# 30 époques fixes : au passage de nettoyage, plateau dès la 3e à la 5e époque et aucun surapprentissage jusqu'à
# la 33e ; l'arrêt anticipé sur la perte de validation s'arrêtait au hasard sur ce plateau (de la 3e à la 23e).
EPOQUES = 30
TAUX_INITIAL, DECROISSANCE = 1e-3, 0.95   # grands pas au début, ajustements fins ensuite (x 0,21 à la 30e)
GRAINE = 0
# Augmentations à la volée (les défauts d'image sont déjà dans les versions dégradées de extraire_caracteres.py)
ROTATION_MAX = 8.0               # degrés : le redressement a déjà corrigé la perspective
DECALAGE_MAX = 2.0               # pixels sur 32 : imprécision du centrage de la découpe
ECHELLE = (0.9, 1.1)             # taille du caractère dans la vignette
PROBA_EPAISSEUR = 0.25           # érosion (traits plus fins) ou dilatation (plus épais) 2 x 2, chacune 1 fois sur 4
# Couleurs des courbes : deux séries, dans l'ordre fixe de la palette de référence
BLEU, ORANGE = "#2a78d6", "#eb6834"
ENCRE, ENCRE_SECONDAIRE, GRILLE = "#0b0b0b", "#52514e", "#e2e1dc"


def augmenter(vignette, rng):
    """Petite transformation géométrique et changement d'épaisseur des traits d'une vignette 32 x 32 (uint8)."""
    bords = np.concatenate([vignette[0], vignette[-1], vignette[:, 0], vignette[:, -1]])
    fond = int(np.median(bords))                 # niveau du fond, pour remplir ce qui entre dans le cadre
    cote = vignette.shape[0]
    matrice = cv2.getRotationMatrix2D(((cote - 1) / 2, (cote - 1) / 2), rng.uniform(-ROTATION_MAX, ROTATION_MAX),
                                      rng.uniform(*ECHELLE))
    matrice[:, 2] += rng.uniform(-DECALAGE_MAX, DECALAGE_MAX, size=2)
    sortie = cv2.warpAffine(vignette, matrice, (cote, cote), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=fond)
    tirage = rng.random()
    if tirage < PROBA_EPAISSEUR:
        sortie = cv2.erode(sortie, np.ones((2, 2), dtype=np.uint8))
    elif tirage < 2 * PROBA_EPAISSEUR:
        sortie = cv2.dilate(sortie, np.ones((2, 2), dtype=np.uint8))
    return sortie


def poids_des_classes(etiquettes):
    """Poids de chaque classe, inversement proportionnel à son effectif (effectif total / (11 x effectif de la
    classe)) : au total, chaque classe pèse autant dans la perte, le « 1 » et « autre » ne dominent plus."""
    effectifs = np.bincount(etiquettes, minlength=len(NOMS_CLASSES))
    return len(etiquettes) / (len(NOMS_CLASSES) * np.maximum(effectifs, 1))


class LotsAugmentes(keras.utils.PyDataset):
    """Lots d'apprentissage : ordre tiré au hasard à chaque époque, vignettes augmentées à la volée, et poids de
    chaque vignette selon sa classe."""

    def __init__(self, vignettes, etiquettes, poids_classes, entree, graine):
        super().__init__()
        self.vignettes, self.etiquettes, self.entree = vignettes, etiquettes, entree
        self.poids = poids_classes[etiquettes].astype(np.float32)
        self.rng = np.random.default_rng(graine)
        self.ordre = self.rng.permutation(len(etiquettes))

    def __len__(self):
        return int(np.ceil(len(self.etiquettes) / TAILLE_LOT))

    def __getitem__(self, k):
        indices = self.ordre[k * TAILLE_LOT:(k + 1) * TAILLE_LOT]
        lot = [augmenter(self.vignettes[i], self.rng) for i in indices]
        return preparer_vignettes(lot, self.entree), self.etiquettes[indices], self.poids[indices]

    def on_epoch_end(self):
        self.ordre = self.rng.permutation(len(self.etiquettes))


def construire_cnn(largeur):
    entree = keras.Input((32, 32, 1), name="vignettes")
    x = entree
    for k in range(3):                           # 32 -> 16 -> 8 -> 4 px ; filtres : largeur, 2 x, 4 x
        # Pas de biais : la BatchNorm qui suit a son propre décalage appris, un biais ici serait redondant
        x = layers.Conv2D(largeur * 2 ** k, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)                      # garde la position des traits (6 et 9 : boucle en bas ou en haut)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    sortie = layers.Dense(len(NOMS_CLASSES), activation="softmax", name="probabilites")(x)
    return keras.Model(entree, sortie, name=f"lecteur_{largeur}")


def entrainer_modele(jeu, apprentissage, validation, entree, largeur):
    """Entraîne un CNN sur les vignettes « apprentissage » (masque booléen), EPOQUES époques. La validation
    éventuelle est seulement mesurée à chaque époque (courbes) : elle ne pilote rien. Renvoie (modèle, historique)."""
    keras.utils.set_random_seed(GRAINE)
    etiquettes = jeu["etiquette"][apprentissage].astype(np.int64)
    lots = LotsAugmentes(jeu[entree][apprentissage], etiquettes, poids_des_classes(etiquettes), entree, GRAINE)
    modele = construire_cnn(largeur)
    modele.compile(optimizer=keras.optimizers.Adam(TAUX_INITIAL), loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    # Calendrier fixe : le taux d'apprentissage ne dépend que du numéro d'époque, jamais de la validation
    calendrier = keras.callbacks.LearningRateScheduler(lambda epoque, taux: TAUX_INITIAL * DECROISSANCE ** epoque)
    donnees_validation = None
    if validation is not None:
        donnees_validation = (preparer_vignettes(jeu[entree][validation], entree),
                              jeu["etiquette"][validation].astype(np.int64))
    historique = modele.fit(lots, validation_data=donnees_validation, epochs=EPOQUES, callbacks=[calendrier],
                            verbose=2)
    return modele, historique.history


def validation_croisee(jeu, entree, largeur, nom, reprendre=False):
    """Un modèle par pli. Chaque pli est enregistré dès qu'il est fini (modèle et historique) : avec reprendre,
    les plis déjà entraînés sont rechargés au lieu d'être réentraînés (calcul interrompu)."""
    dossier = DOSSIER_RUNS / nom
    dossier.mkdir(parents=True, exist_ok=True)
    origine = jeu["version"] == 0
    # Probabilités hors pli, alignées sur le jeu entier (NaN pour les versions dégradées, jamais jugées)
    probabilites = np.full((len(jeu["etiquette"]), len(NOMS_CLASSES)), np.nan, dtype=np.float32)
    resume = {"entree": entree, "largeur": largeur, "plis": {}}
    historiques = {}
    for pli in range(NB_PLIS):
        debut = time.perf_counter()
        apprentissage = jeu["pli"] != pli
        validation = (jeu["pli"] == pli) & origine
        chemin_modele, chemin_historique = dossier / f"pli{pli}.keras", dossier / f"pli{pli}_historique.json"
        if reprendre and chemin_modele.exists():
            print(f"\n=== Pli {pli} : repris (modèle déjà entraîné) ===")
            modele = keras.models.load_model(chemin_modele)
            historique = {}                      # conservé seulement pour les plis entraînés après cet ajout
            if chemin_historique.exists():
                with open(chemin_historique, encoding="utf-8") as fichier:
                    historique = json.load(fichier)
        else:
            print(f"\n=== Pli {pli} : {apprentissage.sum()} vignettes d'apprentissage, {validation.sum()} de "
                  f"validation ===")
            modele, historique = entrainer_modele(jeu, apprentissage, validation, entree, largeur)
            historique = {cle: [float(v) for v in valeurs] for cle, valeurs in historique.items()}
            modele.save(chemin_modele)
            with open(chemin_historique, "w", encoding="utf-8") as fichier:
                json.dump(historique, fichier)
        probabilites[validation] = modele.predict(preparer_vignettes(jeu[entree][validation], entree), verbose=0)
        exactitude = float(np.mean(probabilites[validation].argmax(axis=1) == jeu["etiquette"][validation]))
        resume["plis"][str(pli)] = {"exactitude_validation": exactitude, "duree_s": round(time.perf_counter() - debut)}
        historiques[pli] = historique
        print(f"Pli {pli} : exactitude hors pli {exactitude * 100:.2f} % ({resume['plis'][str(pli)]['duree_s']} s)")
    juges = origine & ~np.isnan(probabilites[:, 0])
    resume["exactitude_hors_pli"] = float(np.mean(probabilites[juges].argmax(axis=1) == jeu["etiquette"][juges]))
    np.savez_compressed(dossier / "hors_pli.npz", probabilites=probabilites)
    with open(dossier / "resume.json", "w", encoding="utf-8") as fichier:
        json.dump(resume, fichier, indent=2, ensure_ascii=False)
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    figure_courbes(historiques, DOSSIER_FIGURES / f"courbes_{nom}.png", nom)
    print(f"\nExactitude hors pli ({juges.sum()} vignettes d'origine) : {resume['exactitude_hors_pli'] * 100:.2f} % ; "
          f"résultats : {dossier}")


def modele_final(jeu, entree, largeur, nom):
    dossier = DOSSIER_RUNS / nom
    dossier.mkdir(parents=True, exist_ok=True)
    print(f"Modèle final : tous les plis, {EPOQUES} époques, même calendrier que les plis")
    modele, _ = entrainer_modele(jeu, np.ones(len(jeu["etiquette"]), dtype=bool), None, entree, largeur)
    modele.save(dossier / "final.keras")
    print(f"Enregistré : {dossier / 'final.keras'}")


def figure_courbes(historiques, chemin, nom):
    """Courbes d'apprentissage : une colonne par pli ; perte en haut, exactitude en bas."""
    figure, axes = plt.subplots(2, len(historiques), figsize=(3.2 * len(historiques), 5.8), sharey="row", squeeze=False)
    for colonne, (pli, historique) in enumerate(sorted(historiques.items())):
        for ligne, (cle, titre) in enumerate((("loss", "perte"), ("accuracy", "exactitude"))):
            axe = axes[ligne][colonne]
            if historique:
                epoques = np.arange(1, len(historique["loss"]) + 1)
                axe.plot(epoques, historique[cle], color=BLEU, linewidth=2, label="apprentissage (vignettes augmentées)")
                axe.plot(epoques, historique["val_" + cle], color=ORANGE, linewidth=2, label="validation (hors pli)")
            else:                                # pli repris sans historique conservé
                axe.text(0.5, 0.5, "historique\nnon conservé", ha="center", va="center", transform=axe.transAxes,
                         color=ENCRE_SECONDAIRE, fontsize=9)
            axe.grid(color=GRILLE, linewidth=0.8)
            for cote in ("top", "right"):
                axe.spines[cote].set_visible(False)
            axe.tick_params(colors=ENCRE_SECONDAIRE, labelsize=8)
            if ligne == 0:
                axe.set_title(f"pli {pli}", fontsize=10, color=ENCRE)
            if colonne == 0:
                axe.set_ylabel(titre, color=ENCRE_SECONDAIRE)
            if ligne == 1:
                axe.set_xlabel("époque", color=ENCRE_SECONDAIRE)
    avec_courbes = [axe for axe in axes[0] if axe.get_legend_handles_labels()[0]]
    if avec_courbes:
        poignees, libelles = avec_courbes[0].get_legend_handles_labels()
        figure.legend(poignees, libelles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    figure.suptitle(f"Courbes d'apprentissage ({nom}) : {EPOQUES} époques fixes ; perte d'apprentissage pondérée "
                    f"par classe, validation seulement mesurée", fontsize=10, color=ENCRE)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    figure.savefig(chemin, dpi=110)
    plt.close(figure)


def main():
    parseur = argparse.ArgumentParser(description="Entraînement du CNN de lecture des caractères.")
    parseur.add_argument("--entree", choices=ENTREES, required=True, help="vignette grise ou masque binaire")
    parseur.add_argument("--largeur", type=int, default=16, help="filtres du 1er bloc (8, 16 ou 32)")
    parseur.add_argument("--nom", help="nom de l'essai (par défaut : <entree>_<largeur>)")
    parseur.add_argument("--final", action="store_true", help="modèle final sur tous les plis")
    parseur.add_argument("--reprendre", action="store_true", help="recharger les plis déjà entraînés (interruption)")
    arguments = parseur.parse_args()
    nom = arguments.nom or f"{arguments.entree}_{arguments.largeur}"
    donnees = np.load(FICHIER_JEU)
    jeu = {cle: donnees[cle] for cle in donnees.files}
    if arguments.final:
        modele_final(jeu, arguments.entree, arguments.largeur, nom)
    else:
        validation_croisee(jeu, arguments.entree, arguments.largeur, nom, arguments.reprendre)


if __name__ == "__main__":
    main()
