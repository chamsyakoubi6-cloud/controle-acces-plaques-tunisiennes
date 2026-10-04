"""Phase 4 : évaluation du CNN sur les caractères, à partir des prédictions hors pli (PC).

Chaque vignette d'origine a été jugée par le modèle du pli qui n'a jamais vu son véhicule
(entrainement/entrainer_cnn.py) : ces mesures valent pour des véhicules inconnus.
  - exactitude (part des vignettes bien classées) et moyenne des rappels par classe (chaque classe compte
    autant : c'est le critère des comparaisons, pour que « 1 » et « autre », nombreux, ne dominent pas) ;
  - pour chaque classe : précision (parmi les vignettes prédites « 7 », part de vrais 7) et rappel (parmi les
    vrais 7, part reconnue) ; principales confusions ; tableau comparatif si plusieurs essais sont donnés ;
  - --confusion : figure de la matrice de confusion ;
  - --bleu : exactitude sur les plaques de validation recolorées en bleu (plaques de location, absentes du jeu
    Kaggle), chacune lue par le modèle du pli qui n'a jamais vu son véhicule ;
  - --desaccords : planches des plaques où le CNN contredit l'étiquette avec une forte confiance. Ce sont soit
    des erreurs du CNN, soit des étiquettes fausses (faute de saisie, coïncidence de segmentation) : on décide
    à l'oeil sur l'image de la plaque, jamais d'après le seul avis du modèle.

Utilisation, depuis la racine du projet :
    python -m entrainement.evaluer_cnn binaire_16 gris_16 --bleu
    python -m entrainement.evaluer_cnn binaire_16 --confusion --desaccords
"""

import argparse
import json

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import matplotlib

matplotlib.use("Agg")  # figures enregistrées dans des fichiers, sans fenêtre
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from entrainement.donnees_kaggle import RACINE, assembler_planche, faire_vignette
from entrainement.evaluer_traitement import cadre_vrai, lire_plaques
from entrainement.extraire_caracteres import DOSSIER_FIGURES, FICHIER_JEU, NB_PLIS
from systeme.binarisation import binariser
from systeme.lecture import AUTRE, NOMS_CLASSES, preparer_vignettes
from systeme.segmentation import decouper_vignettes
from systeme.traitement import traiter_plaque

DOSSIER_RUNS = RACINE / "sorties" / "runs" / "cnn"  # celui de entrainer_cnn.py (importé ici sans TensorFlow)
SEUIL_DESACCORD = 0.9    # le CNN donne au moins 90 % à une autre classe que l'étiquette
PAR_PLANCHE = 20         # plaques par planche de désaccords (2 colonnes x 10 lignes)
ENCRE, ENCRE_SECONDAIRE = "#0b0b0b", "#52514e"


def charger(nom):
    """Jeu de caractères, probabilités hors pli de l'essai, et masque des vignettes jugées (d'origine)."""
    donnees = np.load(FICHIER_JEU)
    jeu = {cle: donnees[cle] for cle in donnees.files}
    probabilites = np.load(DOSSIER_RUNS / nom / "hors_pli.npz")["probabilites"]
    juges = (jeu["version"] == 0) & ~np.isnan(probabilites[:, 0])
    return jeu, np.nan_to_num(probabilites), juges


def matrice_confusion(etiquettes, predites):
    """Lignes : vraie classe ; colonnes : classe prédite."""
    confusion = np.zeros((len(NOMS_CLASSES), len(NOMS_CLASSES)), dtype=np.int64)
    np.add.at(confusion, (etiquettes, predites), 1)
    return confusion


def resume_mesures(confusion):
    """(exactitude, moyenne des rappels par classe, rappels, précisions)."""
    diagonale = np.diag(confusion)
    rappel = diagonale / np.maximum(confusion.sum(axis=1), 1)
    precision = diagonale / np.maximum(confusion.sum(axis=0), 1)
    return diagonale.sum() / confusion.sum(), rappel.mean(), rappel, precision


def afficher_mesures(confusion):
    exactitude, moyenne, rappel, precision = resume_mesures(confusion)
    erreurs_totales = confusion.sum() - np.trace(confusion)
    print(f"Exactitude : {exactitude * 100:.2f} % ({erreurs_totales} erreurs sur {confusion.sum()}) ; "
          f"moyenne des rappels par classe : {moyenne * 100:.2f} %")
    print("   classe  effectif  précision  rappel")
    for k, nom in enumerate(NOMS_CLASSES):
        print(f"   {nom:>6s}  {confusion[k].sum():8d}  {precision[k] * 100:7.1f} %  {rappel[k] * 100:5.1f} %")
    erreurs = sorted(((confusion[i, j], i, j) for i in range(len(NOMS_CLASSES)) for j in range(len(NOMS_CLASSES))
                      if i != j and confusion[i, j]), reverse=True)
    print("Principales confusions (vraie classe -> classe prédite) : "
          + ", ".join(f"{NOMS_CLASSES[i]} -> {NOMS_CLASSES[j]} : {n}" for n, i, j in erreurs[:12]))


def figure_confusion(confusion, chemin, titre):
    """Matrice de confusion en effectifs ; couleur en échelle logarithmique, sinon les erreurs (quelques unités)
    disparaîtraient à côté de la diagonale (des centaines)."""
    figure, axe = plt.subplots(figsize=(7.6, 6.6))
    image = axe.imshow(np.maximum(confusion, 0.1), cmap="Blues", norm=LogNorm(vmin=0.5, vmax=confusion.max()))
    for i in range(len(NOMS_CLASSES)):
        for j in range(len(NOMS_CLASSES)):
            if confusion[i, j]:
                clair = confusion[i, j] > np.sqrt(confusion.max())   # texte blanc sur les cases foncées
                axe.text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=8,
                         color="white" if clair else ENCRE)
    axe.set_xticks(range(len(NOMS_CLASSES)), NOMS_CLASSES)
    axe.set_yticks(range(len(NOMS_CLASSES)), NOMS_CLASSES)
    axe.set_xlabel("classe prédite", color=ENCRE_SECONDAIRE)
    axe.set_ylabel("vraie classe (étiquette)", color=ENCRE_SECONDAIRE)
    axe.set_title(titre, fontsize=10, color=ENCRE, loc="left")
    figure.colorbar(image, ax=axe, label="vignettes (échelle logarithmique)")
    figure.tight_layout()
    figure.savefig(chemin, dpi=120)
    plt.close(figure)


def fond_bleu(jeu, probabilites, juges, nom):
    """Exactitude hors pli sur les plaques de validation recolorées en bleu : chaque plaque est lue par le modèle
    du pli qui n'a jamais vu son véhicule. Les vignettes sont redécoupées aux mêmes positions après la
    binarisation du système, comme pour une vraie plaque bleue."""
    import keras  # import local : TensorFlow n'est chargé que pour ce contrôle
    from entrainement.regler_traitement import BLEUS, simuler_fond_bleu
    with open(DOSSIER_RUNS / nom / "resume.json", encoding="utf-8") as fichier:
        entree = json.load(fichier)["entree"]
    plaques = {f"{p['chemin'].parent.name}/{p['image']}": p for p in lire_plaques("reglage")[0]}
    vignettes = {nom_bleu: [] for nom_bleu in BLEUS}
    indices = []
    for nom_image in sorted(set(jeu["image"][juges].tolist())):
        plaque = plaques[nom_image]
        image = cv2.imread(str(plaque["chemin"]))
        resultat = traiter_plaque(image, cadre_vrai(plaque["chemin"], image.shape[1], image.shape[0]))
        de_la_plaque = np.flatnonzero(juges & (jeu["image"] == nom_image))
        for nom_bleu, couleur in BLEUS.items():
            binaire, gris = binariser(simuler_fond_bleu(resultat["plaque"], couleur))
            grises, binaires = decouper_vignettes(binaire, gris, resultat["candidats"])
            source = grises if entree == "gris" else binaires
            vignettes[nom_bleu] += [source[jeu["position"][i]] for i in de_la_plaque]
        indices += de_la_plaque.tolist()
    indices = np.array(indices)
    etiquettes = jeu["etiquette"][indices].astype(np.int64)
    modeles = [keras.models.load_model(DOSSIER_RUNS / nom / f"pli{pli}.keras") for pli in range(NB_PLIS)]
    resultats = {"fond noir (origine)": probabilites[indices].argmax(axis=1)}
    for nom_bleu in BLEUS:
        x = preparer_vignettes(vignettes[nom_bleu], entree)
        predites = np.zeros(len(indices), dtype=np.int64)
        for pli in range(NB_PLIS):
            dans = jeu["pli"][indices] == pli
            predites[dans] = modeles[pli].predict(x[dans], verbose=0).argmax(axis=1)
        resultats[nom_bleu] = predites
    print(f"Fond bleu simulé ({len(indices)} vignettes, entrée {entree}) :")
    moyennes = {}
    for nom_fond, predites in resultats.items():
        exactitude, moyennes[nom_fond], _, _ = resume_mesures(matrice_confusion(etiquettes, predites))
        print(f"   {nom_fond:20s}: exactitude {exactitude * 100:6.2f} %, "
              f"moyenne des rappels {moyennes[nom_fond] * 100:6.2f} %")
    return moyennes


def nom_court(classe):
    return "a" if classe == AUTRE else str(classe)


def panneau(plaque, nom_image, indices, jeu, probabilites, suspects):
    """Plaque redressée en couleur (on y lit la vérité), puis sa binarisation avec les candidats : vert = chiffre,
    orange = « autre » (étiquettes) ; rouge = désaccord, annoté « étiquette>prédiction confiance »."""
    image = cv2.imread(str(plaque["chemin"]))
    resultat = traiter_plaque(image, cadre_vrai(plaque["chemin"], image.shape[1], image.shape[0]))
    vue = cv2.cvtColor(resultat["binaire"], cv2.COLOR_GRAY2BGR)
    for i in indices:
        blob = resultat["candidats"][jeu["position"][i]]
        etiquette, predite = int(jeu["etiquette"][i]), int(probabilites[i].argmax())
        if i in suspects:
            couleur, texte = (0, 0, 255), f"{nom_court(etiquette)}>{nom_court(predite)} {probabilites[i].max():.2f}"
        else:
            couleur, texte = ((0, 140, 255), "") if etiquette == AUTRE else ((0, 200, 0), "")
        cv2.rectangle(vue, (blob["x"], blob["y"]), (blob["x"] + blob["l"] - 1, blob["y"] + blob["h"] - 1), couleur,
                      2 if i in suspects else 1)
        if texte:
            cv2.putText(vue, texte, (max(0, blob["x"] - 4), max(10, blob["y"] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        couleur, 1)
    titre = f"{nom_image} : saisi {plaque['serie']} TU {plaque['numero']} ; pli {jeu['pli'][indices[0]]}"
    bandeau = faire_vignette(np.zeros((18, 450, 3), dtype=np.uint8), 450, 18, titre)
    return np.vstack([bandeau, resultat["plaque"], vue])


def desaccords(jeu, probabilites, juges, nom):
    predites = probabilites.argmax(axis=1)
    suspects = set(np.flatnonzero(juges & (predites != jeu["etiquette"])
                                  & (probabilites.max(axis=1) >= SEUIL_DESACCORD)).tolist())
    images = sorted({str(jeu["image"][i]) for i in suspects},
                    key=lambda n: (n.split("/")[0], int(n.split("/")[1].split(".")[0])))
    print(f"\nDésaccords confiants (classe prédite différente de l'étiquette, confiance >= {SEUIL_DESACCORD}) : "
          f"{len(suspects)} vignettes sur {len(images)} plaques")
    plaques = {f"{p['chemin'].parent.name}/{p['image']}": p for p in lire_plaques("reglage")[0]}
    panneaux = []
    for nom_image in images:
        indices = np.flatnonzero(juges & (jeu["image"] == nom_image))
        for i in indices:
            if i in suspects:
                print(f"   {nom_image} (saisi {plaques[nom_image]['serie']} TU {plaques[nom_image]['numero']}) : "
                      f"position {jeu['position'][i]}, étiquette {NOMS_CLASSES[jeu['etiquette'][i]]} -> prédit "
                      f"{NOMS_CLASSES[predites[i]]} ({probabilites[i].max():.2f})")
        panneaux.append(panneau(plaques[nom_image], nom_image, indices, jeu, probabilites, suspects))
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(panneaux), PAR_PLANCHE):
        chemin = DOSSIER_FIGURES / f"desaccords_{nom}_{k // PAR_PLANCHE + 1}.jpg"
        cv2.imwrite(str(chemin), assembler_planche(panneaux[k:k + PAR_PLANCHE], 2))
        print(f"Planche : {chemin}")


def main():
    parseur = argparse.ArgumentParser(description="Évaluation du CNN sur les caractères (prédictions hors pli).")
    parseur.add_argument("noms", nargs="+", help="essais à évaluer (dossiers de sorties/runs/cnn/)")
    parseur.add_argument("--confusion", action="store_true", help="figure de la matrice de confusion")
    parseur.add_argument("--bleu", action="store_true", help="exactitude sur fond bleu simulé")
    parseur.add_argument("--desaccords", action="store_true", help="planches des désaccords confiants")
    arguments = parseur.parse_args()
    tableau = []
    for nom in arguments.noms:
        jeu, probabilites, juges = charger(nom)
        print(f"\n===== Essai {nom} : {juges.sum()} vignettes d'origine jugées hors pli =====")
        confusion = matrice_confusion(jeu["etiquette"][juges].astype(np.int64), probabilites[juges].argmax(axis=1))
        afficher_mesures(confusion)
        exactitude, moyenne, _, _ = resume_mesures(confusion)
        ligne = {"nom": nom, "exactitude": exactitude, "moyenne": moyenne}
        if arguments.confusion:
            DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
            figure_confusion(confusion, DOSSIER_FIGURES / f"confusion_{nom}.png",
                             f"Matrice de confusion hors pli ({nom}) : {juges.sum()} vignettes, "
                             f"exactitude {exactitude * 100:.2f} %")
        if arguments.bleu:
            ligne["bleu"] = fond_bleu(jeu, probabilites, juges, nom)
        if arguments.desaccords:
            desaccords(jeu, probabilites, juges, nom)
        tableau.append(ligne)
    if len(tableau) > 1:
        print("\nComparaison (hors pli, vignettes d'origine ; critère : erreur moyenne par classe = 1 - moyenne "
              "des rappels)")
        for ligne in tableau:
            texte = (f"   {ligne['nom']:14s} exactitude {ligne['exactitude'] * 100:6.2f} %, erreur moyenne par classe "
                     f"{(1 - ligne['moyenne']) * 100:5.2f} %")
            for nom_fond, moyenne in ligne.get("bleu", {}).items():
                if nom_fond != "fond noir (origine)":
                    texte += f", {nom_fond} {(1 - moyenne) * 100:5.2f} %"
            print(texte)


if __name__ == "__main__":
    main()
