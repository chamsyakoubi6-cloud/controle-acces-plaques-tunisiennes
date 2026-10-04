"""Phase 4, étape 1 : jeu de caractères étiqueté automatiquement à partir de la vérité terrain (PC).

Pour chaque plaque de réglage, hors véhicules présents au test (même numéro = même plaque physique) :
  1. traitement de la phase 3 tel quel, avec le cadre vrai (l'étiquetage doit être le plus sûr possible) ;
  2. seules les plaques STRICTEMENT segmentées sont gardées (exactement le bon nombre de chiffres probables
     de chaque côté). Le k-ième chiffre probable de la série reçoit le k-ième chiffre saisi de la série, de
     même pour le numéro ; tous les autres candidats de ces plaques deviennent « autre » (classe 10). Les
     « autre » ne viennent que de ces plaques : sur une plaque mal segmentée, un vrai chiffre écarté par les
     filtres serait étiqueté « autre » à tort ;
  3. versions dégradées de la plaque redressée (éclairage, flou, faible résolution, bruit, compression JPEG),
     rebinarisées par la phase 3 et découpées aux positions des blobs de la plaque propre : l'étiquette est
     conservée, et le masque binaire se dégrade comme dans la réalité (un flou appliqué à un masque déjà
     binaire ne ressemblerait à rien de réel). Elles ne servent qu'à l'entraînement, jamais à la validation.
Les véhicules sont répartis en 5 plis pour la validation croisée : toutes les images d'un même véhicule sont
dans le même pli, sinon la validation contiendrait les mêmes chiffres physiques que l'entraînement.
Les plaques de entrainement/exclusions_caracteres.csv (étiquettes fausses repérées à l'oeil, par exemple une
coïncidence de segmentation) ne fournissent aucune vignette ; elles restent dans plis.csv, avec leur pli,
pour l'évaluation de bout en bout (leur numéro saisi est juste).

Sorties (non versionnées) :
  donnees/caracteres/caracteres.npz : vignettes 32 x 32 grises et binaires, étiquette, pli, version, image...
  donnees/caracteres/plis.csv : pli de chaque image de réglage gardée (pour l'évaluation de bout en bout)
  docs/figures/phase4/ : repartition_classes.png, planche_classes.jpg, planche_degradations.jpg et
                         classes/classe_<nom>.jpg (toutes les vignettes d'origine d'une classe)

Utilisation, depuis la racine du projet :
    python -m entrainement.extraire_caracteres
"""

import csv
import random
import time
from collections import Counter

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import matplotlib

matplotlib.use("Agg")  # figures enregistrées dans des fichiers, sans fenêtre
import matplotlib.pyplot as plt
import numpy as np

from entrainement.donnees_kaggle import RACINE, assembler_planche, faire_vignette
from entrainement.evaluer_traitement import cadre_vrai, lire_plaques
from systeme.binarisation import binariser
from systeme.lecture import AUTRE, NOMS_CLASSES
from systeme.segmentation import decouper_vignettes
from systeme.traitement import traiter_plaque

DOSSIER_CARACTERES = RACINE / "donnees" / "caracteres"
FICHIER_JEU = DOSSIER_CARACTERES / "caracteres.npz"
FICHIER_PLIS = DOSSIER_CARACTERES / "plis.csv"
FICHIER_EXCLUSIONS = RACINE / "entrainement" / "exclusions_caracteres.csv"
DOSSIER_FIGURES = RACINE / "docs" / "figures" / "phase4"
NB_PLIS = 5
GRAINE = 0
NB_VERSIONS = 4                                  # versions dégradées par plaque, en plus de l'originale
# Dégradations, dans l'ordre physique de formation de l'image : éclairage, flou optique, échantillonnage par
# le capteur (résolution), bruit, compression ; puis réagrandissement à 450 x 100, comme au redressement.
# Chaque défaut n'est appliqué qu'avec la probabilité PROBA_DEFAUT : une vraie photo cumule rarement tous les
# défauts. Calibrage sur les plaques de réglage (netteté = variance du laplacien ; réglage : p10 67, p25 133,
# médiane 251) : en cumulant tous les défauts sur des plages plus larges, médiane 36, bien sous le p10 ; avec
# les valeurs ci-dessous, médiane 106 et p10 23 : les versions dégradées couvrent la moitié basse des vraies
# plaques et un peu au-delà. Vérifié aussi à l'oeil sur planche_degradations.jpg.
PROBA_DEFAUT = 0.5
GAIN, DECALAGE = (0.6, 1.3), (-30.0, 30.0)       # contraste et luminosité
FLOU_SIGMA = (0.5, 2.0)                          # flou gaussien (px sur la plaque 450 x 100) : mise au point, bougé
REDUCTION = (0.35, 1.0)                          # 0,35 : plaque échantillonnée sur 35 px de haut seulement
BRUIT_SIGMA = (2.0, 8.0)                         # bruit du capteur (niveaux de gris)
QUALITE_JPEG = (30, 90)                          # compression des images du web
TYPES = {"gris": np.uint8, "binaire": np.uint8, "etiquette": np.int8, "pli": np.int8, "version": np.int8,
         "image": str, "vehicule": str, "position": np.int8, "nettete": np.float32}
# Couleurs des figures : une seule série, donc une seule teinte ; « autre » en gris (ce n'est pas un chiffre)
BLEU, GRIS_AUTRE = "#2a78d6", "#8c8b86"
ENCRE, ENCRE_SECONDAIRE, GRILLE = "#0b0b0b", "#52514e", "#e2e1dc"


def vehicule(plaque):
    """Identifiant du véhicule : son numéro complet (deux images au même numéro = la même plaque physique)."""
    return f"{plaque['serie']} TU {plaque['numero']}"


def attribuer_plis(vehicules):
    """Répartit les véhicules en NB_PLIS plis : tirage à graine fixe, puis distribution tour à tour."""
    ordre = sorted(vehicules)                    # ordre fixe avant le tirage : le résultat ne dépend que de la graine
    random.Random(GRAINE).shuffle(ordre)
    return {v: k % NB_PLIS for k, v in enumerate(ordre)}


def etiqueter(resultat, plaque):
    """Étiquette de chaque candidat d'une plaque strictement segmentée (None si elle ne l'est pas)."""
    if len(resultat["serie"]) != len(plaque["serie"]) or len(resultat["numero"]) != len(plaque["numero"]):
        return None
    chiffres = {}
    for groupe, texte in ((resultat["serie"], plaque["serie"]), (resultat["numero"], plaque["numero"])):
        for blob, chiffre in zip(groupe, texte):  # groupes triés de gauche à droite, comme le texte saisi
            chiffres[id(blob)] = int(chiffre)
    return [chiffres.get(id(blob), AUTRE) for blob in resultat["candidats"]]


def degrader(plaque, rng):
    """Version dégradée d'une plaque redressée (BGR 450 x 100), avec des forces tirées dans les plages ci-dessus."""
    hauteur, largeur = plaque.shape[:2]
    image = plaque.astype(np.float32)
    if rng.random() < PROBA_DEFAUT:
        image = image * rng.uniform(*GAIN) + rng.uniform(*DECALAGE)
    if rng.random() < PROBA_DEFAUT:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(*FLOU_SIGMA))
    echelle = rng.uniform(*REDUCTION) if rng.random() < PROBA_DEFAUT else 1.0
    petite = cv2.resize(image, (int(largeur * echelle), int(hauteur * echelle)), interpolation=cv2.INTER_AREA)
    if rng.random() < PROBA_DEFAUT:
        petite = petite + rng.normal(0.0, rng.uniform(*BRUIT_SIGMA), petite.shape)
    petite = np.clip(petite, 0, 255).astype(np.uint8)
    if rng.random() < PROBA_DEFAUT:
        qualite = int(rng.integers(QUALITE_JPEG[0], QUALITE_JPEG[1] + 1))
        _, octets = cv2.imencode(".jpg", petite, [cv2.IMWRITE_JPEG_QUALITY, qualite])
        petite = cv2.imdecode(octets, cv2.IMREAD_COLOR)
    return cv2.resize(petite, (largeur, hauteur), interpolation=cv2.INTER_LINEAR)


def nettete(plaque):
    """Variance du laplacien de la plaque en gris : même mesure de netteté qu'en phase 3 (faible = floue)."""
    return float(cv2.Laplacian(cv2.cvtColor(plaque, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def lire_exclusions():
    """Plaques exclues du jeu de caractères : {image (partie/nom): raison}."""
    with open(FICHIER_EXCLUSIONS, encoding="utf-8", newline="") as fichier:
        return {ligne["image"]: ligne["raison"] for ligne in csv.DictReader(fichier)}


def extraire(plaques, plis, exclusions, noms_exemples, rng):
    """Traite les plaques ; renvoie (jeu de vignettes, lignes de plis.csv, exemples de dégradations, netteté de
    chaque version de plaque)."""
    jeu = {cle: [] for cle in TYPES}
    lignes_plis, exemples = [], []
    nettetes = {"origine": [], "degradee": []}
    for plaque in plaques:
        nom = f"{plaque['chemin'].parent.name}/{plaque['image']}"
        image = cv2.imread(str(plaque["chemin"]))
        resultat = traiter_plaque(image, cadre_vrai(plaque["chemin"], image.shape[1], image.shape[0]))
        etiquettes = etiqueter(resultat, plaque)
        lignes_plis.append({"image": nom, "vehicule": vehicule(plaque), "pli": plis[vehicule(plaque)],
                            "stricte": int(etiquettes is not None), "exclue": int(nom in exclusions)})
        if etiquettes is None or nom in exclusions:
            continue
        versions = [resultat["plaque"]] + [degrader(resultat["plaque"], rng) for _ in range(NB_VERSIONS)]
        exemple = []
        for numero_version, version in enumerate(versions):
            if numero_version == 0:              # vignettes déjà découpées par la segmentation
                binaire = resultat["binaire"]
                grises, binaires = resultat["vignettes_grises"], resultat["vignettes_binaires"]
            else:                                # même binarisation que le système, mêmes positions de découpe
                binaire, gris = binariser(version)
                grises, binaires = decouper_vignettes(binaire, gris, resultat["candidats"])
            valeur = nettete(version)
            nettetes["origine" if numero_version == 0 else "degradee"].append(valeur)
            exemple.append((version, binaire, valeur))
            for position, (grise, binaire_vignette, etiquette) in enumerate(zip(grises, binaires, etiquettes)):
                for cle, donnee in (("gris", grise), ("binaire", binaire_vignette), ("etiquette", etiquette),
                                    ("pli", plis[vehicule(plaque)]), ("version", numero_version), ("image", nom),
                                    ("vehicule", vehicule(plaque)), ("position", position), ("nettete", valeur)):
                    jeu[cle].append(donnee)
        if nom in noms_exemples and len(exemples) < 6:
            exemples.append((nom, exemple))
    return {cle: np.array(valeurs, dtype=TYPES[cle]) for cle, valeurs in jeu.items()}, lignes_plis, exemples, nettetes


def resumer(jeu, lignes_plis, plaques, nettetes):
    """Affiche les effectifs par pli et par classe, le contrôle du biais de sélection et la netteté obtenue."""
    origine = jeu["version"] == 0
    retenues = [l for l in lignes_plis if l["stricte"] and not l["exclue"]]
    print(f"\nPlaques strictement segmentées : {sum(l['stricte'] for l in lignes_plis)} sur {len(lignes_plis)}, "
          f"dont {sum(l['stricte'] and l['exclue'] for l in lignes_plis)} exclue(s) ; retenues : {len(retenues)} "
          f"({len({l['vehicule'] for l in retenues})} véhicules sur {len({l['vehicule'] for l in lignes_plis})})")
    print("\nPlis (vignettes d'origine ; la validation d'un pli n'utilise que celles-ci) :")
    for pli in range(NB_PLIS):
        lignes = [l for l in lignes_plis if l["pli"] == pli]
        effectifs = np.bincount(jeu["etiquette"][origine & (jeu["pli"] == pli)], minlength=len(NOMS_CLASSES))
        print(f"   pli {pli} : {len({l['vehicule'] for l in lignes}):3d} véhicules, {len(lignes):3d} images, "
              f"{sum(1 for l in lignes if l['stricte'] and not l['exclue']):3d} plaques retenues, "
              f"{effectifs[:10].sum():4d} chiffres "
              f"(de {effectifs[:10].min()} à {effectifs[:10].max()} par classe), {effectifs[AUTRE]:4d} « autre »")

    effectifs = np.bincount(jeu["etiquette"][origine], minlength=len(NOMS_CLASSES))
    print(f"\nVignettes d'origine par classe ({effectifs.sum()} au total) :")
    for classe, nom in enumerate(NOMS_CLASSES):
        print(f"   {nom:>5s} : {effectifs[classe]:4d} ({effectifs[classe] / effectifs.sum() * 100:4.1f} %)")
    print(f"   écart : {effectifs[:10].max() / effectifs[:10].min():.1f} fois plus de « {np.argmax(effectifs[:10])} » "
          f"que de « {np.argmin(effectifs[:10])} » ; {effectifs[AUTRE] / effectifs[:10].min():.1f} fois plus de "
          f"« autre »")

    physiques, rang = set(), Counter()           # chiffre physique = (véhicule, rang du chiffre dans le numéro)
    for i in np.flatnonzero(origine & (jeu["etiquette"] != AUTRE)):
        physiques.add((jeu["vehicule"][i], rang[jeu["image"][i]]))
        rang[jeu["image"][i]] += 1
    print(f"   chiffres physiques distincts : {len(physiques)} (un même véhicule apparaît sur plusieurs images)")

    # Biais de sélection : la sélection stricte garde-t-elle les mêmes proportions de chiffres que l'ensemble ?
    toutes = Counter(c for p in plaques for c in p["serie"] + p["numero"])
    print("\nPart de chaque chiffre : toutes les plaques gardées / plaques strictes")
    print("   " + "  ".join(f"{k}: {toutes[str(k)] / sum(toutes.values()) * 100:4.1f}/"
                            f"{effectifs[k] / effectifs[:10].sum() * 100:4.1f}" for k in range(10)))

    print("\nNetteté des plaques redressées (variance du laplacien) : p10 / p25 / médiane / p75")
    for nom, valeurs in nettetes.items():
        print(f"   {nom:9s}: " + " / ".join(f"{v:.0f}" for v in np.percentile(valeurs, [10, 25, 50, 75])))
    print(f"\nVignettes au total : {len(jeu['etiquette'])} ({origine.sum()} d'origine, "
          f"{(~origine).sum()} dégradées pour l'entraînement)")


def figure_repartition(jeu, chemin, sous_titre):
    effectifs = np.bincount(jeu["etiquette"][jeu["version"] == 0], minlength=len(NOMS_CLASSES))
    figure, axe = plt.subplots(figsize=(9, 4.4))
    axe.bar(range(len(NOMS_CLASSES)), effectifs, width=0.6, color=[BLEU] * 10 + [GRIS_AUTRE], zorder=2)
    # Valeurs écrites sur quelques barres seulement : chiffres le plus et le moins fréquents, et « autre »
    for classe in sorted({int(np.argmax(effectifs[:10])), int(np.argmin(effectifs[:10])), AUTRE}):
        axe.annotate(str(effectifs[classe]), (classe, effectifs[classe]), xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, color=ENCRE)
    axe.set_xticks(range(len(NOMS_CLASSES)), NOMS_CLASSES)
    axe.tick_params(colors=ENCRE_SECONDAIRE)
    axe.set_ylabel("vignettes d'origine", color=ENCRE_SECONDAIRE)
    axe.set_title("Jeu de caractères : vignettes par classe\n" + sous_titre, loc="left", fontsize=10, color=ENCRE)
    axe.grid(axis="y", color=GRILLE, linewidth=0.8, zorder=0)
    for cote in ("top", "right"):
        axe.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        axe.spines[cote].set_color(GRILLE)
    figure.tight_layout()
    figure.savefig(chemin, dpi=120)
    plt.close(figure)


def grille(vignettes, colonnes, zoom=2, ecart=2, fond=128):
    """Vignettes 32 x 32 agrandies sans lissage (pixels nets) et rangées en grille, séparées par un liseré gris."""
    cote = vignettes[0].shape[0] * zoom
    nb_lignes = (len(vignettes) + colonnes - 1) // colonnes
    image = np.full((nb_lignes * (cote + ecart) + ecart, colonnes * (cote + ecart) + ecart), fond, dtype=np.uint8)
    for k, v in enumerate(vignettes):
        y, x = ecart + (k // colonnes) * (cote + ecart), ecart + (k % colonnes) * (cote + ecart)
        image[y:y + cote, x:x + cote] = cv2.resize(v, (cote, cote), interpolation=cv2.INTER_NEAREST)
    return image


def planche_classes(jeu, chemin, nb=20):
    """Pour chaque classe, nb vignettes d'origine tirées au hasard : en gris à gauche, les mêmes en binaire à droite."""
    rng = np.random.default_rng(GRAINE)
    origine = np.flatnonzero(jeu["version"] == 0)
    lignes = []
    for classe, nom in enumerate(NOMS_CLASSES):
        indices = origine[jeu["etiquette"][origine] == classe]
        choix = rng.choice(indices, size=min(nb, len(indices)), replace=False)
        gauche = grille([jeu["gris"][i] for i in choix], nb)
        droite = grille([jeu["binaire"][i] for i in choix], nb)
        etiquette = np.zeros((gauche.shape[0], 90), dtype=np.uint8)
        cv2.putText(etiquette, nom, (8, gauche.shape[0] // 2 + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        lignes.append(np.hstack([etiquette, gauche, np.zeros((gauche.shape[0], 16), dtype=np.uint8), droite]))
    cv2.imwrite(str(chemin), np.vstack(lignes))


def planches_completes(jeu, dossier, colonnes=25):
    """Toutes les vignettes d'origine de chaque classe, dans l'ordre du jeu : pour vérifier les étiquettes."""
    dossier.mkdir(parents=True, exist_ok=True)
    origine = jeu["version"] == 0
    for classe, nom in enumerate(NOMS_CLASSES):
        indices = np.flatnonzero(origine & (jeu["etiquette"] == classe))
        cv2.imwrite(str(dossier / f"classe_{nom}.jpg"), grille([jeu["gris"][i] for i in indices], colonnes))


def planche_degradations(exemples, chemin):
    """Pour quelques plaques : l'originale puis ses versions dégradées, avec leur binarisation et leur netteté."""
    cases = []
    for nom, versions in exemples:
        for numero_version, (plaque, binaire, valeur) in enumerate(versions):
            legende = nom if numero_version == 0 else f"version {numero_version}"
            bandeau = faire_vignette(np.zeros((18, 270, 3), dtype=np.uint8), 270, 18, f"{legende} : nettete {valeur:.0f}")
            cases.append(np.vstack([bandeau, faire_vignette(plaque, 270, 60),
                                    faire_vignette(cv2.cvtColor(binaire, cv2.COLOR_GRAY2BGR), 270, 60)]))
    cv2.imwrite(str(chemin), assembler_planche(cases, NB_VERSIONS + 1))


def main():
    debut = time.perf_counter()
    plaques, _ = lire_plaques("reglage")
    plaques_test, _ = lire_plaques("test")       # seuls les numéros saisis du test sont lus, jamais ses images
    vehicules_test = {vehicule(p) for p in plaques_test}
    exclues = Counter(vehicule(p) for p in plaques if vehicule(p) in vehicules_test)
    gardees = [p for p in plaques if vehicule(p) not in vehicules_test]
    print(f"Plaques de réglage saisies : {len(plaques)} ; exclues car leur véhicule est aussi au test : "
          f"{sum(exclues.values())} images de {len(exclues)} véhicules")
    for nom, nombre in sorted(exclues.items()):
        print(f"   {nom} : {nombre} image(s)")

    exclusions = lire_exclusions()
    print(f"Exclues du jeu de caractères (entrainement/exclusions_caracteres.csv) : {len(exclusions)}")
    for nom, raison in sorted(exclusions.items()):
        print(f"   {nom} : {raison}")

    plis = attribuer_plis({vehicule(p) for p in gardees})
    noms_exemples = set(random.Random(GRAINE).sample([f"{p['chemin'].parent.name}/{p['image']}" for p in gardees], 30))
    jeu, lignes_plis, exemples, nettetes = extraire(gardees, plis, exclusions, noms_exemples,
                                                    np.random.default_rng(GRAINE))

    DOSSIER_CARACTERES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FICHIER_JEU, **jeu)
    with open(FICHIER_PLIS, "w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=["image", "vehicule", "pli", "stricte", "exclue"])
        ecrivain.writeheader()
        ecrivain.writerows(lignes_plis)
    resumer(jeu, lignes_plis, gardees, nettetes)

    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    origine = jeu["version"] == 0
    retenues = [l for l in lignes_plis if l["stricte"] and not l["exclue"]]
    figure_repartition(jeu, DOSSIER_FIGURES / "repartition_classes.png",
                       f"{(jeu['etiquette'][origine] != AUTRE).sum()} chiffres et {(jeu['etiquette'][origine] == AUTRE).sum()} "
                       f"« autre », issus de {len(retenues)} plaques strictement segmentées "
                       f"({len({l['vehicule'] for l in retenues})} véhicules)\n"
                       f"(plus {NB_VERSIONS} versions dégradées par plaque, pour l'entraînement seulement)")
    planche_classes(jeu, DOSSIER_FIGURES / "planche_classes.jpg")
    planche_degradations(exemples, DOSSIER_FIGURES / "planche_degradations.jpg")
    planches_completes(jeu, DOSSIER_FIGURES / "classes")
    print(f"\nJeu : {FICHIER_JEU}\nPlis : {FICHIER_PLIS}\nFigures : {DOSSIER_FIGURES} (repartition_classes.png, "
          f"planche_classes.jpg, planche_degradations.jpg, classes/classe_<nom>.jpg : 25 vignettes par ligne)")
    print(f"Durée : {time.perf_counter() - debut:.0f} s")


if __name__ == "__main__":
    main()
