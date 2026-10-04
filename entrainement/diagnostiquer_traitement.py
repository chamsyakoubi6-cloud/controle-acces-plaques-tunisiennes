"""Phase 3, itération ciblée : diagnostic des chiffres perdus, sur le réglage, hors pli (PC).

Chaîne du système : cadre YOLO, traitement de la phase 3, CNN du pli qui n'a jamais vu le véhicule, décision
(option A, seuil 0,6). Une plaque « perd des chiffres » si sa lecture, même rejetée, en a moins que le numéro
saisi. Pour chacune, des indicateurs automatiques (la vérité ne sert qu'à savoir QUELS chiffres manquent) :
  couverture  : part de la largeur du cadre YOLO couverte par le quadrilatère (plaque coupée si trop faible) ;
  débordement : part du quadrilatère qui sort du cadre YOLO (coins pris sur le bord d'un écran, la carrosserie) ;
  pente       : inclinaison de la rangée des candidats de la taille d'un chiffre (plaque restée penchée) ;
  soudures    : grosses composantes qui ne sont pas des candidats (elles touchent le bord ou dépassent la taille
                d'un caractère) et traversent la rangée des chiffres, du côté où un chiffre manque.
Seuils : ce que les plaques BIEN lues ne dépassent presque jamais (1er ou 99e centile). Attribution dans l'ordre :
coupée, coins faux (penchée ou débordante), soudure au liseré, aucun chiffre, autre ; puis vérification à
l'oeil sur une planche par cause. Pour chaque cause : combien de plaques une AUTRE méthode de la cascade
aurait bien lues (oracle : la marge du contrôle de qualité du redressement).
Pour le « 1 » : quels chiffres manquent (alignement lecture / numéro saisi), et si la lecture revient quand on
n'efface pas les longues lignes (hypothèse : un « 1 » très haut effacé avec le liseré).

Utilisation, depuis la racine du projet :
    python -m entrainement.diagnostiquer_traitement gris_8
"""

import argparse
import csv
import difflib
from collections import Counter

import systeme  # réglages d'OpenCV, avant tout import de cv2
import cv2
import numpy as np

from entrainement.donnees_kaggle import assembler_planche, faire_vignette
from entrainement.evaluer_lecture import LecteurPlis, detecter_et_traiter
from entrainement.evaluer_traitement import MODELE_DETECTION, lire_plaques
from entrainement.extraire_caracteres import DOSSIER_FIGURES, FICHIER_PLIS
from systeme.binarisation import binariser
from systeme.decision import SEUIL, lire_plaque
from systeme.detection import DetecteurPlaques
from systeme.redressement import METHODES, couverture_du_cadre, decouper, redresser
from systeme.segmentation import CANDIDAT_AIRE_MIN, composantes, est_candidat, pente_rangee, segmenter

CAUSES = ("coupée", "coins faux", "soudure au liseré", "aucun chiffre", "autre")
CENTILE = 1                      # seuils au 1er (ou 99e) centile des plaques bien lues
ZONES = {"serie": (0.0, 0.45), "numero": (0.50, 1.0)}   # en fraction de la largeur (série 0,08-0,31 et numéro dès
                                                       # 0,59 en médiane, mesuré à l'arrêt 1 de la phase 4)
PAR_PLANCHE = 20


def lire_option_a(resultat, lecteur, pli, seuil=SEUIL):
    """Lecture comme le système (option A : le CNN ne lit que les chiffres probables), avec le modèle hors pli."""
    probables = [b for b in resultat["candidats"] if b["chiffre"]]
    hauteur = resultat["binaire"].shape[0]
    if not probables:
        return lire_plaque([], [], [], hauteur, seuil)
    classes, confiances = lecteur.lire(resultat, pli)
    garder = [k for k, b in enumerate(resultat["candidats"]) if b["chiffre"]]
    return lire_plaque(probables, np.asarray(classes)[garder], np.asarray(confiances)[garder], hauteur, seuil)


def chiffres_lus(decision):
    return "".join(c for c in decision["lecture"] if c.isdigit())


def chiffres_perdus(verite, lu):
    """Indices des chiffres du numéro saisi absents de la lecture (alignement par plus longue sous-suite commune)."""
    apparies = set()
    for bloc in difflib.SequenceMatcher(None, verite, lu, autojunk=False).get_matching_blocks():
        apparies.update(range(bloc.a, bloc.a + bloc.size))
    return [k for k in range(len(verite)) if k not in apparies]


def soudures(resultat, cotes):
    """Grosses composantes hors candidats qui traversent la rangée des chiffres dans la zone des groupes donnés."""
    binaire = resultat["binaire"]
    hauteur, largeur = binaire.shape
    probables = resultat["serie"] + resultat["numero"]
    if probables:
        haut = float(np.median([b["y"] for b in probables]))
        bas = float(np.median([b["y"] + b["h"] for b in probables]))
    else:
        haut, bas = 0.2 * hauteur, 0.8 * hauteur      # rangée d'une plaque bien redressée (chiffres centrés)
    trouvees = []
    for blob in composantes(binaire):
        if est_candidat(blob, hauteur, largeur) or blob["aire"] < CANDIDAT_AIRE_MIN:
            continue
        traverse = min(blob["y"] + blob["h"], bas) - max(blob["y"], haut) >= 0.5 * (bas - haut)
        for cote in cotes:
            debut, fin = ZONES[cote][0] * largeur, ZONES[cote][1] * largeur
            if traverse and min(blob["x"] + blob["l"], fin) - max(blob["x"], debut) > 0:
                trouvees.append(blob)
                break
    return trouvees


def toutes_methodes(image, boite, verite, lecteur, pli):
    """Vrai si au moins une méthode de la cascade (ou le cadre seul) donne la bonne lecture (oracle)."""
    decoupe, _, cadre = decouper(image, boite)
    flou = cv2.GaussianBlur(cv2.cvtColor(decoupe, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    x1, y1, x2, y2 = cadre
    essais = [methode(flou, cadre) for _, methode in METHODES]
    essais.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
    for coins in essais:
        if coins is None:
            continue
        binaire, gris = binariser(redresser(decoupe, coins))
        resultat = segmenter(binaire, gris)
        resultat["binaire"] = binaire
        decision = lire_option_a(resultat, lecteur, pli)
        if decision["statut"] == "lue" and decision["texte"] == verite:
            return True
    return False


def relire_sans_lisere(resultat, lecteur, pli):
    """Même plaque redressée, binarisée SANS effacer les longues lignes, puis lue."""
    binaire, gris = binariser(resultat["plaque"], lisere=False)
    nouveau = segmenter(binaire, gris)
    nouveau["binaire"] = binaire
    return lire_option_a(nouveau, lecteur, pli)


def panneau(e, resultat, decision, indicateurs, cause):
    """Découpe autour du cadre YOLO (bleu) avec le quadrilatère (vert), plaque redressée, binarisation annotée :
    chiffres probables en vert, autres candidats en orange, soudures (grosses composantes) en rouge."""
    image = cv2.imread(e["chemin"])
    x1, y1, x2, y2 = e["boite"]
    vue = image.copy()
    epaisseur = max(2, image.shape[1] // 400)
    cv2.rectangle(vue, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), epaisseur)
    cv2.polylines(vue, [resultat["coins"].astype(np.int32)], True, (0, 255, 0), epaisseur)
    marge_x, marge_y = 0.35 * (x2 - x1), 0.8 * (y2 - y1)
    zone = vue[max(0, int(y1 - marge_y)):int(y2 + marge_y), max(0, int(x1 - marge_x)):int(x2 + marge_x)]
    binaire = cv2.cvtColor(resultat["binaire"], cv2.COLOR_GRAY2BGR)
    for blob in resultat["candidats"]:
        couleur = (0, 200, 0) if blob["chiffre"] else (0, 140, 255)
        cv2.rectangle(binaire, (blob["x"], blob["y"]), (blob["x"] + blob["l"] - 1, blob["y"] + blob["h"] - 1), couleur, 1)
    for blob in indicateurs["soudures"]:
        cv2.rectangle(binaire, (blob["x"], blob["y"]), (blob["x"] + blob["l"] - 1, blob["y"] + blob["h"] - 1),
                      (0, 0, 255), 2)
    pente = "-" if indicateurs["pente"] is None else f"{indicateurs['pente']:+.3f}"
    texte = (f"{e['nom']} {resultat['methode_coins']} lu [{chiffres_lus(decision)}] vrai [{e['verite']}] "
             f"couv {indicateurs['couverture']:.2f} deb {indicateurs['debordement']:.2f} pente {pente}")
    return np.vstack([faire_vignette(np.zeros((18, 450, 3), dtype=np.uint8), 450, 18, texte[:66]),
                      faire_vignette(zone, 450, 120), resultat["plaque"], binaire])


def main():
    parseur = argparse.ArgumentParser(description="Diagnostic des chiffres perdus (réglage, hors pli).")
    parseur.add_argument("nom", help="essai de validation croisée dont on lit les modèles (ex. gris_8)")
    arguments = parseur.parse_args()
    with open(FICHIER_PLIS, encoding="utf-8", newline="") as fichier:
        plis = {ligne["image"]: int(ligne["pli"]) for ligne in csv.DictReader(fichier)}
    plaques = [p for p in lire_plaques("reglage")[0] if f"{p['chemin'].parent.name}/{p['image']}" in plis]
    lecteur, detecteur = LecteurPlis(arguments.nom), DetecteurPlaques(str(MODELE_DETECTION))

    bonnes, pertes = [], []      # indicateurs des plaques bien lues ; (plaque, e, résultat, décision, indicateurs)
    for plaque in plaques:
        e, resultat = detecter_et_traiter(plaque, detecteur)
        e["chemin"] = str(plaque["chemin"])
        if resultat is None:
            continue
        decision = lire_option_a(resultat, lecteur, plis[e["nom"]])
        couv, deb = couverture_du_cadre(resultat["coins"], e["boite"])
        indicateurs = {"couverture": couv, "debordement": deb, "pente": pente_rangee(resultat["candidats"], *resultat["binaire"].shape),
                       "nb_probables": len(resultat["serie"]) + len(resultat["numero"])}
        verite = plaque["serie"] + plaque["numero"]
        if decision["statut"] == "lue" and decision["texte"] == e["verite"]:
            bonnes.append(indicateurs)
        elif len(chiffres_lus(decision)) < len(verite):
            perdus = chiffres_perdus(verite, chiffres_lus(decision))
            cotes = sorted({"serie" if k < len(plaque["serie"]) else "numero" for k in perdus})
            indicateurs.update({"perdus": [verite[k] for k in perdus], "soudures": soudures(resultat, cotes)})
            pertes.append((plaque, e, resultat, decision, indicateurs))

    # Seuils : ce que les plaques bien lues ne dépassent presque jamais
    seuils = {"couverture": float(np.percentile([b["couverture"] for b in bonnes], CENTILE)),
              "debordement": float(np.percentile([b["debordement"] for b in bonnes], 100 - CENTILE)),
              "pente": float(np.percentile([abs(b["pente"]) for b in bonnes if b["pente"] is not None], 100 - CENTILE))}
    print(f"Plaques bien lues : {len(bonnes)} ; plaques qui perdent des chiffres : {len(pertes)} sur {len(plaques)}")
    print(f"Seuils (centiles {CENTILE} / {100 - CENTILE} des plaques bien lues) : couverture < {seuils['couverture']:.2f}, "
          f"débordement > {seuils['debordement']:.2f}, |pente| > {seuils['pente']:.3f}")

    par_cause = {cause: [] for cause in CAUSES}
    for plaque, e, resultat, decision, ind in pertes:
        if ind["couverture"] < seuils["couverture"]:
            cause = "coupée"
        elif ind["debordement"] > seuils["debordement"] or (ind["pente"] is not None and abs(ind["pente"]) > seuils["pente"]):
            cause = "coins faux"
        elif ind["soudures"]:
            cause = "soudure au liseré"
        elif ind["nb_probables"] == 0:
            cause = "aucun chiffre"
        else:
            cause = "autre"
        ind["recuperable"] = toutes_methodes(cv2.imread(e["chemin"]), e["boite"], e["verite"], lecteur, plis[e["nom"]])
        par_cause[cause].append((plaque, e, resultat, decision, ind))

    acceptees = sum(1 for _, _, _, d, _ in pertes if d["statut"] == "lue")
    print(f"\nCauses (plaques ; dont lectures plus courtes ACCEPTÉES ; dont récupérables par une autre méthode de la "
          f"cascade) :")
    for cause, lot in par_cause.items():
        lues = sum(1 for _, _, _, d, _ in lot if d["statut"] == "lue")
        recuperables = sum(1 for *_, ind in lot if ind["recuperable"])
        methodes = Counter(r["methode_coins"] for _, _, r, _, _ in lot)
        print(f"   {cause:18s}: {len(lot):3d} ({len(lot) / len(pertes) * 100:4.1f} %) ; acceptées {lues:2d} sur "
              f"{acceptees} ; récupérables {recuperables:3d} ; méthodes {dict(methodes.most_common())}")

    # Le « 1 » : part parmi les chiffres perdus, et effet de l'effacement des longues lignes
    tous = Counter(c for p in plaques for c in p["serie"] + p["numero"])
    perdus = Counter(c for *_, ind in pertes for c in ind["perdus"])
    print("\nChiffres perdus par classe (part parmi les perdus / parmi tous les chiffres) :")
    print("   " + "  ".join(f"{k}: {perdus[k] / sum(perdus.values()) * 100:4.1f}/{tous[k] / sum(tous.values()) * 100:4.1f}"
                            for k in "0123456789"))
    sans_lisere = Counter()
    for plaque, e, resultat, decision, ind in pertes:
        groupe = "un « 1 » perdu" if "1" in ind["perdus"] else "aucun « 1 » perdu"
        d = relire_sans_lisere(resultat, lecteur, plis[e["nom"]])
        sans_lisere[(groupe, "total")] += 1
        sans_lisere[(groupe, "lue juste")] += d["statut"] == "lue" and d["texte"] == e["verite"]
    for groupe in ("un « 1 » perdu", "aucun « 1 » perdu"):
        print(f"   plaques avec {groupe:18s}: {sans_lisere[(groupe, 'total')]:3d} ; lues justes sans effacer les "
              f"longues lignes : {sans_lisere[(groupe, 'lue juste')]}")

    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    for numero, (cause, lot) in enumerate(par_cause.items(), start=1):
        if lot:
            panneaux = [panneau(e, r, d, ind, cause) for _, e, r, d, ind in lot[:PAR_PLANCHE]]
            chemin = DOSSIER_FIGURES / f"diagnostic_{numero}_{cause.split()[0].replace('é', 'e')}.jpg"
            cv2.imwrite(str(chemin), assembler_planche(panneaux, 2))
            print(f"Planche : {chemin.name} ({min(len(lot), PAR_PLANCHE)} plaques sur {len(lot)})")


if __name__ == "__main__":
    main()
