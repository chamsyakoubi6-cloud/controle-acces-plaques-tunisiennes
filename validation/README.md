# validation/ — photos et vidéos de test (contenu non versionné)

Photos et vidéos filmées au téléphone, **jamais utilisées pour l'entraînement** :
elles servent uniquement à mesurer les performances du système complet (phase 5).
Protocole de prise de vue : `docs/journal.md`, phase 5.

## Convention de nommage

Le nom du fichier donne le vrai numéro, sans espaces, avec `TU` à la place de « تونس » :

- `215TU4567.mp4` → vidéo de la plaque « 215 TU 4567 », dans `videos/`
- `215TU4567_1.jpg`, `215TU4567_2.jpg`, `215TU4567_3.jpg` → photos du même véhicule, dans `photos/`
- étiquettes facultatives après le numéro, séparées par `_` :
  - `bleu` : plaque de location (fond bleu), ex. `215TU4567_bleu.mp4`, `215TU4567_bleu_1.jpg` ;
  - `arriere` : plaque arrière ;
  - toute autre étiquette (`nuit`, `2`...) sert seulement à distinguer les prises.
- la variante « (2) » que Windows ajoute à un nom déjà pris est acceptée : `215TU4567 (2).jpeg`
- noms en ASCII uniquement (sous Windows, `cv2.imread` échoue sur les chemins non ASCII)
- formats : jpg, jpeg, png ; mp4, mov. Pas de HEIC : choisir « Le plus compatible » dans les réglages de
  l'appareil photo de l'iPhone.

## Organisation

- `photos/` : images
- `videos/` : vidéos, un seul véhicule visé par vidéo
- `decoupage.csv` : partie de chaque véhicule, « developpement » ou « test » (écrit par l'outil ci-dessous ;
  un véhicule déjà attribué ne change jamais de partie)

## Outil

```powershell
python -m outils.validation               # inventaire et contrôles : noms, fichiers lisibles, parties
python -m outils.validation --decouper    # tire au sort la partie des véhicules nouveaux
```

Le découpage se fait par véhicule : toutes les prises d'une voiture vont dans la même partie.
« developpement » sert à toutes les décisions de la phase 5 ; « test » est évalué une seule fois, à la fin.
