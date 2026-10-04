# donnees/ — jeux de données d'entraînement (contenu non versionné)

Le contenu de ce dossier reste sur le PC : il est exclu de git, car il est
volumineux et contient des photos de vraies plaques.

| Sous-dossier | Contenu | Phase |
|---|---|---|
| `kaggle_brut/` | jeu Kaggle « Tunisian Licensed Plates » tel que téléchargé (709 images, 1 classe, annotations LabelImg, format à vérifier) | 2 |
| `yolo/` | le même jeu converti au format YOLO, séparé en entraînement et validation | 2 |
| `caracteres/` | imagettes 32×32 en niveaux de gris, rangées par classe (`0` … `9`, `autre`) pour le CNN | 4 |

Règle : aucune image de `validation/` (photos prises au téléphone) ne doit
jamais se retrouver ici.
