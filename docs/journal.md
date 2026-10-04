# Journal de bord

Décisions, versions et résultats, au fil des phases (matière pour le rapport et l'oral).

## 2026-09-29 — Phase 1 : mise en place

### Constats sur le PC

- Versions de Python présentes : 3.12.10 (par défaut), 3.10.4, 2.7 → **3.11.9 installé** via winget
  (installation utilisateur ; la 3.12 reste la version par défaut).
- GPU : RTX 3050 Laptop 4 Go, pilote 610.88, CUDA pilote ≤ 13.3 → PyTorch cu126 compatible,
  pas besoin du CUDA Toolkit (les paquets PyTorch embarquent leur propre CUDA).
- CPU Ryzen 5 5600H, 15 Go de RAM, webcam HP TrueVision HD.

### Versions retenues et raisons

| Paquet | Version | Raison |
|---|---|---|
| torch / torchvision | 2.14.0+cu126 / 0.29.0+cu126 | index pytorch.org (sous Windows, PyPI ne fournit que la version CPU), installés avant le reste |
| numpy | 1.26.4 | famille 1.x comme la Jetson (1.19.4) : NumPy 2 change les règles de promotion des types |
| opencv-python | 4.11.0.86 | les versions 4.12 et suivantes exigent NumPy 2 ; reste proche de la 4.1.1 de la Jetson |
| tensorflow / keras | 2.21.0 / 3.15.1 | à jour et cohérent avec les onnx/protobuf récents ; CPU uniquement sous Windows |
| tf2onnx | 1.17.0 | conversion du CNN ; l'opset 12 n'est plus testé officiellement → vérifié ci-dessous |
| onnx / onnxruntime / onnxslim | 1.23.1 / 1.30.0 / 0.1.97 | export et inférence sur PC |
| ultralytics | 8.4.166 | YOLOv8n |

`pip check` : aucun conflit. Versions exactes : `requirements-pc.lock.txt` (67 paquets).

### Résultats des vérifications

**`outils/verifier_env.py`** : 17 OK, 1 avertissement, 0 échec.
- GPU vu par PyTorch : capacité de calcul 8.6, CUDA 12.6, cuDNN 9.10. Produit matriciel sur GPU
  identique au CPU (écart relatif 7e-7). Échange NumPy 1.26 ↔ PyTorch correct. TensorFlow fonctionne sur CPU.
- Webcam : 1280×720 obtenu, mais seulement **10 images/s** (avertissement).

**Diagnostic de la webcam** (luminosité moyenne 58/255, exposition courte : ce n'est pas un manque de lumière)

| Pilote vidéo | Résolution | Cadence |
|---|---|---|
| DirectShow (MJPG avant ou après la résolution, FPS=30) | 1280×720 | 10,0 i/s |
| DirectShow | 640×480 | 10,0 i/s |
| Media Foundation (MSMF) | 1280×720 | 30 i/s |

Ouverture avec MSMF : 10,3 s par défaut, 1,3 s avec `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0`.
→ Proposition : utiliser MSMF sous Windows (en attente de décision).

**`entrainement/verifier_export_onnx.py`** : les deux exports réussissent.

| Fichier | Opset | IR | Producteur | Entrée → sortie | Comparaison avec l'original |
|---|---|---|---|---|---|
| `cnn_test.onnx` | 12 | 7 | tf2onnx 1.17.0 | (1, 32, 32, 1) → (1, 11) | écart max 3,7e-8 |
| `yolov8n_test.onnx` | 12 | 7 | pytorch 2.14.0 | (1, 3, 640, 640) → (1, 84, 8400) | écart relatif 3,2e-6 ; « person » à 0,90 sur l'image de test |

- L'ancien exporteur de PyTorch (`dynamo=False`, utilisé par Ultralytics) fonctionne encore en 2.14.
- Keras 3 ignorait la taille de lot de `keras.Input` au moment de l'export : corrigé en fixant la signature d'export.
- Les deux fichiers sont gardés dans `sorties/verification/jetson/` pour le test anticipé avec trtexec.

**vermin** (cible Python 3.6) sur `systeme/` et `outils/` : aucune violation (version minimale détectée : 3.5).
Contre-épreuve sur un fichier volontairement incompatible : `dataclasses`, `:=`, `list[int]` et `X | None`
sont bien détectés.

### Pièges rencontrés

- PowerShell coupe l'argument `-t=3.6-` en `-t=3` + `.6-` : il faut le mettre entre guillemets.
- vermin ne détecte `X | Y` et `f"{x=}"` qu'avec `--feature union-types` et `--feature fstring-self-doc`.

### Suite

- Choisir le pilote webcam sous Windows (MSMF proposé).
- Test anticipé sur Jetson dès réception de la carte (voir README.md, section « Jetson Nano »).
- Phase 2, la détection, sur accord.

## 2026-09-30 — Phase 1 (fin) : webcam en Media Foundation

- **Décision** : sous Windows, la webcam passe par Media Foundation (MSMF) et non plus par DirectShow.
- **Test complémentaire** : définie *après* `import cv2`, la variable `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS`
  reste sans effet (1re image en 10,7 s) : OpenCV la lit à son chargement.
- **Réglage placé à un seul endroit** : `systeme/__init__.py`, que Python exécute avant tout module du paquet.
  - Les scripts de `outils/` et `entrainement/` importent `systeme` avant `cv2`.
  - Ils se lancent depuis la racine avec `python -m` (ex. `python -m outils.verifier_env`).
  - Un avertissement signale un `cv2` importé trop tôt.
- **Résultat de `python -m outils.verifier_env`** : 18 OK, 0 avertissement, 0 échec. Webcam en 1280×720
  à 30,2 images/s, 1re image en 1,6 s.
- **`python -m entrainement.verifier_export_onnx`** : inchangé (opset 12 et IR 7 pour les deux fichiers).
- Phase 1 terminée.

## 2026-09-30 — Phase 2 : détection de la plaque

### 1. Inspection du jeu Kaggle (`entrainement/inspecter_donnees.py`)

- 709 images : 567 dans `train`, 142 dans `test`. Annotations Pascal VOC (LabelImg), classe « LP »,
  une plaque par image. Aucun défaut technique : tailles XML exactes, pas de rotation EXIF, aucun cadre invalide.
- **Images couchées** : 9 dans train et 2 dans test ont leurs pixels tournés de 90°, alors que l'EXIF est normal.
  Pour chacune, les deux sens de redressement ont été affichés côte à côte : les 9 se redressent toutes dans
  le sens horaire (`couchees_1.jpg`, `couchees_2.jpg`).
- **Vues de trois quarts** : rapport largeur/hauteur médian des cadres de 3,1, contre 4,5 pour une plaque vue
  de face. Ce sont des cas légitimes (plaques vues de biais), conservés.
- **Doublons** : 7 exacts (MD5) et 11 copies réenregistrées de la même photo, que seul le dHash détecte.
  Répartition des écarts dHash : un creux net entre 2 et 5 bits.
  - Seuil **doublon** : 2 bits ou moins (vérifié à l'oeil, c'est la même photo).
  - Seuil de **regroupement** pour le découpage : 12 bits. C'est un seuil volontairement large, car regrouper
    à tort deux voitures différentes est sans conséquence, alors que séparer deux photos de la même voiture
    serait une fuite de données. Même à 12 bits, le plus gros groupe ne compte que 11 images.
- **Aucune fuite entre train et test** : les 27 paires proches à cheval sur les deux ont été vérifiées à l'oeil,
  ce sont toutes des voitures différentes.
- Limite du dHash : deux photos de la même voiture sous des angles opposés ne sont pas rapprochées (569/570).

### 2. Conversion YOLO (`entrainement/preparer_yolo.py`)

- 18 doublons retirés et 8 images redressées (la 9e était un doublon). Les rotations sont listées dans
  `entrainement/corrections_kaggle.csv` (versionné).
- Découpage avec une graine de 42, par groupes : **439 images en entraînement, 110 en validation (20,0 %)**.
  Les **142 images de test** sont copiées telles quelles.
- Contrôle : les cadres relus depuis les fichiers YOLO tombent sur les plaques, y compris pour les images
  redressées (`preparation/controle_etiquettes_yolo.jpg`).

### 3. Entraînement (`entrainement/entrainer_yolo.py`)

- Point de départ : YOLOv8n pré-entraîné sur COCO. 100 époques au plus, arrêt anticipé après 30 époques
  sans progrès, lot de 16, AMP, graine 0.
- Optimiseur `auto` : AdamW avec un taux initial de 0,002. Déduit de la colonne de taux d'apprentissage
  de `results.csv` (0,001406 à l'époque 31).
- Augmentations : **pas de miroir**, rotation de ±5°, mosaïque (coupée pour les 10 dernières époques),
  échelle ±50 %, translation ±10 %, variations de teinte, saturation et luminosité.

| Entrée | Époques | Meilleure époque | Durée |
|---|---|---|---|
| 320 | 96 (arrêt anticipé) | 66 | 8,7 min |
| 416 | 59 (arrêt anticipé) | 29 | 7,0 min |
| 640 | 100 | - | 15,8 min |

### 4. Évaluation (`entrainement/evaluer_yolo.py`)

**Choix de la taille, sur la validation**

| Entrée | mAP50 | mAP50-95 | Précision | Rappel |
|---|---|---|---|---|
| 320 | 0,995 | 0,724 | 1,000 | 0,996 |
| 416 | 0,993 | 0,743 | 0,991 | 0,996 |
| **640** | 0,995 | **0,769** | 0,988 | 1,000 |

La règle fixée avant les entraînements retient la plus petite taille à moins de 0,02 du meilleur mAP50-95.
Elle désigne **640** : 416 est à 0,026 du meilleur, 320 à 0,045. Le mAP50 est au plafond pour les trois ;
c'est la précision des cadres (mAP50-95) qui départage, et elle compte pour le redressement de la phase 3.

**Seuil de confiance, sur la validation, modèle 640**

| Seuil | 0,25 | 0,40 | 0,50 | 0,60 | 0,65 | 0,70 | 0,80 | 0,85 | 0,90 |
|---|---|---|---|---|---|---|---|---|---|
| Précision | 0,946 | 0,966 | 0,975 | 0,978 | 0,980 | 0,982 | 0,991 | 0,992 | 1,000 |
| Rappel | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,955 | 0,123 |
| F1 | 0,972 | 0,983 | 0,987 | 0,989 | 0,990 | 0,991 | 0,995 | 0,973 | 0,220 |

- F1 maximal à 0,80.
- Le F1 reste à moins de 0,01 du maximum entre 0,44 et 0,83.
- Au-delà de 0,85, le rappel s'effondre.

**Décision : le seuil de confiance reste à 0,5.**
- Le détecteur est le premier étage d'une chaîne en cascade : il doit privilégier le rappel.
- Une fausse détection sera éliminée plus loin par la lecture (CNN) et le contrôle du format de la plaque,
  alors qu'une plaque manquée est perdue pour cette image.
- Passer à 0,65 ne ferait gagner que 0,5 point de précision en validation (0,975 → 0,980), à rappel égal (1,000).
- Le seuil sera réajusté en phase 5 sur les vidéos filmées pour le projet, c'est-à-dire sur les conditions
  réelles d'utilisation.

**Test** (une seule fois, modèle 640) : mAP50 0,984, mAP50-95 0,711, précision 1,000, rappel 0,971.

**Au seuil de fonctionnement** (confiance 0,5, IoU de NMS 0,45, letterbox carré) :
- **138 plaques sur 142 détectées (97,2 %)** ;
- 2 fausses détections ;
- IoU moyenne de 0,865 sur les détections justes.

| Image | Problème | Cause |
|---|---|---|
| 22, 141 | manquées (aucune boîte, même à 0,001) | images couchées, laissées telles quelles dans le test |
| 12 | manquée (score 0,028) | voiture ancienne, plaque de l'ancien format |
| 140 | manquée (score 0,394) | arrière de BMW ; score juste sous le seuil en letterbox carré (0,79 en rectangulaire) |
| 67 | fausse détection (0,70) | panneau « À VENDRE » |
| 126 | fausse détection (0,58) | petit rectangle au bord de l'image (showroom) |

**Découverte** : Ultralytics utilise un letterbox rectangulaire pour une image seule et un letterbox carré
en mode liste. Notre modèle ONNX à entrée fixe travaille en carré, et les mesures du système sont donc
faites en carré (`rect=False`).

Figures (non versionnées, `docs/figures/phase2/`) :
- `comparaison_tailles.png`, `seuil_confiance_validation.png`, `entrainement_640.png` ;
- `test/` (courbes précision-rappel, F1, matrice de confusion) ;
- `echecs_test.jpg`, `exemples_detections_test.jpg` ;
- `inspection/`, `preparation/`.

### 5. Export (`entrainement/exporter_yolo.py`)

- Les trois tailles en ONNX opset 12, entrée fixe, sans NMS : **opset 12 et IR 7** partout. Écart entre
  ONNX Runtime et PyTorch de l'ordre de 1e-6.
- Emplacements : `modeles/detecteur.onnx` (640) et `sorties/verification/jetson/detecteur_{320,416,640}.onnx`.
- Latence d'inférence ONNX Runtime sur le CPU du PC, mesurée dans un processus sans PyTorch :
  **~8,5 ms (320), ~14 ms (416), 33 à 45 ms (640)**. Les rapports suivent ceux des calculs (×1,7 et ×2,4).
  Mesurée avec PyTorch chargé dans le même processus, la latence était environ deux fois plus élevée,
  d'où l'outil `outils/mesurer_latence.py`.

### 6. Chaîne NumPy et démonstration

- `systeme/inference.py`, `systeme/detection.py` et `systeme/sources.py` sont compatibles Python 3.6
  (vermin : aucune violation).
- **Notre chaîne reproduit exactement celle d'Ultralytics** (`verifier_detection_numpy`). Sur les 142 images,
  146 boîtes ont été comparées : IoU minimale de 1,0000, écart de score nul. Au seuil de fonctionnement,
  on retrouve 97,2 % de plaques détectées et 2 fausses détections.
- Démonstration `python -m outils.demo_detection` avec le modèle 640 sur le PC :
  - environ 20 images/s, pour 3,3 ms de pré-traitement, 38 ms d'inférence et 0,2 ms de post-traitement ;
  - aucune fausse détection sur 183 images de la pièce, sans plaque.

### Pièges rencontrés

- `cv2.putText` n'écrit que l'ASCII : « manquée » s'affichait « manqu??e ».
- Un script lu sur l'entrée standard (`python -`) bloque les processus de chargement d'Ultralytics sous Windows.
- Letterbox rectangulaire ou carré selon la façon d'appeler `predict` (voir ci-dessus).

### Suite

- Seuil de confiance : maintenu à 0,5 (priorité au rappel), à réajuster en phase 5 sur les vidéos du projet.

## 2026-09-30 — Phase 3 : traitement classique de la plaque

### Décisions de départ

- Plaque redressée à **450 × 100 px** (rapport 4,5 de la plaque avant, celle que voit la caméra de la barrière).
- Vérité terrain : saisie de tout le jeu (691 plaques) avec `entrainement/saisir_numeros.py` (clavier AZERTY
  accepté). Ordre : 100 plaques de réglage tirées au hasard, puis le reste du réglage, puis le test.
  Fichier hors de git, avec une copie dans OneDrive à chaque séance.
- Plaques de location : aucune dans le jeu Kaggle. Elles seront vérifiées sur des photos réelles, qui ne
  serviront jamais à régler les paramètres, et par une simulation de fond bleu.

### Constats pendant le développement (avant la vérité terrain)

- **Aperçu des plaques** : liseré blanc sur la plupart, vis ou points aux extrémités, cadres de concessionnaire
  avec du texte, au moins deux plaques sur deux lignes (568, 165).
- **Ordre des coins** : l'astuce x + y / y − x échoue au-delà de 45° d'inclinaison (348). Elle est remplacée
  par un tri par angle autour du centre, le bord supérieur étant le grand côté orienté vers la droite.
- **Filtrage des chiffres** : barres du cadre (18 % plus hautes) et fragments du mot arabe (17 % moins hauts)
  passaient pour des chiffres. On exige donc la régularité d'une plaque : hauteur à ±15 % de la médiane,
  centres alignés.
- **Contours dentelés** par les caractères qui touchent le bord de la zone : on prend l'enveloppe convexe
  avant l'approximation polygonale.
- **Mauvais contour choisi** (le trottoir sous la plaque, 257) : le quadrilatère doit contenir le centre du
  cadre, et son centre doit être dans le cadre.
- **Seuils globaux d'Otsu** mis en défaut par les voitures de la teinte de la plaque ou du liseré (130, 199) :
  ajout de la méthode « bords » (Canny, seuils autour de la médiane). Elle est placée après la méthode
  « caractères », qui réussit mieux quand les deux s'appliquent.
- **Répartition des méthodes sur les 691 plaques**, avec les « comptes plausibles » (1 à 3 chiffres de série,
  1 à 4 de numéro), une condition nécessaire mais pas suffisante :

| Méthode | Part des plaques | Comptes plausibles |
|---|---|---|
| liseré | 63,5 % | 75 % |
| zone noire | 5,8 % | 50 % |
| caractères | 16,9 % | 71 % |
| bords | 9,3 % | 45 % |
| cadre brut (dernier recours) | 4,5 % | 10 % |

  Au total, 67,3 % des plaques donnent des comptes plausibles. Le cadre brut ne sert plus qu'aux plaques très
  vues de biais ou très petites.
- **Durée** : 3 à 10 ms par plaque sur le PC pour les trois étapes.

### Réglage sur la vérité terrain (122 plaques de réglage, 1ʳᵉ séance de saisie)

Méthode : chaque paramètre est choisi par **balayage du taux de segmentation correcte**, au centre du plateau
et non sur un pic isolé. Avec 122 plaques, un écart de 2 à 3 points ne représente que 3 ou 4 plaques.

Deux mesures sont suivies :
- **stricte** : exactement le bon nombre de chiffres probables de chaque côté ;
- **aucun chiffre perdu** : au moins le bon nombre, les candidats en trop étant tolérés. Dans le système
  complet, c'est le CNN qui décidera ; il classera « autre » les barres du cadre et les fragments arabes.

| Étape du réglage | Stricte | Aucun chiffre perdu |
|---|---|---|
| Réglages de départ | 34,4 % | - |
| Marge de redressement de 2 % (x) et 5 % (y) | 45,1 % | - |
| Canal minimum, nettoyage 3×3 | 46,7 % | - |
| Hauteur minimale d'un chiffre de 0,45 à 0,30 | 49,2 % | 56,6 % |
| Découpage des caractères soudés (projection verticale) | **50,8 %** | **62,3 %** |

**Justifications mesurées** (figures `balayage_binarisation.png` et `mesures_chiffres.png`) :

- **Marge de redressement.** Le contour trouvé est souvent le bord intérieur du liseré, collé aux caractères.
  Sans marge, les chiffres touchent le bord de l'image redressée et sont écartés. Au-delà de 10 % de marge
  verticale, le taux baisse, car la plaque devient trop petite.
- **Seuillage.** Otsu obtient 39 %. L'adaptatif forme un plateau de 42 à 51 % pour des blocs de 31 à 51 px
  et C de −20 à −10. On retient bloc 31 et C = −10, au sommet. Un bloc de 15 px, plus petit qu'un caractère
  d'environ 60 px, s'effondre à 0–35 %.
- **Apport de chaque étape**, en la retirant seule : nettoyage 3×3 +6,5 points, CLAHE +4,9 points,
  effacement du liseré +1,6 point.
- **Noyau de nettoyage.** 4×4 donne 42,6 % et 5×5 donne 36,1 % : les traits font au moins 7,4 px (p10) et
  une ouverture trop proche de cette épaisseur les ronge.
- **Tolérance de régularité** (hauteur des chiffres par rapport à la médiane) : 10 % → 46,7 % ;
  **15 % → 49,2 %** ; 20 % → 37,7 % ; 25 % → 32,8 %. Un optimum net.
- **Hauteur minimale d'un chiffre.** 0,45 écartait de vrais chiffres, qui paraissent plus petits quand le
  quadrilatère est plus grand que la plaque. **Biais de sélection** : mesurer la hauteur des chiffres sur les
  seules plaques réussies ne voit jamais les chiffres plus petits que le filtre, et « confirme » donc le
  filtre. Seule la mesure directe du taux de réussite est fiable.
- **Écarté par la mesure :**
  - exclure les blobs proches des bords gauche et droit (contre les barres du cadre) : de 49 % à 20 % pour
    une marge de 8 %, car les chiffres extrêmes sont souvent près du bord ;
  - filtrer par taux de remplissage : aucun effet, car les barres légèrement penchées ressemblent à un
    « 1 » ;
  - contrôles géométriques du quadrilatère (côtés courts, côtés opposés) : aucun effet sur le taux, mais
    conservés contre les quadrilatères aberrants.
- **Fond bleu simulé** (62 plaques bien segmentées, fond recoloré) :

| Fond | Canal « luminance » | Canal « minimum » |
|---|---|---|
| bleu foncé | 77 % | 77 % |
| bleu clair | 57 % | **82 %** |

  Sur fond noir, les deux canaux font jeu égal. Le canal minimum est donc retenu : il ne coûte rien et il
  protège sur fond bleu. La vérification sur de vraies photos de location reste à faire.

### Évaluation sur les plaques de réglage (réglages finaux)

- Segmentation : **50,8 % en mesure stricte, 62,3 % sans chiffre perdu**, environ 8 ms par plaque sur le PC.

| Méthode de redressement | Part des plaques | Correctes |
|---|---|---|
| liseré | 60 % | 59 % |
| caractères | 15 % | 50 % |
| bords | 12 % | 40 % |
| zone noire | 10 % | 33 % |
| cadre brut | 3 % | 0 % |

| Catégorie | Correctes |
|---|---|
| sans difficulté mesurée | 58 % |
| floue | 47 % |
| de biais (57 plaques sur 122) | 44 % |
| petite | 20 % |

- **Redressement, contrôle visuel** de 60 plaques tirées au hasard (`redressement_reglage.jpg`) :
  39 bien redressées (65 %), 9 approximatives mais lisibles (15 %), 12 ratées (20 %), presque toutes
  des vues de biais ou sombres.
- **Échecs restants** : chiffres soudés ou coupés sur les plaques floues ou petites, quadrilatères faux sur les
  fortes vues de biais, barres du cadre et fragments arabes pris pour des chiffres. Ces derniers seront rattrapés
  par le CNN, d'où l'écart entre les deux mesures.
- **Limite du jeu** : 47 % des plaques de réglage sont vues de biais (photos de voitures garées). Devant une
  barrière, avec une caméra face à la voiture, on se rapproche du cas « sans difficulté ».

Figures (non versionnées, `docs/figures/phase3/`) : `balayage_binarisation.png`, `mesures_chiffres.png`,
`fond_bleu_simule.jpg`, `echecs_reglage.jpg`, `redressement_reglage.jpg`, et
`etapes_525/295/505/569/633.png` (étape par étape).

### Vérité terrain complète et confirmation des réglages

- Saisie terminée : **691 plaques**, dont 548 de réglage avec un numéro, 1 « autre format », et
  142 plaques de test (toutes lisibles).
- Réglages vérifiés sur les **548 plaques de réglage**, dont 426 jamais vues pendant le réglage :
  52,0 % en mesure stricte et 64,1 % sans chiffre perdu, contre 50,8 % et 62,3 % sur les 122 de départ.
  **Le réglage généralise** au sein de la même source d'images.
- Chaque variante testée reste dans le bruit ou fait pire :
  - C = −15 : 13 plaques perdues en mesure « sans perte » ;
  - tolérance de régularité de 20 % : la mesure stricte chute à 37,6 %, 15 % est confirmé ;
  - sans découpage des caractères soudés : 13 plaques perdues en mesure « sans perte » ;
  - sans effacement du liseré : +4 plaques en stricte, −2 en « sans perte ». C'est du bruit, l'étape est conservée.

  Aucun réglage n'a été modifié.

### Évaluation finale sur le test (une seule fois, 142 plaques)

| Mesure | Réglage (548) | Test, cadres vrais | Test, cadres YOLO |
|---|---|---|---|
| Plaques traitées | 548 | 142 | 138 détectées sur 142 |
| Stricte | 52,0 % | **31,0 %** | **30,3 %** |
| Aucun chiffre perdu | 64,1 % | **49,3 %** | **48,6 %** |

- **Cadres YOLO contre cadres vrais** : la perte est quasi nulle. Le traitement tolère l'imprécision des cadres
  détectés. Au réglage, l'écart était de 0,5 point en stricte et de 1,1 point sans perte.
- **Chute d'environ 20 points par rapport au réglage.** Ce n'est pas un surajustement, puisque le réglage a
  gardé son niveau sur 426 plaques non vues. C'est la **composition du test**, qui compte beaucoup plus
  d'images web, floues et compressées.

#### Pourquoi le test est plus bas

Netteté = variance du laplacien de la plaque redressée, en médiane.

| Source (plus grand côté de l'image) | Réglage : plaques, stricte / sans perte | Test : plaques, stricte / sans perte | Netteté (réglage / test) |
|---|---|---|---|
| ≤ 700 px (images réduites pour le web) | 15 (3 %) : 40 % / 53 % | **87 (61 %)** : 26 % / 43 % | 143 / 105 |
| 701 à 1000 px | 314 (57 %) : 48 % / 61 % | 42 (30 %) : 33 % / 55 % | 208 / 129 |
| 1001 à 2000 px | 219 (40 %) : 59 % / 69 % | 5 (4 %) : 40 % / 60 % | 510 / 235 |
| > 2000 px (photos pleine résolution) | 0 | 8 (6 %) : 62,5 % / 87,5 % | — / 1155 |

- **La netteté compte, plus que la taille de la plaque.** Dans les deux jeux, la réussite monte avec la
  netteté d'une tranche à l'autre. La netteté médiane du test vaut 131, contre 284 au réglage : deux fois moins.
- **La taille de la plaque joue peu.** Les plaques du test sont plus petites (hauteur médiane du cadre : 50 px
  contre 66 px). Mais au réglage, la réussite ne dépend pas de la hauteur, et au test elle est plus basse à
  toutes les hauteurs :

| Hauteur du cadre | Réglage : stricte (plaques) | Test : stricte (plaques) |
|---|---|---|
| moins de 40 px | 52 % (52) | 27 % (52) |
| 40 à 60 px | 57 % (182) | 37 % (43) |
| 60 à 90 px | 49 % (174) | 24 % (34) |
| 90 px et plus | 49 % (140) | 46 % (13) |

- **Angle de vue.** Il pèse dans les deux jeux. Réglage : 60 % de face contre 42 % de biais ; test : 38 %
  contre 17 %. Une plaque est dite de biais quand le rapport l/h de son cadre est < 2,5. L'angle n'explique
  pas la chute, puisque le test compte moins de vues de biais (33 % contre 43 %).
- Les 8 photos pleine résolution du test sont au niveau du réglage, voire au-dessus. L'effectif est faible.

#### Détail du test (cadres vrais)

Les méthodes de repli ne servent qu'aux plaques où le liseré échoue, c'est-à-dire aux plus difficiles.

| Méthode de redressement | Plaques | Correctes |
|---|---|---|
| liseré | 84 | 38 % |
| caractères | 20 | 30 % |
| cadre entier | 11 | 27 % |
| bords | 15 | 13 % |
| zone noire | 12 | 8 % |

| Catégorie | Plaques | Correctes |
|---|---|---|
| sans difficulté mesurée | 66 | 41 % |
| petite | 17 | 29 % |
| floue | 22 | 23 % |
| de biais | 47 | 17 % |

Types d'échec : il y a 98 échecs en mesure stricte.

- 26 plaques n'ont que des candidats en trop, que le CNN devra classer « autre ».
- 72 plaques ont perdu au moins un chiffre. Sur la planche des échecs, ce sont surtout des plaques floues ou
  compressées dont la binarisation casse les caractères, des vues de biais, et les deux images couchées du test.

#### Enseignement pour la phase 5

Le traitement classique demande des images **nettes** et **de face**. Pour la caméra du parking, cela veut dire :

- placer la caméra dans l'axe de la voie ;
- soigner la mise au point ;
- un temps de pose court, contre le flou de bougé ;
- un éclairage suffisant ;
- pas de forte compression.

Au réglage, les plaques de moins de 40 px réussissent aussi bien que les grandes : la taille de la plaque
n'est donc pas le critère principal.

Figures (non versionnées, `docs/figures/phase3/`) : `echecs_test.jpg`, `redressement_test.jpg`,
`echecs_test_yolo.jpg`, `redressement_test_yolo.jpg`.

### Suite

- Vérification sur les photos de plaques de location, quand elles seront prêtes (vérification seulement).
- Test anticipé sur Jetson : 5 fichiers à passer dans trtexec. La latence réelle de la Nano confirmera
  ou non le choix du 640 ; le 416 est prêt en cas de besoin.
- Phase 4 (lecture par le CNN), sur accord. Piste : étiqueter automatiquement les vignettes des 285 plaques de
  réglage segmentées exactement (mesure stricte), à l'aide des numéros saisis.

## 2026-10-01 — Phase 4 : lecture des caractères par le CNN

### Décisions de départ (plan validé)

- Validation croisée à 5 plis **par véhicule** (même numéro saisi = même plaque physique), pas par image.
- Les images de réglage des 5 véhicules présents aussi au test sont exclues de la phase 4. Le test sera
  rapporté sur ses 142 plaques et sur les 124 plaques de véhicules jamais vus.
- Seuil de confiance : le plus petit seuil qui ramène les substitutions (un chiffre confondu, même longueur)
  sous 1 % des plaques. La barrière ne s'ouvre de toute façon que sur une correspondance exacte avec la liste.
- Deux points d'arrêt : après le jeu de caractères, puis avant le test.

### Constats préalables (mesurés sur le réglage)

- **Même véhicule sur plusieurs images.** Il y a 204 véhicules pour 548 images, et jusqu'à 8 photos du même.
  72 véhicules ont des photos à la fois dans le train et dans la val YOLO : le dHash ne rapproche pas deux
  photos d'une même voiture prises sous d'autres angles. Pour la détection, c'est peu grave : le modèle apprend
  la forme d'une plaque. Pour la lecture, ce serait grave, puisque le modèle apprend les chiffres eux-mêmes.
- **5 véhicules sont communs au réglage et au test** (18 images de chaque côté). Le taux de détection de la
  phase 2 au test (97,2 %) compte donc 18 images de véhicules vus à l'entraînement.
- **19 % des plaques de réglage** (106 sur 548) perdent un chiffre à la segmentation tout en gardant un format
  valide. Le CNN lira les chiffres restants avec assurance : aucun seuil de confiance ne peut arrêter ces
  lectures. Un contrôle d'espacement (repérer un trou dans un groupe de chiffres) ne repère que 25 % de ces cas,
  avec 3,9 % de fausses alertes : il est écarté.
- Un découpage unique 80/20 ne laisserait que 16 à 24 exemples en validation pour le chiffre le plus rare,
  d'où le choix de la validation croisée.
- Architectures mesurées (3 blocs convolution 3 x 3, puis Dense 64) :

| Largeur | Paramètres | Calcul par caractère | Époque sur CPU (11 000 vignettes) |
|---|---|---|---|
| 8 | 39 600 | 0,70 M MAC | 1,7 s |
| 16 | 89 900 | 2,57 M MAC | 3,3 s |
| 32 | 225 000 | 9,86 M MAC | 9,4 s |

  Pour comparaison, le détecteur (YOLOv8n en 640) coûte 8,2 GFLOP et compte 3,0 M paramètres. En largeur 16,
  une plaque de 10 candidats représente moins de 1 % de son calcul.

### Jeu de caractères (`entrainement/extraire_caracteres.py`)

- **Étiquetage automatique.** Seules les plaques strictement segmentées sont gardées. Le k-ième chiffre
  probable de chaque groupe reçoit le k-ième chiffre saisi ; les autres candidats de ces plaques deviennent
  « autre ». Le contrôle du biais de sélection est bon : les proportions de chiffres sont les mêmes sur les
  plaques strictes et sur l'ensemble du réglage.
- **Versions dégradées pour l'entraînement.** Chaque plaque redressée reçoit 4 versions dégradées (éclairage,
  flou, faible résolution, bruit, JPEG). Elles sont rebinarisées par la phase 3 et découpées aux positions de
  la plaque propre ; l'étiquette est conservée et le masque binaire se dégrade comme dans la réalité.
  - Premier essai : tous les défauts cumulés, netteté médiane de 36, bien sous le p10 du réglage (67).
  - Version retenue : chaque défaut appliqué une fois sur deux, plages adoucies. Netteté médiane de 106,
    entre le p10 (67) et le p25 (133) du réglage.
- **Après corrections** : 277 plaques retenues (148 véhicules), soit 2 854 vignettes d'origine
  (1 887 chiffres et 967 « autre ») pour 1 004 chiffres physiques distincts, plus 11 416 vignettes dégradées.
  Le « 1 » représente 20 % des chiffres, car 78 % des séries commencent par 1. « autre » représente un tiers
  des vignettes.
- `systeme/segmentation.py` : la découpe des vignettes (`decouper_vignettes`) et le contrôle de régularité
  (`filtrer_regularite`) sont devenus des fonctions partagées. Le comportement est inchangé.

### Bruit d'étiquetage découvert à l'arrêt 1

- **Fautes de saisie dans la vérité terrain.** Elles étaient sans effet en phase 3, qui ne comparait que le
  nombre de chiffres ; toutes conservent la longueur.
  - Repérage sans modèle (`entrainement/verifier_saisie.py --voisines`) : des photos consécutives d'une même
    voiture dont les numéros diffèrent d'un chiffre ou par une inversion. Chaque cas a ensuite été vérifié à
    l'oeil.
  - Corrigées au réglage : train/184, 186, 196, 208, 248, 272, 312, 404, 498, 564, 701, 707 et val/479, 566.
  - Corrigée au test : test/122.
  - val/483 passe en « illisible » : le dernier chiffre est coupé par le bord de l'image, et une vignette de
    chiffre coupé apprendrait une fausse forme au CNN.
  - Ces corrections font tomber le nombre de véhicules de 199 à 185 : chaque faute créait un faux véhicule.
- **Incident de sauvegarde.** Les copies OneDrive d'avant et d'après correction ont été faites dans la même
  seconde ; la seconde a écrasé la première. L'état d'origine reste intact dans la copie de la dernière séance
  de saisie : elle ne diffère du fichier corrigé que par les 16 lignes modifiées, ce qui a été vérifié. Le
  défaut est corrigé : les copies portent maintenant un suffixe (`_avant_correction`, `_apres_correction`).
- **Coïncidence de segmentation.** Sur val/355, les vrais chiffres de la série sont soudés au cadre et ont été
  écartés. Trois fragments du mot arabe ont formé un groupe de 3 et reçu leurs étiquettes. Cette plaque est
  exclue du jeu de caractères (`entrainement/exclusions_caracteres.csv`, versionné : noms d'images seulement).
- **Relecture du test sans modèle** (`verifier_saisie.py --planche test`, 142 plaques relues à l'oeil). Deux
  saisies étaient fausses : test/11 (série) et test/50 (1er chiffre du numéro). L'étudiant les a vérifiées sur
  l'image agrandie, puis elles ont été corrigées. Pour test/22 (image couchée), la saisie est juste. Aucun
  modèle n'a été appliqué au test.

### Passage de nettoyage (validation croisée provisoire : masque binaire, largeur 16)

- **Exactitude hors pli : 97,30 %** (77 erreurs sur 2 854 vignettes), obtenue avec des étiquettes encore
  bruitées. Cette mesure ne sert à aucune décision : son seul but est de repérer les étiquettes douteuses.
- **26 désaccords confiants** (le CNN donne au moins 0,9 à une autre classe que l'étiquette), sur 16 plaques
  (`evaluer_cnn.py --desaccords`). Pré-tri à l'oeil :
  - 6 coïncidences de segmentation, aux étiquettes décalées ;
  - 10 erreurs du CNN : barres du cadre lues « 1 », morceaux de cadre lus comme des chiffres, chiffres
    abîmés par la binarisation ;
  - aucune nouvelle faute de saisie.

  L'étudiant a confirmé ce tri sur la planche. Les 6 plaques ont été exclues (train/184, 257, 284, 336, 367,
  661) et les 10 autres gardées, leur étiquette étant juste. Le jeu nettoyé compte 271 plaques et 143
  véhicules, soit 2 785 vignettes d'origine (1 846 chiffres et 939 « autre »).
- **Courbes d'apprentissage** : un plateau est atteint dès la 3e à la 5e époque, sans surapprentissage
  jusqu'à la 33e. Sur ce plateau plat, l'arrêt anticipé sur la perte de validation s'arrête au hasard
  (meilleures époques de 3 à 23 selon les plis).
- **Décision : 30 époques fixes, sans arrêt anticipé.** Le taux d'apprentissage suivait déjà un calendrier fixe
  (10⁻³ x 0,95^époque), qui avait remplacé la baisse sur plateau du plan. Plus rien ne dépend donc de la
  validation : elle est seulement mesurée. Les modèles des plis et le modèle final suivent exactement la même
  trajectoire.

### Choix de l'entrée : masque binaire ou vignette grise (largeur 16, validation croisée, hors pli)

| Entrée | Exactitude | Erreur moyenne par classe | Bleu foncé simulé | Bleu clair simulé |
|---|---|---|---|---|
| binaire | 98,24 % (49 erreurs) | 1,16 % | 1,54 % | 2,20 % |
| gris | **99,14 % (24 erreurs)** | **0,91 %** | **1,01 %** | **1,94 %** |

- Ces mesures portent sur les 2 785 vignettes d'origine, chacune jugée par le modèle du pli qui n'a pas vu son
  véhicule.
- Erreur moyenne par classe = 1 − moyenne des rappels. C'est le critère fixé d'avance : chaque classe compte
  autant, pour que « 1 » et « autre », nombreux, ne dominent pas.
- **Test de McNemar exact** (les deux modèles comparés vignette par vignette) : 34 vignettes sont fausses pour
  le binaire seul, 9 pour le gris seul, 15 pour les deux. Avec p = 0,00017, l'écart n'est pas dû au hasard.
- **L'erreur qui compte pour la plaque.** Le binaire lit 32 « autre » comme des chiffres, surtout des barres
  du cadre lues 1 ou 7 ; le gris n'en lit que 6. Ces erreurs ajoutent de faux chiffres à une lecture. Le
  binaire perd surtout sur la grande classe « autre » (rappel de 96,6 % contre 99,4 %), qui ne compte que pour
  1/11 dans le critère.

**Déviation par rapport à la règle fixée d'avance.**
- Règle : « l'entrée à l'erreur moyenne par classe la plus basse ; si l'écart est sous 0,5 point, le binaire,
  plus robuste au fond bleu ».
- Écart mesuré : 0,25 point. Prise à la lettre, la règle désigne donc le binaire.
- Hypothèse de la règle : le binaire serait plus robuste au fond bleu des voitures de location, absentes du jeu
  Kaggle.
- Contrôle qui l'a réfutée : celui prévu dans le plan pour la vérifier. Avec un fond recoloré en bleu par la
  fonction de la phase 3, le gris fait moins d'erreurs que le binaire sur les deux bleus (1,01 % contre 1,54 %
  et 1,94 % contre 2,20 %).
- Décision de l'étudiant : **le gris**, en départageant par la robustesse au fond bleu, comme la règle
  l'entendait. Il est aussi significativement meilleur et lit beaucoup moins de barres du cadre comme des
  chiffres.
- Cette décision a été prise sur le réglage seulement, à partir des prédictions hors pli. Le test n'a pas été
  touché.

### Choix de la largeur (entrée grise, validation croisée, hors pli)

| Largeur | Exactitude | Erreur moyenne par classe | Paramètres | Calcul par caractère | Entraînement d'un pli |
|---|---|---|---|---|---|
| **8** | 99,10 % (25 erreurs) | 0,88 % | 39 600 | 0,70 M MAC | ~52 s |
| 16 | 99,14 % (24 erreurs) | 0,91 % | 89 900 | 2,57 M MAC | ~150 s |
| 32 | 99,21 % (22 erreurs) | 0,58 % | 225 000 | 9,86 M MAC | ~6 min |

- Règle fixée d'avance : la plus petite largeur dont l'erreur reste à moins de 0,5 point de la meilleure.
  La largeur 8 en est à 0,30 point. **Largeur 8 retenue**, pour 14 fois moins de calcul que la 32.
- Incident : la tâche de fond de la largeur 32 a atteint sa durée maximale pendant le 5e pli. Les plis 0 à 3
  étaient déjà enregistrés. L'option `--reprendre` de `entrainer_cnn.py` les recharge et n'entraîne que les
  plis manquants. Chaque pli enregistre désormais aussi son historique d'apprentissage.

### Le CNN retenu (gris, largeur 8), hors pli

- Exactitude de 99,10 % (25 erreurs sur 2 785) ; moyenne des rappels par classe de 99,12 %.
- Les confusions attendues entre chiffres sont presque absentes, une seule de chaque sorte au plus
  (5 -> 9, 6 -> 1, 0 -> 8, 2 -> 1). La police des plaques est nette.
- Les erreurs opposent surtout un chiffre et « autre » :
  - des chiffres cassés ou collés au cadre, lus « autre » (1 -> autre : 4, 4 -> autre : 3, 8 -> autre : 3) ;
  - des morceaux de cadre lus comme des chiffres (autre -> 1 : 3, autre -> 4 : 2, autre -> 7 : 2).
- Figure : `confusion_gris_8.png` (non versionnée).

### Lecture de bout en bout, réglage hors pli (gris_8, 529 plaques, cadres YOLO)

Chaque plaque est lue par le modèle du pli qui n'a jamais vu son véhicule. Les 529 plaques ont été détectées.

| Option, au seuil retenu par la règle | Correctes | Rejetées | Substitutions | Plus courtes | Plus longues |
|---|---|---|---|---|---|
| A (chiffres probables), seuil 0,6 | 57,8 % | 20,6 % | 0,8 % | 20,8 % | 0,0 % |
| B (tous les candidats), seuil 0,85 | 57,7 % | 25,5 % | 0,8 % | 16,1 % | 0,0 % |

- **Seuil.** Règle : le plus petit seuil qui ramène les substitutions sous 1 % des plaques. Il vaut 0,6 pour A
  et 0,85 pour B.
- **Option.** Règle : B seulement s'il apporte au moins 0,5 point de lectures correctes. Ce n'est pas le cas
  (57,7 % contre 57,8 %), donc A est retenue. Au même seuil (0,85), B ferait toutefois mieux que A : 57,7 %
  de lectures correctes contre 55,0 %, et 16,1 % de plus courtes contre 17,2 %. Mais le seuil de B doit être
  plus haut pour tenir l'objectif de substitutions.
- **Les lectures « plus courtes » sont le premier défaut** : une plaque sur cinq. Beaucoup ont perdu presque
  tous leurs chiffres (« 1 TU 1 », « 8 TU 49 »), et le format officiel les accepte (série de 1 à 3 chiffres,
  numéro de 1 à 4).
- **Contrôle de vraisemblance des longueurs, mesuré.** Aucune des 547 plaques de réglage saisies n'a une
  série ou un numéro d'un seul chiffre. Les numéros allant de 1 à 9999, un numéro d'un chiffre n'a qu'environ
  0,1 % de chances.

| Règle de longueur (option A, seuil 0,6) | Correctes | Rejetées | Plus courtes |
|---|---|---|---|
| format officiel (série 1-3, numéro 1-4) | 57,8 % | 20,6 % | 20,8 % |
| série >= 2, numéro >= 2 | **57,8 %** | 31,9 % | **9,5 %** |
| série >= 2, numéro >= 3 | 57,5 % | 34,0 % | 7,8 % |
| série >= 2, numéro = 4 | 55,8 % | 39,7 % | 3,8 % |

  La règle « série >= 2, numéro >= 2 » divise par deux les lectures plus courtes sans perdre une seule lecture
  correcte. Les règles plus strictes perdent de vraies lectures (les numéros de 2 ou 3 chiffres existent).
  Parmi les plus courtes restantes, la plupart n'ont perdu qu'un chiffre, dans la série ou dans le numéro :
  leur format reste plausible. C'est le vote sur plusieurs images (phase 5) qui les traitera, et la barrière
  ne s'ouvre que sur une correspondance exacte avec la liste.
- Part des lectures acceptées qui sont justes (option A, seuil 0,6) : 72,8 % avec le format officiel, 84,9 %
  avec la règle « série >= 2, numéro >= 2 ».
- Décision de l'étudiant attendue à l'arrêt 2 (largeur, option et seuil, règle de longueur), avant le test.

### Décisions de l'arrêt 2 (prises sur le réglage hors pli ; le test n'a pas été touché)

- **Largeur 8**, par la règle.
- **Option A, seuil 0,6**, par les règles fixées d'avance.
- **Règle de vraisemblance ajoutée** à `systeme/decision.py` : série et numéro d'au moins 2 chiffres. Une
  lecture qui ne la respecte pas est rejetée avec le motif « improbable », distinct de « format » (hors format
  officiel). Mesures officielles, avec cette règle intégrée :

| Au seuil retenu | Correctes | Rejetées | Substitutions | Plus courtes | Motifs des rejets (plaques) |
|---|---|---|---|---|---|
| **A, seuil 0,6 (retenue)** | 57,8 % | 31,9 % | 0,8 % | 9,5 % | doute 10, format 21, improbable 62, aucun chiffre 76 |
| B, seuil 0,85 (pour mémoire) | 57,7 % | 33,8 % | 0,8 % | 7,8 % | doute 43, format 38, improbable 66, aucun chiffre 32 |

- **B est noté pour la phase 5.** Il donne autant de lectures correctes avec 1,7 point de lectures plus
  courtes en moins, contre 1,9 point de rejets en plus. On en rediscutera si le vote sur plusieurs images le
  justifie, car une lecture fausse pèse sur le vote alors qu'un rejet ne fait que passer son tour.
- **Note pour la phase 5 (étudiant)** : une lecture qui correspond exactement à une plaque de la liste des
  autorisés doit rester acceptée même avec un groupe d'un seul chiffre. Sinon, une voiture autorisée au numéro
  d'un seul chiffre serait refusée à chaque passage. C'est pour cela que `lire_plaque` conserve le texte
  assemblé (« lecture ») même quand il le rejette comme « improbable ».

### Modèle final, export et intégration (avant le test)

- **Modèle final** : vignette grise, largeur 8, 30 époques sur les 143 véhicules (vignettes d'origine et
  dégradées), avec le même calendrier que les plis : `sorties/runs/cnn/gris_8/final.keras`.
- **Export ONNX** (`exporter_cnn.py`) : opset 12, IR 7, en lot de 1 (1, 32, 32, 1) et en lot de 16
  (16, 32, 32, 1). Sur les 2 785 vignettes, l'écart maximal entre Keras et ONNX Runtime est de 1,4·10⁻⁶ et la
  classe prédite est identique dans 100 % des cas. Des copies sont dans `sorties/verification/jetson/` pour
  le test J.
- **Lot de 16 retenu.** L'option A lit 7 vignettes par plaque en médiane, 11 au maximum au réglage. Sur le CPU
  du PC (ONNX Runtime, processus propre), un appel en lot de 1 coûte 0,04 ms, soit 0,28 ms par plaque, contre
  0,12 ms pour l'unique appel en lot de 16. Le modèle du système est donc `modeles/lecteur.onnx` (lot de 16).
  Sur la Jetson, le coût fixe de chaque appel devrait encore plus favoriser le lot de 16 ; le test J le
  confirmera.
- **`systeme/decision.py`.** `lire_numero` (option A, seuil 0,6, règle de vraisemblance) est la seule fonction
  de lecture du système : la démo et l'évaluation du test l'appellent toutes deux.
- **`outils/demo.py`.**
  - Le numéro lu s'affiche sous le cadre : en vert s'il est accepté, en orange avec le motif s'il est rejeté.
  - Sous chaque vignette lue s'affichent la classe et la confiance, ainsi que la latence du CNN.
  - `--sans-lecture` présente la phase 3 seule.
  - Rendu vérifié sur deux plaques de réglage, lues justes avec une confiance d'au moins 0,999 ; la boucle
    webcam a été vérifiée sans affichage.
- **Mesure du jour.** Le détecteur prend 75 ms en médiane, contre 33 ms en phase 2, pendant que Windows Update
  travaille en fond, après deux heures d'entraînements à pleine charge. Le lecteur reste à 0,13 ms. Le
  détecteur est à remesurer quand la machine sera au repos.

### Évaluation finale sur le test (une seule fois)

La chaîne est celle du système, avec tous les choix figés : détecteur ONNX, traitement de la phase 3,
`lire_numero` (vignette grise, largeur 8, option A, seuil 0,6, règle de vraisemblance) et
`modeles/lecteur.onnx` (lot de 16).

| Plaques du test | Correctes | Rejetées | Substitutions | Plus courtes | Motifs des rejets (plaques) |
|---|---|---|---|---|---|
| toutes (142) | **43,0 %** (61) | 36,6 % (52) | 2,1 % (3) | 18,3 % (26) | doute 2, format 13, improbable 22, aucun chiffre 11, non détectée 4 |
| véhicules jamais vus (124) | **43,5 %** (54) | 33,9 % (42) | 2,4 % (3) | 20,2 % (25) | doute 2, format 11, improbable 17, aucun chiffre 10, non détectée 2 |
| images web <= 700 px (87) | 36,8 % | 40,2 % | 1,1 % | 21,8 % | |
| images > 700 px (55) | 52,7 % | 30,9 % | 3,6 % | 12,7 % | |

Aucune lecture plus longue n'a été acceptée.

- **Détection** : 138 plaques sur 142, et 122 sur 124 pour les véhicules jamais vus.
- **CNN seul**, sur les plaques du test strictement segmentées : 96,48 % d'exactitude (16 erreurs sur
  455 vignettes), contre 99,10 % hors pli au réglage.
- **Par rapport au réglage hors pli** : 43,0 % de lectures correctes au lieu de 57,8 %, 18,3 % de plus courtes
  au lieu de 9,5 %, 2,1 % de substitutions au lieu de 0,8 % (3 plaques). Les lectures acceptées sont justes à
  68 % (61 sur 90), contre 85 % au réglage.
- **L'écart a la même cause qu'en phase 3** : les images web, floues et compressées, qui forment 61 % du test.
  Elles donnent 36,8 % de lectures correctes, contre 52,7 % pour les images de plus de 700 px.
- **Les véhicules déjà vus ne faussent pas le résultat** : 43,5 % pour les 124 plaques de véhicules jamais vus,
  contre 43,0 % pour l'ensemble.
- **Les 29 lectures fausses acceptées** (`fausses_test.jpg`) :
  - 26 plus courtes. Le plus souvent, un seul chiffre est perdu, très souvent un « 1 » : trait fin, cassé ou
    soudé au cadre ou à une vis. Sur deux plaques (test/4, test/114), la série entière est perdue et le numéro
    est coupé en deux par un espace interne. Une plaque est coupée par le bord de l'image.
  - 3 substitutions : test/136 (police italique), test/81 (plaque floue et penchée), test/130 (dernier
    chiffre collé à une vis). Les deux premières tiennent autant de la segmentation que du CNN.
- **Enseignements** :
  1. Le CNN n'est pas le maillon faible (96,5 % au test). Les erreurs viennent surtout de la segmentation de
     la phase 3, qui perd des chiffres.
  2. En phase 5, la protection contre les lectures plus courtes vient du vote sur plusieurs images et de la
     correspondance exacte avec la liste des autorisés.
  3. Comme en phase 3, le premier levier reste la qualité d'image de la caméra du parking : netteté, prise de
     vue de face, peu de compression.
  4. Piste pour plus tard : les « 1 » perdus (trait fin, soudé au cadre ou à une vis) méritent un examen des
     filtres de la phase 3.

### Suite

- Vérification sur les photos de plaques de location, quand elles seront prêtes (vérification seulement).
- Test anticipé sur Jetson : passer aussi `lecteur_lot1.onnx` et `lecteur_lot16.onnx` dans trtexec.
- Démo en direct à essayer devant une vraie plaque ; remesurer le détecteur quand la machine sera au repos.
- Phase 5 (système complet), sur accord : sources vidéo, vote sur plusieurs images, liste des autorisés (avec
  les notes de la phase 4), journal des passages, métriques.

## 2026-10-01 — Itération ciblée sur la phase 3 : diagnostic des chiffres perdus

Contexte : lors d'un essai de l'étudiant sur une plaque affichée sur un écran de téléphone, trois échecs typiques
sont apparus : coins « bords » faux (plaque restée penchée), chiffres de la série soudés à un reste du liseré, et
redressement qui coupe la moitié droite. Règles de l'itération :
- tout se règle sur le réglage, hors pli (le CNN des plis gris_8 lit chaque plaque) ;
- le test Kaggle, déjà utilisé, n'est pas retouché ;
- le CNN n'est pas réentraîné ;
- les captures de l'essai servent à vérifier à l'oeil, jamais à régler ;
- la mesure finale se fera en phase 5, sur les photos et vidéos de l'étudiant.

### Dimensionnement (chaîne actuelle)

- **211 plaques de réglage sur 529 perdent des chiffres**, lectures rejetées comprises. 76 n'ont aucun chiffre
  lu, 10 ont un numéro vide.
- Pertes par méthode de redressement : liseré 32 % de ses 321 plaques, caractères 24 % (101), bords 75 % (52),
  zone noire 67 % (30), cadre YOLO seul 100 % (25).
- **Choix de la méthode** (oracle : au moins une méthode de la cascade donne la bonne lecture) :
  - lectures correctes : 57,8 % avec la cascade actuelle, 73,5 % au mieux ;
  - une règle naïve donne déjà 63,1 % : on garde la première méthode dont la rangée de chiffres est plausible
    (série de 2 à 3 chiffres, numéro de 2 à 4), sinon celle qui en donne le plus ;
  - cette règle essaie en moyenne 1,32 méthode par plaque.

### Diagnostic par cause (`entrainement/diagnostiquer_traitement.py`)

- **Seuils** : ce que les plaques bien lues ne dépassent presque jamais (1er ou 99e centile) : couverture du
  cadre YOLO < 0,83 ; débordement hors du cadre > 0,69 ; pente de la rangée > 0,059.
- **Attribution** dans l'ordre : coupée, coins faux, soudure, aucun chiffre, autre. Elle a été vérifiée à l'oeil
  sur les planches `diagnostic_1` à `diagnostic_5`.

| Cause | Plaques | dont lectures courtes acceptées | Récupérables par une autre méthode | Vérification à l'oeil |
|---|---|---|---|---|
| coupée | 8 | 2 | 6 | 7 vraies coupures (méthode « caractères » surtout), 1 plaque sur deux lignes |
| coins faux | 13 | 2 | 3 | environ 9 vrais coins faux, 3 soudures, 1 plaque sur deux lignes |
| soudure au liseré | 102 | 15 | 41 | environ 3 sur 4 sont de vraies soudures, le reste des binarisations chaotiques (flou, stries) |
| aucun chiffre | 27 | 0 | 4 | redressements ratés sur des vues très obliques (voir ci-dessous) |
| autre | 61 | 31 | 24 | mélange (voir ci-dessous) |

- **Soudures.** Elles touchent le plus souvent le premier chiffre de la série, souvent un « 1 », collé au trait
  vertical gauche du liseré. Viennent ensuite des chiffres collés aux traits du haut ou du bas, ou aux vis et
  aux points du cadre.
- **Aucun chiffre.** Repli « cadre » (aucun coin trouvé) ou quadrilatère « bords » penché :
  - sur 11 plaques, les chiffres inclinés sont soudés deux à deux et rejetés par le filtre largeur/hauteur ;
  - sur 8, aucun candidat n'a la taille d'un chiffre ;
  - sur 1, les chiffres sont décalés verticalement.

  L'hypothèse d'un quadrilatère trop serré (chiffres de plus de 90 % de la hauteur) est réfutée.
- **Autre.** On y trouve quatre mécanismes : un « 1 » de tête de série écarté (soudure légère au bord gauche),
  deux chiffres voisins soudés que la scission ne sépare pas, du flou ou un faible contraste, et des polices
  italiques. Cette catégorie regroupe la plupart des lectures courtes acceptées (31 sur 50).
- **Le « 1 » n'est pas surreprésenté** : 17,1 % des chiffres perdus, pour 19,0 % de tous les chiffres.
  L'hypothèse d'un « 1 » très haut effacé avec le liseré est réfutée : sans l'effacement des longues lignes,
  une seule des 122 plaques qui perdent un « 1 » devient juste. Les « 1 » perdus relèvent des soudures au bord
  gauche, pas d'un mécanisme propre.
- **Plaques sur deux lignes.** train/165 et train/538 ont été saisies avec un numéro, alors que la convention de
  saisie les classe en « autre format ». La chaîne, qui suppose une seule rangée, ne peut pas les lire.
- **Captures de l'essai sur écran** (vérification à l'oeil) :
  - 175008 (zone noire) et 175018 (bords) : coins faux, avec un quadrilatère qui déborde sur le bord de
    l'écran ;
  - 175030 (liseré) : coins faux et série soudée au trait du haut ;
  - 175046 (liseré) : plaque coupée, le numéro est hors du quadrilatère.

Décisions de l'arrêt : l'étudiant valide l'attribution. train/165 et train/538 (plaques sur deux lignes) passent
en « autre format », avec les copies datées. Elles n'avaient fourni aucune vignette au jeu de caractères : leur
segmentation n'était pas stricte, donc rien n'est à reprendre pour le CNN. Le jeu de caractères et les plis ne
sont pas régénérés, pour garder la correspondance avec les modèles hors pli gris_8.

### Étape 2 : contrôle du redressement par la rangée de caractères (`regler_redressement.py`)

`systeme/traitement.py` essaie les propositions de coins de la cascade dans l'ordre, et garde la première dont la
rangée de chiffres est bonne. Si aucune ne l'est, il garde celle qui donne le plus de chiffres probables.
Variantes du contrôle :
- groupes : série de 2 à 3 chiffres probables, numéro de 2 à 4 ;
- pente : groupes, et rangée horizontale, avec une pente inférieure à 0,059 ;
- geometrie : pente, et quadrilatère qui couvre au moins 83 % du cadre YOLO sans en déborder de plus de 69 % ;
- complete : geometrie, et au moins 6 chiffres probables (98 % des plaques de réglage en ont 6 ou 7).

Mesures sur 527 plaques de réglage, hors pli (option A, seuil 0,6) :

| Variante | Correctes | Rejetées | Substitutions | Plus courtes | Gains / pertes | Acceptées justes | Méthodes essayées | Traitement (médiane / p95) |
|---|---|---|---|---|---|---|---|---|
| aucun (cascade d'origine) | 58,1 % | 31,9 % | 0,8 % | 9,3 % | — | 85,2 % | 1,00 | 8,0 / 12,5 ms |
| groupes | 63,4 % | 22,8 % | 0,9 % | 12,9 % | +45 / −17 | 82,1 % | 1,52 | 9,0 / 26,3 ms |
| pente | 63,6 % | 22,6 % | 0,9 % | 12,9 % | +46 / −17 | 82,2 % | 1,54 | 9,4 / 27,2 ms |
| geometrie | 64,1 % | 23,0 % | 0,9 % | 12,0 % | +48 / −16 | 83,2 % | 1,58 | 9,5 / 27,4 ms |
| complete | **65,5 %** | 22,8 % | 0,9 % | 10,8 % | **+53 / −14** | 84,8 % | 1,65 | 9,8 / 30,2 ms |

- **Le gain est réel.** Il apparaît dans chacun des 5 plis : avec « complete », +6,4, +4,6, +9,0, +7,1 et
  +9,7 points.
- **La règle fixée d'avance n'est pas respectée.** Elle demandait que les lectures plus courtes n'augmentent pas,
  en part de toutes les plaques. Or toutes les variantes les augmentent, de 1,5 à 3,6 points : le contrôle
  transforme des rejets en lectures acceptées, dont certaines plausibles mais incomplètes.
- **La variante « complete »** a été ajoutée après les quatre premières, pour viser cette cause. Elle limite la
  hausse à 1,5 point, et la part des lectures acceptées qui sont justes reste presque la même (84,8 % contre
  85,2 %).
- **Le système garde la cascade d'origine (« aucun ») jusqu'à la décision de l'étudiant.**
- **Vérification à l'oeil sur les 6 captures de l'essai sur écran**, sans aucun réglage dessus :
  - sans contrôle : 0 lecture juste, 2 fausses, 4 rejets ;
  - avec « complete » : 3 justes, 3 fausses (plus courtes), aucun rejet ;
  - la capture 175018 perd le « 1 » de tête, soudé au bord gauche, dans les deux cas : c'est le travail de
    l'étape 3.

**Décision de l'étudiant : « complete »** (`CONTROLE = "complete"` dans `systeme/traitement.py`). Déviation par
rapport à la règle fixée d'avance, notée avec ses trois raisons :
1. La règle mesurait les lectures plus courtes en part de toutes les plaques. Cette part augmente mécaniquement
   dès qu'on accepte plus de plaques. La mesure pertinente est la fiabilité d'une lecture acceptée, qui ne bouge
   pas : 84,8 % contre 85,2 %.
2. « complete » a été conçue après avoir vu les résultats. Son gain devra être confirmé en phase 5, sur les
   photos et vidéos de l'étudiant, seules données encore vierges.
3. Pour les prochaines règles (étape 3, phase 5), le garde-fou est directement la fiabilité des lectures
   acceptées.

### Étape 3 : séparer les chiffres soudés aux restes du liseré (règle fixée avant la mesure)

Deux options, mesurées par-dessus le contrôle « complete » :
- **bords** (`binariser(lisere_bords=True)`). Une seconde passe d'effacement des lignes, avec des lignes plus
  courtes que la première (15 % de la largeur, 50 % de la hauteur), cherchées seulement dans les bandes du
  bord : 15 % en haut et en bas, 5 % à gauche et à droite.
- **rangee** (`segmenter(detacher=True)`). Pour chaque grosse composante qui n'est pas un candidat et traverse
  la rangée des chiffres, on efface ce qui dépasse la rangée en haut ou en bas : un trait horizontal du liseré.
  Dans les bandes de côté, on efface aussi les colonnes blanches de part en part qui débordent la rangée en haut
  ET en bas : un trait vertical du liseré. Un chiffre ne dépasse jamais la rangée. La seconde segmentation n'est
  gardée que si elle donne plus de chiffres probables que la première.

Règle :
- On retient l'option qui donne le plus de lectures correctes.
- Deux conditions : les substitutions restent sous 1 % des plaques, et la fiabilité des lectures acceptées ne
  baisse pas de plus de 1 point par rapport à la référence (« complete » sans séparation). Ce point
  correspond à peu près à l'incertitude de mesure : environ ± 1,8 point sur 400 lectures acceptées.
- À 0,5 point près, on retient la plus simple. Dans l'ordre : bords, puis rangee.

### Étape 3 : mesures (`regler_soudures.py`, 527 plaques, par-dessus « complete »)

| Option | Correctes | Substitutions | Plus courtes | Fiabilité des acceptées | Gains / pertes |
|---|---|---|---|---|---|
| référence (« complete » seule) | 65,5 % | 0,9 % | 10,8 % | 84,8 % | — |
| bords | 65,5 % | **1,1 %** | 10,8 % | 84,6 % | +3 / −3 |
| rangee | 65,3 % | 0,9 % | 11,0 % | 84,5 % | +3 / −4 |

- **La règle garde la référence** : aucune option n'apporte de lectures correctes en plus, et « bords » dépasse
  1 % de substitutions. Les deux options restent dans le code, désactivées (`LISERE_BORDS = False`,
  `DETACHER = False`), pour pouvoir les retester en phase 5 sur des données vierges.
- **Pourquoi « rangee » échoue.** Le mécanisme a été examiné sur les soudures typiques du diagnostic et sur la
  capture 175018. Le détachement fonctionne : le « 1 » de tête est séparé du trait gauche et devient un candidat.
  Mais le filtre de régularité le rejette ensuite :
  - un reste du liseré, resté collé dans la marge de la rangée, le rend trop haut (train/515 : 71 px pour
    environ 62) ;
  - ou bien il est coupé en deux (train/279 : 37 px).

  Le nombre de chiffres probables ne monte donc pas, et la seconde segmentation est écartée. Piste pour plus
  tard : recouper le chiffre détaché aux limites exactes de la rangée. Elle n'a pas été essayée ici : ce serait
  encore régler sur les données déjà regardées.

### Bilan de l'itération (réglage, hors pli ; le test n'a pas été touché)

- **Retenu : contrôle du redressement « complete ».** Pas de séparation des soudures.
- **Lecture de bout en bout** (527 plaques, option A, seuil 0,6) :
  - lectures correctes : 65,5 % contre 58,1 % (+53 / −14 plaques, gain dans les 5 plis) ;
  - rejets : 22,8 % contre 31,9 % ;
  - substitutions : 0,9 % ;
  - lectures plus courtes : 10,8 % contre 9,3 % ;
  - fiabilité des lectures acceptées : 84,8 % contre 85,2 %.
- **Segmentation de la phase 3** (545 plaques) :
  - cadres vrais : 67,3 % en mesure stricte et 75,2 % sans chiffre perdu, contre 52,0 % et 64,1 % ;
  - cadres YOLO : 65,5 % et 72,7 %, contre 51,5 % et 63,0 %.

  Le contrôle choisit moins souvent le liseré (45 % des plaques) et plus souvent « caractères » (25 %).
- **Plaques qui perdent des chiffres** (diagnostic relancé) : 165 au lieu de 211.

| Cause | Avant | Après |
|---|---|---|
| coupée | 8 | 2 |
| coins faux | 13 | 16 |
| soudure au liseré | 102 | 79 |
| aucun chiffre | 27 | 22 |
| autre | 61 | 46 |

  Les seuils du diagnostic sont recalculés sur les nouvelles plaques bien lues ; ils sont un peu plus stricts,
  d'où la légère hausse des coins faux. **Les soudures au liseré restent la première cause.**
- **Temps du traitement sur PC** : 9,9 ms en médiane (8,0 avant) et 30 ms au 95e centile (12,5 avant), quand la
  cascade va jusqu'au bout. À mesurer sur la Jetson en phase 6.
- **Captures de l'essai sur écran** (vérification à l'oeil) : 3 lectures justes sur 6 au lieu de 0. La capture
  175018 perd toujours le « 1 » de tête soudé au bord gauche.
- **À confirmer en phase 5** sur les photos et vidéos de l'étudiant : « complete » a été conçue après avoir vu
  les premiers résultats.
- **Non retouché** : le CNN (`lecteur.onnx`, modèles gris_8), le jeu de caractères et ses plis. Si le jeu de
  caractères est régénéré en phase 5, il reflétera le nouveau traitement.

## 2026-10-02 — Phase 5 : système complet et évaluation

### Décisions de départ (plan validé)

- **Données vierges** : photos et vidéos de l'étudiant, découpées **par véhicule** en deux parties.
  « Développement » sert à toutes les décisions de la phase ; « test final » est évalué une seule fois.
  - Objectif : 60 véhicules filmés (30 + 30), chacun avec une vidéo et 3 photos, dont au moins 6 plaques bleues.
  - Les 4 véhicules photographiés avant le protocole (10 photos, aucune vidéo) vont d'office en développement.
    Certains ont été vus pendant l'essai sur écran. Mon inventaire du plan n'en comptait que 3 ; la décision de
    l'étudiant s'applique aussi au quatrième.
- **Attente maximale de 3 s**, au-delà « non lu ».
- **Fermeture** : 3 s après que la plaque qui a ouvert a quitté l'image, avec une ouverture d'au moins 5 s.
  Pour le rapport : une vraie installation exige un capteur de présence (boucle au sol) pour la sécurité de la
  fermeture. La caméra perd la plaque avant dès que la voiture s'engage sous la barrière.
- **Liste simulée** :
  - la moitié des véhicules est autorisée ;
  - deux listes croisées : chaque véhicule est évalué une fois autorisé, une fois non ;
  - une liste adverse, décrite dans le plan.
- **Flask 2.0.3 et sa famille**, épinglés à l'identique sur le PC et la Jetson. L'interface web vient en dernier.
- **Contrat d'API** : celui du plan, avec trois ajouts demandés par l'étudiant, qui met la maquette à jour.
  - `materiel` (« PC » ou « Jetson ») dans /etat ;
  - les latences par étape : detection, redressement, binarisation, segmentation, lecture ;
  - la `confiance` de la lecture en tête dans chaque piste.

```
GET /video          flux MJPEG (multipart/x-mixed-replace), image annotée
GET /etat           {"heure", "materiel": "PC"|"Jetson", "images_par_seconde",
                     "latences_ms": {"detection", "redressement", "binarisation", "segmentation", "lecture"},
                     "barriere": {"etat": "ouverte"|"fermee", "depuis_s"},
                     "pistes": [{"id", "meneur": "215 TU 4567", "confiance", "voix", "voix_requises", "statut",
                                 "age_s"}],
                     "derniere_decision": {"heure", "numero", "decision": "ouverture"|"refus"|"non lu",
                                           "autorise", "delai_s"}}
GET /journal?n=50   [{"heure", "numero", "decision", "autorise", "voix", "delai_s", "capture": "/capture/..."}]
GET /etape/<nom>    JPEG ; nom = decoupe | coins | redressee | binaire | segmentation | vignettes
```

- **Garde-fou de toutes les règles** : la fiabilité des lectures acceptées.

### Protocole de prise de vue (fixé avant de filmer)

L'étudiant simule l'approche en marchant vers des voitures garées, dans le parking de l'école, avec l'accord de
l'administration. Adaptations :
- **Tenue du téléphone** :
  - en paysage (16:9, comme la webcam), à deux mains, coudes contre le corps, à environ 1 m de haut ;
  - légèrement incliné vers le bas (environ 10°), pour que la plaque reste au centre de l'image jusqu'à la fin ;
  - sans zoom ;
  - stabilisation du téléphone activée : elle corrige la marche, un défaut propre au protocole qu'une caméra
    fixe n'aurait pas.
- **Trajet** :
  - départ à 8 à 10 m, dans l'axe du véhicule ;
  - marche lente et régulière (environ un pas par seconde) jusqu'à environ 2 m ;
  - puis 4 s immobile, comme une voiture arrêtée devant la barrière. 4 s plutôt que 2 (modifié avant le
    tournage) : une plaque lue pour la première fois au point d'arrêt doit pouvoir atteindre l'attente maximale
    de 3 s avant la fin de la vidéo ;
  - une vidéo de 10 à 15 s par véhicule, un seul véhicule visé.
- **Vidéo** : 1080p ou 720p à 30 images/s, en H.264 (.mp4) si le téléphone le propose.
- **Photos** : 3 par véhicule, en paysage et depuis 1 m de haut :
  - de face à environ 2 m ;
  - de face à environ 4 m ;
  - en biais (20 à 30°) à environ 3 m.
- **Plaque avant de préférence** : c'est elle que verra la caméra de la barrière. Si seule l'arrière est
  accessible, on la filme quand même, avec l'étiquette `arriere` dans le nom.
- **Conditions** : surtout de jour, avec de l'ombre, du contre-jour et quelques prises en fin de journée.
- **Plaques voisines** : celles des voitures garées à côté apparaîtront souvent dans l'image. L'évaluation ne
  retiendra que les pistes du véhicule visé ; la règle sera fixée avant la mesure et vérifiée à l'oeil.
- **Noms** : `215TU4567.mp4` (ou `.MOV`, format de l'iPhone de l'étudiant) dans `validation/videos/` ;
  `215TU4567_1.jpg`, `_2`, `_3` dans `validation/photos/`.
  - Étiquettes : `bleu` pour une plaque de location (`215TU4567_bleu.mp4`, `215TU4567_bleu_1.jpg`) et `arriere`.
  - La forme « (2) » ajoutée par Windows est acceptée.
- **Confidentialité** : tout reste en local (`validation/` est hors de git) ; éviter de filmer les personnes.
- **Images réduites à 1280 px de large** avant le système, la résolution de la webcam : l'évaluation mesure ce
  que verra le système, pas la pleine résolution du téléphone.
- **Limites à noter dans le rapport** :
  - la caméra bouge au lieu de la voiture ;
  - un piéton approche plus lentement qu'une voiture ;
  - l'optique du téléphone est meilleure que celle d'une webcam USB.

  Les deux derniers points rendent les mesures un peu optimistes ; l'arrêt de 2 s en fin de vidéo, lui,
  correspond bien à la voiture qui attend devant la barrière.

### Définitions du système (fixées avant toute mesure)

- **Sources** :
  - le temps d'une vidéo est le sien (numéro de l'image / images par seconde) : les mesures sont reproductibles
    et ne dépendent pas de la vitesse du PC ;
  - en évaluation, la cadence simulée est de 10 images traitées par seconde, et de 5 pour la sensibilité.
- **Suivi** :
  - une détection prolonge la piste dont le dernier cadre la recouvre le plus (IoU ≥ 0,3, appariement glouton) ;
  - une piste se ferme après 1 s sans détection.
- **Voix** : une lecture acceptée (« lue »), ou une lecture « improbable » égale **exactement** à une plaque
  autorisée. Les rejets « doute », « format » et « aucun chiffre » ne votent jamais.
- **Fenêtre** : les W dernières voix de la piste.
- **Deux variantes de décision** :
  - **(a) majorité** : la lecture en tête décide dès qu'elle a au moins K voix **et** plus de la moitié des voix
    de la fenêtre. La décision est une ouverture si cette lecture est dans la liste, un refus sinon ;
  - **(b) liste d'abord** : ouverture dès qu'une plaque autorisée a K voix dans la fenêtre, même sans majorité.
    Le refus attend l'échéance : il faut alors une lecture en tête avec K voix et la majorité, sinon c'est
    « non lu ».
- **Chronomètre** : il part à la **première voix** de la piste, pas à sa première détection.
  - Pourquoi : la plaque est détectée de loin, bien avant d'être lisible. Compter l'approche comme de l'attente
    ferait déclarer « non lue » une voiture qui approche encore.
  - Limite : une voiture arrêtée dont la plaque ne donne jamais de lecture acceptée attend sans décision. Elle est
    notée « non lue » quand elle repart. Avec une boucle au sol, le chronomètre partirait à son arrivée.
- **Échéance** : D secondes après la première voix, avec D ≤ 3 s (attente maximale). Sans décision à ce moment,
  la piste est « non lue » (variante a) ; pour la variante b, voir ci-dessus.
- **Une décision par piste.** Une piste qui disparaît sans décision est « non lue » si elle a eu au moins une
  lecture en forme de plaque (« lue », « doute » ou « improbable »). Sinon, rien n'est noté : fausse détection
  probable, ou plaque restée trop loin.
- **Délai** : de la première voix à la décision. Le délai depuis la première détection est rapporté à titre
  d'information.
- **Barrière** : elle s'ouvre sur une décision d'ouverture. Elle se ferme quand l'ouverture dure depuis au moins
  5 s **et** qu'aucune plaque qui l'a ouverte n'a été détectée depuis 3 s.
- **Paramètres mesurés sur la partie développement (arrêt B)** :
  - seuil du détecteur ;
  - variante de vote ;
  - K de 1 à 5, W parmi 5, 10 et 20, D parmi 1, 2 et 3 s.

  Valeurs provisoires d'ici là : majorité, K = 3, W = 10, D = 3 s, seuil 0,5. La règle de choix est celle du
  plan : zéro ouverture à tort avec la liste adverse, puis une fiabilité des décisions au moins égale à celle
  des lectures acceptées image par image, puis le plus d'ouvertures correctes, et enfin, à 2 points près, le
  délai médian des ouvertures correctes le plus court.

### Logique du système (écrite avant les données, vérifiée sur des scénarios synthétiques)

- **Modules** :
  - `systeme/sources.py`, `suivi.py`, `vote.py`, `action.py` et `chaine.py` ;
  - `outils/validation.py` (inventaire, découpage) et `outils/verifier_logique.py` ;
  - `config/autorises.exemple.csv` (fictif).
- **Temps d'une vidéo** : l'horodatage de chaque image dans le fichier, car certains téléphones filment à cadence
  variable en basse lumière. À défaut, on avance d'une image à la cadence nominale.
- **Vérifications** (`python -m outils.verifier_logique`) : 35 scénarios construits à la main, tous conformes
  aux définitions ci-dessus.
  - suivi : approche, trou toléré, piste fermée ;
  - vote : les deux variantes, chronomètre, échéance, voix « improbable », fenêtre ;
  - barrière : ouverture minimale, départ de la plaque, seconde ouverture ;
  - passage complet ;
  - liste des autorisés ;
  - sources : cadence simulée, réduction à 1280 px, photos.
- **Essais de fonctionnement**, sans aucun réglage : la chaîne complète tourne sur une approche synthétique
  (zoom progressif sur une photo de développement) et sur le dossier de photos. Le journal des passages et les
  captures sont écrits, et l'état est conforme au contrat de /etat.
- **Taille des plaques que le détecteur voit bien** :
  - Dans le jeu Kaggle d'entraînement, la plaque occupe une part médiane de 0,19 de la largeur de l'image
    (0,35 au 95e centile).
  - Zoom synthétique sur les 10 photos de développement, la plaque occupant une part donnée de la largeur d'une
    image 1280 x 720 :
    - entre 8 % et 20 % : score d'au moins 0,79 pour les 10 photos ;
    - à 25 % et à 30 % : une photo sur 10 passe sous 0,5 ;
    - à 45 % : quatre photos sur 10 ;
    - à 6 % : quatre photos sur 10 aussi. De plus, une vraie caméra à cette distance donne une image moins
      nette que ce zoom.
  - Conséquence pour le protocole : s'arrêter vers 2 m, pas plus près. Avec l'objectif principal d'un
    téléphone en 16:9, la plaque occupe alors environ 17 % de la largeur.
  - Pour la phase 6 : placer la caméra pour que la plaque, au point d'arrêt, occupe 10 à 20 % de la largeur
    de l'image.
- **Latence du détecteur, remesurée** (PC sur batterie, mode « Performances élevées ») :
  - 39,9 ms en médiane dans un processus propre, dans la fourchette de la phase 2. Les 75 ms du 2026-10-01
    venaient donc de la charge de la machine.
  - Dans la chaîne complète : 45 à 55 ms. Les appels du CNN ajoutent environ 5 ms ; le décodage de la vidéo dans
    le même processus n'a pas d'effet mesurable.
  - Sans effet sur l'évaluation, qui suit le temps de la vidéo.
- **Données de développement** :
  - Le quatrième véhicule photographié avant le protocole est mis en développement avec les trois autres
    (confirmé par l'étudiant).
  - Sur les 4 photos du véhicule qui en a 4, un chiffre du numéro était faux dans le nom des fichiers : voir
    « Correction de la vérité terrain » plus bas.

### Évaluation : définitions et règle de choix (réponses de l'étudiant, fixées avant toute mesure)

- **Deux temps** (`outils/evaluer.py`, `outils/mesures.py`) :
  - **Perception**, lente et mise en cache. Chaque vidéo est lue à 10 images/s, au temps de la vidéo, et
    réduite à 1280 px de large. Sur chaque image, le détecteur tourne au seuil 0,25 ; un seuil plus haut se
    rejoue en écartant les détections plus faibles, ce qui revient au même, car la NMS ne fait disparaître une
    boîte qu'au profit d'une boîte plus sûre. Puis le traitement et le CNN lisent **chaque** plaque détectée.
  - **Décision**, rapide. Le suivi et le vote du système (`ControleAcces`, le même code qu'en service) sont
    rejoués sur ces lectures, pour chaque réglage et chaque liste.
- **Pistes visées** (plaques voisines) : règle géométrique, indépendante des lectures.
  - La plus grande détection des 2 dernières secondes donne la position horizontale de la plaque visée (à
    défaut, le centre de l'image).
  - Une piste est « visée » si sa position horizontale médiane en est à moins de 15 % de la largeur de l'image ;
    sinon, c'est une « autre plaque ».
  - Une planche de contrôle par vidéo permet de vérifier ce classement à l'oeil. Toute ouverture sur une autre
    plaque est vérifiée sur sa capture.
- **Listes croisées** :
  - L1 est une moitié des véhicules de la partie, tirée au sort avec une graine fixe ; L2 est l'autre moitié.
  - Liste k = autorisés de Lk + variantes adverses des non autorisés, moins les vrais numéros des non autorisés.
    Sinon, une lecture juste d'un non autorisé compterait comme une ouverture à tort.
  - Chaque véhicule filmé passe donc une fois autorisé et une fois non autorisé.
- **Variantes adverses** (décision de l'étudiant, avant toute mesure) : toutes les variantes **à une modification
  près**, c'est-à-dire un chiffre supprimé, remplacé par un autre ou ajouté, dans la série ou dans le numéro.
  - Seules les variantes au format officiel sont gardées. Pour un numéro de 3 + 4 chiffres, on en compte 68 (aucun
    ajout n'y est possible) ; pour un numéro de 2 + 3 chiffres, 111.
  - Le test ne dépend plus d'aucune hypothèse sur les confusions du CNN : c'est le pire cas pour une erreur d'un
    seul chiffre.
  - Le plan prévoyait un chiffre en moins et des confusions proches (1<->7, 0<->8, 3<->8, 5<->6).
- **Issue d'un passage** (une vidéo, une liste ; décisions des pistes visées) :
  - « ouverture à tort » : une ouverture sur un autre numéro que le vrai, quelle que soit la liste ;
  - sinon « ouverture » : une ouverture sur le vrai numéro ;
  - sinon « refus » : lecture juste, ou fausse ;
  - sinon « non lu ».
- **Indicateurs** :
  - taux d'ouverture correcte = passages autorisés ouverts / passages autorisés ;
  - ouvertures à tort = nombre de passages, toutes listes confondues : il doit rester nul ;
  - fiabilité des décisions = ouvertures et refus au vrai numéro / ouvertures et refus, sur les pistes visées
    et les deux listes ;
  - fiabilité des lectures image par image = lectures acceptées justes / lectures acceptées, sur les images des
    pistes visées. Ce sont les mêmes données ;
  - délai d'une décision = de la première voix de la piste à la décision. Il ne dépasse jamais l'attente maximale
    D, et un « non lu » à l'échéance compte pour D. Les décisions sans voix n'ont pas de délai.
- **Balayage** : variante × seuil du détecteur (0,25 ; 0,35 ; 0,5 ; 0,65) × K (1 à 5) × W (5, 10, 20) × D (1, 2,
  3 s), soit 360 réglages, à 10 images/s.
- **Règle de choix** (précisions de l'étudiant intégrées) :
  1. **Admissibles** : zéro ouverture à tort ; fiabilité des décisions au moins égale à celle des lectures image
     par image.
  2. **Bons** :
     - admissibles ;
     - nombre de véhicules ouverts correctement à **un véhicule près** du niveau considéré, en partant du meilleur
       admissible. C'était l'intention de l'étudiant derrière « à 2 points près » : avec environ 30 vidéos, un
       véhicule pèse 3,3 points ;
     - au moins 90 % des décisions des pistes visées prises en moins de 3 s.
  3. **Pas de maximum isolé** : un bon réglage n'est gardé que si au moins la moitié de ses voisins de grille
     sont bons. Un voisin est un cran de plus ou de moins sur un seul paramètre (seuil, K, W ou D), dans la même
     variante. Si aucun bon réglage ne passe cette étape, on descend au taux d'ouverture suivant et on reprend à
     l'étape 2. Sans cette descente, un pic isolé tout en haut empêcherait tout choix : les réglages du plateau
     seraient à plus de 2 points de lui (vu sur un test construit à la main).
  4. **Le plus sûr** : le K le plus grand.
  5. **Le centre du plateau** : le plus de bons voisins.
  6. **Égalités restantes** : le plus d'ouvertures correctes, puis la variante majorité, puis l'ordre de la grille.

  Rapport : le nombre de réglages à chaque étape, le voisinage du réglage retenu, la sensibilité à 5 images/s.
- **Autres décisions de l'arrêt B** (acceptées par l'étudiant) :
  - **« complete »** est confirmée si, sur les photos de développement, elle donne au moins autant de lectures
    correctes que « aucun », sans baisser la fiabilité des lectures acceptées de plus de 1 point ; sinon,
    discussion ;
  - **soudures** : règle de l'itération ciblée, sur les photos de développement. On retient l'option qui donne
    le plus de lectures correctes, avec moins de 1 % de substitutions et une fiabilité qui ne baisse pas de plus
    de 1 point ; à 0,5 point près, la plus simple ;
  - **CNN** : jamais entraîné sur les données de validation. Si ses erreurs viennent de vignettes absentes du jeu
    Kaggle, on discutera d'une réextraction du jeu Kaggle avec le traitement actuel.
- **Test final** : une seule fois, avec le réglage retenu. « complete » y est comparée à « aucun » pour le
  rapport seulement.

### Outil d'évaluation (écrit avant les données, vérifié sur des cas construits à la main)

- `outils/evaluer.py` (perception, bilans, planches, balayage, comparaisons, test final) et `outils/mesures.py`
  (listes, pistes visées, rejeu, indicateurs, règle ; sans image).
- **Vérifications** : `python -m outils.verifier_logique` passe 48 contrôles, dont 13 nouveaux sur les mesures :
  - variantes adverses et listes croisées ;
  - classement des pistes visées ;
  - issues des passages ;
  - règle de choix : plateau, pic isolé, contrainte des 3 s, ouverture à tort ;
  - McNemar.
- **Correction de la règle avant toute mesure** : avec la première rédaction, un pic isolé tout en haut
  empêchait tout choix, comme l'a montré un cas construit à la main. D'où la descente au taux d'ouverture
  suivant (étape 3 de la règle).
- **Banc d'essai** (fichiers dans un dossier temporaire, rien dans `validation/`) : deux approches simulées par
  un zoom sur des photos de développement, dont une avec une plaque voisine décalée sur le côté.
  - Tous les modes tournent.
  - La planche classe bien la voisine en « autre plaque ».
  - Ces vidéos sont trop nettes pour mesurer quoi que ce soit : seul le fonctionnement est vérifié.
- **Durées sur le PC** :
  - perception : environ 14 s pour 12 s de vidéo, mise en cache ;
  - balayage des 360 réglages : environ 6 s par vidéo, soit 3 à 4 min pour 30 vidéos.
- **Questions tranchées par l'étudiant, toujours avant toute mesure** :
  - variantes adverses à une modification près ;
  - « à un véhicule près » au lieu de « à 2 points près ».

  Ses deux précisions sur le balayage sont confirmées telles qu'écrites : plateau, et K le plus grand tant que
  90 % des décisions restent sous 3 s.

### Démo OpenCV du système complet (`outils/demo.py`, avant les données)

- **Mode par défaut : le système complet** (`systeme/chaine.py`). Les modes des phases 2 à 4 restent disponibles
  pour la soutenance : `--detection-seule`, `--sans-lecture`, et `--sans-vote`, la lecture image par image de
  la phase 4.
- **Source** (`--source`) :
  - la webcam, par défaut ;
  - une vidéo, lue au temps de la vidéo, à 10 images traitées par seconde comme l'évaluation ;
  - des photos, chacune répétée 1 s, comme une voiture arrêtée.
- **Fenêtre** :
  - chaque plaque suivie a un cadre et son numéro de piste ;
  - en haut à gauche, l'état de chaque piste : vote en cours (lecture en tête, voix sur K, chronomètre depuis la
    première voix), puis la décision ;
  - à droite, la plus grande plaque visible, détaillée avec sa dernière lecture ;
  - en bas, la barrière (simulée), le réglage du vote et les 4 dernières décisions.

  Touches : q, espace (pause) et s (capture).
- **Vérifié** : sur les deux vidéos du banc d'essai, la démo sans affichage prend **exactement les mêmes
  décisions** que le rejeu de l'évaluation (pistes, numéros, voix). Ce que montre la démo sur une vidéo est donc
  ce que mesure l'évaluation.
- **Vitesse sur le PC, sans affichage** :
  - 18 à 20 images/s en mode complet, car une piste décidée n'est plus lue ;
  - 15 images/s en lecture image par image.
- **`Chaine`** garde la dernière lecture de chaque piste active (capture du journal, affichage) : la plaque
  détaillée est la plus grande plaque visible, même déjà décidée. Les essais ont montré qu'avec « la dernière
  plaque lue », le panneau pouvait détailler une plaque voisine qui n'était plus à l'image.

### Correction de la vérité terrain (avant toute mesure)

- **Véhicule aux 4 photos** (photographié avant le protocole, en développement) : un chiffre du numéro était
  faux dans le nom des 4 fichiers. Le système lisait juste.
  - Une première vérification sur la voiture avait conclu que le nom était juste.
  - L'image de contrôle a tranché : le « 4 » de la série à côté du chiffre contesté, découpés dans les pixels
    de la photo d'origine (pleine résolution, sans redressement ni binarisation), agrandis sans lissage.
    L'image est dans `sorties/verification/`, non versionnée.
  - Corrections : les 4 fichiers renommés, `validation/decoupage.csv` (même partie, développement) et
    `config/autorises.csv`.
- Aucune mesure n'avait encore été faite sur ces données : la correction ne change ni les règles ni les
  résultats.
- **Leçon pour le tournage** : le nom de chaque fichier est la vérité terrain, et il est saisi à la main. Le
  relire sur la photo de face à 2 m au moment de nommer. En phase 4, la vérité terrain du jeu Kaggle contenait
  elle aussi environ 3 % de fautes de saisie.

## 2026-10-03 — Phase 5 : interface web locale (maquette de l'outil de design branchée)

- **Page servie** : `interface/maquette/Supervision parking.dc.html`, exportée de l'outil de design. L'export
  d'origine est committé tel quel (351e700), pour que l'écart reste visible dans git.
  - La page interroge elle-même `/etat` (toutes les 300 ms), `/journal?n=50` (toutes les 2 s), `/etape/<nom>`
    (chaque seconde, vue technique) et affiche `/video`.
  - Sans réponse de `/etat`, elle passe en « mode démonstration », avec des données simulées.
- **Seules modifications de la page**, sans rien changer à l'apparence :
  1. **React 18.3.1 servi en local** (`interface/maquette/vendor/`). Ce sont les fichiers d'unpkg, dont les
     empreintes SRI inscrites dans `support.js` ont été vérifiées. La page fonctionne sans internet, sur le PC
     comme sur la Jetson.
  2. **La décision affichée pendant 6 s dépend de l'état réel de la barrière**, demande de l'étudiant.
     - « Autorisé, barrière ouverte » n'apparaît que tant que `/etat` dit la barrière ouverte.
     - Si elle se referme avant la fin des 6 s, la page passe au résumé « Dernière décision : … », un état
       prévu par la maquette. C'est le cas normal : la barrière peut se refermer 5 s après l'ouverture.
- **Vérifié dans un vrai navigateur** (Edge sans fenêtre, états fabriqués), sur 6 cas :
  - « Autorisé, barrière ouverte » seulement si la barrière est ouverte ;
  - refus, non lu et fenêtre de 6 s inchangés.

  La page d'origine, sur les deux cas « barrière fermée », affichait « Autorisé, barrière ouverte » : le test
  distingue bien les deux versions.
- **Côté système, l'API suit ce qu'attend la maquette** :
  - `statut` vaut « en_cours » pour une piste non décidée ; la confiance reste vide tant qu'il n'y a pas de voix ;
  - les heures sont au format HH:MM:SS. Le journal CSV garde la date complète, que `/journal` renvoie à part ;
  - les pistes sont triées : plaques vues d'abord, la plus grande en tête, car la maquette détaille la première ;
  - `/journal` renvoie des nombres et des booléens, alors que le CSV est du texte ;
  - l'ordre des temps de traitement est conservé : Flask 2.0 triait les clés du JSON par ordre alphabétique ;
  - `/etape/<nom>` renvoie du PNG. Les étapes sont recalculées en mode debug une fois par nouvelle lecture ;
    « binaire » est l'image binaire réellement utilisée par la segmentation.
- **Image de contrôle des coins** : une découpe étroite est agrandie à 480 px avant le dessin. Sinon, le texte et
  les traits couvraient la plaque une fois l'image affichée en grand.
- **Fil du système** :
  - il lit la source (vidéo et photos au rythme réel, `--boucle` pour une démonstration) ;
  - après chaque image, il dépose un instantané sous verrou ; Flask ne fait que le lire ;
  - pendant les attentes (écart entre deux photos, pause entre deux boucles, fin de la source), la barrière
    continue de vivre en temps réel.
- **Sécurité** :
  - écoute sur 127.0.0.1 par défaut ; l'option `--ecoute`, prévue pour la phase 6, affiche un avertissement ;
  - jamais le mode debug de Flask ;
  - seuls la page, ses scripts et les captures au nom vérifié sont servis. Vérifié : un chemin `../`, encodé ou
    non, ou un fichier inconnu renvoient 404.
- **Flask** : la famille 2.0.3 est installée sur le PC.
  - Jinja2 passe de 3.1.6 à 3.0.3 et MarkupSafe de 3.0.3 à 2.0.1, car PyTorch demande « jinja2 » sans version
    minimale.
  - `pip check` ne signale rien, PyTorch reste en +cu126 et le lock est régénéré.
- **Contrôles** : `verifier_logique.py` passe 57 contrôles, dont 7 nouveaux sur l'interface (format de `/etat` et
  de `/journal`, fichiers refusés), sans navigateur ni modèle.
- **Pas encore essayé** :
  - la webcam en direct dans l'interface, qui passe par le même code que la démo ;
  - l'interface sur la Jetson (phase 6).
- **Ouverture automatique du navigateur** (demande de l'étudiant) :
  - Le serveur est créé avec `make_server` de Werkzeug au lieu de `app.run`, qui bloque. Dès sa création, il
    écoute : on sait qu'il est prêt, sans délai arbitraire.
  - Le navigateur est ouvert ensuite. Sa demande attend dans la file du serveur, qui y répond aussitôt.
  - Si le port est déjà pris, le programme s'arrête avec un message, avant d'allumer la caméra et sans ouvrir le
    navigateur.
  - **Désactivée** par `--sans-navigateur`, et par défaut sur la Jetson et sous Linux sans affichage graphique
    (SSH). Le module `webbrowser` y lancerait un navigateur en mode texte qui bloquerait la console. L'adresse est
    alors affichée.
  - Vérifié sans ouvrir de vrai navigateur, en remplaçant `webbrowser.open` par une fonction d'essai :
    - démarrage normal : page et `/etat` répondent en 0,02 s au moment de l'appel ;
    - avec `--sans-navigateur` : aucun appel ;
    - port occupé : arrêt, aucun appel.

    Le choix par défaut est contrôlé dans `verifier_logique.py`, qui passe maintenant 58 contrôles.
- **Décision de l'étudiant** : pas d'option pour forcer l'ouverture du navigateur sur la Jetson. Un navigateur y
  serait trop lourd pour 2 Go de RAM partagés entre le processeur et le GPU. En phase 6, la page de la Jetson
  s'ouvrira depuis le navigateur du PC (option `--ecoute`).

### Points pour l'arrêt B (constats de l'étudiant sur l'interface, rien n'est réglé maintenant)

1. **Une plaque détectée mais jamais lue**. Elle appartient à l'un des quatre véhicules photographiés avant le
   protocole, en développement.
   - Déjà observé le 2026-10-02 sur ses deux photos : « aucun chiffre » sur l'une, une lecture hors format sur
     l'autre ; jamais une lecture en forme de plaque.
   - Conséquence : aucune voix, donc aucune échéance. La page reste « En attente », et rien n'est noté au départ de
     la voiture.
   - À analyser à l'arrêt B dans la vue technique : découpe, coins, binarisation, segmentation, vignettes.
2. **Une voiture arrêtée devant la barrière avec une plaque illisible doit aboutir à « non lu »**, pour alerter le
   gardien. Aujourd'hui, l'échéance ne part qu'à la première voix (définitions du système, « Limite »).

   **Règle proposée**, à mesurer et à décider à l'arrêt B, avant toute mesure de comparaison :
   - **Une piste qui a des voix garde la règle actuelle** : échéance D après la première voix. Les plaques
     lisibles ne changent donc pas de traitement.
   - **Une piste sans aucune voix est déclarée « non lue »** quand sa plaque est « à portée de lecture » depuis
     T secondes.
     - « À portée de lecture » : le cadre de la plaque occupe au moins X % de la largeur de l'image.
     - T = 3 s, l'attente maximale décidée par l'étudiant.
     - Pourquoi la taille : c'est le signe le plus simple que la voiture est assez près pour être lue. Sur
       l'approche, la plaque est détectée de loin bien avant d'être lisible ; une échéance qui partirait de la
       détection déclencherait trop tôt.
   - **Choix de X sur le développement** :
     - X = la largeur à laquelle 90 % des plaques visées finalement lues ont déjà eu leur première lecture juste.
       L'évaluation l'affiche déjà, colonne « 1re juste (largeur) ».
     - Contrôle : la plaque arrêtée doit dépasser X dans presque toutes les vidéos, colonne « largeur max ». Sinon,
       une voiture arrêtée trop loin n'atteindrait jamais X.
   - **Fermeture d'une piste sans décision** : « non lu » si elle a eu une lecture en forme de plaque, **ou** si
     elle a atteint X. C'est le cas d'une voiture venue devant la barrière et repartie sans être lue.
   - **Variante à comparer** : départ à l'arrêt de la voiture, quand le cadre de la plaque reste presque immobile
     (taille et position) pendant 1 s.
     - Plus proche de l'idée « voiture arrêtée ».
     - Mais plus fragile : le cadre détecté tremble d'une image à l'autre, et une voiture qui attend dans une
       file, encore loin, déclencherait l'alerte.
   - **Garde-fous** :
     - la règle ne doit faire perdre aucune ouverture correcte sur le développement. Elle n'agit que sur les
       pistes sans voix ; une perte signifierait une voix arrivée après l'alerte ;
     - elle doit transformer en « non lu » les véhicules de développement jamais lus. Rapporter aussi le délai
       d'alerte, compté depuis le passage de X.
   - **Phase 6** : X dépend de la position de la caméra (la plaque au point d'arrêt doit occuper 10 à 20 % de la
     largeur) ; à revérifier avec la webcam réelle.

**Décisions de l'étudiant, fixées avant toute mesure** :
- **Départ de l'échéance** :
  - On garde la règle par la taille, sauf si la variante « arrêt » fait mieux sur les deux garde-fous à la fois :
    aucune ouverture correcte perdue, **et** plus de véhicules jamais lus transformés en « non lu », ou les mêmes
    plus vite.
  - À égalité, la taille, plus simple et plus robuste.
  - Seuils de la variante « arrêt », proposés pour confirmation : sur la dernière seconde, l'aire du cadre varie
    de moins de 10 % et son centre bouge de moins de 2 % de la largeur de l'image.
- **X** : 90 %. Si ce X fait perdre une ouverture correcte, on passe à 95 %, sans autre réglage.

### Photos montrées plus longtemps (`--duree-photo`, démo et interface)

- **Demande de l'étudiant** : garder chaque photo affichée plus longtemps (par exemple 5 s), pour montrer l'alerte
  « Lecture incertaine » à la soutenance.
- **L'option remplace `--repetitions`** : elle donne la durée en secondes (défaut : 1 s), convertie en passages à
  10 images/s.
- **Changement dans `systeme/sources.py`** : chaque photo est suivie de 10 s sans image, au lieu d'un écart fixe
  de 10 s entre les débuts de deux photos. Une photo montrée plus de 10 s aurait chevauché la suivante, et le temps
  aurait reculé.
- **L'alerte elle-même** viendra avec la règle d'échéance de l'arrêt B. Avec les règles actuelles, une plaque qui
  ne donne aucune voix n'a pas d'échéance, quelle que soit la durée d'affichage (vérifié sur la plaque jamais lue :
  « en cours », sans voix, pendant ses 5 s).
- **Constat de vitesse** : sur ces photos en portrait (1280 x 1707 après réduction), avec le PC sur batterie et
  Windows Update au travail (processus TiWorker), la détection prend 100 à 120 ms.
  - L'interface ne traite alors que 7,2 images/s au lieu de 10 : une photo de 5 s dure environ 7 s à l'écran.
  - Les décisions, comptées en temps de la source, ne changent pas.
  - Pour la soutenance : PC sur secteur, mises à jour terminées.

### Règle d'alerte codée, X provisoire pour les démonstrations (décision de l'étudiant)

- **Seuils de la variante « arrêt » confirmés par l'étudiant**, avant toute mesure : sur la dernière seconde, l'aire
  du cadre varie de moins de 10 % et son centre bouge de moins de 2 % de la largeur de l'image.
- **Code** :
  - dans `systeme/vote.py` : `DEPART_ALERTE` (« taille »), `ATTENTE_ALERTE` (3 s), `PORTEE_LECTURE` et
    `PORTEE_PROVISOIRE`, la fonction `arretee()` (variante) ;
  - dans la classe `Vote` : `observer()` fait partir l'alerte ; `examiner()` déclare « non lu » une piste sans voix
    à l'échéance ; `fermer()` note aussi une piste dont l'alerte était partie ;
  - une piste qui a des voix garde l'échéance du vote ;
  - `ControleAcces.associer` reçoit la largeur de l'image.
- **X provisoire = 10 % de la largeur**, réservé aux démonstrations et marqué comme tel dans le code et dans la
  console de la démo et de l'interface.
  - Il est choisi sans mesure, entre le bas de la plage où le détecteur voit bien les plaques (8 %) et la taille
    de la plaque au point d'arrêt (environ 17 %).
  - Sur les photos de développement, les plaques en occupent environ 15 %.
- **L'évaluation n'emploie jamais ce X** :
  - tant qu'il est provisoire, le bilan et le balayage tournent sans l'alerte, et l'annoncent ;
  - le test final refuse de tourner. C'est le seul refus, confirmé par l'étudiant : le balayage et la mesure de X
    doivent pouvoir tourner à l'arrêt B, et ils n'emploient pas ce X ;
  - `python -m outils.evaluer --portee` mesure X : 90e centile de la largeur de la plaque à sa première lecture
    juste, 95e si le 90e fait perdre une ouverture correcte. Il donne aussi la part des vidéos où la plaque arrêtée
    atteint X, mesure la variante « arrêt », applique la règle de choix et indique les valeurs à inscrire.
  - « Ou plus vite » veut dire, confirmé par l'étudiant : autant de « non lu », avec un délai médian plus court.
- **Ordre de l'arrêt B** :
  1. balayage du vote, sans l'alerte (elle n'agit que sur les pistes sans voix ; son effet sur les ouvertures est
     mesuré ensuite) ;
  2. `--portee`, avec le réglage retenu ;
  3. inscription de X et du départ dans `systeme/vote.py` ;
  4. test final.
- **Vérifié** :
  - 11 scénarios construits à la main, dans `verifier_logique.py` (71 contrôles) ;
  - dans l'interface, sur deux photos montrées 5 s : la plaque jamais lue passe en « non lu » environ 3 s après
    son apparition (« Lecture incertaine » sur la page), avec sa capture au journal ;
  - la démo prend toujours les mêmes décisions que le rejeu de l'évaluation ;
  - `--portee` tourne sur le banc d'essai, ce qui vérifie le fonctionnement seulement : les vidéos synthétiques
    sont lues dès 4 % de largeur.

## 2026-10-04 — Phase 5 terminée, sans évaluation sur vidéos réelles

### Changement de plan (décision de l'étudiant)

- Aucune vidéo ne sera filmée : la collecte de données s'arrête. La phase 5 se termine sans l'arrêt B (mesures et
  décisions sur la partie développement) et sans test final.
- La vidéo à deux véhicules mentionnée par l'étudiant n'est plus dans `validation/videos/` : le dossier est vide
  (dernière modification le 2026-10-04 à 10 h 26), il n'y a rien à supprimer. Il reste 5 photos des 4 véhicules
  de développement.

### Réglages figés par raisonnement (non mesurés)

- **Vote** (`systeme/vote.py`) :
  - variante majorité, la plus simple. « Liste d'abord » ouvrirait sans majorité, un risque qu'aucune mesure ne
    justifie ici ;
  - K = 3 voix : une lecture isolée peut être fausse (chiffre perdu, confusion), mais la même erreur trois fois de
    suite est bien plus rare. À 10 images/s, trois voix ne coûtent que 0,3 s ;
  - W = 10 voix : à peu près la dernière seconde de lectures. La plaque se rapproche, et les lectures récentes sont
    les meilleures ;
  - D = 3 s : l'attente maximale décidée par l'étudiant.
- **Seuil du détecteur** 0,5 : choix de la phase 2 (priorité au rappel), conservé.
- **Alerte « plaque sans voix »** :
  - départ « taille » : « non lu » 3 s après que la plaque occupe X = 10 % de la largeur de l'image ;
  - ce X est pris entre le bas de la plage où le détecteur voit bien les plaques (8 %) et la taille de la plaque
    au point d'arrêt (environ 17 %) ;
  - la variante « arrêt » reste codée mais n'est pas retenue. Elle est plus fragile : le cadre détecté tremble,
    et une voiture qui attend loin dans une file déclencherait l'alerte. La règle de l'étudiant ne l'aurait
    retenue que si elle faisait mieux sur les deux garde-fous, ce qui n'a pas pu être mesuré.
- La mention « provisoire » disparaît de la démo et de l'interface.
- `PORTEE_PROVISOIRE` devient `PORTEE_MESUREE = False`. Les outils d'évaluation n'emploient toujours pas un X non
  mesuré : bilan et balayage sans l'alerte, test final refusé.
- **Traitement** : contrôle du redressement « complete » conservé, sans confirmation sur des données vierges ;
  options de séparation des soudures désactivées.

### Limite (pour le rapport)

**Le système complet n'est pas évalué sur des vidéos réelles.**
- Le taux d'ouverture correcte, les ouvertures à tort, la fiabilité des décisions et les délais ne sont pas mesurés.
- Le gain de « complete » n'est pas confirmé sur des données vierges.
- Le seuil du détecteur n'est pas réajusté.
- Les plaques de location (fond bleu) ne sont vérifiées que sur un fond bleu simulé (phases 3 et 4) : aucune
  vraie photo de location n'a été prise.

Ce qui est établi :
- les mesures des phases 2 à 4 sur le jeu Kaggle :
  - test de la phase 4, une lecture par image : 43,0 % de lectures correctes, 2,1 % de substitutions ;
  - réglage hors pli, après l'itération ciblée : 65,5 % de lectures correctes, fiabilité des lectures acceptées
    84,8 % ;
- la logique du système, vérifiée par 71 contrôles sur des scénarios synthétiques
  (`python -m outils.verifier_logique`) ;
- l'accord exact entre la démo et le rejeu de l'évaluation, sur les deux vidéos synthétiques du banc d'essai ;
- les essais de l'interface (dont la règle de la barrière, testée dans un vrai navigateur) ;
- les essais de fonctionnement sur les photos de développement.

### Perspective : le protocole prévu, prêt à l'emploi

- **Données** :
  - 60 véhicules filmés selon le protocole de prise de vue du 2026-10-02 (marche vers des voitures garées,
    arrêt de 4 s, 3 photos), dont au moins 6 plaques bleues ;
  - découpage développement / test par véhicule (`python -m outils.validation --decouper`).
- **Mesures** :
  - listes croisées, avec variantes adverses à une modification près ;
  - pistes visées par la règle géométrique ;
  - règles de choix fixées avec l'étudiant : plateau, à un véhicule près, K le plus grand tant que 90 % des
    décisions restent sous 3 s.
- **Ordre** :
  1. `python -m outils.evaluer --planches` (contrôle des données) ;
  2. `--balayage` ;
  3. `--portee` : X (90e centile, ou 95e si une ouverture correcte est perdue) et départ de l'alerte (« arrêt »
     seulement s'il fait mieux sur les deux garde-fous), puis inscription dans `systeme/vote.py` avec
     `PORTEE_MESUREE = True` ;
  4. `--comparer aucun` (« complete » contre « aucun », sur les photos de développement), puis
     `--comparer bords,rangee` (séparation des soudures), avec les règles du 2026-10-02 ;
  5. `--test`, une seule fois.
- **À analyser aussi** : la plaque de développement détectée mais jamais lue (vue technique).
- **Outils gardés** : `outils/evaluer.py`, `outils/mesures.py`, `outils/validation.py`, et leurs contrôles dans
  `outils/verifier_logique.py`.

### Bilan de la phase 5

- **Système complet** (PC et Jetson, Python 3.6) :
  - sources : webcam, vidéo au temps de la vidéo, photos ;
  - suivi par recouvrement ;
  - vote et alerte « plaque sans voix » ;
  - barrière simulée (fermeture au moins 5 s après l'ouverture, 3 s après le départ de la plaque) ;
  - journal des passages avec captures ;
  - `systeme/chaine.py`.
- **Démonstrations** :
  - démo OpenCV, avec les modes des phases 2 à 4 ;
  - interface web locale (Flask 2.0.3, maquette de l'outil de design, 127.0.0.1, ouverture automatique du navigateur
    sur le PC).
- **Contrôles** : `python -m outils.verifier_logique`, 71 contrôles ; vermin passe.
- **Suite** : phase 6, le portage sur la Jetson, avec l'accord de l'étudiant.
