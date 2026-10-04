"""Code du système de lecture de plaques (contrôle d'accès d'un parking).

Ce paquet tourne sur le PC ET sur la Jetson Nano : il doit rester compatible
Python 3.6, NumPy 1.19 et OpenCV 4.1.1, et ne jamais importer TensorFlow,
PyTorch ni Ultralytics. Les modules du pipeline arrivent à partir de la
phase 2 (voir README.md, section « Pipeline »).

Ce fichier est aussi l'endroit UNIQUE où l'on règle l'environnement d'OpenCV.
Python l'exécute avant n'importe quel module du paquet, donc avant leurs
« import cv2 ». Un script situé hors du paquet (outils/, entrainement/) doit
simplement importer « systeme » avant « cv2 ».
"""

import os
import sys
import warnings

# Webcam sous Windows, pilote Media Foundation (MSMF) : avec ses transformations
# matérielles, l'ouverture prend environ 10 s ; sans, 1,3 s (mesuré en phase 1).
# OpenCV lit cette variable à son chargement : elle doit donc être définie AVANT
# le premier « import cv2 » (vérifié : définie après, elle reste sans effet).
# Sans effet sous Linux (Jetson), où la webcam passe par V4L2.
# setdefault : une valeur déjà définie par l'utilisateur est respectée.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

# Garde-fou : si OpenCV est déjà chargé, le réglage ci-dessus arrive trop tard.
if "cv2" in sys.modules:
    warnings.warn("OpenCV (cv2) a été importé avant le paquet systeme : ses réglages "
                  "d'environnement n'auront pas d'effet. Importer systeme avant cv2.")
