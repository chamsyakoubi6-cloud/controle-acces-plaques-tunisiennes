#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vérification de l'environnement : bibliothèques, GPU et webcam.

Ce script tourne sur le PC (Windows, Python 3.11) ET sur la Jetson Nano
(Python 3.6) : il reste donc compatible Python 3.6 (contrôle : vermin).

Utilisation, depuis la racine du projet :
    python -m outils.verifier_env                 tout vérifier
    python -m outils.verifier_env --afficher      + aperçu webcam en direct (touche q pour quitter)
    python -m outils.verifier_env --sans-camera   sans la webcam
    python -m outils.verifier_env --camera 1      autre webcam

Code de sortie : 0 s'il n'y a aucun échec, 1 sinon.
"""

import argparse
import importlib
import io
import os
import platform
import sys
import time

# Réglages d'environnement d'OpenCV, faits une seule fois dans systeme/__init__.py :
# cet import doit précéder tout import de cv2. Lancer depuis la racine avec
# « python -m outils.verifier_env », sinon Python ne trouve pas le paquet systeme.
import systeme

# Moins de messages techniques de TensorFlow (doit être défini avant son import).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Racine du projet : ce fichier est dans outils/, la racine est le dossier parent.
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_SORTIE = os.path.join(RACINE, "sorties", "verification")

# Bibliothèques attendues : (nom du module Python, rôle dans le projet)
BIBLIOTHEQUES_PC = [
    ("numpy", "calcul sur tableaux"),
    ("cv2", "OpenCV, traitement d'image"),
    ("onnx", "lecture/écriture des fichiers ONNX"),
    ("onnxruntime", "inférence ONNX sur PC"),
    ("torch", "PyTorch, entraînement de YOLO"),
    ("torchvision", "complément de PyTorch"),
    ("ultralytics", "YOLOv8"),
    ("tensorflow", "entraînement du CNN"),
    ("keras", "API du CNN (Keras 3)"),
    ("tf2onnx", "conversion TensorFlow vers ONNX"),
]
BIBLIOTHEQUES_JETSON = [
    ("numpy", "calcul sur tableaux"),
    ("cv2", "OpenCV de JetPack"),
    ("tensorrt", "inférence sur le GPU"),
    ("pycuda", "mémoire GPU depuis Python"),
]

# Résultats accumulés : liste de (statut, sujet, détail).
# Statuts : OK, INFO (simple information), AVERT (à surveiller), ECHEC (à corriger).
resultats = []


def noter(statut, sujet, detail=""):
    """Enregistre un résultat et l'affiche aussitôt."""
    resultats.append((statut, sujet, detail))
    print("  [{:<5}] {:<26} {}".format(statut, sujet, detail))


def forcer_utf8_console():
    """Python 3.6 sous une locale « C » (certaines sessions SSH sur la Jetson)
    écrit la console en ASCII : le moindre accent ferait planter print().
    Dans ce cas seulement, on rouvre la sortie standard en UTF-8."""
    encodage = (sys.stdout.encoding or "").lower()
    if encodage in ("ascii", "ansi_x3.4-1968"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)


def detecter_profil():
    """Jetson si le processeur est un ARM 64 bits (aarch64), PC sinon."""
    if platform.machine() == "aarch64":
        return "jetson"
    return "pc"


def importer(nom_module):
    """Importe un module ; renvoie (module, None) ou (None, message d'erreur).
    On attrape toute exception : une DLL manquante lève autre chose qu'ImportError."""
    try:
        return importlib.import_module(nom_module), None
    except Exception as erreur:
        return None, "{}: {}".format(type(erreur).__name__, erreur)


# --------------------------------------------------------------- 1. Système

def verifier_systeme(profil):
    print("\n1. Système")
    noter("INFO", "Python", "{} ({})".format(platform.python_version(), sys.executable))
    noter("INFO", "Système", "{} {} ({})".format(platform.system(), platform.release(), platform.machine()))
    noter("INFO", "Profil", profil.upper())
    version = sys.version_info[:2]
    if profil == "pc" and version != (3, 11):
        noter("AVERT", "Version de Python", "le venv du projet est prévu en 3.11")
    if profil == "jetson" and version != (3, 6):
        noter("AVERT", "Version de Python", "JetPack 4.6 fournit Python 3.6")


# --------------------------------------------------------- 2. Bibliothèques

def lire_version(nom_module, module):
    """Numéro de version d'un module (pycuda le range ailleurs que les autres)."""
    if nom_module == "pycuda":
        return getattr(module, "VERSION_TEXT", "?")
    return str(getattr(module, "__version__", "?"))


def controler_version(nom_module, version, profil):
    """Contraintes de version importantes pour le projet ; renvoie (statut, remarque)."""
    if profil == "pc":
        if nom_module == "numpy" and not version.startswith("1."):
            return "AVERT", "NumPy 2 ne se comporte pas comme la 1.19.4 de la Jetson"
        if nom_module == "cv2" and not version.startswith("4."):
            return "AVERT", "OpenCV 4.x attendu (Jetson : 4.1.1)"
        if nom_module == "torch" and "+cu" not in version:
            return "ECHEC", "version CPU : réinstaller depuis l'index pytorch.org (cu126)"
    else:
        if nom_module == "numpy" and version == "1.19.5":
            return "ECHEC", "la 1.19.5 plante sur la Nano : installer la 1.19.4"
        if nom_module == "numpy" and version != "1.19.4":
            return "AVERT", "1.19.4 attendue"
        if nom_module == "cv2" and not version.startswith("4.1.1"):
            return "AVERT", "4.1.1 (JetPack) attendue : opencv-python installé par pip ?"
        if nom_module == "tensorrt" and not version.startswith("8.2"):
            return "AVERT", "TensorRT 8.2 attendu (JetPack 4.6)"
    return "OK", ""


def verifier_bibliotheques(profil):
    print("\n2. Bibliothèques")
    if profil == "pc":
        liste = BIBLIOTHEQUES_PC
    else:
        liste = BIBLIOTHEQUES_JETSON
    for nom_module, role in liste:
        module, erreur = importer(nom_module)
        if module is None:
            noter("ECHEC", nom_module, "{} - {}".format(role, erreur))
            continue
        version = lire_version(nom_module, module)
        statut, remarque = controler_version(nom_module, version, profil)
        detail = "{:<16} {}".format(version, role)
        if remarque:
            detail += " - " + remarque
        noter(statut, nom_module, detail)

    if profil == "pc":
        onnxruntime, erreur = importer("onnxruntime")
        if onnxruntime is not None:
            fournisseurs = onnxruntime.get_available_providers()
            noter("INFO", "ONNX Runtime", "fournisseurs : " + ", ".join(fournisseurs))


# ------------------------------------------------------ 3. Calcul (profil PC)

def verifier_gpu_pytorch():
    torch, erreur = importer("torch")
    if torch is None:
        noter("ECHEC", "PyTorch", erreur)
        return
    if not torch.cuda.is_available():
        noter("ECHEC", "GPU vu par PyTorch", "torch.cuda.is_available() = False (version CPU ou pilote ?)")
        return
    proprietes = torch.cuda.get_device_properties(0)
    noter("OK", "GPU vu par PyTorch", "{}, {:.1f} Go".format(proprietes.name, proprietes.total_memory / 1024 ** 3))
    noter("INFO", "Capacité de calcul", "{}.{}".format(proprietes.major, proprietes.minor))
    noter("INFO", "CUDA / cuDNN (PyTorch)", "{} / {}".format(torch.version.cuda, torch.backends.cudnn.version()))

    # Un vrai calcul sur le GPU : produit matriciel, comparé au même calcul sur le CPU.
    a = torch.rand(512, 512)
    b = torch.rand(512, 512)
    sur_cpu = a @ b
    sur_gpu = (a.cuda() @ b.cuda()).cpu()
    ecart_relatif = ((sur_cpu - sur_gpu).abs().max() / sur_cpu.abs().max()).item()
    if ecart_relatif < 1e-3:
        noter("OK", "Calcul sur le GPU", "produit 512x512 identique au CPU (écart relatif {:.1e})".format(ecart_relatif))
    else:
        noter("ECHEC", "Calcul sur le GPU", "écart relatif {:.1e} avec le CPU".format(ecart_relatif))


def verifier_numpy_pytorch():
    """Aller-retour NumPy -> PyTorch -> NumPy : vérifie que PyTorch accepte NumPy 1.x."""
    torch, erreur = importer("torch")
    if torch is None:
        return
    try:
        import numpy as np
        tableau = np.arange(6, dtype=np.float32).reshape(2, 3)
        retour = torch.from_numpy(tableau).numpy()
        if np.array_equal(tableau, retour):
            noter("OK", "Échange NumPy/PyTorch", "NumPy " + np.__version__)
        else:
            noter("ECHEC", "Échange NumPy/PyTorch", "données différentes après l'aller-retour")
    except Exception as erreur:
        noter("ECHEC", "Échange NumPy/PyTorch", "{}: {}".format(type(erreur).__name__, erreur))


def verifier_tensorflow():
    tf, erreur = importer("tensorflow")
    if tf is None:
        noter("ECHEC", "TensorFlow", erreur)
        return
    types_appareils = [appareil.device_type for appareil in tf.config.list_physical_devices()]
    if "GPU" in types_appareils:
        noter("INFO", "TensorFlow", "voit un GPU (inattendu sous Windows natif)")
    else:
        noter("OK", "TensorFlow", "CPU uniquement (normal sous Windows natif)")
    # Petit calcul pour vérifier que TensorFlow fonctionne réellement.
    matrice = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    produit = tf.matmul(matrice, matrice).numpy()
    if produit.tolist() == [[7.0, 10.0], [15.0, 22.0]]:
        noter("OK", "Calcul TensorFlow", "produit matriciel correct")
    else:
        noter("ECHEC", "Calcul TensorFlow", "résultat inattendu : {}".format(produit.tolist()))


# -------------------------------------------------- 3. Calcul (profil Jetson)

def lire_meminfo():
    """Lit /proc/meminfo (Linux) ; valeurs en kio, ex. {"MemTotal": 2027300, ...}."""
    valeurs = {}
    with open("/proc/meminfo", encoding="utf-8") as fichier:
        for ligne in fichier:
            cle, reste = ligne.split(":", 1)
            valeurs[cle] = int(reste.split()[0])
    return valeurs


def verifier_gpu_jetson():
    try:
        import pycuda.driver as cuda
        cuda.init()
        appareil = cuda.Device(0)
        noter("OK", "GPU vu par pycuda", "{}, {:.1f} Go".format(appareil.name(), appareil.total_memory() / 1024 ** 3))
    except Exception as erreur:
        noter("ECHEC", "GPU vu par pycuda", "{}: {}".format(type(erreur).__name__, erreur))

    try:
        import tensorrt as trt
        journal = trt.Logger(trt.Logger.WARNING)
        constructeur = trt.Builder(journal)
        noter("OK", "TensorRT", "builder créé, FP16 rapide : {}".format(constructeur.platform_has_fast_fp16))
    except Exception as erreur:
        noter("ECHEC", "TensorRT", "{}: {}".format(type(erreur).__name__, erreur))

    chemin_trtexec = "/usr/src/tensorrt/bin/trtexec"
    if os.path.exists(chemin_trtexec):
        noter("OK", "trtexec", chemin_trtexec)
    else:
        noter("AVERT", "trtexec", "absent de /usr/src/tensorrt/bin")

    # Mémoire : la Nano 2 Go a besoin d'un swap pour construire les moteurs TensorRT.
    memoire = lire_meminfo()
    ram_go = memoire.get("MemTotal", 0) / 1024 ** 2
    swap_go = memoire.get("SwapTotal", 0) / 1024 ** 2
    noter("INFO", "RAM", "{:.1f} Go".format(ram_go))
    if swap_go >= 2:
        noter("OK", "Swap", "{:.1f} Go".format(swap_go))
    else:
        noter("AVERT", "Swap", "{:.1f} Go : en activer environ 4 Go avant trtexec".format(swap_go))


# ---------------------------------------------------------------- 4. Webcam

def verifier_webcam(index, largeur, hauteur, afficher):
    cv2, erreur = importer("cv2")
    sources, erreur_sources = importer("systeme.sources")
    if cv2 is None or sources is None:
        noter("ECHEC", "Webcam", erreur or erreur_sources)
        return
    if sys.platform.startswith("win"):
        reglage = os.environ.get("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "(absent)")
        noter("INFO", "Réglage MSMF", "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=" + reglage)

    # Ouverture par la même fonction que le système (pilote MSMF sous Windows ou V4L2 sur la
    # Jetson, format MJPG, résolution demandée) : la vérification teste le vrai chemin du code.
    debut_ouverture = time.time()
    capture, nom_pilote = sources.ouvrir_webcam(index, largeur, hauteur)
    if not capture.isOpened():
        noter("ECHEC", "Ouverture de la webcam",
              "index {} via {} : branchement ? accès à la caméra autorisé ?".format(index, nom_pilote))
        return

    # On lit 10 images et on ne garde que la dernière : l'exposition automatique n'est
    # pas encore réglée sur les premières. La première image mesure le délai d'ouverture.
    lu, image, delai = False, None, None
    for _ in range(10):
        lu, image = capture.read()
        if lu and delai is None:
            delai = time.time() - debut_ouverture
    if not lu or image is None:
        noter("ECHEC", "Lecture d'images", "aucune image reçue")
        capture.release()
        return
    noter("OK", "Ouverture de la webcam",
          "index {} via {}, 1re image en {:.1f} s".format(index, nom_pilote, delai))

    # Résolution réelle, lue sur l'image : capture.get() renvoie parfois la valeur
    # demandée même quand la caméra ne l'applique pas.
    hauteur_lue, largeur_lue = image.shape[:2]
    if (largeur_lue, hauteur_lue) == (largeur, hauteur):
        noter("OK", "Résolution", "{}x{}".format(largeur_lue, hauteur_lue))
    else:
        noter("ECHEC", "Résolution", "{}x{} reçu au lieu de {}x{}".format(largeur_lue, hauteur_lue, largeur, hauteur))

    # Cadence réelle : nombre d'images lues pendant environ 3 secondes.
    nb_images = 0
    debut = time.time()
    while time.time() - debut < 3.0:
        lu, image_courante = capture.read()
        if not lu:
            break
        image = image_courante
        nb_images += 1
    cadence = nb_images / (time.time() - debut)
    if cadence >= 20:
        noter("OK", "Cadence", "{:.1f} images/s".format(cadence))
    else:
        noter("AVERT", "Cadence",
              "{:.1f} images/s (peu de lumière = pose plus longue = moins d'images/s)".format(cadence))

    # Format réellement utilisé : le code FOURCC range 4 caractères dans un entier de 32 bits.
    # Certains pilotes (MSMF) renvoient un code qui n'est pas du texte lisible.
    code = int(capture.get(cv2.CAP_PROP_FOURCC))
    format_video = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    if not format_video.strip().isalnum():
        format_video = "non communiqué par le pilote"
    noter("INFO", "Format vidéo", format_video)

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    chemin_image = os.path.join(DOSSIER_SORTIE, "webcam.jpg")
    cv2.imwrite(chemin_image, image)
    noter("INFO", "Image enregistrée", chemin_image)

    if afficher:
        apercu_webcam(cv2, capture)
    capture.release()


def apercu_webcam(cv2, capture):
    """Aperçu en direct, cadence affichée sur l'image ; touche q pour quitter."""
    print("  Aperçu en direct : touche q dans la fenêtre pour quitter.")
    precedent = time.time()
    while True:
        lu, image = capture.read()
        if not lu:
            break
        maintenant = time.time()
        cadence = 1.0 / max(maintenant - precedent, 1e-6)
        precedent = maintenant
        cv2.putText(image, "{:.0f} images/s".format(cadence), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Webcam (q pour quitter)", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# ------------------------------------------------------------------ Bilan

def afficher_bilan():
    """Résumé final ; renvoie le code de sortie du script (0 = aucun échec)."""
    compte = {"OK": 0, "INFO": 0, "AVERT": 0, "ECHEC": 0}
    for statut, _, _ in resultats:
        compte[statut] += 1
    print("\nBilan : {} OK, {} avertissement(s), {} échec(s)".format(
        compte["OK"], compte["AVERT"], compte["ECHEC"]))
    for statut, sujet, detail in resultats:
        if statut in ("AVERT", "ECHEC"):
            print("  [{}] {} : {}".format(statut, sujet, detail))
    if compte["ECHEC"] == 0:
        return 0
    return 1


def main():
    forcer_utf8_console()
    parseur = argparse.ArgumentParser(description="Vérifie l'environnement du projet plaques (PC ou Jetson).")
    parseur.add_argument("--camera", type=int, default=0, help="index de la webcam (défaut : 0)")
    parseur.add_argument("--largeur", type=int, default=1280, help="largeur demandée (défaut : 1280)")
    parseur.add_argument("--hauteur", type=int, default=720, help="hauteur demandée (défaut : 720)")
    parseur.add_argument("--afficher", action="store_true", help="aperçu en direct de la webcam (q pour quitter)")
    parseur.add_argument("--sans-camera", action="store_true", help="ne pas tester la webcam")
    arguments = parseur.parse_args()

    profil = detecter_profil()
    print("Vérification de l'environnement - profil {}".format(profil.upper()))
    verifier_systeme(profil)
    verifier_bibliotheques(profil)

    if profil == "pc":
        print("\n3. Calcul : GPU vu par PyTorch, NumPy/PyTorch, TensorFlow")
        verifier_gpu_pytorch()
        verifier_numpy_pytorch()
        verifier_tensorflow()
    else:
        print("\n3. Calcul : GPU (pycuda), TensorRT, mémoire")
        verifier_gpu_jetson()

    print("\n4. Webcam")
    if arguments.sans_camera:
        noter("INFO", "Webcam", "non testée (--sans-camera)")
    else:
        verifier_webcam(arguments.camera, arguments.largeur, arguments.hauteur, arguments.afficher)

    return afficher_bilan()


if __name__ == "__main__":
    sys.exit(main())
