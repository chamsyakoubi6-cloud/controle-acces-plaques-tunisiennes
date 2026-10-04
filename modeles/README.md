# modeles/ — poids et modèles (contenu non versionné)

| Fichier (nom indicatif) | Produit par | Utilisé sur | Phase |
|---|---|---|---|
| `yolov8n.pt` | poids pré-entraînés Ultralytics (téléchargés) | PC | 2 |
| `detecteur.pt` | entraînement de YOLOv8n, 1 classe « plaque » | PC | 2 |
| `detecteur.onnx` | export ONNX opset 12 | PC puis Jetson | 2 |
| `lecteur.keras` | entraînement du CNN (11 classes) | PC | 4 |
| `lecteur.onnx` | export tf2onnx opset 12 | PC puis Jetson | 4 |
| `*.engine` | `trtexec --fp16` à partir des `.onnx` | **Jetson uniquement** | 6 |

Un moteur TensorRT (`.engine`) n'est pas portable : il se construit sur la Jetson
elle-même, avec la version de TensorRT qui l'exécutera.
