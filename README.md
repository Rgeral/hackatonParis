# Hackathon Defense Paris 7-9/11/2025

### Subject : Synteza-1 - Detecting and tracking of FPV drones with Cameras and AI

- **Problem statement** : To develop a fully autonomous system based on computer vision (AI models or CV trackers) capable of detecting and tracking small FPV drones in real time at a distance of 500 meters, with at least 50% accuracy – without external communication or human intervention.
- **Context** : Small FPV drones are virtually invisible to traditional detection systems. Until now, most units have relied on visual detection, and there are no technological solutions at the tactical level to counter them. Synteza provides a tool and 3d models of drones to generate labeled datasets for computer vision models training.



## Project installation

### 1. Créer env virtuel
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installer dépendances
```bash
pip install --upgrade pip
pip install -r requirement.txt
pip install -r requirements-dev.txt
```

### 3. Activer pre-commit (optionnel)
```bash
pre-commit install
```

### 4. Tester que ça marche
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
```

### 5. Pour GPU/CUDA (carte NVIDIA)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
