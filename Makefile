# Makefile pour le projet de détection de drones FPV
# Usage: make <command>

.PHONY: help venv install split clean

# Variables
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
VENV_DIR := .venv

# Couleurs pour l'affichage
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

## help: Affiche l'aide
help:
	@echo "$(BLUE)🚀 Commandes disponibles :$(NC)"
	@echo ""
	@echo "  $(GREEN)make venv$(NC)        - Crée l'environnement virtuel Python"
	@echo "  $(GREEN)make install$(NC)     - Installe les dépendances"
	@echo "  $(GREEN)make split$(NC)       - Split le dataset en train/val/test (70/20/10)"
	@echo "  $(GREEN)make clean$(NC)       - Nettoie les fichiers temporaires"
	@echo "  $(GREEN)make help$(NC)        - Affiche cette aide"
	@echo ""

## venv: Crée l'environnement virtuel
venv:
	@echo "$(BLUE)📦 Création de l'environnement virtuel...$(NC)"
	python3 -m venv $(VENV_DIR)
	@echo "$(GREEN)✅ Environnement virtuel créé dans $(VENV_DIR)$(NC)"

## install: Installe les dépendances
install: venv
	@echo "$(BLUE)📥 Installation des dépendances...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirement.txt
	@echo "$(GREEN)✅ Dépendances installées$(NC)"

## split: Split le dataset en train/val/test
split:
	@echo "$(BLUE)🔀 Lancement du split du dataset...$(NC)"
	@echo "$(YELLOW)Source:$(NC)"
	@echo "  - Images positives: Datasets/dataset/images (2500)"
	@echo "  - Images négatives: Datasets/dataset/negatives_images (1000)"
	@echo "$(YELLOW)Destination:$(NC)"
	@echo "  - Datasets/splited_dataset/"
	@echo ""
	$(PYTHON) src/split/split_dataset.py
	@echo ""
	@echo "$(GREEN)✅ Split terminé !$(NC)"
	@echo "$(YELLOW)Prochaine étape : Augmentation du train set$(NC)"

## clean: Nettoie les fichiers temporaires
clean:
	@echo "$(BLUE)🧹 Nettoyage...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"
