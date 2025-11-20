# Dockerfile pour l'application FastAPI avec uv
FROM python:3.11.11-slim

WORKDIR /app

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de configuration pyproject.toml et le code
COPY pyproject.toml ./
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY main.py ./

# Créer un environnement virtuel et installer les dépendances avec uv
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# Créer le répertoire pour les fichiers uploadés
RUN mkdir -p /app/uploads

# Exposer le port
EXPOSE 8000

# Activer l'environnement virtuel et exécuter l'application
CMD ["/bin/bash", "-c", ". .venv/bin/activate && python main.py"]
