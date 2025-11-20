#!/bin/bash
#
# Script pour lancer l'entraînement du modèle de prédiction au moment de la réservation.
#
# Ce script envoie une requête POST à l'endpoint /api/v1/fit-booking-model
# pour recalculer les statistiques historiques et ré-entraîner le modèle.
# Assurez-vous que le backend est en cours d'exécution avant de lancer ce script.

API_URL="http://localhost:8000/api/v1/fit-booking-model"

echo "🚀 Lancement de l'entraînement du modèle de réservation..."
echo "URL de l'API: $API_URL"
echo ""

# Envoi de la requête POST avec curl
# -f : Échoue silencieusement sur les erreurs HTTP (affiche un message d'erreur)
# -s : Mode silencieux (ne montre pas la barre de progression)
# -S : Affiche les erreurs même en mode silencieux
# -X POST : Spécifie la méthode de requête
# -H "Content-Type: application/json" : Spécifie le header
response=$(curl -fsS -X POST "$API_URL")

# Vérifier le code de sortie de curl
if [ $? -eq 0 ]; then
  echo "✅ Entraînement terminé avec succès !"
  echo ""
  echo "Réponse du serveur :"
  echo "$response"
else
  echo "❌ Erreur lors de la communication avec le serveur."
  echo "Veuillez vérifier que le backend est démarré et accessible à l'adresse $API_URL."
fi
