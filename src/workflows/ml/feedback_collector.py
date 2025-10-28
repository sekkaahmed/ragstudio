"""
Module de collecte de feedback pour l'amélioration continue du modèle.

Ce module permet de capturer les prédictions, les résultats réels,
et les métriques de performance pour réentraîner le modèle.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np

LOGGER = logging.getLogger(__name__)


class FeedbackSource(str, Enum):
    """Source du feedback."""
    AUTO = "auto"  # Automatique (métriques de performance)
    USER = "user"  # Correction manuelle utilisateur
    VALIDATION = "validation"  # Validation A/B test
    PRODUCTION = "production"  # Données de production


@dataclass
class FeedbackEntry:
    """Entrée de feedback pour une prédiction."""

    # Identifiants
    feedback_id: str
    timestamp: float
    source: FeedbackSource

    # Document original
    doc_id: Optional[str]
    text: str
    text_length: int

    # Profil du document
    profile: Dict[str, Any]

    # Prédiction du modèle
    predicted_strategy: str
    prediction_confidence: float
    all_probabilities: Dict[str, float]

    # Stratégie réelle utilisée et résultats
    actual_strategy_used: Optional[str] = None
    actual_performance: Optional[Dict[str, float]] = None

    # Feedback utilisateur
    user_override: Optional[str] = None
    user_rating: Optional[int] = None  # 1-5
    user_comment: Optional[str] = None

    # Métriques de chunking
    chunk_count: Optional[int] = None
    avg_chunk_size: Optional[float] = None
    processing_time: Optional[float] = None
    retrieval_quality: Optional[float] = None  # Score de qualité RAG

    # Métadonnées
    model_version: Optional[str] = None
    environment: Optional[str] = None  # dev, staging, prod

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeedbackEntry:
        """Créer depuis un dictionnaire."""
        return cls(**data)


class FeedbackCollector:
    """
    Collecteur de feedback pour l'amélioration continue.

    Collecte les prédictions, résultats réels, et feedback utilisateur
    pour réentraîner le modèle périodiquement.
    """

    def __init__(
        self,
        feedback_dir: str = "data/feedback",
        model_version: str = "v1.0",
        environment: str = "production"
    ):
        """
        Initialiser le collecteur de feedback.

        Args:
            feedback_dir: Répertoire de stockage du feedback
            model_version: Version du modèle actuel
            environment: Environnement d'exécution
        """
        self.feedback_dir = Path(feedback_dir)
        self.model_version = model_version
        self.environment = environment

        # Créer les répertoires
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir = self.feedback_dir / "pending"
        self.validated_dir = self.feedback_dir / "validated"
        self.rejected_dir = self.feedback_dir / "rejected"

        for dir_path in [self.pending_dir, self.validated_dir, self.rejected_dir]:
            dir_path.mkdir(exist_ok=True)

        LOGGER.info("FeedbackCollector initialized (dir=%s, version=%s)",
                   feedback_dir, model_version)

    def record_prediction(
        self,
        text: str,
        profile: Dict[str, Any],
        predicted_strategy: str,
        confidence: float,
        all_probabilities: Dict[str, float],
        doc_id: Optional[str] = None,
        source: FeedbackSource = FeedbackSource.PRODUCTION
    ) -> str:
        """
        Enregistrer une prédiction pour feedback futur.

        Args:
            text: Texte du document
            profile: Profil du document
            predicted_strategy: Stratégie prédite
            confidence: Confiance de la prédiction
            all_probabilities: Probabilités pour toutes les stratégies
            doc_id: ID optionnel du document
            source: Source du feedback

        Returns:
            ID du feedback créé
        """
        # Générer un ID unique
        feedback_id = self._generate_feedback_id()

        # Créer l'entrée de feedback
        entry = FeedbackEntry(
            feedback_id=feedback_id,
            timestamp=time.time(),
            source=source,
            doc_id=doc_id,
            text=text[:1000],  # Limiter la taille stockée
            text_length=len(text),
            profile=profile,
            predicted_strategy=predicted_strategy,
            prediction_confidence=confidence,
            all_probabilities=all_probabilities,
            model_version=self.model_version,
            environment=self.environment
        )

        # Sauvegarder en pending
        self._save_feedback(entry, self.pending_dir)

        LOGGER.debug("Recorded prediction: %s (strategy=%s, confidence=%.3f)",
                    feedback_id, predicted_strategy, confidence)

        return feedback_id

    def update_feedback(
        self,
        feedback_id: str,
        actual_strategy: Optional[str] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        user_override: Optional[str] = None,
        user_rating: Optional[int] = None,
        user_comment: Optional[str] = None
    ) -> bool:
        """
        Mettre à jour une entrée de feedback avec les résultats réels.

        Args:
            feedback_id: ID du feedback à mettre à jour
            actual_strategy: Stratégie réellement utilisée
            performance_metrics: Métriques de performance
            user_override: Correction manuelle de l'utilisateur
            user_rating: Note de l'utilisateur (1-5)
            user_comment: Commentaire de l'utilisateur

        Returns:
            True si la mise à jour a réussi
        """
        # Charger le feedback pending
        entry = self._load_feedback(feedback_id, self.pending_dir)

        if entry is None:
            LOGGER.warning("Feedback not found: %s", feedback_id)
            return False

        # Mettre à jour les champs
        if actual_strategy:
            entry.actual_strategy_used = actual_strategy

        if performance_metrics:
            entry.actual_performance = performance_metrics
            entry.chunk_count = performance_metrics.get("chunk_count")
            entry.avg_chunk_size = performance_metrics.get("avg_chunk_size")
            entry.processing_time = performance_metrics.get("processing_time")
            entry.retrieval_quality = performance_metrics.get("retrieval_quality")

        if user_override:
            entry.user_override = user_override
            entry.source = FeedbackSource.USER

        if user_rating:
            entry.user_rating = user_rating

        if user_comment:
            entry.user_comment = user_comment

        # Sauvegarder la mise à jour
        self._save_feedback(entry, self.pending_dir)

        LOGGER.debug("Updated feedback: %s", feedback_id)

        return True

    def validate_feedback(self, feedback_id: str, is_valid: bool = True) -> bool:
        """
        Valider ou rejeter une entrée de feedback.

        Args:
            feedback_id: ID du feedback
            is_valid: True pour valider, False pour rejeter

        Returns:
            True si l'opération a réussi
        """
        # Charger le feedback pending
        entry = self._load_feedback(feedback_id, self.pending_dir)

        if entry is None:
            return False

        # Déplacer vers validated ou rejected
        target_dir = self.validated_dir if is_valid else self.rejected_dir
        self._save_feedback(entry, target_dir)

        # Supprimer de pending
        self._delete_feedback(feedback_id, self.pending_dir)

        status = "validated" if is_valid else "rejected"
        LOGGER.info("Feedback %s: %s", status, feedback_id)

        return True

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Obtenir des statistiques sur le feedback collecté.

        Returns:
            Statistiques détaillées
        """
        pending_count = len(list(self.pending_dir.glob("*.json")))
        validated_count = len(list(self.validated_dir.glob("*.json")))
        rejected_count = len(list(self.rejected_dir.glob("*.json")))

        # Charger les feedbacks validés pour statistiques détaillées
        validated_entries = self._load_all_feedback(self.validated_dir)

        stats = {
            "total_pending": pending_count,
            "total_validated": validated_count,
            "total_rejected": rejected_count,
            "total_feedback": pending_count + validated_count + rejected_count,
            "model_version": self.model_version,
            "environment": self.environment
        }

        if validated_entries:
            # Statistiques sur les prédictions
            correct_predictions = sum(
                1 for e in validated_entries
                if e.predicted_strategy == e.actual_strategy_used
            )

            stats.update({
                "accuracy": correct_predictions / len(validated_entries),
                "avg_confidence": np.mean([e.prediction_confidence for e in validated_entries]),
                "sources": self._count_by_field(validated_entries, "source"),
                "predicted_strategies": self._count_by_field(validated_entries, "predicted_strategy"),
                "actual_strategies": self._count_by_field(validated_entries, "actual_strategy_used"),
            })

            # Ratings utilisateur
            ratings = [e.user_rating for e in validated_entries if e.user_rating]
            if ratings:
                stats["avg_user_rating"] = np.mean(ratings)

        return stats

    def export_for_training(
        self,
        output_file: str,
        min_confidence_threshold: float = 0.0,
        include_user_overrides: bool = True
    ) -> int:
        """
        Exporter les feedbacks validés pour réentraînement.

        Args:
            output_file: Fichier de sortie (JSONL)
            min_confidence_threshold: Confiance minimale pour inclure
            include_user_overrides: Inclure les corrections utilisateur

        Returns:
            Nombre d'exemples exportés
        """
        # Charger tous les feedbacks validés
        entries = self._load_all_feedback(self.validated_dir)

        # Filtrer et transformer pour l'entraînement
        training_samples = []

        for entry in entries:
            # Déterminer la "vraie" stratégie
            true_strategy = None

            # Priorité 1: Override utilisateur
            if include_user_overrides and entry.user_override:
                true_strategy = entry.user_override

            # Priorité 2: Stratégie réellement utilisée avec bonne performance
            elif entry.actual_strategy_used:
                # Vérifier si la performance était acceptable
                if entry.actual_performance:
                    quality = entry.actual_performance.get("retrieval_quality", 0)
                    if quality >= 0.7:  # Seuil de qualité acceptable
                        true_strategy = entry.actual_strategy_used

            # Priorité 3: Prédiction avec haute confiance
            elif entry.prediction_confidence >= min_confidence_threshold:
                true_strategy = entry.predicted_strategy

            # Si on a une vraie stratégie, créer un exemple d'entraînement
            if true_strategy:
                sample = {
                    "doc_id": entry.doc_id or entry.feedback_id,
                    "text": entry.text,
                    "length_tokens": entry.profile.get("length_tokens", 0),
                    "hierarchy_depth": entry.profile.get("hierarchy_depth", 1),
                    "has_tables": entry.profile.get("has_tables", False),
                    "has_headings": entry.profile.get("has_headings", False),
                    "has_lists": entry.profile.get("has_lists", False),
                    "structure_score": entry.profile.get("structure_score", 0.0),
                    "lang": entry.profile.get("lang", "unknown"),
                    "type": entry.profile.get("type", "unknown"),
                    "best_strategy": true_strategy,
                    # Métadonnées pour traçabilité
                    "_feedback_id": entry.feedback_id,
                    "_source": entry.source.value,
                    "_confidence": entry.prediction_confidence,
                    "_timestamp": entry.timestamp
                }

                training_samples.append(sample)

        # Sauvegarder en JSONL
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        LOGGER.info("Exported %d training samples to %s",
                   len(training_samples), output_file)

        return len(training_samples)

    def _generate_feedback_id(self) -> str:
        """Générer un ID unique pour le feedback."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = os.urandom(4).hex()
        return f"feedback_{timestamp}_{random_suffix}"

    def _save_feedback(self, entry: FeedbackEntry, directory: Path) -> None:
        """Sauvegarder une entrée de feedback."""
        file_path = directory / f"{entry.feedback_id}.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)

    def _load_feedback(self, feedback_id: str, directory: Path) -> Optional[FeedbackEntry]:
        """Charger une entrée de feedback."""
        file_path = directory / f"{feedback_id}.json"

        if not file_path.exists():
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return FeedbackEntry.from_dict(data)

    def _delete_feedback(self, feedback_id: str, directory: Path) -> None:
        """Supprimer une entrée de feedback."""
        file_path = directory / f"{feedback_id}.json"

        if file_path.exists():
            file_path.unlink()

    def _load_all_feedback(self, directory: Path) -> List[FeedbackEntry]:
        """Charger tous les feedbacks d'un répertoire."""
        entries = []

        for file_path in directory.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                entries.append(FeedbackEntry.from_dict(data))
            except Exception as e:
                LOGGER.warning("Failed to load feedback from %s: %s", file_path, e)

        return entries

    def _count_by_field(self, entries: List[FeedbackEntry], field: str) -> Dict[str, int]:
        """Compter les entrées par valeur de champ."""
        counts = {}

        for entry in entries:
            value = getattr(entry, field, None)
            if value:
                value_str = value.value if isinstance(value, Enum) else str(value)
                counts[value_str] = counts.get(value_str, 0) + 1

        return counts


# Instance globale
_COLLECTOR_INSTANCE: Optional[FeedbackCollector] = None


def get_feedback_collector(
    feedback_dir: str = "data/feedback",
    model_version: str = "v1.0",
    environment: str = "production"
) -> FeedbackCollector:
    """
    Obtenir l'instance globale du collecteur de feedback.

    Args:
        feedback_dir: Répertoire de stockage
        model_version: Version du modèle
        environment: Environnement

    Returns:
        FeedbackCollector instance
    """
    global _COLLECTOR_INSTANCE

    if _COLLECTOR_INSTANCE is None:
        _COLLECTOR_INSTANCE = FeedbackCollector(
            feedback_dir, model_version, environment
        )

    return _COLLECTOR_INSTANCE


if __name__ == "__main__":
    # Test du collecteur
    collector = FeedbackCollector()

    # Exemple de prédiction
    feedback_id = collector.record_prediction(
        text="Texte de test pour la prédiction",
        profile={"length_tokens": 50, "hierarchy_depth": 2},
        predicted_strategy="semantic",
        confidence=0.85,
        all_probabilities={"semantic": 0.85, "recursive": 0.1, "parent_child": 0.05}
    )

    print(f"Feedback enregistré: {feedback_id}")

    # Mise à jour avec résultats réels
    collector.update_feedback(
        feedback_id,
        actual_strategy="semantic",
        performance_metrics={"retrieval_quality": 0.9, "chunk_count": 5}
    )

    # Validation
    collector.validate_feedback(feedback_id, is_valid=True)

    # Statistiques
    stats = collector.get_feedback_stats()
    print(f"Statistiques: {json.dumps(stats, indent=2)}")
