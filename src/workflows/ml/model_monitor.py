"""
Système de monitoring et métriques pour le modèle de scoring.

Suit les performances du modèle en production et détecte les dérives.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque

import numpy as np

LOGGER = logging.getLogger(__name__)


class ModelMonitor:
    """
    Moniteur de performance du modèle en production.

    Collecte et analyse les métriques de prédiction pour détecter
    les dérives de performance et les anomalies.
    """

    def __init__(
        self,
        metrics_dir: str = "data/metrics",
        window_size: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialiser le moniteur.

        Args:
            metrics_dir: Répertoire de stockage des métriques
            window_size: Taille de la fenêtre glissante pour calculs
            alert_thresholds: Seuils d'alerte personnalisés
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size

        # Seuils d'alerte par défaut
        self.alert_thresholds = alert_thresholds or {
            "min_confidence": 0.6,
            "max_confidence_std": 0.3,
            "min_accuracy": 0.7,
            "max_prediction_time": 5.0,  # secondes
        }

        # Métriques en mémoire (fenêtre glissante)
        self.predictions_window = deque(maxlen=window_size)

        # Statistiques par stratégie
        self.strategy_stats = defaultdict(lambda: {
            "count": 0,
            "correct": 0,
            "total_confidence": 0.0,
            "prediction_times": []
        })

        # Fichier de métriques journalières
        self.daily_metrics_file = self.metrics_dir / "daily_metrics.jsonl"

        # Alertes actives
        self.active_alerts = []

        LOGGER.info("ModelMonitor initialized (window_size=%d)", window_size)

    def record_prediction(
        self,
        predicted_strategy: str,
        confidence: float,
        actual_strategy: Optional[str] = None,
        prediction_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Enregistrer une prédiction pour monitoring.

        Args:
            predicted_strategy: Stratégie prédite
            confidence: Confiance de la prédiction
            actual_strategy: Stratégie réelle (si connue)
            prediction_time: Temps de prédiction en secondes
            metadata: Métadonnées additionnelles
        """
        prediction_record = {
            "timestamp": time.time(),
            "predicted_strategy": predicted_strategy,
            "confidence": confidence,
            "actual_strategy": actual_strategy,
            "prediction_time": prediction_time,
            "metadata": metadata or {}
        }

        # Ajouter à la fenêtre
        self.predictions_window.append(prediction_record)

        # Mettre à jour les statistiques par stratégie
        stats = self.strategy_stats[predicted_strategy]
        stats["count"] += 1
        stats["total_confidence"] += confidence

        if prediction_time is not None:
            stats["prediction_times"].append(prediction_time)

        if actual_strategy is not None:
            if actual_strategy == predicted_strategy:
                stats["correct"] += 1

        # Vérifier les seuils d'alerte
        self._check_alerts(prediction_record)

        LOGGER.debug("Recorded prediction: %s (confidence=%.3f)",
                    predicted_strategy, confidence)

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Obtenir les métriques actuelles (fenêtre glissante).

        Returns:
            Métriques agrégées
        """
        if not self.predictions_window:
            return {
                "window_size": 0,
                "message": "No predictions recorded yet"
            }

        predictions = list(self.predictions_window)

        # Métriques globales
        total_predictions = len(predictions)
        confidences = [p["confidence"] for p in predictions]
        prediction_times = [p["prediction_time"] for p in predictions if p["prediction_time"]]

        metrics = {
            "window_size": total_predictions,
            "timestamp": datetime.now().isoformat(),

            # Confiance
            "confidence": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "median": np.median(confidences)
            },

            # Performance temporelle
            "prediction_time": {
                "mean": np.mean(prediction_times) if prediction_times else None,
                "std": np.std(prediction_times) if prediction_times else None,
                "max": np.max(prediction_times) if prediction_times else None
            }
        }

        # Accuracy (si actual_strategy disponible)
        predictions_with_actual = [p for p in predictions if p["actual_strategy"]]

        if predictions_with_actual:
            correct = sum(
                1 for p in predictions_with_actual
                if p["predicted_strategy"] == p["actual_strategy"]
            )
            metrics["accuracy"] = correct / len(predictions_with_actual)

        # Distribution des stratégies
        strategy_distribution = defaultdict(int)
        for p in predictions:
            strategy_distribution[p["predicted_strategy"]] += 1

        metrics["strategy_distribution"] = dict(strategy_distribution)

        # Statistiques par stratégie
        strategy_metrics = {}
        for strategy, stats in self.strategy_stats.items():
            if stats["count"] > 0:
                strategy_metrics[strategy] = {
                    "count": stats["count"],
                    "avg_confidence": stats["total_confidence"] / stats["count"],
                    "accuracy": stats["correct"] / stats["count"] if stats["count"] > 0 else None,
                }

                if stats["prediction_times"]:
                    strategy_metrics[strategy]["avg_prediction_time"] = np.mean(
                        stats["prediction_times"]
                    )

        metrics["by_strategy"] = strategy_metrics

        return metrics

    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Obtenir le résumé des métriques pour une journée.

        Args:
            date: Date cible (aujourd'hui si None)

        Returns:
            Résumé journalier
        """
        if date is None:
            date = datetime.now()

        # Filtrer les prédictions de la journée
        day_start = datetime(date.year, date.month, date.day).timestamp()
        day_end = day_start + 86400  # +24h

        daily_predictions = [
            p for p in self.predictions_window
            if day_start <= p["timestamp"] < day_end
        ]

        if not daily_predictions:
            return {
                "date": date.strftime("%Y-%m-%d"),
                "total_predictions": 0,
                "message": "No predictions for this day"
            }

        # Calculer les métriques
        confidences = [p["confidence"] for p in daily_predictions]

        summary = {
            "date": date.strftime("%Y-%m-%d"),
            "total_predictions": len(daily_predictions),
            "avg_confidence": np.mean(confidences),
            "strategy_distribution": {}
        }

        # Distribution des stratégies
        for p in daily_predictions:
            strategy = p["predicted_strategy"]
            summary["strategy_distribution"][strategy] = \
                summary["strategy_distribution"].get(strategy, 0) + 1

        # Accuracy si disponible
        with_actual = [p for p in daily_predictions if p["actual_strategy"]]
        if with_actual:
            correct = sum(
                1 for p in with_actual
                if p["predicted_strategy"] == p["actual_strategy"]
            )
            summary["accuracy"] = correct / len(with_actual)

        return summary

    def detect_drift(self, lookback_days: int = 7) -> Dict[str, Any]:
        """
        Détecter une dérive de performance du modèle.

        Args:
            lookback_days: Nombre de jours à analyser

        Returns:
            Résultats de détection de dérive
        """
        now = datetime.now()
        cutoff_time = (now - timedelta(days=lookback_days)).timestamp()

        # Filtrer les prédictions récentes
        recent_predictions = [
            p for p in self.predictions_window
            if p["timestamp"] >= cutoff_time
        ]

        if len(recent_predictions) < 30:  # Minimum pour analyse
            return {
                "drift_detected": False,
                "reason": "insufficient_data",
                "sample_count": len(recent_predictions)
            }

        # Séparer en deux périodes
        mid_point = len(recent_predictions) // 2
        period1 = recent_predictions[:mid_point]
        period2 = recent_predictions[mid_point:]

        # Comparer les métriques
        conf1 = np.mean([p["confidence"] for p in period1])
        conf2 = np.mean([p["confidence"] for p in period2])

        confidence_drift = abs(conf1 - conf2)

        # Comparer l'accuracy si disponible
        accuracy_drift = None

        with_actual1 = [p for p in period1 if p["actual_strategy"]]
        with_actual2 = [p for p in period2 if p["actual_strategy"]]

        if with_actual1 and with_actual2:
            acc1 = sum(
                1 for p in with_actual1
                if p["predicted_strategy"] == p["actual_strategy"]
            ) / len(with_actual1)

            acc2 = sum(
                1 for p in with_actual2
                if p["predicted_strategy"] == p["actual_strategy"]
            ) / len(with_actual2)

            accuracy_drift = abs(acc1 - acc2)

        # Détection de dérive
        drift_detected = False
        reasons = []

        if confidence_drift > 0.15:  # Seuil de dérive de confiance
            drift_detected = True
            reasons.append(f"Confidence drift: {confidence_drift:.3f}")

        if accuracy_drift and accuracy_drift > 0.1:  # Seuil de dérive d'accuracy
            drift_detected = True
            reasons.append(f"Accuracy drift: {accuracy_drift:.3f}")

        return {
            "drift_detected": drift_detected,
            "reasons": reasons,
            "metrics": {
                "confidence_drift": confidence_drift,
                "accuracy_drift": accuracy_drift,
                "period1_samples": len(period1),
                "period2_samples": len(period2),
                "period1_confidence": conf1,
                "period2_confidence": conf2
            }
        }

    def _check_alerts(self, prediction_record: Dict[str, Any]) -> None:
        """Vérifier les seuils d'alerte pour une prédiction."""
        alerts = []

        # Vérifier la confiance minimale
        if prediction_record["confidence"] < self.alert_thresholds["min_confidence"]:
            alerts.append({
                "type": "low_confidence",
                "severity": "warning",
                "message": f"Low confidence: {prediction_record['confidence']:.3f}",
                "timestamp": prediction_record["timestamp"]
            })

        # Vérifier le temps de prédiction
        if prediction_record["prediction_time"]:
            if prediction_record["prediction_time"] > self.alert_thresholds["max_prediction_time"]:
                alerts.append({
                    "type": "slow_prediction",
                    "severity": "warning",
                    "message": f"Slow prediction: {prediction_record['prediction_time']:.3f}s",
                    "timestamp": prediction_record["timestamp"]
                })

        # Ajouter les alertes
        for alert in alerts:
            self.active_alerts.append(alert)
            LOGGER.warning("Alert: %s", alert["message"])

        # Limiter le nombre d'alertes en mémoire
        if len(self.active_alerts) > 100:
            self.active_alerts = self.active_alerts[-100:]

    def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtenir les alertes actives.

        Args:
            severity: Filtrer par sévérité (warning, error, critical)

        Returns:
            Liste des alertes
        """
        if severity:
            return [a for a in self.active_alerts if a["severity"] == severity]

        return self.active_alerts

    def clear_alerts(self) -> int:
        """
        Effacer les alertes actives.

        Returns:
            Nombre d'alertes effacées
        """
        count = len(self.active_alerts)
        self.active_alerts = []

        LOGGER.info("Cleared %d alerts", count)

        return count

    def save_daily_metrics(self) -> None:
        """Sauvegarder les métriques journalières sur disque."""
        summary = self.get_daily_summary()

        with open(self.daily_metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')

        LOGGER.info("Saved daily metrics")

    def export_metrics_report(
        self,
        output_file: str,
        days: int = 30
    ) -> None:
        """
        Exporter un rapport de métriques.

        Args:
            output_file: Fichier de sortie (JSON)
            days: Nombre de jours à inclure
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "current_metrics": self.get_current_metrics(),
            "drift_analysis": self.detect_drift(lookback_days=days),
            "active_alerts": self.get_active_alerts(),
            "strategy_statistics": {}
        }

        # Statistiques par stratégie
        for strategy, stats in self.strategy_stats.items():
            if stats["count"] > 0:
                report["strategy_statistics"][strategy] = {
                    "total_predictions": stats["count"],
                    "avg_confidence": stats["total_confidence"] / stats["count"],
                    "accuracy": stats["correct"] / stats["count"] if stats["count"] > 0 else None
                }

        # Sauvegarder le rapport
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        LOGGER.info("Exported metrics report to %s", output_file)


# Instance globale
_MONITOR_INSTANCE: Optional[ModelMonitor] = None


def get_model_monitor(
    metrics_dir: str = "data/metrics",
    window_size: int = 1000
) -> ModelMonitor:
    """
    Obtenir l'instance globale du moniteur.

    Args:
        metrics_dir: Répertoire de métriques
        window_size: Taille de la fenêtre

    Returns:
        ModelMonitor instance
    """
    global _MONITOR_INSTANCE

    if _MONITOR_INSTANCE is None:
        _MONITOR_INSTANCE = ModelMonitor(metrics_dir, window_size)

    return _MONITOR_INSTANCE


if __name__ == "__main__":
    # Test du moniteur
    monitor = ModelMonitor()

    # Simuler des prédictions
    strategies = ["semantic", "recursive", "parent_child", "late"]

    for i in range(100):
        strategy = np.random.choice(strategies)
        confidence = np.random.uniform(0.6, 0.95)
        actual = strategy if np.random.random() > 0.2 else np.random.choice(strategies)

        monitor.record_prediction(
            predicted_strategy=strategy,
            confidence=confidence,
            actual_strategy=actual,
            prediction_time=np.random.uniform(0.1, 2.0)
        )

    # Afficher les métriques
    metrics = monitor.get_current_metrics()
    print(json.dumps(metrics, indent=2))

    # Détecter la dérive
    drift = monitor.detect_drift()
    print(f"\nDrift detection: {json.dumps(drift, indent=2)}")

    # Exporter un rapport
    monitor.export_metrics_report("data/metrics/test_report.json")
    print("\nReport exported")
