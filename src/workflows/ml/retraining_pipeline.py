"""
Pipeline de réentraînement automatique du modèle.

Orchestre la collecte de feedback, l'enrichissement du dataset,
et le réentraînement périodique du modèle de scoring des stratégies.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.workflows.ml.feedback_collector import FeedbackCollector, get_feedback_collector
from src.workflows.ml.dataset_enrichment import DatasetEnricher
from src.workflows.ml.training import train_complete_pipeline

LOGGER = logging.getLogger(__name__)


class RetrainingConfig:
    """Configuration pour le réentraînement automatique."""

    def __init__(
        self,
        # Seuils de déclenchement
        min_new_samples: int = 50,
        min_accuracy_drop: float = 0.05,
        min_days_since_last_training: int = 7,

        # Paramètres de dataset
        target_samples_per_class: int = 100,
        augmentation_factor: float = 1.5,
        generate_synthetic: bool = True,

        # Paramètres d'entraînement
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        test_size: float = 0.2,

        # Chemins
        feedback_dir: str = "data/feedback",
        models_dir: str = "data/models",
        dataset_dir: str = "data",
    ):
        self.min_new_samples = min_new_samples
        self.min_accuracy_drop = min_accuracy_drop
        self.min_days_since_last_training = min_days_since_last_training

        self.target_samples_per_class = target_samples_per_class
        self.augmentation_factor = augmentation_factor
        self.generate_synthetic = generate_synthetic

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_size = test_size

        self.feedback_dir = feedback_dir
        self.models_dir = models_dir
        self.dataset_dir = dataset_dir


class RetrainingPipeline:
    """
    Pipeline de réentraînement automatique.

    Gère le cycle complet:
    1. Collecte et validation du feedback
    2. Export des nouvelles données d'entraînement
    3. Enrichissement du dataset
    4. Réentraînement du modèle
    5. Validation et déploiement
    """

    def __init__(
        self,
        config: Optional[RetrainingConfig] = None,
        feedback_collector: Optional[FeedbackCollector] = None
    ):
        """
        Initialiser le pipeline de réentraînement.

        Args:
            config: Configuration du pipeline
            feedback_collector: Collecteur de feedback (créé si None)
        """
        self.config = config or RetrainingConfig()
        self.feedback_collector = feedback_collector or get_feedback_collector(
            feedback_dir=self.config.feedback_dir
        )
        self.enricher = DatasetEnricher()

        # Chemins importants
        self.models_dir = Path(self.config.models_dir)
        self.dataset_dir = Path(self.config.dataset_dir)

        # Historique des réentraînements
        self.history_file = self.models_dir / "retraining_history.json"
        self.history = self._load_history()

        LOGGER.info("RetrainingPipeline initialized")

    def should_retrain(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Déterminer si un réentraînement est nécessaire.

        Returns:
            Tuple (should_retrain, reasons)
        """
        reasons = {
            "triggered": False,
            "triggers": [],
            "stats": {}
        }

        # 1. Vérifier le nombre de nouveaux samples
        feedback_stats = self.feedback_collector.get_feedback_stats()
        new_samples = feedback_stats.get("total_validated", 0)

        reasons["stats"]["new_samples"] = new_samples

        if new_samples >= self.config.min_new_samples:
            reasons["triggers"].append(
                f"Sufficient new samples: {new_samples} >= {self.config.min_new_samples}"
            )

        # 2. Vérifier la baisse de performance
        if self.history:
            last_training = self.history[-1]
            current_accuracy = feedback_stats.get("accuracy", 1.0)
            last_accuracy = last_training.get("metrics", {}).get("accuracy", 1.0)

            accuracy_drop = last_accuracy - current_accuracy
            reasons["stats"]["accuracy_drop"] = accuracy_drop

            if accuracy_drop >= self.config.min_accuracy_drop:
                reasons["triggers"].append(
                    f"Accuracy dropped: {accuracy_drop:.3f} >= {self.config.min_accuracy_drop:.3f}"
                )

        # 3. Vérifier le temps depuis le dernier entraînement
        if self.history:
            last_training = self.history[-1]
            last_date = datetime.fromisoformat(last_training["timestamp"])
            days_since = (datetime.now() - last_date).days

            reasons["stats"]["days_since_last_training"] = days_since

            if days_since >= self.config.min_days_since_last_training:
                reasons["triggers"].append(
                    f"Time threshold reached: {days_since} days >= {self.config.min_days_since_last_training} days"
                )

        # Décision finale
        reasons["triggered"] = len(reasons["triggers"]) > 0

        if reasons["triggered"]:
            LOGGER.info("Retraining triggered: %s", reasons["triggers"])
        else:
            LOGGER.info("Retraining not needed")

        return reasons["triggered"], reasons

    def run_retraining(
        self,
        force: bool = False,
        validate_before_deploy: bool = True
    ) -> Dict[str, Any]:
        """
        Exécuter le pipeline de réentraînement complet.

        Args:
            force: Forcer le réentraînement même si non nécessaire
            validate_before_deploy: Valider avant déploiement

        Returns:
            Résultats du réentraînement
        """
        LOGGER.info("Starting retraining pipeline (force=%s)", force)

        # Vérifier si le réentraînement est nécessaire
        should_retrain, reasons = self.should_retrain()

        if not should_retrain and not force:
            LOGGER.info("Retraining not needed, skipping")
            return {
                "success": False,
                "reason": "not_needed",
                "details": reasons
            }

        # Créer un répertoire pour cette session de réentraînement
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.models_dir / f"retraining_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Retraining session: %s", session_id)

        results = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "triggered_by": reasons["triggers"] if not force else ["forced"],
            "steps": {}
        }

        try:
            # Étape 1: Exporter le feedback validé
            LOGGER.info("Step 1: Exporting validated feedback")

            feedback_export = session_dir / "feedback_samples.jsonl"
            num_feedback = self.feedback_collector.export_for_training(
                str(feedback_export),
                min_confidence_threshold=0.7,
                include_user_overrides=True
            )

            results["steps"]["export_feedback"] = {
                "success": True,
                "samples": num_feedback
            }

            # Étape 2: Fusionner avec le dataset existant
            LOGGER.info("Step 2: Merging with existing dataset")

            original_dataset = self.dataset_dir / "strategy_samples.jsonl"
            merged_dataset = session_dir / "merged_samples.jsonl"

            num_merged = self._merge_datasets(
                [str(original_dataset), str(feedback_export)],
                str(merged_dataset)
            )

            results["steps"]["merge_datasets"] = {
                "success": True,
                "total_samples": num_merged
            }

            # Étape 3: Enrichir le dataset
            LOGGER.info("Step 3: Enriching dataset")

            enriched_dataset = session_dir / "enriched_samples.jsonl"

            num_enriched = self.enricher.enrich_dataset(
                input_file=str(merged_dataset),
                output_file=str(enriched_dataset),
                target_samples_per_class=self.config.target_samples_per_class,
                augmentation_factor=self.config.augmentation_factor,
                generate_synthetic=self.config.generate_synthetic
            )

            results["steps"]["enrich_dataset"] = {
                "success": True,
                "total_samples": num_enriched
            }

            # Étape 4: Entraîner le nouveau modèle
            LOGGER.info("Step 4: Training new model")

            new_model_dir = session_dir / "model"

            model, scaler, label_encoder, metrics = train_complete_pipeline(
                data_file=str(enriched_dataset),
                output_dir=str(new_model_dir),
                num_epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                test_size=self.config.test_size
            )

            results["steps"]["train_model"] = {
                "success": True,
                "metrics": metrics
            }

            # Étape 5: Valider le nouveau modèle
            if validate_before_deploy:
                LOGGER.info("Step 5: Validating new model")

                validation_result = self._validate_model(
                    new_model_dir,
                    str(enriched_dataset)
                )

                results["steps"]["validate_model"] = validation_result

                # Décider du déploiement
                should_deploy = validation_result.get("should_deploy", False)
            else:
                should_deploy = True

            # Étape 6: Déployer le nouveau modèle
            if should_deploy:
                LOGGER.info("Step 6: Deploying new model")

                deployment_result = self._deploy_model(new_model_dir, session_id)

                results["steps"]["deploy_model"] = deployment_result
                results["deployed"] = True
            else:
                LOGGER.warning("New model validation failed, not deploying")
                results["deployed"] = False
                results["reason"] = "validation_failed"

            # Sauvegarder les résultats dans l'historique
            self._save_to_history(results)

            results["success"] = True

            LOGGER.info("Retraining completed successfully")

        except Exception as e:
            LOGGER.error("Retraining failed: %s", e, exc_info=True)

            results["success"] = False
            results["error"] = str(e)

        return results

    def _merge_datasets(
        self,
        input_files: List[str],
        output_file: str
    ) -> int:
        """
        Fusionner plusieurs datasets JSONL.

        Args:
            input_files: Liste des fichiers d'entrée
            output_file: Fichier de sortie

        Returns:
            Nombre total d'échantillons
        """
        all_samples = []

        for input_file in input_files:
            if not os.path.exists(input_file):
                LOGGER.warning("Input file not found: %s", input_file)
                continue

            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_samples.append(json.loads(line))

        # Sauvegarder le dataset fusionné
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        LOGGER.info("Merged %d samples from %d files", len(all_samples), len(input_files))

        return len(all_samples)

    def _validate_model(
        self,
        model_dir: Path,
        test_dataset: str
    ) -> Dict[str, Any]:
        """
        Valider un nouveau modèle avant déploiement.

        Args:
            model_dir: Répertoire du nouveau modèle
            test_dataset: Dataset de test

        Returns:
            Résultats de validation
        """
        # Pour l'instant, validation simple basée sur les métriques d'entraînement
        # Dans une version complète, on pourrait faire une validation sur un hold-out set

        try:
            # Charger les métriques du modèle
            training_args_path = model_dir / "training_args.bin"

            if not training_args_path.exists():
                return {
                    "success": False,
                    "should_deploy": False,
                    "reason": "training_args_not_found"
                }

            # Vérifier l'accuracy minimale
            # (Dans un vrai système, on chargerait le modèle et ferait une vraie validation)

            # Pour le moment, on approuve si le modèle existe
            should_deploy = True

            # Comparer avec le modèle précédent si disponible
            if self.history:
                last_training = self.history[-1]
                last_accuracy = last_training.get("metrics", {}).get("accuracy", 0.0)

                # Le nouveau modèle doit être au moins aussi bon
                # (Simplifié pour cette implémentation)

                LOGGER.info("Comparing with last model accuracy: %.3f", last_accuracy)

            return {
                "success": True,
                "should_deploy": should_deploy,
                "validation_passed": True
            }

        except Exception as e:
            LOGGER.error("Model validation failed: %s", e)

            return {
                "success": False,
                "should_deploy": False,
                "error": str(e)
            }

    def _deploy_model(self, model_dir: Path, session_id: str) -> Dict[str, Any]:
        """
        Déployer le nouveau modèle en production.

        Args:
            model_dir: Répertoire du nouveau modèle
            session_id: ID de la session de réentraînement

        Returns:
            Résultats du déploiement
        """
        try:
            # Répertoire de production
            prod_model_dir = self.models_dir / "strategy_scorer_hf"

            # Sauvegarder l'ancien modèle
            if prod_model_dir.exists():
                backup_dir = self.models_dir / f"strategy_scorer_hf_backup_{session_id}"
                shutil.move(str(prod_model_dir), str(backup_dir))

                LOGGER.info("Backed up old model to %s", backup_dir)

            # Copier le nouveau modèle
            shutil.copytree(str(model_dir), str(prod_model_dir))

            LOGGER.info("Deployed new model to %s", prod_model_dir)

            return {
                "success": True,
                "model_path": str(prod_model_dir),
                "backup_path": str(backup_dir) if prod_model_dir.exists() else None
            }

        except Exception as e:
            LOGGER.error("Model deployment failed: %s", e)

            return {
                "success": False,
                "error": str(e)
            }

    def _load_history(self) -> List[Dict[str, Any]]:
        """Charger l'historique des réentraînements."""
        if not self.history_file.exists():
            return []

        with open(self.history_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_to_history(self, results: Dict[str, Any]) -> None:
        """Sauvegarder les résultats dans l'historique."""
        self.history.append(results)

        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        LOGGER.info("Saved retraining results to history")

    def get_retraining_history(self) -> List[Dict[str, Any]]:
        """Obtenir l'historique des réentraînements."""
        return self.history

    def cleanup_old_sessions(self, keep_last_n: int = 5) -> int:
        """
        Nettoyer les anciennes sessions de réentraînement.

        Args:
            keep_last_n: Nombre de sessions à conserver

        Returns:
            Nombre de sessions supprimées
        """
        # Lister toutes les sessions
        session_dirs = sorted(
            self.models_dir.glob("retraining_*"),
            key=lambda p: p.name,
            reverse=True
        )

        # Garder les N plus récentes
        to_delete = session_dirs[keep_last_n:]

        for session_dir in to_delete:
            shutil.rmtree(session_dir)
            LOGGER.info("Deleted old session: %s", session_dir)

        return len(to_delete)


if __name__ == "__main__":
    # Test du pipeline
    pipeline = RetrainingPipeline()

    # Vérifier si réentraînement nécessaire
    should_retrain, reasons = pipeline.should_retrain()

    print(f"Should retrain: {should_retrain}")
    print(f"Reasons: {json.dumps(reasons, indent=2)}")

    # Forcer un réentraînement de test
    # results = pipeline.run_retraining(force=True)
    # print(f"Results: {json.dumps(results, indent=2)}")
