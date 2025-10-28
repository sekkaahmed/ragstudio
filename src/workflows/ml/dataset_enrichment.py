"""
Module d'enrichissement automatique du dataset d'entraînement.

Génère de nouveaux exemples synthétiques et augmente les données
existantes pour améliorer la couverture du modèle.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class DatasetEnricher:
    """
    Enrichisseur de dataset pour augmenter les données d'entraînement.

    Génère des exemples synthétiques, augmente les données existantes,
    et équilibre les classes pour améliorer la performance du modèle.
    """

    def __init__(self, seed: int = 42):
        """
        Initialiser l'enrichisseur.

        Args:
            seed: Graine aléatoire pour reproductibilité
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Templates par stratégie
        self.strategy_templates = self._load_strategy_templates()

        LOGGER.info("DatasetEnricher initialized (seed=%d)", seed)

    def _load_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Charger les templates pour chaque stratégie."""

        return {
            "recursive": {
                "description": "Documents courts, peu structurés, texte continu",
                "characteristics": {
                    "length_tokens_range": (20, 100),
                    "hierarchy_depth": 1,
                    "has_tables": False,
                    "has_headings": False,
                    "structure_score_range": (0.0, 0.3),
                },
                "text_patterns": [
                    "court_article", "breve", "paragraphe_simple", "texte_continu"
                ]
            },
            "semantic": {
                "description": "Documents moyens, bien structurés, sémantiquement riches",
                "characteristics": {
                    "length_tokens_range": (100, 300),
                    "hierarchy_depth_range": (2, 3),
                    "has_headings": True,
                    "structure_score_range": (0.6, 0.9),
                },
                "text_patterns": [
                    "fiche_technique", "article_structure", "documentation_produit"
                ]
            },
            "parent_child": {
                "description": "Documents longs, hiérarchie complexe, sections dépendantes",
                "characteristics": {
                    "length_tokens_range": (200, 600),
                    "hierarchy_depth_range": (3, 5),
                    "has_headings": True,
                    "structure_score_range": (0.7, 0.95),
                },
                "text_patterns": [
                    "rapport_technique", "guide_complet", "manuel_utilisateur"
                ]
            },
            "late": {
                "description": "Documents avec code/tables, structure technique",
                "characteristics": {
                    "length_tokens_range": (150, 400),
                    "hierarchy_depth_range": (2, 4),
                    "has_tables": True,
                    "structure_score_range": (0.8, 0.95),
                },
                "text_patterns": [
                    "api_documentation", "config_technique", "guide_developpeur"
                ]
            }
        }

    def enrich_dataset(
        self,
        input_file: str,
        output_file: str,
        target_samples_per_class: int = 50,
        augmentation_factor: float = 2.0,
        generate_synthetic: bool = True
    ) -> int:
        """
        Enrichir un dataset existant.

        Args:
            input_file: Fichier JSONL d'entrée
            output_file: Fichier JSONL de sortie enrichi
            target_samples_per_class: Nombre cible d'exemples par stratégie
            augmentation_factor: Facteur de multiplication par augmentation
            generate_synthetic: Générer des exemples synthétiques

        Returns:
            Nombre total d'exemples dans le dataset enrichi
        """
        LOGGER.info("Enriching dataset from %s to %s", input_file, output_file)

        # Charger le dataset original
        original_samples = self._load_jsonl(input_file)

        LOGGER.info("Loaded %d original samples", len(original_samples))

        # Analyser la distribution des classes
        class_distribution = self._analyze_class_distribution(original_samples)

        LOGGER.info("Class distribution: %s", class_distribution)

        # Collection des exemples enrichis
        enriched_samples = list(original_samples)  # Commencer avec les originaux

        # 1. Augmentation des données existantes
        if augmentation_factor > 1.0:
            augmented = self._augment_existing_samples(
                original_samples, augmentation_factor
            )
            enriched_samples.extend(augmented)
            LOGGER.info("Added %d augmented samples", len(augmented))

        # 2. Génération d'exemples synthétiques pour équilibrer
        if generate_synthetic:
            synthetic = self._generate_synthetic_samples(
                class_distribution, target_samples_per_class
            )
            enriched_samples.extend(synthetic)
            LOGGER.info("Added %d synthetic samples", len(synthetic))

        # 3. Sauvegarder le dataset enrichi
        self._save_jsonl(enriched_samples, output_file)

        LOGGER.info("Enriched dataset saved: %d total samples", len(enriched_samples))

        # Afficher la nouvelle distribution
        new_distribution = self._analyze_class_distribution(enriched_samples)
        LOGGER.info("New class distribution: %s", new_distribution)

        return len(enriched_samples)

    def _augment_existing_samples(
        self,
        samples: List[Dict[str, Any]],
        factor: float
    ) -> List[Dict[str, Any]]:
        """
        Augmenter les exemples existants par variation.

        Args:
            samples: Exemples originaux
            factor: Facteur de multiplication

        Returns:
            Liste d'exemples augmentés
        """
        augmented = []
        num_to_generate = int(len(samples) * (factor - 1))

        LOGGER.info("Generating %d augmented samples", num_to_generate)

        for _ in range(num_to_generate):
            # Sélectionner un exemple au hasard
            original = random.choice(samples)

            # Créer une variation
            augmented_sample = self._create_variation(original)

            if augmented_sample:
                augmented.append(augmented_sample)

        return augmented

    def _create_variation(self, original: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Créer une variation d'un exemple existant.

        Args:
            original: Exemple original

        Returns:
            Exemple varié ou None
        """
        variation = dict(original)

        # Générer un nouvel ID
        variation["doc_id"] = f"{original['doc_id']}_var_{random.randint(1000, 9999)}"

        # Variations du texte (légères modifications)
        text = original["text"]

        # Augmentation textuelle basique
        augmentations = [
            self._add_noise_to_text,
            self._paraphrase_simple,
            self._adjust_formatting,
        ]

        augmentation_fn = random.choice(augmentations)
        variation["text"] = augmentation_fn(text)

        # Variations des features (dans une marge de ±10%)
        if "length_tokens" in variation:
            variation["length_tokens"] = self._add_noise_to_metric(
                variation["length_tokens"], 0.1
            )

        if "structure_score" in variation:
            variation["structure_score"] = self._add_noise_to_metric(
                variation["structure_score"], 0.1, min_val=0.0, max_val=1.0
            )

        # Ajouter métadonnée
        variation["_augmented"] = True

        return variation

    def _add_noise_to_text(self, text: str) -> str:
        """Ajouter du bruit léger au texte."""
        # Ajouter/retirer espaces aléatoires
        words = text.split()

        if len(words) > 5:
            # Retirer quelques mots au hasard (max 10%)
            num_to_remove = max(1, len(words) // 10)
            for _ in range(num_to_remove):
                if words:
                    words.pop(random.randint(0, len(words) - 1))

        return " ".join(words)

    def _paraphrase_simple(self, text: str) -> str:
        """Paraphraser simplement le texte."""
        # Remplacements simples
        replacements = {
            "est": "se trouve être",
            "a": "possède",
            "avec": "comportant",
            "pour": "afin de",
            "très": "extrêmement",
            "bon": "excellent",
            "grand": "important",
        }

        paraphrased = text
        for old, new in replacements.items():
            if random.random() < 0.3:  # 30% de chance de remplacement
                paraphrased = re.sub(
                    r'\b' + old + r'\b', new, paraphrased, count=1
                )

        return paraphrased

    def _adjust_formatting(self, text: str) -> str:
        """Ajuster le formatage du texte."""
        # Variations de ponctuation
        variations = [
            lambda t: t.replace(".", ". \n"),  # Ajouter retours ligne
            lambda t: t.replace("\n\n", "\n"),  # Réduire espaces
            lambda t: t.upper() if len(t) < 50 else t,  # Mettre en majuscules si court
        ]

        variation_fn = random.choice(variations)
        return variation_fn(text)

    def _add_noise_to_metric(
        self,
        value: float,
        noise_factor: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> float:
        """Ajouter du bruit à une métrique."""
        noise = value * noise_factor * (2 * random.random() - 1)
        noisy_value = value + noise

        if min_val is not None:
            noisy_value = max(min_val, noisy_value)

        if max_val is not None:
            noisy_value = min(max_val, noisy_value)

        return noisy_value

    def _generate_synthetic_samples(
        self,
        current_distribution: Dict[str, int],
        target_per_class: int
    ) -> List[Dict[str, Any]]:
        """
        Générer des exemples synthétiques pour équilibrer les classes.

        Args:
            current_distribution: Distribution actuelle des classes
            target_per_class: Nombre cible d'exemples par classe

        Returns:
            Liste d'exemples synthétiques
        """
        synthetic_samples = []

        for strategy, current_count in current_distribution.items():
            # Calculer le nombre d'exemples à générer
            num_to_generate = max(0, target_per_class - current_count)

            if num_to_generate > 0:
                LOGGER.info("Generating %d synthetic samples for %s",
                           num_to_generate, strategy)

                for i in range(num_to_generate):
                    sample = self._generate_synthetic_sample(strategy, i)
                    synthetic_samples.append(sample)

        return synthetic_samples

    def _generate_synthetic_sample(
        self,
        strategy: str,
        index: int
    ) -> Dict[str, Any]:
        """
        Générer un exemple synthétique pour une stratégie.

        Args:
            strategy: Stratégie cible
            index: Index de l'exemple

        Returns:
            Exemple synthétique
        """
        template = self.strategy_templates[strategy]
        chars = template["characteristics"]

        # Générer un ID unique
        doc_id = f"synthetic_{strategy}_{index:04d}"

        # Générer les caractéristiques
        length_tokens = random.randint(*chars.get("length_tokens_range", (50, 200)))

        hierarchy_depth = chars.get("hierarchy_depth")
        if hierarchy_depth is None:
            hierarchy_depth = random.randint(*chars.get("hierarchy_depth_range", (1, 3)))

        has_tables = chars.get("has_tables", random.choice([True, False]))
        has_headings = chars.get("has_headings", random.choice([True, False]))
        has_lists = random.choice([True, False])

        structure_score = random.uniform(*chars.get("structure_score_range", (0.0, 1.0)))

        lang = random.choice(["fr", "en"])

        # Générer le texte synthétique
        text_pattern = random.choice(template["text_patterns"])
        text = self._generate_text_from_pattern(
            text_pattern, length_tokens, has_headings, has_tables
        )

        # Créer l'exemple
        sample = {
            "doc_id": doc_id,
            "text": text,
            "length_tokens": length_tokens,
            "hierarchy_depth": hierarchy_depth,
            "has_tables": has_tables,
            "has_headings": has_headings,
            "has_lists": has_lists,
            "structure_score": structure_score,
            "lang": lang,
            "type": text_pattern,
            "best_strategy": strategy,
            "_synthetic": True
        }

        return sample

    def _generate_text_from_pattern(
        self,
        pattern: str,
        length_tokens: int,
        has_headings: bool,
        has_tables: bool
    ) -> str:
        """
        Générer un texte synthétique basé sur un pattern.

        Args:
            pattern: Pattern de texte
            length_tokens: Longueur cible en tokens
            has_headings: Inclure des titres
            has_tables: Inclure des tables

        Returns:
            Texte synthétique
        """
        text_parts = []

        # Titre principal
        if has_headings:
            text_parts.append(f"# {self._generate_heading(pattern)}\n\n")

        # Introduction
        text_parts.append(self._generate_paragraph(pattern, length_tokens // 3))

        # Sections
        if has_headings:
            num_sections = max(1, length_tokens // 100)
            for i in range(num_sections):
                text_parts.append(f"\n## {self._generate_heading(pattern, i+1)}\n\n")
                text_parts.append(self._generate_paragraph(pattern, length_tokens // (num_sections + 1)))

        # Table si nécessaire
        if has_tables:
            text_parts.append("\n" + self._generate_table() + "\n")

        # Conclusion
        text_parts.append("\n" + self._generate_paragraph(pattern, length_tokens // 4))

        return "".join(text_parts)

    def _generate_heading(self, pattern: str, index: int = 0) -> str:
        """Générer un titre."""
        headings = {
            "fiche_technique": [
                "Caractéristiques Techniques", "Spécifications", "Performance",
                "Équipements", "Tarifs"
            ],
            "rapport_technique": [
                "Introduction", "Analyse", "Résultats", "Recommandations", "Conclusion"
            ],
            "api_documentation": [
                "Endpoints", "Authentication", "Examples", "Error Codes", "Rate Limiting"
            ],
            "article_structure": [
                "Introduction", "Contexte", "Analyse", "Perspectives", "Conclusion"
            ]
        }

        pattern_headings = headings.get(pattern, ["Section", "Détails", "Information"])

        if index < len(pattern_headings):
            return pattern_headings[index]
        else:
            return f"Section {index + 1}"

    def _generate_paragraph(self, pattern: str, target_tokens: int) -> str:
        """Générer un paragraphe synthétique."""
        sentences = {
            "fiche_technique": [
                "Le véhicule est équipé d'un moteur performant.",
                "Les dimensions sont optimisées pour le confort.",
                "La consommation est réduite grâce aux technologies modernes.",
                "Les équipements de sécurité sont de série.",
            ],
            "rapport_technique": [
                "L'analyse des données montre une tendance positive.",
                "Les performances dépassent les objectifs fixés.",
                "Les recommandations visent à améliorer l'efficacité.",
                "Les résultats sont conformes aux attentes.",
            ],
            "api_documentation": [
                "The API uses REST principles for resource access.",
                "Authentication is required for all endpoints.",
                "Rate limiting prevents abuse of the service.",
                "Error codes follow HTTP standard conventions.",
            ],
        }

        pattern_sentences = sentences.get(
            pattern,
            [
                "Ce document présente les informations essentielles.",
                "Les détails sont fournis dans les sections suivantes.",
                "Les données sont issues d'analyses approfondies.",
            ]
        )

        # Générer le paragraphe
        paragraph = []
        tokens_count = 0

        while tokens_count < target_tokens:
            sentence = random.choice(pattern_sentences)
            paragraph.append(sentence)
            tokens_count += len(sentence.split())

        return " ".join(paragraph)

    def _generate_table(self) -> str:
        """Générer une table Markdown synthétique."""
        headers = ["Paramètre", "Valeur", "Description"]
        rows = [
            ["CPU", "Intel i7", "Processeur haute performance"],
            ["RAM", "16 GB", "Mémoire vive"],
            ["Storage", "512 GB SSD", "Stockage rapide"],
        ]

        table_lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["-------"] * len(headers)) + "|",
        ]

        for row in rows:
            table_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(table_lines)

    def _analyze_class_distribution(
        self,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Analyser la distribution des classes."""
        distribution = {}

        for sample in samples:
            strategy = sample.get("best_strategy", "unknown")
            distribution[strategy] = distribution.get(strategy, 0) + 1

        return distribution

    def _load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Charger un fichier JSONL."""
        samples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        return samples

    def _save_jsonl(self, samples: List[Dict[str, Any]], file_path: str) -> None:
        """Sauvegarder un fichier JSONL."""
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # Test de l'enrichisseur
    enricher = DatasetEnricher()

    # Enrichir le dataset existant
    num_samples = enricher.enrich_dataset(
        input_file="data/strategy_samples.jsonl",
        output_file="data/strategy_samples_enriched.jsonl",
        target_samples_per_class=20,
        augmentation_factor=1.5,
        generate_synthetic=True
    )

    print(f"Dataset enrichi: {num_samples} exemples au total")
