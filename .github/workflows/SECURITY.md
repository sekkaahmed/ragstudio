# 🔒 Security Tools for AI/ML Projects

Ce projet utilise une **stack de sécurité complète** adaptée aux projets d'IA/ML.

---

## 📋 Outils de sécurité intégrés

### 1. **Secrets Detection** 🔑

**Outils:**
- **Gitleaks** - Détecte les secrets dans le code (API keys, tokens, passwords)
- **TruffleHog** - Scan des secrets avec vérification

**Détecte:**
- ✅ API keys OpenAI, Anthropic, HuggingFace
- ✅ AWS/GCP/Azure credentials
- ✅ Database passwords
- ✅ Private keys
- ✅ Tokens dans l'historique Git

**Quand:** Sur chaque PR

---

### 2. **SAST (Static Application Security Testing)** 🔍

**Outils:**
- **Bandit** - Security linter Python
- **Semgrep** - Pattern matching avec règles AI/ML

**Détecte:**
- ✅ SQL injection
- ✅ Hardcoded secrets
- ✅ Unsafe deserialization (pickle files)
- ✅ Path traversal
- ✅ YAML/JSON injection
- ✅ Unsafe file operations

**Règles spécifiques IA:**
- Chargement de modèles non vérifiés
- Désérialisation de données ML dangereuses
- Exécution de code dynamique

**Quand:** Sur chaque PR

---

### 3. **Dependency Security** 📦

**Outils:**
- **Safety** - Vérifie les vulnérabilités PyPI
- **Pip-audit** - Alternative à Safety
- **Snyk** - Scan des dépendances avec base de données complète

**Vérifie:**
- ✅ torch, transformers, tensorflow
- ✅ langchain, openai, anthropic
- ✅ numpy, scipy, pandas
- ✅ Toutes les dépendances transitives

**Pourquoi important pour l'IA:**
- Les librairies ML ont souvent des vulnérabilités critiques
- Supply chain attacks sur des modèles pré-entraînés
- Backdoors dans les poids de modèles

**Quand:** Sur chaque PR + release

---

### 4. **Supply Chain Security** 🔗

**Outils:**
- **Dependency Review** - GitHub native
- **SBOM Generation** - Software Bill of Materials (CycloneDX)

**Génère:**
- ✅ Liste complète des dépendances
- ✅ Versions exactes
- ✅ Licences
- ✅ Hashes de vérification

**Utilité:**
- Traçabilité complète
- Audit de conformité
- Détection de tampering

**Quand:** Sur chaque PR + release

---

### 5. **License Compliance** ⚖️

**Outils:**
- **pip-licenses** - Extraction des licences

**Vérifie:**
- ❌ Bloque GPL, AGPL, LGPL (copyleft)
- ✅ Autorise MIT, Apache, BSD

**Pourquoi critique pour l'IA:**
- Beaucoup de modèles ML ont des licences restrictives
- HuggingFace models peuvent être non-commerciales
- Evite les problèmes légaux

**Quand:** Sur chaque PR

---

### 6. **CodeQL (Advanced SAST)** 🧠

**Outil:**
- **GitHub CodeQL** - Analyse sémantique du code

**Analyse:**
- ✅ Data flow analysis
- ✅ Taint tracking
- ✅ Control flow analysis
- ✅ Security patterns

**Queries:**
- `security-extended` - Vulnérabilités étendues
- `security-and-quality` - Qualité + sécurité

**Quand:** Sur chaque PR

---

### 7. **Trivy (Vulnerability Scanner)** 🛡️

**Outil:**
- **Aqua Trivy** - Scanner universel

**Scanne:**
- ✅ Filesystem
- ✅ Dependencies Python
- ✅ OS packages
- ✅ Containers (si applicable)

**Sévérité:**
- CRITICAL, HIGH, MEDIUM

**Quand:** Sur chaque PR + release

---

## 🚨 Cas d'usage spécifiques IA/ML

### Risque 1: Model Poisoning
**Outil:** Bandit + Semgrep
**Détecte:** Chargement de modèles non vérifiés
```python
# ❌ Dangereux
model = torch.load("model.pth")  # Détecté par Bandit

# ✅ Sûr
model = torch.load("model.pth", map_location="cpu", weights_only=True)
```

### Risque 2: Data Exfiltration
**Outil:** Semgrep + CodeQL
**Détecte:** Envoi de données sensibles
```python
# ❌ Dangereux - Détecté
requests.post(UNKNOWN_URL, data=user_data)

# ✅ Sûr - Whitelisted URLs only
```

### Risque 3: Pickle Deserialization
**Outil:** Bandit
**Détecte:** Utilisation non sécurisée de pickle
```python
# ❌ Dangereux
import pickle
data = pickle.load(file)  # Détecté

# ✅ Sûr
import json
data = json.load(file)
```

### Risque 4: API Key Leaks
**Outil:** Gitleaks + TruffleHog
**Détecte:** Tokens dans le code
```python
# ❌ Dangereux - Détecté immédiatement
OPENAI_API_KEY = "sk-proj-abc123..."

# ✅ Sûr
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

---

## 🔧 Configuration requise

### Secrets GitHub (optionnels)

**SNYK_TOKEN** (recommandé):
1. Créer compte sur https://snyk.io
2. Générer token API
3. Ajouter dans GitHub Secrets

**Sans token Snyk:**
Les autres outils fonctionnent sans configuration!

---

## 📊 Rapports générés

Chaque PR génère:
- 📄 Bandit JSON report
- 📄 Semgrep JSON report
- 📄 Safety JSON report
- 📄 Pip-audit JSON report
- 📄 SBOM (CycloneDX JSON)
- 📄 Licenses JSON/Markdown
- 📄 Trivy SARIF

**Accès:**
GitHub Actions → Artifacts

---

## ✅ Best Practices

### Pour les contributeurs:

1. **Avant de commit:**
   ```bash
   # Scan local
   gitleaks detect --source .
   bandit -r src/
   ```

2. **Tester les dépendances:**
   ```bash
   safety check
   pip-audit
   ```

3. **Vérifier les licences:**
   ```bash
   pip-licenses --fail-on="GPL;AGPL"
   ```

### Pour l'admin:

1. **Review Security tab** sur GitHub régulièrement
2. **Vérifier les Dependabot alerts**
3. **Auditer le SBOM** avant chaque release
4. **Valider les licences** des nouvelles dépendances

---

## 🆘 En cas d'alerte

### Vulnérabilité CRITICAL trouvée:

1. **Ne pas merger la PR**
2. **Identifier la dépendance:** Regarder le rapport
3. **Chercher un patch:**
   ```bash
   pip install --upgrade <package>
   ```
4. **Si pas de patch:** Trouver une alternative

### Secret détecté:

1. **STOP immédiatement**
2. **Révoquer le secret** (OpenAI, AWS, etc.)
3. **Supprimer de l'historique:**
   ```bash
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch <file>' HEAD
   ```
4. **Forcer un nouveau secret**

### License non-compatible:

1. **Identifier la dépendance**
2. **Chercher une alternative** avec licence compatible
3. **Ou négocier** une licence commerciale

---

## 📚 Ressources

- [OWASP Top 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [HuggingFace Model Cards](https://huggingface.co/docs/hub/model-cards)
- [Microsoft AI Security Best Practices](https://www.microsoft.com/en-us/security/business/ai-machine-learning)

---

## 🎯 Résumé

| Risque | Outil | Quand |
|--------|-------|-------|
| Secrets in code | Gitleaks, TruffleHog | Chaque PR |
| Code vulnerabilities | Bandit, Semgrep, CodeQL | Chaque PR |
| Dependency CVEs | Safety, Pip-audit, Snyk | Chaque PR |
| Supply chain | SBOM, Dependency Review | Chaque PR |
| License issues | pip-licenses | Chaque PR |
| Container vulns | Trivy | Chaque PR + Release |

**Résultat:** Stack de sécurité de niveau **entreprise** pour projet IA! 🛡️
