# CI/CD Workflows

Ce projet utilise 2 pipelines GitHub Actions:

## 1. PR Validation (Contributeurs)

**Fichier:** `pr-validation.yml`

**Déclenchement:**
- Pull Requests vers `main`
- Push sur branches (sauf `main`)

**Actions:**
- ✅ Lint (Ruff, Black, Flake8)
- ✅ Tests (pytest sur Python 3.10, 3.11, 3.12)
- ✅ Security scan (Bandit, Safety, Trivy)
- ✅ Build package

**Pas de déploiement** - uniquement validation

---

## 2. Release & Deploy (Admin)

**Fichier:** `release-deploy.yml`

**Déclenchement:**
- Tags `v*.*.*` (ex: v0.1.2)
- Manuel via workflow_dispatch

**Actions:**
1. ✅ Validation complète
2. 🏗️ Build package
3. 🧪 Publish to TestPyPI (manuel)
4. 🚀 Publish to PyPI (automatique sur tag)
5. 📦 Create GitHub Release
6. ✅ Post-deployment tests

---

## Configuration requise

### Secrets GitHub (Settings → Secrets and variables → Actions)

**Pour PyPI:**
1. Aller sur https://pypi.org/manage/account/token/
2. Créer un token API
3. Configurer les environments dans GitHub:
   - Environment: `pypi`
   - Secret: Configuré automatiquement avec Trusted Publisher

**Pour TestPyPI:**
1. Aller sur https://test.pypi.org/manage/account/token/
2. Créer un token API
3. Environment: `testpypi`

### Trusted Publisher (recommandé)

Au lieu de secrets, configurez Trusted Publisher:

**PyPI:**
1. Aller sur: https://pypi.org/manage/project/ragctl/settings/publishing/
2. Add publisher:
   - Owner: `horiz-data`
   - Repository: `ragstudio`
   - Workflow: `release-deploy.yml`
   - Environment: `pypi`

**TestPyPI:**
Même chose sur: https://test.pypi.org/manage/project/ragctl/settings/publishing/

---

## Utilisation

### Pour les contributeurs

```bash
# Créer une branche
git checkout -b feature/my-feature

# Faire vos modifications
git add .
git commit -m "feat: my feature"

# Pousser et créer une PR
git push origin feature/my-feature
```

Le workflow `pr-validation.yml` se déclenche automatiquement.

### Pour l'admin (release)

**Option 1: Automatique (recommandé)**
```bash
# Créer et pousser un tag
git tag v0.1.3
git push origin v0.1.3
```

Le workflow `release-deploy.yml` se déclenche et publie sur PyPI automatiquement.

**Option 2: Manuel**
1. Aller sur GitHub Actions
2. Choisir "Release & Deploy (Admin)"
3. Cliquer "Run workflow"
4. Choisir l'environnement (testpypi ou pypi)

---

## Protection de la branche main

Avec la branche `main` protégée, le workflow garantit:
- ✅ Toutes les PRs passent les tests
- ✅ Aucun code non testé en production
- ✅ Sécurité vérifiée automatiquement

---

## Monitoring

Vérifier les workflows sur:
```
https://github.com/horiz-data/ragstudio/actions
```
