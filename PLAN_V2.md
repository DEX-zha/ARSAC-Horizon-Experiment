# Plan V2 — ARSAC Horizon (refonte corrigeant l'audit `AUDIT_MATH.md`)

Objectif produit : une bibliothèque + CLI qui, pour une série temporelle (chaotique ou réelle),
fournit une **borne inférieure calibrée L(x) de l'horizon de prédictibilité** d'un modèle donné,
avec P(H ≥ L) ≥ 1−α vérifiée honnêtement, des unités physiques claires, et des diagnostics
qui détectent eux-mêmes les régimes où la borne n'est pas fiable.

Principe directeur : **chaque quantité a une unité (temps physique, pas, temps de Lyapunov)
et chaque garantie a des hypothèses vérifiées par un test.**

---

## État d'avancement (2026-07-05)

Réalisé par la campagne de correctifs (AUDIT_MATH.md) puis la campagne théorique
(docs/THEORY.md, docs/theory/*.md, studies/*.py) :

- **Phase 0** — partiellement ✅ : générateurs corrigés (DDE Mackey-Glass, tau en
  unités de temps, dt par système), tests physiques bloquants avec λ vs littérature
  (`tests/test_physics_chaos.py`). Reste : λ de référence par Benettin/QR en CI.
- **Phase 1** — ✅ : Rosenstein auto-paramétré validé ; embedding théorique MI+FNN
  (`src/horizon_embedding.py`) intégré comme défaut des estimateurs de chaos
  (4/4 systèmes plus proches de λ_lit) ; produits QR de compagnons disponibles
  (`src/horizon_ftle.py`, rangé comme feature : plus faible que l'existant).
- **Phase 2** — ✅ : définition d'horizon unifiée (erreur au pas h, K consécutifs) ;
  censure explicite (théorème de conservativité, garde `label_identified`,
  perte de Powell sous `--censored-quantile`) ; **Hmax auto en temps de
  Lyapunov** (max(3 T_λ, 1.2·ln(τ/e₀)/λ₁), borné [30, 400] + budget de données)
  et tolérance par défaut absolue 0.4·std (validé : `studies/study_lyap_hmax.py`).
  Limite documentée : plafond de coût 400 pas (Rössler et modèles quasi parfaits
  restent partiellement censurés).
- **Phase 3** — partiellement ✅ : direction de calibration réparée ; marge conforme
  correcte ; arbre en split temporel ; bootstrap de blocs pour `coverage_lb`.
  Rangé sur critères chiffrés : calibration disjointe et conforme pondéré
  (`src/horizon_conformal_beyond.py`, à réévaluer sur données réelles à dérive).
  Reste : sous-couverture Mackey-Glass au profil rapide, studentisation du bootstrap.
- **Phase 4** — ◻ : à faire (métriques en T_λ, baselines, paper regénéré).
- **Phase 5** — partiellement ✅ (MVP 2026-07-05) : API `HorizonEstimator`
  (`src/horizon_estimator.py`) pour séries utilisateur — `fit(series)` →
  `lower_bounds_`, `coverage_`, `report()` avec diagnostics (`label_identified`,
  `horizon_certified`, niveau de garantie). Reste : forecaster fourni par
  l'utilisateur, monitoring/recalibration en ligne, rationalisation CLI.

---

## Phase 0 — Vérité terrain physique (fondation, ~1 semaine)

Corrige : A1, A2, A3, A4, F3, F5.

1. **Générateurs validés** (`arsac/systems/`)
   - Un module par système avec ses paramètres canoniques ET ses constantes de référence :
     | Système | λ₁ (littérature) | dt d'échantillonnage | Temps de Lyapunov |
     |---|---|---|---|
     | Lorenz (10, 28, 8/3) | ≈ 0.906 /u.t. | 0.02 | ≈ 1.10 u.t. (55 pas) |
     | Rössler (0.2, 0.2, 5.7) | ≈ 0.071 /u.t. | 0.20 | ≈ 14 u.t. (70 pas) |
     | Mackey-Glass (τ=17) | ≈ 0.006 /u.t. | 1.0 (intégré à 0.1) | ≈ 170 u.t. (170 pas) |
     | Logistique (r=4) | ln 2 ≈ 0.693 /itération | 1 | ≈ 1.4 itérations |
   - `dt`, `tau`, `warmup` en **unités de temps physiques**, convertis en pas en interne
     (`n_delay = round(tau/dt_int)`). Plus aucun `--dt` global partagé.
   - Mackey-Glass : intégrateur DDE correct — méthode des pas, RK4 avec interpolation
     cubique de l'historique aux instants retardés intermédiaires, `dt_int ≤ 0.1`,
     sous-échantillonnage vers le `dt` d'échantillonnage.
   - Lorenz/Rössler : `solve_ivp(..., dense_output=True)` + `sol.sol(t)` sur une grille
     `np.linspace` exacte ; warmup ≥ 20 u.t. ; supprimer le flag `--integrator` fantôme.
   - Option bruit d'observation contrôlé (σ_obs) pour les études de robustesse.

2. **Exposants de Lyapunov de référence par méthode directe** (`arsac/systems/lyapunov_gt.py`)
   - On possède les équations : calculer λ₁ par Benettin/QR sur le **Jacobien exact de l'ODE**
     (et le Jacobien analytique de la logistique). C'est la vérité terrain interne,
     recalculée en CI et comparée à la littérature (tolérance ±5 %).

3. **Tests physiques bloquants** (`tests/physics/`)
   - λ₁ (Benettin) dans la fourchette de référence pour chaque système.
   - Mackey-Glass τ=17 : test « est chaotique » (λ₁ > 0.003) — aurait détecté le bug A1.
   - Statistiques d'attracteur stables entre seeds (moyenne/std/min/max de x).
   - Absence de point fixe/périodicité (test de récurrence) après warmup.

**Critère de sortie** : les 4 systèmes passent les tests physiques ; tout PR touchant
`systems/` les exécute.

---

## Phase 1 — Estimateurs de chaos corrects (~1 semaine)

Corrige : B1, B2, B3, B4.

1. **Embedding fondé sur la théorie** (`arsac/embedding.py`)
   - `lag` : premier minimum de l'information mutuelle (fallback : premier zéro de l'autocorrélation).
   - `dim` : faux plus proches voisins (FNN < 1 %).
   - La recherche par MSE de validation peut rester pour le *modèle*, mais les estimateurs
     de chaos utilisent l'embedding théorique, pas celui qui minimise le MSE.

2. **Rosenstein réparé** (`arsac/lyapunov.py`)
   - Fenêtre de Theiler = max(période moyenne estimée, temps de décorrélation), en pas.
   - Suivi de la divergence sur ≥ 2 temps de Lyapunov (calculé depuis la référence Phase 0
     pour les systèmes connus ; sinon auto-adaptatif).
   - Détection automatique de la région linéaire (plus long segment où R² ≥ 0.99 sur ⟨ln d(k)⟩)
     au lieu du fit fixe 1→10.
   - Sortie : `lambda_per_time` (unité explicite), `lyapunov_time = 1/λ`, diagnostic de fit
     (R², longueur du segment). **Renommage partout** : plus de `lyap_time` ambigu.
   - Validation croisée en CI contre le Benettin de Phase 0 (±20 %).

3. **Croissance locale par produits de Jacobiens** (`arsac/local_growth.py`)
   - Dynamique embarquée = matrice compagnon C(x) avec dernière ligne ∇f(x).
   - Exposant local sur k pas : produits C(x_t)·…·C(x_{t+k}) avec re-orthonormalisation QR.
   - Modèles Torch en `eval()` strict (backward RNN via `torch.backends.cudnn.flags(enabled=False)`).
   - `estimate_expansion_quantile` / `estimate_error_growth` : fit de pente sur la partie
     linéaire de ⟨ln d(k)⟩ / ⟨ln e(h)⟩, troncature avant saturation (seuil = fraction du
     diamètre de l'attracteur), plus de moyenne de ratios bruts.

**Critère de sortie** : sur Lorenz, λ_Rosenstein, λ_Benettin et l'exposant des produits QR
du modèle appris concordent à ±20 % ; `chaos_quick_test.py` (réécrit sur ces estimateurs)
classe correctement les 4 systèmes + un AR(1) témoin non chaotique.

---

## Phase 2 — Une seule définition d'horizon, censure gérée (~1 semaine)

Corrige : C1, C2, C3, D2.

1. **Définition unique** (`arsac/horizon_label.py`)
   - `H_w = min{ h : |ŷ_{t+h} − y_{t+h}| > τ pour K pas consécutifs }`, erreur **au pas h**
     (pas de RMSE cumulé), K=2 par défaut. Utilisée partout : labels, horizon global
     (= médiane des H_w), calibration probabiliste. Suppression de `window_horizons`
     et du cumul de `build_horizon_dataset`.
   - Tolérance : `τ = ρ · std(attracteur)` avec ρ ∈ {0.2, 0.4} (convention « valid time »
     de la littérature). Suppression de la tolérance relative à l'erreur un-pas par fenêtre.
   - `Hmax ≥ 3 temps de Lyapunov` (en pas, par système, depuis Phase 0).

2. **Censure explicite**
   - Label = couple `(H_w, censored)` ; `censored = (H_w == Hmax)`.
   - Régression quantile avec pinball censuré (la perte ne pénalise pas q̂ > Hmax pour
     les censurés) ; scores conformes calculés uniquement là où la comparaison est
     informative, sinon comptés conservativement.
   - Le run est **invalidé** (exit code ≠ 0, pas un warning) si p_sat > 0.5 :
     « Hmax trop court ou système non chaotique ».

3. **Horizon théorique proprement étiqueté**
   - `horizon_lyapunov = (1/λ₁) ln(τ / e₀)` conservé mais exporté comme *référence
     heuristique* ; `horizon_model` (récurrence g·e+δ) rétrogradé en diagnostic,
     plus jamais présenté comme borne (D2).

**Critère de sortie** : test unitaire « une seule définition » (les trois anciens chemins
donnent le même H_w sur un cas synthétique) ; sur Lorenz avec Hmax = 3 T_λ, p_sat < 10 %.

---

## Phase 3 — Calibration honnête (~2 semaines)

Corrige : D1, E1, E2, E3, E4.

1. **Chemin probabiliste : direction réparée ou suppression**
   - Recommandation : **supprimer** `bound_mode=probabilistic` (redondant et irréparable
     en l'état) et garder la voie conforme unique. Si conservé : `scale = quantile_α(h_real/h_model)`
     plafonné à 1, hit = `h_real ≥ L`, suppression du plancher 1.0.

2. **Conformal adapté aux séries temporelles** (`arsac/conformal.py`)
   - **Fenêtres de calibration disjointes** : stride ≥ (dim−1)·lag + Hmax. C'est la correction
     principale ; elle réduit n mais rend les scores ≈ indépendants.
   - Marge one-sided : `L(x) = q̂_α(x) − c·ŝ(x)`, c au rang ⌈(n+1)(1−α)⌉ (déjà correct),
     calculée **une seule fois** sur un jeu de calibration vierge.
   - Suppression de la cascade offset → marge → guard → debias. Si un recentrage est
     souhaité : split de la calibration en deux moitiés temporelles (offset sur C₁,
     marge sur C₂). Idem arbre conforme : structure sur C₁, quantiles de feuilles sur C₂.
   - Pool CV : ne jamais concaténer train et calib (discontinuité + fuite in-sample).
     Soit vrai CV+ (le test est prédit par l'agrégat des modèles de folds), soit split
     simple — mais correct.
   - Option « beyond exchangeability » (Barber et al. 2023) : poids décroissants avec
     l'ancienneté + terme de couverture perdue borné, pour les séries à dérive.

3. **Incertitude de la couverture rapportée**
   - Remplacer Wilson par un **bootstrap par blocs circulaires** (taille de bloc ≥ portée
     d'autocorrélation des hits, estimée par Politis-White) → `coverage_lb` honnête.
   - Rapporter aussi la couverture par bloc temporel (dérive visible).

4. **Évaluation train/calib/test avec gaps**
   - Insérer un gap ≥ (dim−1)·lag + Hmax entre chaque split pour éliminer la fuite
     d'information aux jonctions.

**Critère de sortie** : expérience de validation à grande échelle — 50 seeds Lorenz :
la couverture empirique test tombe dans [1−α−2σ_boot, 1] pour ≥ 95 % des seeds,
sans guard ni debias ; tightness médiane ≥ 0.5.

---

## Phase 4 — Évaluation scientifique et paper (~1 semaine)

Corrige : E4, F2, F4 + rend les résultats comparables.

1. **Métriques en temps de Lyapunov** : tous les horizons exportés en {pas, u.t., T_λ}.
   Les systèmes deviennent comparables entre eux (un horizon de 1.5 T_λ sur Rössler
   et Lorenz veut dire la même chose).
2. **Baselines** obligatoires : (a) horizon Lyapunov théorique (Phase 0), (b) climatologie
   (prédire la moyenne), (c) borne naïve constante = quantile α des H_w de calibration.
   La valeur ajoutée de L(x) = gain de tightness vs (c) à couverture égale.
3. **Protocoles de stress** : bruit d'observation croissant, changement de régime
   (ρ de Lorenz 28 → 24.5), calibration sur un régime / test sur l'autre → montrer
   que les diagnostics détectent la perte de couverture.
4. **README/paper regénérés depuis les runs** : script qui produit tables et figures
   depuis les CSV (aucun chiffre copié à la main) ; `paper.tex` réécrit avec un LaTeX
   valide ; retirer toute claim « ≥ 1−α garanti » au profit de « couverture empirique
   avec IC par bootstrap de blocs, hypothèses documentées ».

---

## Phase 5 — En faire un outil réellement utile (~2 semaines)

1. **API bibliothèque** (le cœur de la valeur) :
   ```python
   from arsac import HorizonEstimator
   est = HorizonEstimator(model=my_forecaster, alpha=0.1, tolerance=0.4)  # ×std
   est.fit(series_train, series_calib)          # accepte des données réelles
   L = est.predict(x_windows)                    # borne inférieure par fenêtre
   report = est.diagnostics()                    # p_sat, couverture, dérive, n_eff
   ```
   - Modèle-agnostique via un protocole `predict(x) -> y` (sklearn, torch, fonction).
   - Fonctionne sur données utilisateur (pas seulement les 4 systèmes jouets) ;
     les estimateurs de chaos deviennent des diagnostics optionnels.
2. **Monitoring en ligne** : couverture glissante sur les données récentes,
   alerte + recalibration quand elle sort de l'IC (c'est le cas d'usage « deployment
   & safety checks » que le README promet sans le livrer).
3. **Rationalisation** : les ~100 flags CLI deviennent 4 dataclasses de config
   (`SystemConfig`, `ModelConfig`, `LabelConfig`, `CalibrationConfig`) sérialisées
   en YAML ; wandb sort du cœur (callback optionnel) ; suppression du code mort (F1)
   et des chemins redondants ; packaging `pyproject.toml`, CI avec les tests physiques.

---

## Ordre d'exécution et dépendances

```
Phase 0 (générateurs + λ de référence)
   └→ Phase 1 (estimateurs)      — dépend des λ de référence pour se valider
        └→ Phase 2 (labels)      — dépend des T_λ pour fixer Hmax
             └→ Phase 3 (calibration) — dépend de labels propres
                  └→ Phase 4 (éval/paper)
                       └→ Phase 5 (produit)
```

Budget total estimé : ~7-8 semaines à temps plein, chaque phase livrable et testable seule.

## Ce qui est volontairement abandonné

- `bound_mode=probabilistic` (garantie inversée, redondant) → diagnostic seulement.
- Guard + debias + offset empilés → une marge conforme propre sur données disjointes.
- `--integrator` pour Lorenz/Rössler, `--dt` global, tolérance relative par fenêtre.
- Le verdict binaire de `chaos_quick_test.py` → remplacé par un rapport chiffré
  (λ ± IC, comparaison à la référence).
