# Le plancher physique de prédictibilité : atteint et prouvé par mesure directe

Étude `studies/study_chaos_floor.py`, runs des 2026-07-05 (CSV : `outputs/chaos_floor.csv`).
Résultat principal : **un forecaster appris depuis les données atteint le plancher
physique de prédictibilité de Lorenz**, prouvé par comparaison appariée contre la
vraie dynamique perturbée, avec critères pré-enregistrés et réplication.

---

## 1. Le dispositif de preuve (pas d'hypothèses)

**Problème** : affirmer qu'un modèle « touche la limite du chaos » exige une
définition opérationnelle de la limite. La nôtre est mesurée, pas postulée :

**Jumeau one-shot.** Pour chaque fenêtre de test, on intègre la *vraie* dynamique
de Lorenz depuis l'état complet exact de la fenêtre, perturbé d'une erreur de la
taille de l'erreur un-pas du modèle (e₀_w, appariée fenêtre par fenêtre). Son
horizon H_twin (même observable x, même tolérance τ=0.4σ, même règle K=2) est
**le plancher physique à ce niveau d'erreur** : aucun prédicteur ne peut battre
systématiquement la vraie dynamique à erreur initiale égale.

**Jumeau à injection.** Un forecaster pas-à-pas réinjecte une erreur ~e₀ à
*chaque* pas. Le second jumeau injecte un bruit isotrope de taille e₀ après
chaque pas de la vraie dynamique : c'est la borne équitable pour tout
prédicteur séquentiel. Théorie dérivée (sommation géométrique des injections
amplifiées) : **H_inj ≈ ln(τ·λ₁·dt/e₀)/(λ₁·dt)** pas.

**Critères pré-enregistrés** : plancher touché ⟺ médiane de ρ_w = H_modèle_w/H_jumeau_w ≥ 0.8.

**Contrôles positifs** : le taux effectif du jumeau one-shot doit être λ₁ —
mesuré : R_twin = Λ_eff/λ₁ ∈ {0.96, 0.97, 1.02, 1.03} sur tous les runs.
L'appareil mesure bien ce qu'il prétend. Les deux membres de chaque paire de
jumeaux utilisent le même intégrateur (RK4 pas fixe h=dt/4, distance mutuelle),
donc l'erreur d'intégration est de mode commun.

## 2. Résultats (Lorenz, dt=0.01, τ=0.4σ, K=2, 250 fenêtres/run)

### La transition model-limited → chaos-limited (courbe mesurée)

| Forecaster (accès) | e₀ médian (σ) | H médian (T_λ) | ρ one-shot | R = Λ_eff/λ₁ |
|---|---|---|---|---|
| Linéaire AR (observable) | ~1e-2 | 0.2 | ~0.06 | 41 |
| MLP (observable) | — | ~0.6 | — | 8.8 |
| NG-RC degré 3 (observable) | 1.1e-6 | 1.8 | ≤ 0.16 | 7.2 |
| NG-RC degré 3 (état complet) | 1.8e-7 | 6.5 | 0.49 | 2.25 |
| NG-RC degré 5 (état complet) | 1.2e-10 | 15.5 | 0.68 | 1.45 |
| **NG-RC degré 6 (état complet)** | **1.9e-10** | **18.8** | **0.83** | **1.15** |
| *Plancher (jumeau one-shot)* | *(apparié)* | *22.4–22.7* | *1* | *0.96–1.02* |

### Verdicts pré-enregistrés, deux seeds indépendants

| | Seed 0 | Seed 1 (CI décalée, fenêtres/directions nouvelles) |
|---|---|---|
| ρ one-shot (seuil 0.8) | **0.834** [IQR 0.767–0.911] | **0.862** [0.805–0.922] |
| ρ injection (seuil 0.8) | **0.911** [0.843–1.001] | **0.959** [0.872–1.002] |
| R modèle | 1.15 | 1.21 |
| H modèle | 2080 pas = 18.8 T_λ | 2001 pas = 18.1 T_λ |
| Verdicts | FLOOR TOUCHED ×2 | FLOOR TOUCHED ×2 |

### La loi du plancher d'injection, dérivée puis vérifiée

Mesure vs formule H_inj = ln(τ·λ₁·dt/e₀)/(λ₁·dt) : rapport 1.12–1.21 sur tous
les runs (biais systématique dans le sens attendu : le bruit isotrope se projette
partiellement hors de la direction instable, facteur géométrique O(1)).
Conséquence quantitative : le coût irréductible d'un prédicteur pas-à-pas vs le
one-shot est ln(1/(λ₁dt))/ln(τ/e₀) — il s'annule **logarithmiquement** quand
e₀→0, ce que la progression degré 3→6 vérifie (ρ : 0.49 → 0.68 → 0.83).

## 3. Ce que cela établit

1. **Existence constructive** : un modèle appris (ridge polynomial degré 6 sur
   l'état, 84 coefficients, aucune connaissance des équations) prévoit Lorenz
   pendant ~18–19 temps de Lyapunov et son erreur croît au taux de Lyapunov
   (R=1.15–1.21) — il a *épuisé* le contenu prédictif du système au sens du
   jumeau, à 14–17 % près.
2. **Méthodologie de preuve réutilisable** : le couple (jumeau apparié,
   diagnostic R avec contrôle positif) donne un critère opérationnel et
   falsifiable de « saturation de la prédictibilité » applicable à tout système
   dont on peut simuler la dynamique — et R seul reste calculable sur données
   réelles (λ₁ estimé par Rosenstein).
3. **Décomposition quantitative de la perte de prédictibilité** :
   H_modèle ≈ H_chaos(e₀) − coût d'injection(e₀, λdt) − inefficacité systématique,
   chaque terme mesuré séparément.

## 4. Limites honnêtes et falsifiabilité

- **Un seul système** (Lorenz), données propres, sans bruit d'observation.
- **Accès à l'état complet requis** : en observable seul, le même cadre plafonne
  à R≈7 (écart ×7 au plancher) — l'embedding de retards dégrade le Jacobien
  appris. Question ouverte falsifiable n°1 : le plancher est-il atteignable
  depuis l'observable seul (reconstruction d'état + modèle) ?
- **Famille polynomiale sur champ polynomial** : le champ de Lorenz est
  quadratique — le ridge polynomial peut capturer flot ET Jacobien. Question
  falsifiable n°2 : Rössler (quadratique aussi) devrait suivre ; Mackey-Glass
  (retard, non polynomial) est le vrai test de généralité.
- τ et K fixés (0.4σ, K=2) ; robustesse en τ non balayée à ce stade.
- Le seuil 0.8 est pré-enregistré mais conventionnel ; les IQR encadrent 0.8.

## 5. Prochaines étapes vers la revendication forte

1. Répliquer sur Rössler et Mackey-Glass (état complet) — même protocole.
2. Balayer τ ∈ {0.2, 0.8} et le bruit d'observation (le plancher DOIT descendre
   avec le bruit selon la même loi d'injection — prédiction testable).
3. Observable-only : état reconstruit (retards + carte inverse apprise) → R ?
4. Dériver le facteur géométrique O(1) de la loi d'injection (projection
   isotrope → direction instable + transitoire d'alignement).

---

## 6. Extension au bruit d'observation : le plancher atteignable (2026-07-05, suite)

Étude `studies/study_noisy_floor.py` (CSV : `outputs/noisy_floor.csv`,
copie : `docs/theory/data/noisy_floor.csv`). Trois affirmations pré-enregistrées,
testées sur Lorenz à bruit synthétique CONNU σ ∈ {1e-3, 1e-2, 3e-2} std :

| σ | C1 : σ̂/σ ∈ [0.5, 2] | C2 : plancher/loi ∈ [0.8, 1.3] | C3 : ρ_noisy (borne OK) |
|---|---|---|---|
| 1e-3 | 1.84 ✓ | 0.99 ✓ | 0.967 → TOUCHÉ |
| 1e-2 | 1.84 ✓ | 0.91 ✓ | 0.787 → NEAR |
| 3e-2 | 1.59 ✓ | 0.88 ✓ | 0.633 → NEAR |

Conséquences :
1. **La loi one-shot se transporte au bruit** : le plancher atteignable sous
   bruit d'observation σ est H_reach ≈ ln(τ/σ)/(λ₁·dt), vérifié à 12 % près.
   C'est le pont qui rend le diagnostic honnête sur données réelles, où les
   jumeaux sont impossibles (pas de vraie dynamique disponible).
2. **σ̂ par résidus localement linéaires** (src/horizon_noise.py) est étalonné :
   surestimation systématique ×1.6–1.8 (biais de courbure, robuste MAD) — une
   borne supérieure, donc H_reach estimé conservativement (dans le bon sens).
3. **Décomposition produit** (HorizonEstimator.report) : `margin_real` =
   H_reach/H_médian sépare la marge d'amélioration réelle du modèle de la part
   irréductible due au bruit — la réponse chiffrée à « vaut-il la peine
   d'améliorer mon modèle ? » sur données réelles.
Limites : estimateur σ̂ biaisé vers le haut (facteur ~1.7 mesuré sur Lorenz —
non recalibré volontairement, à revalider sur d'autres systèmes) ; λ₁ estimé
par Rosenstein sur données réelles a sa propre incertitude, non propagée.

---

## 7. Réplication multi-systèmes (2026-07-05, suite) — la frontière de généralité

Étude `studies/study_floor_multisystem.py` (CSV : `docs/theory/data/chaos_floor_multisystem.csv`).
Même protocole (jumeaux appariés, contrôles positifs, seuils pré-enregistrés) :

| Système | Classe du champ | ρ one-shot | ρ injection | R modèle | R jumeau (contrôle) | Verdict |
|---|---|---|---|---|---|---|
| Lorenz | polynomial (quadratique) | 0.834 / 0.862 | 0.911 / 0.959 | 1.15 / 1.21 | 0.96–1.03 ✓ | **TOUCHÉ** ×2 seeds |
| Rössler | polynomial (quadratique) | 0.601 | 0.707 | 1.70 | 1.01 ✓ | NEAR |
| Mackey-Glass | **non polynomial** (Hill) + retard | 0.045 | — | 20.9 | 0.91 ✓ | MODEL-LIMITED |

Faits établis :
1. **La loi d'injection est répliquée sur un second système** : Rössler
   mesure/théorie = 1.12 (Lorenz : 1.12–1.21). Avec le transport au bruit
   (§6 : 0.88–0.99), la loi H ≈ ln(τ·[λdt]·/ε)/(λdt) tient sur 2 systèmes
   × 3 régimes (injection modèle, bruit d'observation, one-shot).
2. **Le protocole est fiable partout** : contrôles positifs R_jumeau ∈
   [0.91, 1.03] sur les trois systèmes ; la garde anti-censure a invalidé
   d'elle-même un premier verdict Rössler trop optimiste (jumeaux censurés).
3. **Structurel** : à petit λ·dt (Rössler), le coût d'injection plafonne
   ρ_one-shot ≈ 0.74 pour TOUT prédicteur pas-à-pas à e₀ ≈ 1e-10 — le
   seuil 0.8 one-shot y est inatteignable en float64 ; la comparaison
   équitable est le plancher d'injection (0.707 mesuré, NEAR).
4. **La frontière de généralité est fonctionnelle, pas dimensionnelle** :
   Mackey-Glass échoue non par le retard (l'état délai de dim 20 est bien
   appris à un pas : e₀ = 2e-5) mais parce que le ridge polynomial ne capture
   pas le JACOBIEN de la nonlinéarité de Hill le long de l'attracteur
   (croissance 21× λ₁). Prédiction falsifiable : une famille rationnelle ou
   à noyaux doit combler cet écart — c'est le prochain test.
