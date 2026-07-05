# Embedding fondé sur la théorie (information mutuelle + faux plus proches voisins)

Point 6 du plan — module `src/horizon_embedding.py`, étude `studies/study_embedding.py`,
tests `tests/test_embedding.py`.

## Problème

Les estimateurs de chaos du pipeline (Rosenstein via `estimate_lyapunov`,
expansion locale, FTLE) travaillent dans un espace de plongement à retard
`(dim, lag)`. Aujourd'hui, `_lyapunov_metrics` (`src/horizon_experiment_core.py:226-227`)
réutilise par défaut le couple `(dim, lag)` sélectionné par la recherche
val-MSE du *forecaster* (`best["dim"]`, `best["lag"]`, bornée à `lag_max=8` pas).
C'est une erreur d'objectif (audit A3/B1) : le MSE de validation favorise des
lags courts qui rendent les coordonnées quasi colinéaires — un plongement
dégénéré au sens de Takens. À `dt = 0.01` (Lorenz), `lag ≤ 8` pas = 0.08 u.t.,
très en dessous du temps de décorrélation : les « voisins » de Rosenstein sont
des points de la même portion d'orbite et λ estimé perd son sens. Le naïf
`(dim=3, lag=1)` utilisé comme défaut de secours a le même défaut.

## Théorie importée (énoncés précis avec références)

1. **Théorème de Takens / embedologie** (Takens 1981 ; Sauer, Yorke &
   Casdagli, *Embedology*, J. Stat. Phys. 65, 1991). Pour un attracteur de
   dimension boîte `d_A`, l'application de plongement à retard
   `x_t ↦ (x_t, x_{t+τ}, …, x_{t+(m−1)τ})` est génériquement un plongement dès
   que `m > 2 d_A`. Le théorème est silencieux sur le choix de `τ` (toute
   valeur générique convient asymptotiquement), mais à données finies un `τ`
   trop court donne des coordonnées redondantes et un `τ` trop long des
   coordonnées décorrélées (repliement).

2. **Choix du retard par information mutuelle** (Fraser & Swinney, *Independent
   coordinates for strange attractors from mutual information*, Phys. Rev. A
   33(2):1134, 1986). Prescription : `τ* = premier minimum local` de la courbe
   `I(τ) = I(x_t ; x_{t+τ})`, l'information mutuelle estimée par histogramme.
   Ce `τ*` rend les coordonnées « aussi indépendantes que possible tout en
   restant dynamiquement liées ». Repli standard quand `I(τ)` n'a pas de
   minimum : premier passage de l'autocorrélation sous `1/e` (Kantz &
   Schreiber, *Nonlinear Time Series Analysis*, 2e éd., 2004, §3.3).

3. **Choix de la dimension par faux plus proches voisins** (Kennel, Brown &
   Abarbanel, *Determining embedding dimension for phase-space reconstruction
   using a geometrical construction*, Phys. Rev. A 45(6):3403, 1992). Pour
   chaque `d`, on prend le plus proche voisin `j` de chaque point `i` dans le
   plongement de dimension `d` (distance `R_d`), et l'écart de la coordonnée
   supplémentaire `Δ = |x_{i+dτ} − x_{j+dτ}|`. Le voisin est **faux** si
   `Δ / R_d > r_tol` (critère 1, `r_tol = 15`) ou si
   `sqrt(R_d² + Δ²) / R_A > a_tol` (critère 2, `a_tol = 2`, `R_A` = taille de
   l'attracteur ≈ écart-type de la série). Prescription : `m* = premier d`
   avec fraction de faux voisins < 1 %.

4. **Fenêtre de Theiler** (Theiler, Phys. Rev. A 34:2427, 1986). Les voisins
   temporellement proches (`|i−j| ≤ W`) sont exclus pour ne pas confondre
   corrélation temporelle et proximité géométrique. `W` = premier zéro de
   l'autocorrélation (même recette que `estimate_lyapunov`).

5. **Cas des applications discrètes fortement mélangeantes** (Kantz &
   Schreiber 2004, §3.3). Pour une application (logistique r=4), l'ACF tombe
   à ~0 dès le lag 1 et la courbe `I(τ)` décroît vers le plancher de bruit
   d'estimation sans minimum significatif : le « premier minimum local » est
   un artefact. La prescription standard est alors `τ = 1`.

## Adaptation à notre cadre

- **Cible** : uniquement les estimateurs de chaos. Le forecaster garde sa
  recherche val-MSE (c'est le bon critère pour la prédiction) ; seuls
  `lyap_dim`/`lyap_lag` (et à terme `expansion_dim`/`expansion_lag`) doivent
  provenir du plongement théorique quand l'utilisateur les laisse à `None`.
- **MI par histogramme** : 32 bins partagés sur l'étendue de la série,
  courbe `I(τ)` pour `τ = 0..max_lag` (nats) ; minimum local = premier
  `τ ≥ 1` avec `I(τ) < I(τ−1)` et `I(τ) ≤ I(τ+1)`.
- **Repli** : sans minimum local avant `max_lag`, premier lag avec ACF < 1/e ;
  si l'ACF ne croise jamais 1/e, `max_lag`.
- **FNN** : critères 1 et 2 de Kennel, `R_A = std(série)`, fenêtre de Theiler
  auto (premier zéro ACF, borné à `[10, n/10]`), sous-échantillonnage seedé à
  2000 points de référence pour tenir le budget CPU.
- **Garde-fou « application »** (point 5 ci-dessus, ajouté après la première
  passe de validation) : si aucune dimension n'atteint 1 % de faux voisins au
  lag choisi par la MI, le plongement n'a pas déplié les données (embedding
  de type bruit). On refait alors le FNN à `lag = 1` et on le garde s'il
  atteint une fraction plus basse. Sans ce garde-fou, la logistique recevait
  `(dim=5, lag=12)` et λ s'effondrait à 0.021 (litt. 0.693) ; le FNN signale
  lui-même l'échec (min = 0.198 ≫ 0.01).

## Algorithme

`src/horizon_embedding.py`, numpy pur, fonctions pures et seedées :

1. `mutual_information_lag(series, max_lag=100, bins=32)` →
   `(best_lag, mi_curve)` : MI par histogramme pour chaque lag, premier
   minimum local, repli ACF < 1/e.
2. `false_nearest_neighbors(series, lag, max_dim=10, rtol=15.0, atol=2.0,
   theiler=None, max_points=2000, seed=0)` → `(best_dim, fnn_fractions)` :
   critère de Kennel avec fenêtre de Theiler ; `best_dim` = premier `d` avec
   FNN < 1 %, sinon le `d` de fraction minimale.
3. `select_embedding(series, max_dim=10, max_lag=100, …)` → `dict(dim, lag,
   mi_curve, fnn_fractions)` : enchaîne 1 puis 2, avec le garde-fou `lag = 1`
   si le FNN échoue au lag MI.

Complexité : MI `O(max_lag · n)` ; FNN `O(max_dim · max_points · n)` —
~1 s par système aux tailles de l'étude.

## Validation numérique (chiffres)

`python studies/study_embedding.py` — seed 0, CPU, **24.5 s** au total.
Systèmes aux dt audités : Lorenz (dt=0.01, n=6000), Rössler (dt=0.05, n=8000),
Mackey-Glass τ=17 (dt=1.0, n=3000), logistique r=4 (n=4000).

Plongements sélectionnés :

| système | dim | lag | lag·dt (u.t.) | FNN à dim choisie |
|---|---|---|---|---|
| lorenz | 3 | 16 | 0.16 | 0.000 |
| rossler | 3 | 27 | 1.35 | 0.004 |
| mackey_glass | 3 | 12 | 12.0 | 0.003 |
| logistic | 1 | 1 (garde-fou) | 1.0 | 0.000 |

Sanity : Lorenz `lag·dt = 0.16` u.t. ∈ [0.1, 0.2] attendu — **OK**.
Mackey-Glass `dim = 3` hors de la fourchette attendue 4-7 — **écart assumé** :
le critère de Kennel est légitimement satisfait à d=3 (0.3 % de faux voisins,
dim. de Kaplan-Yorke ≈ 2.1 pour τ=17) et l'estimation de λ s'améliore quand
même (ci-dessous).

λ par unité de temps, plongement théorique vs naïf `(dim=3, lag=1)`,
`estimate_lyapunov` en paramètres auto (seed 0) :

| système | littérature | théorique | naïf | rel.err théo | rel.err naïf | plus proche |
|---|---|---|---|---|---|---|
| lorenz | 0.906 | **0.8755** | 0.7303 | 0.034 | 0.194 | oui |
| rossler | 0.071 | **0.0623** | 0.1384 | 0.123 | 0.949 | oui |
| mackey_glass | 0.006 | **0.0044** | 0.0041 | 0.270 | 0.310 | oui |
| logistic | 0.693 | **0.6866** | 0.6508 | 0.009 | 0.061 | oui |

Robustesse (3 conditions initiales par système, 12 runs) : le plongement
théorique est plus proche de la littérature **12/12**, aucun run dégradé
(pire variation de l'erreur relative : **−0.040**, i.e. même le pire cas
s'améliore de 4 points). Cas notable : logistique x0=0.23, le naïf s'effondre
(λ = 0.0026, rel.err 0.996) tandis que le théorique tient (0.6815, rel.err
0.017). Tests : `tests/test_embedding.py`, 9 tests (sinus → lag ≈ T/4 ;
Hénon → dim 2, benchmark de Kennel 1992 ; logistique → dim 1 ; repli ACF ;
garde-fou application ; déterminisme ; séries dégénérées ; smoke Lorenz) ;
suite complète : 203 tests verts en ~8 s.

## Bénéfice projet

- Rössler passe d'une erreur relative de **0.95 à 0.12** sur λ₁ : l'horizon
  théorique `horizon_from_lyapunov` et le temps de Lyapunov exporté dans les
  CSV deviennent utilisables pour ce système (avant, λ était surestimé ×2).
- Lorenz : rel.err 0.19 → 0.03 ; logistique : 0.06 → 0.01 (et robuste aux
  conditions initiales là où le naïf peut s'effondrer) ; Mackey-Glass :
  amélioration marginale (0.31 → 0.27) mais aucun coût.
- Correction directe du point B1 de l'audit (λ Rosenstein sans signification
  faute de plongement correct) sans toucher au forecaster.
- Coût : ~1 s par série ; aucune dépendance nouvelle.

## Critère de décision

Intégrer `select_embedding` comme source par défaut de `lyap_dim`/`lyap_lag`
si **≥ 2 systèmes** se rapprochent de la valeur de littérature et **aucun**
ne se dégrade de plus de 50 points d'erreur relative.

Mesuré : **4/4** systèmes plus proches (seed 0), **0** dégradation
(pire variation : −0.040) ; confirmé 12/12 sur 3 seeds.

## Verdict

**INTÉGRER.** Câblage une-ligne proposé (non appliqué ici) dans
`_lyapunov_metrics` (`src/horizon_experiment_core.py:226-227`) : quand
`args.lyap_dim`/`args.lyap_lag` sont `None`, remplacer le repli
`best["dim"]`/`best["lag"]` par

```python
emb = select_embedding(np.concatenate([data.train_raw, data.val_raw]))
lyap_dim = args.lyap_dim if args.lyap_dim is not None else emb["dim"]
lyap_lag = args.lyap_lag if args.lyap_lag is not None else emb["lag"]
```

(sur la série *brute*, comme l'appel existant à `estimate_lyapunov`).
Réserves : (1) le garde-fou `lag = 1` est indispensable pour les applications
discrètes — ne pas intégrer la MI seule ; (2) pour Mackey-Glass le gain est
marginal et `dim=3` est sous la fourchette usuelle 4-7 : surveiller ce système
lors de l'intégration ; (3) `expansion_dim`/`expansion_lag` (défaut actuel :
mêmes valeurs que lyap) devraient suivre le même chemin dans un second temps.
