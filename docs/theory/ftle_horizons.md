# FTLE : distribution des exposants de Lyapunov locaux → distribution des horizons

Point 3 du programme théorique. Code : `src/horizon_ftle.py` ; étude : `studies/study_ftle.py`
(seedée, ~37 s CPU) ; tests : `tests/test_ftle.py`.

## Problème

Les labels d'horizon `H_w` (premier pas où l'erreur de rollout dépasse la tolérance,
`build_horizon_dataset`) présentent sur Lorenz une forte dispersion et des queues de
« slack » lourdes : deux fenêtres voisines sur l'attracteur peuvent avoir des horizons
très différents. La borne conforme actuelle traite cette dispersion comme du bruit
homoscédastique. Question : la distribution des exposants de Lyapunov **locaux**
(finite-time Lyapunov exponents, FTLE) explique-t-elle la distribution des `H_w`,
et un FTLE du modèle appris est-il une feature utile pour la régression quantile
qui produit `L(x)` ?

## Théorie importée (énoncés précis avec références)

1. **FTLE et grandes déviations.** Pour un système ergodique chaotique, l'exposant
   local `λ_T(x) = (1/T) ln ‖DΦ_T(x) v‖` (v aligné sur la direction la plus dilatante)
   fluctue autour de `λ₁` ; le théorème de grandes déviations pour les produits de
   matrices donne `P(λ_T ≈ a) ≍ exp(−T I(a))` avec `I(λ₁) = 0`, et par CLT
   `Var(λ_T) ≈ σ²/T` (Grassberger, Badii & Politi, *J. Stat. Phys.* 51 (1988) ;
   Ott, *Chaos in Dynamical Systems*, ch. 9 ; Pikovsky & Politi, *Lyapunov Exponents*,
   CUP 2016, ch. 5-6). Pour Lorenz (σ=10, ρ=28, β=8/3), `λ₁ ≈ 0.906` par unité de temps.
2. **Horizon de prédictibilité local.** Si une erreur initiale `e₀` croît localement
   comme `e(t) ≈ e₀ exp(λ_T(x) t)`, le temps de franchissement d'une tolérance `tol` est
   `H(x) ≈ ln(tol/e₀) / λ_T(x)` (argument classique du « temps de Lyapunov », cf.
   Boffetta, Cencini, Falcioni & Vulpiani, *Phys. Rep.* 356 (2002), §3). La distribution
   de `H_w` hériterait donc de celle de `1/λ_T` : les queues lourdes de `H_w` (fenêtres
   localement peu dilatantes → horizon long) seraient l'image des queues basses de `λ_T`.
3. **Calcul du FTLE.** Méthode de Benettin : propager une base tangente par les équations
   variationnelles (ou par les Jacobiennes du système discret), ré-orthonormaliser par QR
   à chaque pas, accumuler `ln |R₁₁|` (Benettin, Galgani, Giorgilli & Strelcyn,
   *Meccanica* 15 (1980)). Pour la dynamique **apprise** en espace d'embedding, la
   Jacobienne exacte du buffer de rollout `b' = (b₂,…,b_w, f(x(b)))` est une matrice
   compagnon dont la dernière ligne porte `∇f` aux colonnes `i·lag` — cela corrige le
   point B4 de l'audit (`‖∇f‖` seul n'est pas un facteur d'expansion).

## Adaptation à notre cadre

- **FTLE du modèle** (`ftle_along_series`) : produit de matrices compagnons le long du
  rollout **prédit** (le même rollout qui définit `H_w`), QR à chaque pas (une colonne :
  `ln R₁₁` = log de la norme du vecteur propagé), `k = 100` pas (`T = k·dt = 1.0` u.t.
  à `dt = 0.01`). Exposant par pas ; division par `dt` pour l'unité de temps.
- **FTLE vérité terrain** (`lorenz_ftle_ground_truth`) : équations variationnelles de
  Lorenz (état + base tangente 3×3), RK4 couplé, QR à chaque pas, numpy pur vectorisé
  sur les points. Ancré aux **mêmes points de trajectoire** que les fenêtres du modèle
  (la trajectoire RK4 3D est conservée, `lorenz_trajectory`).
- **Hypothèses déclarées honnêtement** :
  1. le FTLE du modèle mesure la carte **apprise**, pas le flot vrai — pour `LinearAR`
     la Jacobienne est constante, donc son FTLE est constant par construction (variance
     résiduelle = bruit du vecteur initial aléatoire) ;
  2. l'embedding de retard est un changement de coordonnées qui distord les normes
     locales (perturbe les valeurs à T fini, pas l'exposant asymptotique) ;
  3. le rollout prédit s'écarte de la trajectoire vraie bien avant `k = 100` pas quand
     `H_w ≤ 30` : le FTLE à `k = 100` intègre de l'information au-delà de l'échelle du label
     (d'où la sensibilité `k = 30` dans l'étude) ;
  4. les labels sont censurés à `Hmax = 30` (`p_sat` jusqu'à 0.75 pour le MLP), ce qui
     tronque toute corrélation atteignable (audit C3).

## Algorithme

```
model_jacobian_row(model, x)      # ∇f exact (LinearAR), autograd eval-mode (Torch),
                                  # différences finies centrales sinon
companion_matrix(g, dim, lag)     # taille w=(dim−1)·lag+1 ; décalage + g aux colonnes i·lag
ftle_along_series(model, s, dim, lag, k, stride, max_windows, seed):
    pour chaque fenêtre échantillonnée :
        v ← vecteur unitaire aléatoire (seedé)
        pour k pas : rollout prédit ; C ← compagnon(∇f(x_t)) ; v ← C v ;
                     log_sum += ln‖v‖ ; v ← v/‖v‖
    retourne (log_sum/k par fenêtre, indices de départ)
lorenz_ftle_ground_truth(points, T, dt):
    RK4 couplé (x, V∈R^{3×3}) sur T ; QR à chaque pas ; λ_T = Σ ln|R₁₁| / T
```

## Validation numérique (chiffres)

Étude `studies/study_ftle.py` : Lorenz `dt = 0.01`, série 4000 pts (train 2400 / val 400 /
éval 1200), `dim = 4`, `lag = 10`, labels `build_horizon_dataset` (`Hmax = 30`,
tolérance absolue 0.4 en unités standardisées, `stride = 2`, 535 fenêtres alignées),
`LinearAR` + MLP (32 neurones, ≤ 20 epochs), 2 seeds (0, 1). Runtime total : **37 s**.

**(1) Estimateur vérité terrain** (300 points d'attracteur) :

| T (u.t.) | mean λ_T | std λ_T | var·T |
|---|---|---|---|
| 0.5 | 0.530 | 2.331 | 2.72 |
| 1.0 | 0.712 | 1.220 | 1.49 |
| 2.0 | 0.747 | 0.717 | 1.03 |
| 5.0 | 0.814 | 0.326 | 0.53 |
| 10.0 | 0.854 | 0.155 | 0.24 |
| 20.0 | 0.876 | 0.081 | 0.13 |

La moyenne converge de façon monotone vers `λ₁ = 0.906` (biais ≈ O(1/T), transitoire
d'alignement de la base tangente initialisée à l'identité). La variance décroît avec T ;
plus vite que 1/T sur cette plage (terme transitoire O(1/T²) superposé au terme CLT).
L'estimateur est validé.

**(2) FTLE du modèle vs vérité terrain** (mêmes points, k = 100, T = 1.0) :

| modèle / seed | mean FTLE modèle (u.t.⁻¹) | std | mean vérité | std | Pearson | Spearman |
|---|---|---|---|---|---|---|
| linear / 0 | −1.106 | 0.869 | +0.744 | 1.192 | −0.009 | −0.000 |
| linear / 1 | −0.911 | 0.820 | +0.695 | 1.202 | +0.003 | +0.003 |
| mlp / 0 | −0.636 | 1.420 | +0.744 | 1.192 | +0.214 | +0.242 |
| mlp / 1 | −0.249 | 1.670 | +0.695 | 1.202 | −0.125 | −0.145 |

Le FTLE du modèle est **négatif en moyenne** (cartes apprises contractantes : rayon
spectral de la compagnonne ≈ e^{−0.01} pour LinearAR) alors que le flot vrai dilate à
+0.7/u.t. : la carte apprise un-pas ne reproduit pas l'expansion du flot. Pour LinearAR
la corrélation est nulle **par construction** (Jacobienne constante). Pour le MLP elle
est faible et de signe instable entre seeds.

**(3) Corrélations avec les labels H_w** (Spearman, 535 fenêtres ; `p_sat` : linear
0.27/0.25, mlp 0.58/0.75) :

| feature | linear s0 | linear s1 | mlp s0 | mlp s1 |
|---|---|---|---|---|
| FTLE modèle k=100 | +0.083 | −0.054 | **−0.297** | −0.120 |
| FTLE modèle k=30 (sensibilité) | +0.095 | −0.053 | −0.337 | −0.302 |
| FTLE vérité (oracle) | −0.090 | −0.112 | −0.308 | −0.199 |
| jac_mean (existant) | nan (constant) | nan | **−0.646** | −0.400 |
| resid1 (existant) | −0.325 | −0.343 | +0.202 | +0.204 |
| ln(tol/e₀)/λ_T (théorie directe, oracle) | +0.167 | +0.204 | +0.265 | +0.141 |

Le signe négatif attendu (plus de dilatation → horizon plus court) est présent pour le
MLP, mais `|ρ| < 0.3` en moyenne et **toujours plus faible que `jac_mean`**, qui est
déjà un proxy d'expansion locale à courte échéance (norme de ∇f moyennée sur 3 pas).
Le prédicteur « théorie directe » `ln(tol/e₀)/λ_T`, même avec le λ oracle du flot vrai,
ne dépasse pas ρ ≈ +0.27. Sur le sous-ensemble non censuré les corrélations s'effondrent
ou changent de signe (effet de sélection de la censure).

**(4) Perte pinball (GradientBoostingRegressor quantile α = 0.1, CV 5 plis contigus)** :

| jeu de features | linear s0 | linear s1 | mlp s0 | mlp s1 |
|---|---|---|---|---|
| quantile constant (baseline) | 1.127 | 1.152 | 1.569 | 1.642 |
| features existantes | 0.809 | 0.853 | 0.638 | 0.964 |
| + FTLE modèle k=100 | 0.824 (−1.9 %) | 0.897 (−5.2 %) | 0.633 (+0.9 %) | 0.983 (−2.0 %) |
| + FTLE modèle k=30 | 0.817 (−1.1 %) | 0.887 (−4.0 %) | 0.644 (−1.0 %) | 1.022 (−6.0 %) |
| + FTLE vérité (oracle) | 0.815 (−0.8 %) | 0.846 (+0.8 %) | 0.670 (−5.0 %) | 0.979 (−1.6 %) |

Gain moyen du FTLE modèle k=100 : **linear −3.6 %, mlp −0.5 %** (des pourcentages
positifs = amélioration ; ici la feature dégrade ou ne change rien). Même l'oracle
(FTLE du flot vrai) n'apporte aucun gain — le GBR sur les features existantes capte
déjà le signal d'expansion locale via `jac_mean`/`resid1`.

## Bénéfice projet

Bénéfice **mesuré** de la feature FTLE : nul (aucun gain de pinball ≥ 5 %, corrélation
plus faible que la feature existante `jac_mean`). Bénéfices réels mais indirects :

1. **Explication mécaniste validée à moitié** : la dilatation locale de la carte
   *apprise* à courte échéance pilote bien `H_w` (c'est `jac_mean`, ρ jusqu'à −0.65,
   la feature la plus corrélée) — cela confirme le mécanisme « expansion locale →
   horizon » de la théorie. En revanche la version longue-échelle (`k = 100 ≫ Hmax = 30`)
   et la version « flot vrai » n'ajoutent rien : à ces échelles d'horizon (0.3 u.t.
   ≈ 0.27 temps de Lyapunov), `H_w` mesure l'accumulation d'erreur du modèle, pas la
   limite chaotique de l'attracteur (cohérent avec l'audit A3).
2. **Estimateur vérité terrain réutilisable** : `lorenz_ftle_ground_truth` fournit le
   λ₁ de référence (Benettin/variational, prévu en Phase 0 du PLAN_V2) et des λ_T
   locaux pour de futures études ; `companion_matrix` + produits QR corrigent le
   point B4 (jacobien de la dynamique embarquée).
3. **Mise en garde chiffrée** : le FTLE d'un modèle linéaire est constant par
   construction — toute heuristique « expansion locale » basée sur la carte apprise
   est vide pour les modèles linéaires, et la carte apprise peut être contractante
   (FTLE moyen −0.2 à −1.1/u.t.) alors que le flot dilate (+0.9/u.t.).

## Critère de décision

Intégrer le FTLE comme feature optionnelle si, sur l'étude seedée :
`gain de pinball ≥ 5 %` **ou** (`|Spearman(FTLE, H_w)| ≥ 0.3` **et** plus fort que
`|Spearman(jac_mean, H_w)|`). Mesuré (moyenne 2 seeds, FTLE k=100) :

- linear : gain −3.6 % ; |ρ| = 0.069 (jac_mean : constant, nan) → critère non rempli.
- mlp : gain −0.5 % ; |ρ| = 0.209 < 0.3 et < 0.523 (jac_mean) → critère non rempli.
- Sensibilité k=30 et oracle vérité terrain : également sous les seuils.

## Verdict

**SHELVE.** La feature FTLE (k = 100 comme k = 30, modèle comme oracle) n'améliore pas
la régression quantile des horizons et est dominée par `jac_mean`, déjà présente. La
théorie FTLE → horizons est qualitativement confirmée dans sa version courte échéance
(le signal est réel mais déjà capté), et quantitativement inopérante aux échelles
d'horizon actuelles (`Hmax = 0.27` temps de Lyapunov, censure 25-75 %). À réexaminer
uniquement si le pipeline passe à `Hmax ≥ 2-3` temps de Lyapunov avec censure gérée
(Phase 2 du PLAN_V2), où la limite chaotique — donc λ_T — redevient le facteur dominant.
Si l'intégration était décidée : ajouter dans `build_horizon_dataset` un flag
`use_ftle` (défaut False) qui appellerait `ftle_along_series` sur les mêmes fenêtres et
concaténerait la colonne aux features — ne pas l'implémenter en l'état.
