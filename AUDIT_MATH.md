# Audit mathématique — ARSAC Horizon Experiment

Audit complet des erreurs de mathématiques et incohérences liées aux attracteurs chaotiques.
Chaque point donne : le fichier/la ligne, le problème, pourquoi c'est faux, et la correction attendue.

Légende sévérité : 🔴 invalide les résultats · 🟠 biaise fortement · 🟡 incohérence/fragilité.

---

## A. Génération des systèmes chaotiques

### A1. 🔴 Mackey-Glass n'est pas chaotique tel que généré (bug `tau` × `dt`)
- **Où** : `src/horizon_utils.py:114-145` (`generate_mackey_glass`), `src/horizon_cli.py:213` (`--dt` par défaut 0.01, partagé entre tous les systèmes), `config.yaml:131` (`dt: 0.01`) et `config.yaml:139` (`tau: 17`).
- **Problème** : dans le générateur, le délai est indexé en **pas** (`x_tau = x[t - tau]`), donc le délai physique vaut `tau * dt`. L'équation de Mackey-Glass n'est chaotique que pour un délai τ ≳ 16.8 **unités de temps**. Avec le `dt=0.01` injecté par le CLI/config, le délai effectif est 17 × 0.01 = **0.17** unité de temps : le système converge vers un point fixe / régime trivialement prévisible.
- **Preuve interne** : `paper.tex`, table 1 — Mackey-Glass : `pSatMed = 1.000`, `TightMed = 1.000`, `SlackP90 = 0.000`. Toutes les fenêtres saturent à `Hmax` : la série est parfaitement prévisible, donc non chaotique. Le paper interprète cela comme « saturation et comportement instable » alors que c'est le générateur qui est cassé.
- **Correction** : exprimer `tau` en unités de temps, indexer le délai par `round(tau / dt)`, et donner à chaque système son propre `dt` (ne jamais partager un `--dt` global).

### A2. 🔴 Le « RK4 » de Mackey-Glass n'est pas un RK4
- **Où** : `src/horizon_utils.py:131-139`.
- **Problème** : une équation différentielle à retard (DDE) exige d'évaluer le terme retardé aux instants intermédiaires (`t+dt/2 − τ`, `t+dt − τ`) par interpolation de l'historique. Ici `x_tau` est gelé pour les 4 étages (k1..k4). L'ordre de convergence retombe à ~1-2 : l'étiquette « RK4 » est mathématiquement fausse, et l'attracteur simulé diffère de l'attracteur de Mackey-Glass canonique.
- **Correction** : méthode des pas pour DDE avec interpolation (cubique/Hermite) de l'historique, pas d'intégration ≤ 0.1, sous-échantillonnage ensuite.

### A3. 🟠 Échelles de temps incohérentes entre systèmes (dt partagé, horizons en pas)
- **Où** : `src/horizon_cli.py:213`, `config.yaml` (`horizon_max: 30`), `src/horizon_data.py`.
- **Problème** : à `dt=0.01`, `horizon_max=30-60` pas = 0.3-0.6 unité de temps.
  - Lorenz : λ₁ ≈ 0.906 → temps de Lyapunov ≈ 1.1 u.t. ≈ **110 pas**. L'horizon mesuré (≤ 60 pas) est bien en dessous de l'échelle où le chaos agit : on mesure l'accumulation d'erreur du modèle, pas la limite de prédictibilité de l'attracteur.
  - Rössler : λ₁ ≈ 0.071 → temps de Lyapunov ≈ 14 u.t. ≈ **1400 pas**. `horizon_max=60` couvre 4 % d'un temps de Lyapunov : l'expérience ne teste rien de chaotique.
  - Par ailleurs le générateur Rössler a un défaut interne `dt=0.05` mais reçoit 0.01 du CLI, et la recherche d'embedding est plafonnée à `lag_max=8` pas = 0.08 u.t. — coordonnées quasi colinéaires, embedding dégénéré au sens de Takens.
- **Correction** : définir dt, lag, horizon et fenêtre de Theiler **par système**, en unités de temps de Lyapunov (ex. `Hmax ≈ 2-3 / λ₁` converti en pas).

### A4. 🟡 Flag `--integrator` mensonger et warmup marginal
- **Où** : `src/horizon_utils.py:29-72`, `config.yaml` (`integrator: rk4`, `warmup: 200`).
- **Problème** : Lorenz/Rössler utilisent toujours SciPy RK45 quel que soit le flag (documenté dans le README mais la config affiche `rk4` comme si c'était effectif). Warmup de 200 pas = 2 u.t. pour éliminer le transitoire de Lorenz depuis (1,1,1) : marginal. `np.arange(0, total*dt, dt)` en flottant peut produire un point de plus ou de moins que prévu.
- **Correction** : un seul intégrateur documenté par système ; warmup en unités de temps (≥ 10 u.t. Lorenz) ; `np.linspace` avec nombre de points explicite.

---

## B. Estimateurs de chaos (Lyapunov, expansion, Jacobien)

### B1. 🔴 Rosenstein mal paramétré → λ estimé sans signification
- **Où** : `src/horizon_utils.py:317-376` (`estimate_lyapunov`), défauts `src/horizon_cli.py:195-198` (`max_t=25`, `theiler=10`, fit 1→10).
- **Problèmes** :
  1. **Fenêtre de Theiler = 10 pas = 0.1 u.t.** à dt=0.01 : très inférieure au temps d'autocorrélation. Les « plus proches voisins » sont des points de la même portion de trajectoire → la divergence mesurée est celle du flot le long de l'orbite, pas la divergence chaotique → λ sous-estimé, voire aléatoire. Rosenstein recommande une fenêtre ≥ la période moyenne.
  2. **Plage de fit 1→10 pas = 0.1 u.t.** : sur Lorenz, la croissance attendue sur cette plage est e^(0.906×0.1) ≈ 1.09, soit 9 % — noyé dans le bruit d'estimation. Il faut suivre la divergence sur ≥ 1-2 temps de Lyapunov (des centaines de pas à ce dt).
  3. Pas de détection de la **région linéaire** (avant saturation à la taille de l'attracteur), pas de filtrage des voisins initialement trop éloignés.
  4. Le paramètre `dt` est accepté puis ignoré (`_ = dt`, ligne 375) : le λ retourné est par pas, et la conversion est laissée à l'appelant — piège d'unités (voir B2).
- **Correction** : theiler auto (premier zéro de l'autocorrélation ou minimum de l'information mutuelle), `max_t` en temps de Lyapunov, fit automatique de la région linéaire, validation contre les valeurs de référence (Lorenz 0.906, Rössler 0.071, logistique ln 2 ≈ 0.693, Mackey-Glass τ=17 ≈ 0.006).

### B2. 🟡 Confusion d'unités et de nommage autour de λ
- **Où** : `src/horizon_experiment_core.py:237` (`lyap_time = lyap_step / dt`).
- **Problème** : la formule est correcte (λ par unité de temps = λ par pas / dt) mais la variable s'appelle `lyap_time`, ce qui en dynamique désigne le **temps de Lyapunov** (1/λ) — l'inverse de ce qui est calculé. Toute lecture des CSV (`lyapunov_time`) prête à contresens.
- **Correction** : nommer `lyap_per_step`, `lyap_per_unit_time`, `lyapunov_time = 1/λ`.

### B3. 🟠 `estimate_expansion_quantile` et `estimate_error_growth` : saturation ignorée
- **Où** : `src/horizon_utils.py:217-270`, `src/horizon_metrics.py:71-120`.
- **Problème** : les ratios de divergence `d1/d0` (état) et `err_h/err_{h-1}` (erreur) sont moyennés sans exclure le régime saturé (distance plafonnée au diamètre de l'attracteur, erreur plafonnée à ~2·std). La moyenne des log-ratios inclut la phase plate → sous-estimation systématique de la croissance ; à l'inverse `eps=1e-8` produit des ratios explosifs quand une erreur passe près de 0. Le quantile 0.95 de ces ratios est ensuite traité comme un taux de croissance appliqué à chaque pas (voir D2).
- **Correction** : ne moyenner que dans la région de croissance linéaire (log), tronquer avant saturation, estimer la croissance par régression sur log e(h) plutôt que par ratios successifs.

### B4. 🟠 Le « Jacobien » n'est pas le Jacobien de la dynamique embarquée
- **Où** : `src/horizon_metrics.py:167-238` (`estimate_jacobian_growth`), `285-318` (`jacobian_norm`).
- **Problème** : le code calcule ‖∇f(x)‖, le gradient de la sortie scalaire du modèle un-pas. Or la dynamique dans l'espace d'embedding est le décalage `(x₂,…,x_d, f(x))` dont le Jacobien est une matrice compagnon : sa plus grande valeur singulière est ≥ 1 par construction et n'est pas ‖∇f‖. Utiliser ‖∇f‖ comme facteur d'expansion par pas (source `jacobian` de `--growth-source`) n'a pas de justification : le vrai taux local s'obtient par produit de Jacobiens le long de la trajectoire avec re-orthonormalisation (QR / Benettin). En outre, pour les LSTM, le gradient est calculé en mode `train()` (`horizon_metrics.py:192-194`), ce qui active le dropout et rend le « Jacobien » stochastique.
- **Correction** : construire la matrice compagnon, propager les produits QR sur k pas pour un exposant local ; geler le modèle en `eval()` (utiliser `cudnn.flags(enabled=False)` si nécessaire pour le backward RNN).

---

## C. Définitions d'horizon incohérentes entre elles et avec la spec

### C1. 🟠 Trois définitions différentes de « l'horizon » coexistent
1. `horizon_from_rmse` (`horizon_metrics.py:508-513`) : premier h où le RMSE **moyenné sur toutes les fenêtres** dépasse τ (horizon global).
2. `build_horizon_dataset` (`horizon_metrics.py:405`) : `rmse_by_h = sqrt(cumsum(errors)/h)` — RMSE **cumulé** de 1 à h pour une fenêtre, avec K dépassements consécutifs. **Contredit AGENT.MD §2.1**, qui définit `rmse_w(h)` comme l'erreur au pas h, pas la moyenne cumulée. Le cumul lisse et retarde le franchissement → H_w systématiquement surestimé.
3. `window_horizons` (`horizon_metrics.py:474-505`) : première erreur ponctuelle |e_h| ≥ τ, sans K consécutifs.

Le chemin probabiliste calibre avec la définition 3 alors que le chemin conforme apprend la définition 2 : les « horizons » comparés dans les CSV ne mesurent pas la même chose.
- **Correction** : une seule définition documentée (erreur au pas h + K consécutifs recommandé), partagée par tous les modules, avec test unitaire de cohérence.

### C2. 🟠 Tolérance « relative » par fenêtre basée sur un seul échantillon
- **Où** : `horizon_metrics.py:406-409` (`tolerance_local = rmse_by_h[0] * error_factor`).
- **Problème** : `rmse_by_h[0]` pour une fenêtre est **une seule réalisation** de l'erreur un-pas. Si le modèle tombe juste sur cette fenêtre (erreur ≈ 0), τ_w ≈ 0 et H_w = 1 alors que la prédiction est excellente — labels contaminés par le bruit. De plus « relatif » signifie ailleurs (`_tolerance_from_mode`, `horizon_experiment_core.py:183-184`) un multiple du RMSE global : même mot, deux définitions.
- **Correction** : τ défini comme fraction de l'échelle de l'attracteur (ex. 0.4 × std de la série), constant par dataset ; c'est aussi la convention de la littérature (e.g. « valid time »).

### C3. 🟠 La saturation à Hmax est de la censure, traitée comme une valeur exacte
- **Où** : `horizon_metrics.py:334/341` (retourne `len(rmse)` quand τ n'est jamais franchi), régression quantile et conformal en aval.
- **Problème** : H_w = Hmax signifie « H_w ≥ Hmax » (donnée censurée à droite), pas « H_w = Hmax ». La régression pinball et les scores conformes traitent la valeur censurée comme exacte → biais vers le bas de q̂ et couverture faussée dès que p_sat > 0 (Mackey-Glass : p_sat = 1.0, tout le pipeline tourne sur des labels dégénérés ; seul un warning est prévu).
- **Correction** : soit Hmax ≫ horizon typique (≥ 3 temps de Lyapunov) pour rendre la censure rare, soit gestion explicite de la censure (pinball censuré / conformal pour données censurées), et invalidation du run quand p_sat dépasse un seuil.

---

## D. Chemin probabiliste : garanties inversées

### D1. 🔴 La « calibration de couverture » est inversée par rapport à l'objectif annoncé
- **Où** : `src/horizon_experiment_probabilistic.py:151-155` (`_calibration_scale`) et `159-173` (`_coverage_from_ratios`).
- **Problème** : le README promet une **borne inférieure** L avec P(H ≥ L) ≥ 1−α. Or :
  - `scale = quantile_{1−α}(h_real / h_model)` avec plancher `calibration_floor = 1.0` → le facteur ne peut qu'**augmenter** la borne ;
  - la couverture compte un succès quand `h_model * scale >= h_real`, c'est-à-dire quand la borne **dépasse** l'horizon réel.
  On calibre donc P(borne ≥ H_réel) ≈ 1−α : une borne *supérieure* — exactement l'inverse d'une garantie de sécurité. Faire confiance aux prédictions jusqu'à `horizon_cal` est dangereux au-delà de `h_real`, et c'est le cas ~95 % du temps par construction.
- **Correction** : pour une borne inférieure, scale = quantile_α (h_real/h_model) plafonné à 1, hit quand `h_real >= h_model * scale`, et supprimer le plancher à 1.0.

### D2. 🟠 `horizon_model` présenté comme une borne alors que le taux de croissance est un quantile empirique
- **Où** : `horizon_utils.py:174-208` (`horizon_from_model_bound_by_growth`), utilisé avec `growth_q` = quantile 0.95 des croissances moyennes par fenêtre.
- **Problème** : la récurrence e_{t+1} ≤ g·e_t + δ ne fournit une borne que si g majore la croissance **partout**. Un quantile 0.95 de moyennes par fenêtre n'est ni un sup ni une borne de Lipschitz : l'« horizon garanti » n'a pas de statut mathématique (l'algèbre interne de la récurrence est, elle, correcte).
- **Correction** : présenter h_model comme une heuristique, ou construire une vraie borne (Lipschitz du modèle + bruit borné), ou passer entièrement par la voie conforme.

---

## E. Pipeline conforme : hypothèses violées et réutilisation des données

### E1. 🟠 Échangeabilité brisée : fenêtres chevauchantes + série temporelle
- **Où** : `build_horizon_dataset` (stride=1 par défaut), toute la calibration (`horizon_experiment_conformal_calibration.py`).
- **Problème** : les fenêtres consécutives partagent (dim−1)·lag + Hmax − 1 points : les labels H_w voisins sont fortement corrélés, et calib/test sont des segments temporels adjacents. La garantie conforme exige l'échangeabilité ; le « block-quantile » (`horizon_conformal.py:32-60`, quantile de quantiles par blocs) est une heuristique sans théorème. Le quantile conforme de base (`conformal_quantile`, rang ⌈(n+1)(1−α)⌉) est correct, mais appliqué hors de ses hypothèses.
- **Correction** : fenêtres de calibration **disjointes** (stride ≥ longueur de fenêtre + Hmax), et/ou conformal « beyond exchangeability » (Barber et al. 2023) avec poids de décroissance temporelle ; documenter que la couverture est conditionnelle au régime.

### E2. 🟠 Le pool CV mélange train et calibration, avec discontinuité temporelle
- **Où** : `horizon_experiment_conformal_data.py:111-114` (`_cv_dataset` : `concatenate([train_std, calib_series])`).
- **Problème** : (1) le segment val est physiquement entre train et calib ; la concaténation crée un saut — les fenêtres à cheval sur la jonction sont des trajectoires qui n'existent pas, avec labels faux. (2) Les fenêtres issues de train sont **in-sample** pour le forecaster (entraîné sur train+val) : leurs H_w sont optimistes, ce qui biaisse les scores de calibration vers des marges trop petites. (3) Le modèle qui prédit le test (`_predict_test_from_fit`) est réentraîné sur tout le pool, pas les modèles de folds : ce n'est pas le CV+ de Barber et al., la garantie affichée dans le README (« CV+ cross-fitting ») n'est pas celle implémentée.
- **Correction** : calibration uniquement sur des fenêtres hors-train, exclusion des fenêtres chevauchant une jonction, ou vrai CV+ (agrégation des prédictions de folds sur le test).

### E3. 🟠 Réutilisations successives du même jeu de calibration
- **Où** : `_apply_offset_calibration` (`horizon_experiment_conformal_data.py:229-246`), puis marge conforme, puis `_coverage_guard`, puis debias (`horizon_experiment_conformal_calibration.py:304-364`) — quatre ajustements séquentiels sur les mêmes données.
- **Problème** : chaque étape consomme les mêmes scores : la garantie en échantillon fini disparaît, la couverture rapportée sur calib est mécaniquement proche de la cible (sur-apprentissage de la calibration). Même problème pour l'arbre conforme : `ConformalTreeEstimator.fit` (`horizon_conformal.py:109-122`) construit l'arbre **et** calcule les quantiles de feuilles sur les mêmes données.
- **Correction** : découper la calibration (offset sur une moitié, marge conforme sur l'autre ; arbre sur une moitié, quantiles sur l'autre) ou supprimer offset/guard/debias au profit d'une seule marge bien définie.

### E4. 🟠 Intervalle de confiance de Wilson invalide sous dépendance
- **Où** : `src/horizon_scientific_eval.py:95-104` (`_wilson_lower_bound`).
- **Problème** : Wilson suppose des Bernoulli i.i.d. Les hits de couverture proviennent de fenêtres chevauchantes fortement autocorrélées : n_effectif ≪ n, la borne inférieure rapportée (`coverage_lb`) est largement trop optimiste — précisément la métrique vendue comme « scientifique ».
- **Correction** : bootstrap par blocs (taille de bloc ≥ portée de corrélation) ou correction de taille d'échantillon effective.

---

## F. Divers (code et publication)

- **F1** 🟡 Code mort : `horizon_metrics.py:277-282`, bloc dupliqué inatteignable après le `return` de `gated_rollout`.
- **F2** 🟡 `paper.tex` ne compile pas : quasi toutes les commandes ont des doubles backslashes (`\\documentclass`, `\\usepackage`, …) mélangés à des simples (`\texttt`, `\mathbb`) ; auteurs placeholders ; la table publie les résultats Mackey-Glass dégénérés (A1) et une couverture min de 0.000 sans explication.
- **F3** 🟡 Les tests ne testent aucune physique : `tests/test_physics_simple.py` vérifie forme et absence de NaN. Aucun test ne valide λ contre les valeurs connues, ni le caractère chaotique des séries générées, ni la cohérence des définitions d'horizon.
- **F4** 🟡 `README.md` promet « P(H_window ≥ L(x)) ≥ 1 − α » (garantie), `AGENT.MD` dit « = 1 − α (empirical coverage) » : compte tenu de E1-E3, seule la formulation empirique est défendable.
- **F5** 🟡 `set_seed` (`horizon_utils.py:15`) ne seede CUDA que si déjà initialisé ; `chaos_quick_test.py` hérite du bug A1 (dt=0.01 + tau=17) et son verdict « chaotic/non-chaotic » utilise les estimateurs mal paramétrés de B1/B3.

---

## Synthèse

| Sévérité | Points |
|---|---|
| 🔴 Invalide les résultats | A1 (Mackey-Glass non chaotique), A2 (faux RK4 DDE), B1 (λ Rosenstein sans signification), D1 (garantie inversée) |
| 🟠 Biais fort | A3, B3, B4, C1, C2, C3, D2, E1, E2, E3, E4 |
| 🟡 Incohérences | A4, B2, F1-F5 |

Conclusion : les nombres publiés (README « Performance », `paper.tex`) ne mesurent pas ce qu'ils prétendent mesurer. Sur Mackey-Glass le système n'est pas chaotique ; sur Lorenz/Rössler les horizons sont mesurés à des échelles de temps où le chaos n'agit pas encore ; l'estimateur de Lyapunov n'est pas fiable ; et les deux mécanismes de « garantie » (probabiliste et conforme) sont respectivement inversé et privé de ses hypothèses. Le plan de refonte est dans `PLAN_V2.md`.
