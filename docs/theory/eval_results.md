# Évaluation comparative de l'intégration des études théoriques

Date : 2026-07-05. Agent : intégration (post-études P1..P6).

## Ce qui a été intégré

| Point | Verdict étude | Action d'intégration |
|---|---|---|
| P1 conformal sous dépendance | shelve | **Rien** (aucun flag `--calib-disjoint` / `--conformal-weight-half-life` ajouté ; les deux critères chiffrés de l'étude échouent). Module `src/horizon_conformal_beyond.py` disponible en opt-in futur. |
| P2 censure à Hmax | conditional | Intégré avec la précondition de l'étude (les 3 pièces ensemble) : `cap` optionnel dans `train_quantile_mlp` (Powell), flag `--censored-quantile` (défaut OFF) qui propage `cap=horizon_max` vers l'entraînement quantile ET plafonne les prédictions dans `_compute_scores` / `_calib_interval` / `_test_predictions` ; garde de saturation **toujours active** → `stats['label_identified']` + warning. |
| P3 FTLE | shelve | Rien (pas de `--feature-ftle`). |
| P4 borne certifiée | integrate | `certified_horizon` calculé sur la série de calibration dans `_experiment_setup` (les deux pipelines) ; export `horizon_certified`, `lipschitz_G`, `delta_sup` dans stats + 3 colonnes CSV (en-tête auto-versionné) ; exceptions gardées (LSTM → 0.0 + warning). |
| P5 Politis-White | shelve | Rien de nouveau ; le câblage posé par l'agent d'étude dans `horizon_scientific_eval.py` (mesurablement neutre, différences sous le bruit MC) est laissé en place et vérifié : import OK, suite verte. |
| P6 embedding théorique | integrate | `_lyapunov_metrics` utilise `select_embedding` (IM + FNN) quand `--lyap-dim` ET `--lyap-lag` sont tous deux absents ; les args explicites gagnent ; `expansion_dim/lag` héritent automatiquement (ils défaillent déjà sur `lyap.dim/lag`). Guard try/except → repli sur l'embedding du modèle. |

## Protocole du benchmark

`python -m src.horizon_experiment` en boucle (config.yaml + overrides rapides) :
modèle **linear**, datasets lorenz / rossler / mackey_glass, seeds 0..4,
`series_len=2500`, `warmup=400`, `horizon_max=20`, `horizon_samples=300`,
`mlp_epochs=15`, `quantile_ensemble=1`, `bound_mode=horizon_conformal`
(alpha=0.05, horizon_quantile=0.10, mode bins). CSV bruts :
`outputs/theory_eval_{baseline,censored,sat_baseline,sat_censored}.csv`.

Bras : (a) `baseline` = tous les nouveaux flags OFF ; (c) `censored` =
`--censored-quantile`. Le bras (b) `+calib-disjoint` est **sans objet** (P1
shelve : le flag n'existe pas). Le bras (d) « tous les flags validés » est
**identique à (c)** : le seul flag ajouté par l'intégration est
`--censored-quantile` (P4 et P6 sont toujours actifs, présents dans les deux
bras). Profil `sat_*` : régime de saturation provoqué (`horizon_max=10`,
`error_factor=30`) pour mesurer le flag dans le régime qu'il vise
(p_sat > 0).

## Résultats (médiane / min sur 5 seeds)

Profil standard (p_sat = 0 partout — la censure ne mord jamais) :

| Dataset | Bras | cov méd | cov min | tightness méd | slack_p90 méd | h_cert méd | runtime bras |
|---|---|---|---|---|---|---|---|
| lorenz | baseline | 0.967 | 0.933 | 0.791 | 3.527 | 2.0 | 17.3 s |
| lorenz | censored | 0.967 | 0.933 | 0.791 | 3.527 | 2.0 | 17.1 s |
| rossler | baseline | 0.970 | 0.960 | 1.053 | 2.542 | 1.0 | — |
| rossler | censored | 0.970 | 0.960 | 1.053 | 2.542 | 1.0 | — |
| mackey_glass | baseline | 0.898 | 0.847 | 0.915 | 2.139 | 2.0 | — |
| mackey_glass | censored | 0.898 | 0.847 | 0.915 | 2.139 | 2.0 | — |

Profil saturation (p_sat_calib méd : lorenz 0.157, rossler 0.077, mg 0.037) :

| Dataset | Bras | cov méd | cov min | tightness méd | slack_p90 méd | h_cert méd | runtime bras |
|---|---|---|---|---|---|---|---|
| lorenz | sat_baseline | 0.990 | 0.980 | 0.737 | 4.655 | 2.0 | 16.5 s |
| lorenz | sat_censored | 0.990 | 0.980 | 0.737 | 4.655 | 2.0 | 16.6 s |
| rossler | sat_baseline | 1.000 | 0.955 | 0.763 | 4.094 | 2.0 | — |
| rossler | sat_censored | 1.000 | 0.955 | 0.763 | 4.094 | 2.0 | — |
| mackey_glass | sat_baseline | 0.972 | 0.954 | 0.950 | 3.144 | 2.0 | — |
| mackey_glass | sat_censored | 0.972 | 0.954 | 0.950 | 3.144 | 2.0 | — |

Diagnostics certifiés (P4, constants par dataset — série et modèle linéaire
déterministes à travers les seeds) : lorenz `G=6.493`, `delta_sup=0.00249` ;
rossler `G=8.255`, `delta_sup=0.00318` ; mackey_glass `G=20.481`,
`delta_sup=0.00693`. `h_cert` (1–2 pas) reste très en dessous de la borne
conforme (`horizon_model_cal` ≈ 2.6–4.8) : diagnostic de plancher garanti,
pas un remplacement — conforme au verdict de l'étude P4.

## Lecture honnête

1. **`--censored-quantile` est un no-op mesuré sur ces 4 profils**
   (différences ≤ 1e-7 sur toutes les métriques, dues au `torch.minimum`
   dans la perte). Explication conforme à l'étude P2 : le bénéfice n'existe
   que quand la censure touche le **quantile cible** ; ici la cible est le
   quantile bas Q_0.10 de H, qui ne s'approche de Hmax que si p_sat > 0.9.
   Même le profil provoqué (p_sat ≤ 0.17) reste loin de ce régime. Le flag
   est donc **sans risque** (aucune dégradation, aucun gain) et ne devient
   utile que sur des régimes quasi totalement saturés — que la garde
   détecte désormais.
2. **La garde de saturation (toujours active) fonctionne** : sur un run
   dégénéré (mackey_glass, `error_factor=2000`, p_sat=1.0 — le mode de
   défaillance historique de l'audit C3), `label_identified=False` +
   warning « Q_0.05(H) sits in the censored region at Hmax=10 ». Coût nul.
3. **P4 et P6 n'affectent pas les métriques conformes** (mêmes chiffres que
   l'ancien pipeline sur les colonnes coverage/tightness ; P6 ne change que
   `lyapunov_*`/`horizon_theory`, P4 n'ajoute que des colonnes). Suite de
   tests complète : 212 passed (203 avant intégration + 9 nouveaux).
4. Observations pré-existantes (identiques dans tous les bras, à traiter en
   Phase 3, PAS des régressions de cette intégration) : mackey_glass
   sous-couvre la cible 0.95 au profil standard (méd 0.898, min 0.847) ;
   rossler affiche tightness > 1 au profil standard (médiane de L au-dessus
   de la médiane de H_w avec couverture 0.96–0.98 : distribution de labels
   très asymétrique aux réglages rapides).

## Recommandation de défauts

- `--censored-quantile` : laisser **OFF par défaut** (aucun gain mesuré ici),
  l'activer sur données à p_sat élevé ; la garde `label_identified` signale
  ces régimes automatiquement.
- Export certifié (P4) et embedding théorique (P6) : **ON** (toujours
  actifs), coût mesuré ≈ +0.3 s/run au total, aucune métrique dégradée.

## Post-intégration : Hmax auto (temps de Lyapunov) + tolérance absolue 0.4·std

Validation `studies/study_lyap_hmax.py` (2026-07-05) : modèle linéaire, 5 seeds,
α=0.05, ratios 0.6/0.15/0.15/0.10, Hmax auto = max(3 T_λ, 1.2·ln(τ/e₀)/λ₁)
borné [30, 400] et par le budget de données.

| dataset | Hmax auto | cov med [min] | tight med | p_sat | h_win_med (pas) |
|---|---|---|---|---|---|
| lorenz | 400 | 0.941 [0.937] | 0.765 | 0.000 | 24 |
| rossler | 400 (cible 846, cap sans effet : p_sat=0) | 0.965 [0.938] | 0.800 | 0.000 | 35 |
| mackey_glass | 291 (budget) | 0.960 [0.957] | 0.816 | 0.000 | 11 |

Lecture : la sous-couverture Mackey-Glass du profil rapide (0.898 [0.847]) était
un artefact de labels (tolérance relative ×10 sur l'erreur un-pas + Hmax=20) —
elle disparaît avec des labels à l'échelle de l'attracteur. L'anomalie
tightness > 1 de Rössler disparaît aussi (0.80). Point restant : Lorenz
sous-couvre légèrement mais systématiquement (0.937–0.943 sur 5/5 seeds vs
cible 0.95) — candidat pour la calibration disjointe (P1) en conditions réelles.

## Diagnostic de la sous-couverture Lorenz (studies/study_calib_thinning.py)

Sweep d'amincissement de la calibration, Lorenz, 5 seeds, α=0.05 :

| stride | cov med [min] | tightness | n_calib |
|---|---|---|---|
| 1 | 0.941 [0.937] | 0.765 | 1200 |
| 12 | 0.941 [0.932] | 0.765 | 116 |
| 24 | 0.922 [0.889] | 0.772 | 58 |
| 48 | 0.731 [0.708] | 0.811 | 29 |

Lecture : la décorrélation n'améliore rien (0.941 identique à stride 12 avec
10× moins de fenêtres) — la corrélation sérielle rend les fenêtres redondantes,
pas biaisantes. Le déficit (~1 pt) est un shift calib→test le long de la
trajectoire. Remède validé (5 seeds) : α_cal = 0.035 pour livrer 0.95 →
couverture 0.951 [0.947], tightness 0.758 (coût < 1 %).

## Ablation des mécanismes conservateurs (studies/study_tightness_ablation.py)

Lorenz + Mackey-Glass (+confirmation Rössler), 5 seeds, α=0.05, linéaire :

| mécanisme | effet mesuré | décision |
|---|---|---|
| coverage guard | +1.2 pt de couverture (Lorenz 0.941→0.929 sans lui) | **conservé** |
| debias (0.4) | −0.5 à −0.7 pt de couverture pour ≤0.009 de tightness | **retiré** (défaut 0.0) |
| bins=2 vs global | global : Lorenz 0.961 [0.953] (seul à tenir la cible au min) mais min Rössler 0.938 < bins 0.943 | bins=2 conservé (couverture conditionnelle) ; `conformal_mode: global` documenté pour régimes type Lorenz |
| block_quantile 0.9→0.5 | −0.2 pt, rien gagné | 0.9 conservé |

Défauts après ablation (no-debias) : Lorenz 0.946 [0.943], Rössler 0.967 [0.943],
Mackey-Glass 0.967 [0.957] — amélioration uniforme de la couverture à tightness
inchangée (±0.003). Le debias corrigeait une sur-couverture qui n'existe plus
depuis la correction des labels (audit C1/C2) : mécanisme historique retiré.

## Benchmark définitif (outputs/benchmark_final.csv, 2026-07-05)

40 runs (4 systèmes × 5 seeds × {linéaire, MLP}), α=0.05, défauts post-ablation.
Table complète : outputs/benchmark_final_table.md (regénérée par
studies/make_results_tables.py). Faits saillants : couverture à la cible sur
logistique/Mackey-Glass/Lorenz-MLP ; Lorenz-linéaire garde son écart de ~1 pt
(remède α_cal documenté) ; **résultat négatif Rössler+MLP** (0.813 [0.700]) —
le split de test ne couvre que ~4 T_λ, fluctuation côté test irréductible ;
budget de données requis pour les systèmes lents. Les modèles forts voient plus
loin (H_w méd 58 vs 23 sur Lorenz, 200 vs 11 sur MG) mais la borne est alors
plus conservatrice (tightness 0.46–0.62) → la tightness pour modèles forts est
la prochaine optimisation.
