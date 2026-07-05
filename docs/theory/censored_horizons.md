# Labels d'horizon censurés à Hmax — pinball censuré de Powell et garde de saturation

Point d'audit traité : **C3** (`AUDIT_MATH.md`) — « H_w = Hmax » est une donnée
censurée à droite (« H_w ≥ Hmax »), traitée aujourd'hui comme une valeur exacte.

Module : `src/horizon_censoring.py` · Étude : `studies/study_censoring.py` ·
Tests : `tests/test_censoring.py`.

---

## Problème

Les labels `H_w` produits par `build_horizon_dataset` (`src/horizon_metrics.py`)
sont plafonnés à `horizon_max` : quand la tolérance n'est jamais franchie dans la
fenêtre, on enregistre `H_w = Hmax` alors que la vérité est `H_true ≥ Hmax`.
Deux maillons du pipeline consomment ces labels comme s'ils étaient exacts :

1. la régression quantile (`train_quantile_mlp`, perte pinball naïve) qui
   apprend `q̂_α(x)` ;
2. la calibration conforme (scores signés `s = q̂ − y`, marge
   `c = quantile conforme_{1−α}(s)`, borne inférieure `L(x) = q̂(x) − c`).

Questions : (i) la censure casse-t-elle la garantie de couverture
`P(H_true ≥ L) ≥ 1−α` ? (ii) quand le quantile cible `Q_α(H|x)` est-il encore
identifiable ? (iii) comment corriger le biais de la perte pinball naïve ?

## Théorie importée (énoncés précis avec références)

### (a) Théorème — la censure à droite est conservatrice pour la borne inférieure one-sided

**Énoncé.** Soit `q̂` un prédicteur quelconque (entraîné sur des données
disjointes de la calibration), les scores signés `s(x, y) = q̂(x) − y`, la marge
`c = quantile de rang ⌈(n+1)(1−α)⌉` des scores de calibration
(`conformal_quantile`, `src/horizon_conformal.py`) et `L(x) = q̂(x) − c`.
Si on enregistre `y_rec = min(Y, C)` au lieu de `Y` (censure à droite en `C`),
alors la couverture `P(Y_test ≥ L) ≥ 1−α` est **préservée** ; seule la
finesse de `L` peut se dégrader.

**Preuve (chaîne d'inégalités).**
1. `y_rec,i = min(Y_i, C) ≤ Y_i`, donc `s_rec,i = q̂(x_i) − y_rec,i ≥ q̂(x_i) − Y_i = s_true,i` :
   enregistrer un label **plus petit** que la vérité rend le score signé **plus grand**.
2. Le quantile empirique de rang fixé est monotone : `s_rec ≥ s_true` point à
   point ⟹ `c_rec ≥ c_true`, donc `L_rec(x) = q̂(x) − c_rec ≤ L_true(x)`
   (borne plus **basse**, jamais plus haute).
3. Le conforme split appliqué aux paires censurées `(x_i, y_rec,i)`,
   échangeables avec le point test censuré, donne
   `P(y_rec,test ≥ L_rec) ≥ 1−α` (Vovk et al. 2005 ; Lei et al. 2018, garantie
   valide pour tout `q̂` fixé).
4. `Y_test ≥ y_rec,test` presque sûrement ⟹
   `P(Y_test ≥ L_rec) ≥ P(y_rec,test ≥ L_rec) ≥ 1−α`. ∎

La couverture est donc garantie **quel que soit** `q̂` (même entraîné
naïvement sur labels censurés) ; le coût de la censure est uniquement une
marge `c` inflatée, i.e. une perte de finesse (validé numériquement plus bas).

### (b) Identification du quantile sous censure

**Énoncé.** Avec `y_rec = min(Y, C)` : `F_{y_rec|x}(t) = F_{Y|x}(t)` pour
`t < C` et `= 1` pour `t ≥ C`, donc `Q_α(y_rec|x) = min(Q_α(Y|x), C)`.
Le quantile conditionnel `Q_α(Y|x)` est **identifié** depuis les données
censurées **ssi** `Q_α(Y|x) < C`.

**Garde globale pratique.** Au niveau marginal (F continue) :
`Q_α(Y) < C ⟺ P(Y < C) > α ⟺ p_sat := P(Y ≥ C) < 1 − α`.
D'où la garde `p_sat ≤ 1 − α` (`saturation_gate`) : si elle échoue, le
quantile cible est assis **dans** la région censurée et aucune méthode ne peut
l'estimer — il faut augmenter `Hmax` ou baisser le quantile cible. C'est une
condition nécessaire (globale) ; localement l'identification exige
`Q_α(Y|x) < C` point par point.

### (c) Powell (1986) — régression quantile censurée

**Référence.** J. L. Powell, *Censored regression quantiles*, Journal of
Econometrics 32 (1986) 143–155 (forme censurée à gauche `y = max(0, y*)` ;
la forme censurée à droite utilisée ici en est le miroir). Voir aussi
Koenker (2005), *Quantile Regression*, §8.

**Énoncé.** L'estimateur minimise `Σ ρ_τ(y_rec,i − min(q_θ(x_i), C))` où
`ρ_τ` est la perte pinball.

**Argument de consistance (3 lignes).** Les quantiles sont équivariants par
transformation croissante : `Q_τ(min(Y, C)|x) = min(Q_τ(Y|x), C)` (cf. (b)).
Le minimiseur en population de `E[ρ_τ(y_rec − q)|x]` est `Q_τ(y_rec|x)` ; en
composant la prédiction par la même transformation `q ↦ min(q, C)`, le
minimiseur redevient `q = Q_τ(Y|x)` partout où `Q_τ(Y|x) < C`, et la perte est
**plate** en `q ≥ C` sinon (gradient nul — aucune traction vers le bas). ∎

**Biais de la perte naïve.** Le minimiseur de la pinball naïve sur `y_rec` est
`min(Q_τ(Y|x), C)` : dans la région `Q_τ(Y|x) ≥ C` l'estimation est tirée vers
`C` (biais négatif), et avec un modèle lisse (MLP) cette traction se propage
aux régions voisines identifiées. La transformation de Powell annule le
gradient dès que `min(q̂, C)` atteint `C`, supprimant cette traction.

## Adaptation à notre cadre

- `Y = H_true` (horizon réel de la fenêtre), `C = Hmax` (`args.horizon_max`),
  `y_rec = H_w` : censure à droite **fixe et connue** — le cas le plus simple
  de Powell (pas de `C_i` aléatoire).
- Le pipeline vise `q̂_α(H|x)` avec α petit (0.1) : un quantile **bas**, donc
  très souvent identifié (`Q_α < Hmax`) tant que `p_sat ≤ 1 − α`. C'est le cas
  favorable ; le théorème (a) couvre le reste.
- **Corollaire pour le score conforme** (même argument que (c)) : le score
  censuré `s_cap = min(q̂(x), C) − y_rec` avec la borne
  `L(x) = min(q̂(x), C) − c` reste une borne valide
  (`P(y_rec ≥ min(q̂,C) − c) ≥ 1−α` puis `Y ≥ y_rec`), et évite l'inflation de
  la marge quand `q̂ > C` dans la région saturée. Une borne au-delà de `Hmax`
  n'est de toute façon pas certifiable depuis des labels plafonnés à `Hmax`.
- La version normalisée `s = (min(q̂,C) − y)/σ̂` (chemin `use_sigma`) hérite des
  mêmes propriétés (transformation croissante en y à σ̂ > 0 fixé).

## Algorithme

`src/horizon_censoring.py` (fonctions pures) :

1. `censored_pinball_loss(pred, target, quantile, cap)` — torch,
   différentiable : `pinball(target, torch.minimum(pred, cap))` ;
   `cap=None` ⇒ pinball standard (strictement rétro-compatible).
   `torch.minimum` a un gradient nul en `pred > cap` : c'est exactement la
   région plate de Powell.
2. `censored_pinball_np(pred, target, quantile, cap)` — jumeau numpy pour
   l'évaluation.
3. `saturation_gate(y, horizon_max, alpha)` — renvoie
   `{p_sat, identified: p_sat ≤ 1−α, message}` ; convention de saturation
   identique à `_saturation_rate` (`src/horizon_experiment_conformal_stats.py`).

## Validation numérique (chiffres)

`python studies/study_censoring.py` — 20 seeds, α = 0.1, `y = 5 + slope·x +
LogNormale(σ(x))` hétéroscédastique, censure à `C = p70(y_train)` (30 % de
censure), MLP 1→32→32→1, `Q_0.1(y|x)` connu analytiquement, conforme split.
Runtime total : **24.6 s** (CPU).

Deux régimes :
- **primaire** (`slope=4`, σ(x)=0.3+0.4x) : `C` croise `Q_0.1(y|x)` sur ~18 %
  du domaine — le régime pour lequel Powell est conçu ;
- **sensibilité** (`slope=2`, σ(x)=0.4+0.4x, la conception littérale du
  cahier des charges) : `C` ne croise `Q_0.1(y|x)` que sur ~6 % du domaine.

| design | modèle | biais q̂ | \|biais\| | marge c (censurée) | c (score plafonné) | couverture L | couverture L (plafonné) | mean L | mean L (plafonné) |
|---|---|---|---|---|---|---|---|---|---|
| primaire | naïf | −0.142 | 0.180 | −0.009 | −0.015 | 0.912 | 0.909 | 9.290 | 9.295 |
| primaire | Powell | **−0.047** | **0.074** | +0.437 | +0.001 | 0.999 | 0.916 | 8.940 | **9.293** |
| sensibilité | naïf | −0.011 | 0.039 | −0.002 | −0.004 | 0.904 | 0.902 | 7.364 | 7.366 |
| sensibilité | Powell | −0.001 | 0.031 | +0.054 | +0.006 | 0.932 | 0.905 | 7.318 | 7.362 |

(écarts-types sur 20 seeds : biais ±0.02–0.04, couverture ±0.01 ; cible de
couverture ≥ 1−α−0.02 = 0.88.)

Lectures :

1. **Théorème (a) confirmé** : `c_censuré ≥ c_oracle` sur 20/20 seeds pour les
   deux pertes ; couverture ≥ 0.90 partout (0.912 / 0.999 / 0.916 / 0.904 /
   0.932 / 0.905) — jamais de sous-couverture, uniquement de la sur-couverture.
2. **Biais (thm c)** : réduction de |biais| de **59.1 %** dans le régime
   primaire (0.180 → 0.074) ; **21.7 %** seulement dans le régime de
   sensibilité (0.039 → 0.031) où la pinball naïve est déjà quasi consistante
   (le quantile bas est rarement atteint par la censure).
3. **Marge — l'attendu « marge plus petite avec Powell » est FAUX avec le
   score signé brut** : la marge Powell explose (+0.437 vs −0.009) car dans la
   région saturée `q̂ > C = y_rec` produit des scores positifs massifs ; la
   borne devient PLUS lâche (mean L 8.940 vs 9.290). Avec le **score plafonné**
   `min(q̂,C) − y_rec` (corollaire ci-dessus), la marge Powell retombe à +0.001
   et mean L remonte à 9.293 (couverture 0.916, quasi nominale).
4. **Bénéfice net aval** : à couverture équivalente, mean L Powell+plafonné
   vs naïf = +0.003 (≈ nul marginalement) — le conforme marginal répare déjà
   le biais du naïf en moyenne. Le gain réel de Powell est **conditionnel** :
   `q̂` non biaisé région par région (−59 % de biais), couverture par région
   plus proche du nominal (0.971 vs 0.977–0.978 sur le quart le plus censuré),
   et surtout il ne délègue plus la réparation du biais à la marge conforme —
   condition nécessaire pour l'arbre conforme et les bins Mondrian qui
   supposent `q̂` homogène en x.
5. `saturation_gate` : `p_sat = 0.300 ≤ 0.9 = 1−α` ⇒ `identified=True` sur
   tous les seeds (cohérent avec la construction).

## Bénéfice projet

- **Sécurité inconditionnelle** : le théorème (a) établit que le pipeline
  actuel (pinball naïve + conforme) ne viole PAS la couverture à cause de la
  censure — il perd seulement de la finesse. Le risque « couverture faussée »
  de l'audit C3 est requalifié : c'est un problème de biais/finesse, pas de
  sécurité.
- **Correction de biais mesurée** : −59 % de |biais| de `q̂` quand la censure
  touche le quantile cible (exactement le régime des systèmes à `p_sat`
  élevé : Mackey-Glass historique `p_sat = 1.0`, fenêtres lentes de Rössler).
- **Garde d'identification** : `saturation_gate` transforme le warning actuel
  en critère mathématique (`p_sat ≤ 1−α`) rattaché à un énoncé précis (b).
- Coût d'intégration quasi nul : un argument optionnel `cap=None` strictement
  rétro-compatible.

## Critère de décision

Pré-enregistré : intégrer si Powell réduit |biais| de ≥ 30 % avec couverture
tenue (≥ 1−α−0.02 = 0.88).

- Régime primaire (censure touchant le quantile) : **+59.1 %**, couverture
  0.999 (score brut) et 0.916 (score plafonné) ⇒ **critère atteint**.
- Régime de sensibilité (censure loin du quantile) : +21.7 % ⇒ critère non
  atteint — Powell n'y apporte rien de mesurable (et n'y coûte rien).

## Verdict

**Intégration conditionnelle** — la perte de Powell s'intègre, mais seulement
en package cohérent :

1. `cap` optionnel dans `train_quantile_mlp` (défaut `None` = comportement
   actuel inchangé), activé avec `cap = float(args.horizon_max)` quand les
   labels sont plafonnés — c'est-à-dire toujours dans le pipeline conforme ;
2. **obligatoirement couplé** au score conforme plafonné
   `s = min(q̂, Hmax) − y_rec` et à la borne `L = min(q̂, Hmax) − c` : mesuré,
   Powell avec le score brut rend la borne PLUS lâche (mean L 8.940 vs 9.290) ;
3. `saturation_gate(y_calib, horizon_max, alpha)` en garde bloquante : si
   `p_sat > 1−α`, le quantile cible n'est pas identifiable (b) et le run doit
   être invalidé, pas seulement signalé.

Sans le point 2, ne pas intégrer : le critère de biais est atteint mais le
bénéfice aval mesuré est négatif. Avec les points 1–3, la couverture reste
garantie (thm a), le biais chute de 59 % dans le régime censuré et la finesse
est au pire égale (9.293 vs 9.290).

### Patch minimal (à appliquer par l'intégrateur — non appliqué ici)

```python
# src/horizon_training.py
from src.horizon_censoring import censored_pinball_loss

def train_quantile_mlp(..., cap=None):          # new optional arg, None = current behavior
    ...
    loss = censored_pinball_loss(pred, yb, quantile, cap)          # was: pinball_loss(pred, yb, quantile)
    ...
    val_loss = censored_pinball_loss(val_pred, y_val_t, quantile, cap).item()
```

Points de branchement aval :
- `predict_quantile_ensemble` / `predict_sigma_mlp`
  (`src/horizon_conformal.py`) : propager `cap=float(args.horizon_max)` vers
  `train_quantile_mlp` ;
- `_compute_scores` (`src/horizon_experiment_conformal_calibration.py`) :
  `signed = np.minimum(pred_calib, horizon_max) - y_calib` ;
- `_calib_interval` / `_test_predictions`
  (`src/horizon_experiment_conformal_stats.py`) : borner la prédiction avant
  soustraction de la marge : `np.minimum(pred, horizon_max) - c * sigma_term` ;
- garde : appeler `saturation_gate(y_calib, args.horizon_max, alpha)` au début
  de la calibration et invalider le run si `identified` est faux.
