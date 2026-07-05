# Longueur de bloc automatique (Politis-White) pour le bootstrap de couverture

## Problème

L'audit (`AUDIT_MATH.md`, point E4) a remplacé la borne de Wilson — invalide sous
dépendance — par un bootstrap par blocs mouvants circulaires pour la borne
inférieure `coverage_lb` de la couverture empirique
(`src/horizon_scientific_eval.py::_block_bootstrap_lower_bound`). Les hits de
couverture proviennent de fenêtres de test chevauchantes d'une même trajectoire :
ils sont fortement autocorrélés.

La longueur de bloc y était fixée par une heuristique arbitraire :
`b = max(10, round(4·n^(1/3)))`, soit `b = 32` pour `n = 500`, sans aucun lien
avec la dépendance réelle de la série de hits. Un bloc trop court sous-estime la
variance de la moyenne (LB trop optimiste) ; un bloc trop long réduit le nombre
de blocs effectifs (LB instable). `PLAN_V2.md` (Phase 3.3) proposait d'estimer
la portée d'autocorrélation par Politis-White. Question mesurable : cette règle
automatique améliore-t-elle la calibration `P(LB ≤ couverture vraie) = 1 − α_ci` ?

## Théorie importée

- **Politis & Romano (1995)**, *Bias-corrected nonparametric spectral
  estimation*, J. Time Series Analysis 16(1) : noyau **flat-top trapézoïdal**
  `λ(t) = 1` pour `|t| ≤ 1/2`, `λ(t) = 2(1−|t|)` pour `1/2 < |t| ≤ 1`, `0`
  sinon. Les fenêtres flat-top donnent des estimateurs spectraux à biais
  d'ordre supérieur.
- **Politis & White (2004)**, *Automatic block-length selection for the
  dependent bootstrap*, Econometric Reviews 23(1), 53-70, avec la correction de
  **Patton, Politis & White (2009)**, Econometric Reviews 28(4), 372-375.
  Énoncé : pour le bootstrap (stationnaire / blocs mouvants) de la moyenne
  d'une série stationnaire à mélange suffisamment rapide, la longueur de bloc
  minimisant le MSE de l'estimateur bootstrap de la variance est

  `b_opt = (2 G² / D)^(1/3) · n^(1/3) · (1 + o(1))`

  où `G = Σ_k |k| γ_k` (dérivée généralisée de la densité spectrale en 0),
  `γ_k` l'autocovariance au lag `k`, et `D` une constante dépendant du schéma
  de bootstrap ; pour le bootstrap stationnaire (correction PPW 2009) :
  `D = 2 g(0)²` avec `g(0) = Σ_k γ_k`.
- **Règle empirique de bande** (PPW) : `m̂` = plus petit lag tel que les `K_n`
  autocorrélations suivantes vérifient toutes `|ρ_k| < c·√(log10(n)/n)`
  (`c = 2`, `K_n = max(5, ⌈√log10(n)⌉)`) ; bande `M = 2·m̂` ; `G` et `g(0)`
  estimés par fenêtre flat-top de bande `M`.

## Adaptation à notre cadre

- La série d'entrée est la suite **binaire** des hits de couverture (0/1) des
  fenêtres de test ; stationnarité approximative supposée à l'échelle du run.
- Nous appliquons les constantes du bootstrap stationnaire (`D = 2 ĝ(0)²`,
  spécification projet) au bootstrap par **blocs circulaires** implémenté dans
  `_block_bootstrap_lower_bound`. Le facteur exact blocs-circulaires
  (`D = (4/3) g(0)²`) ne changerait `b` que d'un facteur `(3/2)^(1/3) ≈ 1.14` —
  négligeable devant le clamp.
- Sortie **clampée à `[10, n//3]`** : minimum 10 (compatible avec le plancher
  existant du code appelant), maximum `n//3` (au moins ~3 blocs indépendants).
- Cas dégénérés (série constante ou numériquement constante, `n < 2`, valeurs
  non finies, `M = 0`, `D ≤ 0`) → retourne 10.

## Algorithme (implémenté dans `src/horizon_blocklen.py`)

Entrée : série `x` (hits), constante `c = 2`.

1. `γ_k` = autocovariances biaisées (normalisation `1/n`), `ρ_k = γ_k/γ_0`.
2. Seuil `s = c·√(log10(n)/n)` ; `K_n = max(5, ⌈√log10(n)⌉)` ;
   `m̂` = plus petit `m ≥ 0` tel que `|ρ_k| < s` pour les `K_n` lags
   `k = m+1..m+K_n` (recherche jusqu'à `⌈√n⌉ + K_n` ; repli : dernier lag
   significatif).
3. `M = 2·m̂` (cap à `n−1`).
4. `Ĝ = Σ_{|k|≤M} λ(k/M)·|k|·γ_k` ; `D̂ = 2·(Σ_{|k|≤M} λ(k/M)·γ_k)²`,
   `λ` = flat-top trapézoïdal.
5. `b_opt = ⌈(2Ĝ²/D̂)^(1/3) · n^(1/3)⌉`, clampé à `[10, n//3]`.

Fonctions pures numpy, déterministes, testées dans `tests/test_blocklen.py`
(12 tests : noyau, autocovariances, séries constantes/dégénérées, i.i.d.,
croissance avec la dépendance, ordre de grandeur AR(1) φ=0.9, clamps,
déterminisme, intégration avec `_block_bootstrap_lower_bound`).

## Validation numérique

Protocole (`studies/study_blocklen.py`, seed 20260705, ~30 s) : hits binaires
`1{z_t ≤ q}` avec `z` AR(1) gaussien, `q` fixé pour un taux marginal ≈ 0.90
(imite les hits de couverture à α = 0.1) ; vraie moyenne estimée par simulation
longue (n = 2 000 000) ; `n = 500`, 200 réplications ; LB bootstrap à
`α_ci = 0.05` via la fonction de production `_block_bootstrap_lower_bound`
(1000 rééchantillons), **appariée** (mêmes hits, même graine bootstrap) entre
(a) heuristique `b = 32` et (b) Politis-White. Nominal : **0.95**, SE
Monte-Carlo ≈ 0.015.

| φ | vraie moyenne | b old | b PW (moy [min,max]) | calib old | calib PW | moy(vrai−LB) old | moy(vrai−LB) PW | désaccords appariés old/PW |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.9003 | 32 | 10.0 [10,10] | 0.940 | 0.935 | 0.0213 | 0.0217 | 2 / 1 |
| 0.5 | 0.9003 | 32 | 10.0 [10,12] | 0.935 | 0.935 | 0.0298 | 0.0299 | 0 / 0 |
| 0.9 | 0.8991 | 32 | 20.7 [10,55] | 0.815 | 0.805 | 0.0587 | 0.0575 | 4 / 2 |

(« désaccords appariés a/b » : nombre de réplications couvertes seulement par
l'heuristique / seulement par PW.)

Sweep diagnostique à φ = 0.9 (mêmes 200 réplications, longueurs de bloc fixes) :

| b | 10 | 20 | 32 | 50 | 80 | 120 | 166 |
|---|---|---|---|---|---|---|---|
| calibration | 0.785 | 0.800 | 0.815 | 0.835 | 0.830 | 0.800 | 0.785 |

**Aucune** longueur de bloc n'atteint 0.95 à φ = 0.9 : le déficit de
calibration (~0.12-0.16) n'est pas imputable au choix de `b` mais à la méthode
elle-même — LB percentile d'un bootstrap par blocs qui sous-estime la variance
de la moyenne à `n = 500` sous forte dépendance (biais connu O(b/n) + absence
de studentisation), cf. Hall, Horowitz & Jing (1995), Lahiri (2003).

## Bénéfice projet

- **Bénéfice mesuré : nul.** À φ = 0.5, les deux règles produisent des
  décisions de couverture strictement identiques (0 désaccord sur 200). À
  φ = 0 et φ = 0.9 les écarts (≤ 0.010) sont sous le bruit Monte-Carlo
  (± 0.015) ; au point estimé, PW est même légèrement *plus loin* du nominal à
  φ = 0.9 (0.805 vs 0.815, effet net apparié −2/200).
- Bénéfice de principe seulement : remplace une constante arbitraire (4) par
  une règle publiée et auto-adaptative. Cela ne se traduit par aucune
  amélioration de la calibration de `coverage_lb` dans le régime d'usage
  (n ≈ 500, hits ~0.9).

## Critère de décision

« Intégrer si la calibration PW est plus proche du nominal pour φ = 0.5 et
φ = 0.9 sans dégrader φ = 0. »

- φ = 0.5 : égalité exacte (0.935 vs 0.935) — pas « plus proche ».
- φ = 0.9 : PW plus éloigné (0.805 vs 0.815).
- φ = 0.0 : pas de dégradation significative (0.935 vs 0.940, dans le bruit).

**Critère NON satisfait.**

## Verdict

**SHELVE** (ne pas faire de Politis-White le défaut de `coverage_lb`).

- Le module `src/horizon_blocklen.py` reste disponible (pur, testé,
  déterministe) pour tout usage futur du bootstrap par blocs ; l'édit demandé
  dans `_block_bootstrap_lower_bound` (défaut `block_len=None` → PW avec repli
  heuristique) est en place et **mesurablement neutre** (différences sous le
  bruit MC) — le retirer ou le garder ne change pas les résultats rapportés ;
  si le verdict est appliqué strictement, le retour à l'heuristique est un
  changement de 2 lignes (et suppression du test
  `test_scientific_eval_uses_politis_white_when_block_len_none`).
- Le vrai levier pour une `coverage_lb` honnête sous forte dépendance n'est
  pas la longueur de bloc : c'est la **méthode d'intervalle** (bootstrap par
  blocs *studentisé* ou calibré, ou sous-échantillonnage de Politis-Romano),
  et/ou davantage de fenêtres de test. À documenter comme limite connue :
  avec des hits fortement autocorrélés (φ ≈ 0.9), la LB actuelle à
  α_ci = 0.05 se comporte comme une LB à α_ci ≈ 0.17-0.20, quelle que soit la
  longueur de bloc.
