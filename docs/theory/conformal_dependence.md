# Conformal sous dépendance temporelle (au-delà de l'échangeabilité)

Module : `src/horizon_conformal_beyond.py` · Étude : `studies/study_conformal_dependence.py`
· Tests : `tests/test_conformal_beyond.py` · Audit : point E1 de `AUDIT_MATH.md`.

## Problème

Notre borne inférieure d'horizon est `L(x) = q̂(x) − c`, où `c` est le quantile conforme
des scores de non-conformité `s_i = q̂(x_i) − H_i` calculés sur des fenêtres de calibration.
Le théorème split-conformal exige que les scores de calibration et de test soient
**échangeables**. Or nos fenêtres d'horizon se chevauchent le long d'une trajectoire unique :
les scores voisins sont fortement autocorrélés, et calibration/test sont des segments
temporels adjacents. L'échangeabilité est violée, donc la garantie
`P(H_real ≥ L) ≥ 1 − α` affichée n'est, en l'état, qu'**empirique** (audit E1).
Deux questions : (i) que peut-on récupérer comme garantie approchée, et à quel coût ?
(ii) que se passe-t-il et que peut-on borner en cas de **dérive** de la distribution des
scores (changement de régime) ?

## Théorie importée

**(a) Théorème split-conformal** (Vovk, Gammerman, Shafer, *Algorithmic Learning in a
Random World*, 2005 ; Lei, G'Sell, Rinaldo, Tibshirani, Wasserman, JASA 2018).
Si les scores `s_1, …, s_n, s_test` sont échangeables, alors pour
`c = s_(⌈(n+1)(1−α)⌉)` (statistique d'ordre), `P(s_test ≤ c) ≥ 1 − α`.
La preuve repose exclusivement sur l'échangeabilité des rangs — la corrélation sérielle
de nos fenêtres chevauchantes la casse. Aucune vitesse de dégradation n'est fournie par
le théorème : hors hypothèses, il ne dit **rien**.

**(b) Conformal beyond exchangeability** (Barber, Candès, Ramdas, Tibshirani,
*Conformal prediction beyond exchangeability*, Annals of Statistics 51(2), 2023).
Avec des poids **fixes** (indépendants des données) `w_i ∈ [0,1]` sur les scores de
calibration, poids normalisés `w̃_i = w_i / (Σ_j w_j + 1)` et le point test recevant la
masse `w̃_{n+1} = 1/(Σ_j w_j + 1)` placée à `+∞`, le quantile pondéré
`c = inf{ s : Σ_{i : s_i ≤ s} w̃_i ≥ 1 − α }` satisfait

```
P(s_test ≤ c) ≥ 1 − α − Σ_{i=1}^n w̃_i · d_TV(Z, Z^i)
```

où `d_TV(Z, Z^i)` est la distance en variation totale entre la suite des données `Z` et
la suite `Z^i` où les points `i` et `n+1` sont échangés. Des poids décroissants avec
l'ancienneté rendent petits les termes associés aux points anciens (les plus « dérivés »),
bornant la perte de couverture sous dérive. **Attention** : `d_TV(Z, Z^i)` porte sur les
**suites jointes**, pas sur les marginales ; sous dépendance sérielle il est > 0 même sans
dérive et n'est pas estimable directement (conséquence mesurée plus bas).

**(c) Validité approchée sous mélange** (dans l'esprit de Chernozhukov, Wüthrich, Zhu,
*Exact and robust conformal inference methods for predictive machine learning with
dependent data*, COLT 2018). Si le processus des scores est stationnaire et β-mélangeant,
des scores séparés d'un écart ≥ longueur de décorrélation sont approximativement
échangeables : la couverture du conformal appliqué aux scores **amincis** (thinning)
est `1 − α − o(1)`, le terme d'erreur décroissant avec le rapport écart/longueur de
mélange. Le coût est la réduction de `n` (ici : ~60 points au lieu de 500).

## Adaptation à notre cadre

- Les « données » sont les scores `s_i = q̂(x_i) − H_i` des fenêtres de calibration,
  ordonnés temporellement (le plus récent en dernier). Nous modélisons leur dépendance
  par un AR(1) gaussien de coefficient `φ = 0.9` (longueur de décorrélation ≈ 10 pas),
  proxy conservateur du chevauchement de fenêtres à stride 1.
- **Thinning** : `disjoint_indices(n_windows, gap)` sélectionne les fenêtres
  `0, gap, 2·gap, …` ; dans le pipeline réel, `gap ≥ (dim−1)·lag + Hmax` rend les fenêtres
  physiquement disjointes (déjà prévu par PLAN_V2 Phase 3).
- **Pondération** : `decay_weights(n, half_life)` avec `w_i = 0.5^((n−1−i)/half_life)` ;
  poids fixes au sens de Barber (aucune dépendance aux valeurs des scores).
- **Quantile pondéré** : `weighted_conformal_quantile(scores, alpha, weights)` implémente
  la convention de la masse test à `+∞` : si la masse cumulée de calibration n'atteint
  jamais `1 − α`, la fonction renvoie `+inf` (borne non informative — la réponse honnête
  quand `n` est trop petit ; l'implémentation historique `conformal_quantile` écrête au
  score max, différence documentée et testée).
- **Reporting** : `coverage_gap_bound(weights, dtv)` renvoie le terme de perte
  `Σ w̃_i · d_TV_i` (ou `Σ w̃_i` si `d_TV` inconnu, multiplicateur du pire cas).

## Algorithme

```
# Calibration sous dépendance (stationnaire) :
idx    = disjoint_indices(n_windows, gap)          # gap >= longueur de décorrélation
c      = weighted_conformal_quantile(scores[idx], alpha)      # poids uniformes
L(x)   = q_hat(x) - c

# Sous suspicion de dérive :
w      = decay_weights(n, half_life)               # récents plus lourds
c      = weighted_conformal_quantile(scores, alpha, w)
loss   = coverage_gap_bound(w, dtv_estime)         # reporting honnête, pas un certificat
```

Complexité : un tri, O(n log n). Aucune dépendance hors numpy ; fonctions pures et seedables
(déterministes).

## Validation numérique

Protocole (`studies/study_conformal_dependence.py`, seed 1234, confirmation seed 20260705) :
scores AR(1) gaussiens `φ = 0.9` (variance marginale 1), `n_calib = 500`, `n_test = 500`
(le test **prolonge** la chaîne de calibration, comme dans le pipeline), 300 réplications,
`α = 0.10`, cible dégradée 0.88. Colonnes : couverture test empirique par réplication
(moyenne, écart-type, `P(cov < 0.88)`), couverture « vraie » conditionnelle à la
calibration `Φ(c − µ_test)` (isole l'effet calibration du bruit côté test), marge moyenne `c`.
Durée totale : **< 1 s** (600 réplications × 2 scénarios).

### Scénario 1 — dépendance stationnaire

| méthode | cov moy. | σ(cov) | P(cov<0.88) | cov vraie | P(vraie<0.88) | marge c |
|---|---|---|---|---|---|---|
| overlapping (n=500) | 0.8869 | 0.0630 | 0.4167 | 0.8891 | 0.3500 | 1.2603 |
| **disjoint gap=8 (n=63)** | 0.8986 | 0.0619 | 0.3367 | 0.8997 | 0.2867 | 1.3331 |
| weighted hl=n/4 | 0.8864 | 0.0667 | 0.4000 | 0.8884 | 0.4033 | 1.2725 |
| **oracle Φ⁻¹(0.9)** | 0.8961 | 0.0430 | **0.3100** | 0.9000 | 0.0000 | 1.2816 |
| disjoint gap=4 (n=125) | 0.8922 | 0.0632 | 0.3667 | 0.8943 | 0.3333 | 1.2929 |
| disjoint gap=16 (n=32) | 0.9025 | 0.0721 | 0.2967 | 0.9037 | 0.2900 | 1.3877 |
| disjoint gap=25 (n=20) | 0.8982 | 0.0793 | 0.3433 | 0.9007 | 0.2967 | 1.3972 |

Lectures :
1. La dépendance coûte **≈ 1.1 point de couverture moyenne** au conformal standard
   (0.8891 vs 0.90 nominal) — c'est la violation quantifiée de l'échangeabilité.
2. Le thinning (gap=8, 63 points) **restaure la couverture moyenne nominale**
   (0.8997 ; confirmation seed 2 : 0.9011) pour **+5.8 %** de marge (+4.8 % au seed 2).
3. Mais `P(cov < 0.88)` ne passe que de 0.4167 à 0.3367 (**×1.24** ; seed 2 : ×1.12),
   loin du ×2 requis. Cause structurelle : l'**oracle** (vrai quantile) a lui-même
   `P(cov < 0.88) = 0.31` — la fluctuation vient du côté **test** (500 points corrélés,
   n_eff ≈ 26) et aucune calibration ne peut la supprimer. Sur la métrique purifiée
   `P(Φ(c) < 0.88)` (oracle = 0), le gain n'est encore que ×1.22 : l'amincissement
   corrige le **biais** mais pas la **variance** de l'estimateur de quantile (l'information
   totale de la chaîne est la même).

### Scénario 2 — dérive (+0.5 σ au milieu de la calibration, test décalé)

| méthode | cov moy. | σ(cov) | P(cov<0.88) | cov vraie | marge c |
|---|---|---|---|---|---|
| standard (uniforme) | 0.8391 | 0.0795 | 0.6500 | 0.8447 | 1.5451 |
| **weighted hl=n/4** | 0.8618 | 0.0834 | 0.5167 | 0.8672 | 1.6689 |
| weighted hl=n/2 | 0.8537 | 0.0793 | 0.5700 | 0.8593 | 1.6192 |
| weighted hl=n/8 | 0.8655 | 0.0918 | 0.4800 | 0.8709 | 1.7119 |
| weighted hl=n/16 | 0.8589 | 0.1060 | 0.4967 | 0.8638 | 1.7209 |
| oracle (décalé) | 0.8958 | 0.0426 | 0.3000 | 0.9000 | 1.7816 |

Gains appariés (erreur-type appariée sur 300 réplications) :
- `hl = n/4` : **+2.28 ± 0.23 points** (confirmation : +2.61) ; inflation de marge
  stationnaire : **+0.97 %** (confirmation : −0.35 %).
- `hl = n/8` (réglage optimal du grid) : **+2.64 ± 0.34** (confirmation : **+2.99 ± 0.32**) ;
  inflation stationnaire +1.22 % (confirmation : −0.64 %). Moyenne des deux seeds ≈ 2.8 pts.

Borne de Barber avec le d_TV **marginal** exact du décalage gaussien
(`d_TV(N(0,1), N(0.5,1)) = 2Φ(0.25) − 1 = 0.1974`, appliqué à la moitié pré-dérive) :
standard ≥ 0.8015 (perte 0.0985) ; `hl=n/4` ≥ 0.8607 (perte 0.0393) ; `hl=n/8` ≥ 0.8885
(perte 0.0115).

**Mise en garde mesurée** : la couverture observée de `hl=n/8` (0.8709) est **inférieure**
au plancher plug-in 0.8885, et en stationnaire le conformal uniforme (0.8891) est sous le
plancher `d_TV = 0` (0.90). Ce n'est pas une violation du théorème : le `d_TV` de Barber
porte sur les **suites échangées**, et la dépendance AR(1) y contribue positivement même à
marginales identiques. Conclusion pratique : sous dépendance sérielle, brancher le d_TV
marginal dans `coverage_gap_bound` donne une **indication structurelle** (comparer des
schémas de poids), **pas un certificat**. Le seul usage certifié du bound suppose des
scores indépendants à dérive près.

## Bénéfice projet

Mesuré, aux réglages prescrits :
1. **Thinning disjoint** : supprime le biais de couverture moyenne dû à la dépendance
   (+1.1 pt, retour au nominal 0.900) pour +5-6 % de marge ; fonde la seule affirmation
   théorique défendable (validité approchée sous mélange). Ne réduit `P(cov < 0.88)` que
   de ×1.2 (critère : ×2) — plafond structurel démontré par l'oracle (0.31).
2. **Poids décroissants** : +2.3 à +3.0 points de couverture sous dérive de +0.5 σ, pour
   ~1 % d'inflation de marge en stationnaire (critère : ≥ 3 points ; atteint seulement à
   la limite, `2.99 ± 0.32` au seed de confirmation, moyenne inter-seeds ≈ 2.8).
3. **Reporting** : `coverage_gap_bound` chiffre la perte de garantie sous dérive
   (0.0985 → 0.0115 en passant d'uniforme à `hl=n/8`), avec la limite d'interprétation
   ci-dessus.

**Ce que le README peut désormais affirmer légitimement** :
- au lieu de « garantie ≥ 1 − α » : « couverture **approximativement valide sous hypothèse
  de mélange** lorsque la calibration utilise des fenêtres **disjointes**
  (gap ≥ longueur de décorrélation) ; en simulation AR(1) φ=0.9, la couverture moyenne
  seed-level est nominale (0.900 vs 0.889 pour les fenêtres chevauchantes), la
  fluctuation par seed restant dominée par la dépendance du jeu de test » ;
- sous dérive : « le quantile conforme pondéré (Barber et al. 2023) borne la perte de
  couverture par `Σ w̃_i·d_TV` ; cette borne n'est un certificat que pour des scores
  indépendants — sous dépendance sérielle elle est indicative » ;
- ne **pas** affirmer : garantie finie-échantillon inconditionnelle, ni réduction du
  risque de sous-couverture par seed (non obtenue : ×1.2 seulement).

## Critère de décision

Prescrits par la directive, appliqués tels quels :
- **disjoint** : intégrer si `P(cov < 0.88)` réduit d'un facteur ≥ 2 vs overlapping.
  Résultat : ×1.24 (seed 1), ×1.12 (seed 2) → **non atteint**. De plus le critère est
  structurellement inatteignable ici : l'oracle plafonne à 0.31 vs 0.4167 (×1.34 max).
  Sur la métrique purifiée (côté calibration seule) : ×1.22 → non atteint également.
- **weighted** : intégrer si gain de couverture sous dérive ≥ 3 points avec ≤ 10 %
  d'inflation de marge stationnaire. Résultat : +2.28 ± 0.23 (hl=n/4, prescrit) ;
  +2.64/+2.99 (hl=n/8, réglé puis confirmé sur seed indépendant) ; inflation ~1 % ≪ 10 %
  → **non atteint** (à la marge).

## Verdict

**Shelve conditionnel** (pas d'intégration par défaut dans le pipeline) :
- Les deux critères d'intégration chiffrés échouent honnêtement aux réglages prescrits.
- On **retient** du module : (i) la correction de langage du README (ci-dessus), qui ne
  coûte rien et est exigée par l'honnêteté scientifique ; (ii) `disjoint_indices` comme
  brique de la Phase 3 de PLAN_V2 (fenêtres disjointes), justifiée par la suppression
  mesurée du biais moyen (+1.1 pt) — mais en sachant qu'elle ne réduit pas le risque
  par seed ; (iii) `weighted_conformal_quantile` + `coverage_gap_bound` comme **option
  opt-in** pour les régimes à dérive suspectée (gain reproductible ~2.3-3.0 pts pour ~1 %
  de marge), à réévaluer sur les vrais scores d'horizon (Lorenz avec changement de régime
  ρ 28 → 24.5, protocole prévu en Phase 4) où la dérive peut être plus forte que +0.5 σ.
- Le module reste dans `src/` avec ses 19 tests : fonctions pures, zéro dépendance
  nouvelle, zéro impact sur le pipeline existant (suite complète verte).
