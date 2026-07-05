# Borne d'horizon certifiée (non statistique) par constante de Lipschitz du modèle

Module : `src/horizon_certified.py` · Étude : `studies/study_certified.py` ·
Tests : `tests/test_certified.py`

## Problème

Le pipeline actuel produit une borne inférieure **statistique** L(x) de l'horizon de
prédictibilité : par construction conforme, elle est violée sur une fraction α des
fenêtres, et elle repose sur des hypothèses distributionnelles (échangeabilité) que
l'audit (`AUDIT_MATH.md`, points E1-E3) montre fragiles sur séries temporelles. L'audit
D2 pointe en outre que `horizon_model` (récurrence `e_{t+1} <= g·e_t + δ` avec `g` =
quantile 0.95 de croissances empiriques) est présenté comme une borne alors que `g`
n'y majore rien : « construire une vraie borne (Lipschitz du modèle + bruit borné) »
est précisément la correction suggérée. Ce document construit cette borne : un
horizon certifié `h_cert` tel que **chaque** fenêtre (pas 1−α d'entre elles) vérifie
`H_w >= h_cert`, sans aucune hypothèse distributionnelle.

## Théorie importée (énoncés précis avec références)

**T1 — Norme d'opérateur ℓ∞ d'une matrice** (Horn & Johnson, *Matrix Analysis*, §5.6) :
pour `W ∈ R^{m×n}`, `||W||_∞ := sup_{v≠0} ||Wv||_∞ / ||v||_∞ = max_i Σ_j |W_ij|`
(plus grande somme absolue de ligne). Pour une fonction scalaire différentiable `f`,
la constante de Lipschitz par rapport à la norme sup est
`L_∞(f) = sup_x ||∇f(x)||_1` (dualité ℓ∞/ℓ1).

**T2 — Borne produit par couches pour les réseaux de neurones**
(Szegedy et al., *Intriguing properties of neural networks*, ICLR 2014, §4.3 ;
Virmaux & Scaman, *Lipschitz regularity of deep neural networks*, NeurIPS 2018) :
pour `f = W_k ∘ σ ∘ W_{k-1} ∘ … ∘ σ ∘ W_1` avec σ 1-lipschitzienne appliquée
élément par élément (Tanh, ReLU), et pour toute norme d'opérateur subordonnée,
`Lip(f) <= Π_i ||W_i||`. C'est une **borne supérieure** : le calcul exact est
NP-difficile (Virmaux & Scaman 2018, thm. 2) et le produit peut être lâche pour les
réseaux profonds ; notre MLP (`MLPPredictor`, activation **Tanh** vérifiée dans
`src/horizon_models.py`) a **2 couches cachées** — la borne reste modérément lâche
(mesuré ci-dessous). Pour la comparaison, la constante ℓ2 s'obtient par le produit
des normes spectrales (`torch.linalg.svdvals`).

**T3 — Lemme de Grönwall discret / récurrence affine**
(standard ; cf. Holte, *Discrete Gronwall lemma and applications*, 2009) : si
`e_{h+1} <= G·e_h + δ` avec `G >= 1`, `δ >= 0`, `e_0 = 0`, alors
`e_h <= δ·(G^h − 1)/(G − 1)` (et `e_h <= h·δ` si `G = 1`). L'algèbre de cette
forme close est déjà implémentée et correcte dans
`horizon_from_model_bound_by_growth` (`src/horizon_utils.py`) — l'audit D2 en
validait l'algèbre interne, seul le `g` injecté était illégitime.

## Adaptation à notre cadre

Le pas autorégressif dans l'espace d'embedding est le décalage
`F(x) = (x_2, …, x_d, f(x))` (l'audit B4 rappelle que son jacobien est une matrice
compagnon, pas `∇f`). En **norme sup**, la structure compagnon se borne exactement :

**Proposition.** `||F(x) − F(y)||_∞ <= max(1, L_∞(f)) · ||x − y||_∞`.

*Preuve.* Les `d−1` premières coordonnées de `F(x) − F(y)` sont `x_{i+1} − y_{i+1}`,
de valeur absolue `<= ||x − y||_∞`. La dernière est `f(x) − f(y)`, de valeur absolue
`<= L_∞(f)·||x − y||_∞` par définition de `L_∞`. Le max des deux donne
`max(1, L_∞(f))·||x − y||_∞`. ∎

Soit `δ = sup |f(état vrai) − valeur vraie suivante|` le résidu un-pas sur états
vrais, et `E_h` l'erreur sup (sur tout le buffer de fenêtre) après `h` pas de
rollout, `E_0 = 0`. À chaque pas, la nouvelle valeur prédite vérifie
`|f(x̂) − x_vrai| <= |f(x̂) − f(x)| + |f(x) − x_vrai| <= L_∞·E_h + δ`, les autres
coordonnées étant décalées sans croissance ; d'où `E_{h+1} <= G·E_h + δ` avec
`G = max(1, L_∞(f))`. La preuve vaut pour tout `lag >= 1` (l'erreur est mesurée sur
le buffer complet, dont chaque nouveau terme dépend de `d` entrées du buffer).
Par T3, `E_h <= δ·(G^h − 1)/(G − 1)`, et l'erreur du pas `h` (dernière coordonnée)
est `<= E_h`. Donc :

`h_cert := min{ h >= 1 : δ·(G^h − 1)/(G − 1) >= τ }`  ⟹  pour tout `h < h_cert`,
l'erreur du pas `h` est `< τ`, donc **toute** étiquette `H_w` (premier pas où
l'erreur atteint τ, y compris avec K pas consécutifs) vérifie `H_w >= h_cert`.

**Maillon faible (honnête).** `δ` est un **sup empirique** des résidus sur
train+val+calib. Le résultat est donc « certifié modulo la validité de la borne de
résidu sur l'attracteur » : si une région visitée au test produit un résidu un-pas
`> δ`, la garantie peut y céder. C'est qualitativement différent d'une borne par
quantile : aucune hypothèse distributionnelle, validité pour **chaque** fenêtre (pas
1−α d'entre elles), et le seul point de foi est un sup de fonction continue sur un
attracteur compact, estimé sur ~6400 états. Sur le segment même où `δ` est mesuré,
`H_w >= h_cert` est un théorème (testé : `test_certified_horizon_sound_on_same_segment`).

## Algorithme

`src/horizon_certified.py` (fonctions pures, seedées côté étude) :

1. `lipschitz_linf(model, input_dim)` — `LinearAR` : `||w||_1` sur `weights[:-1]`
   (gradient constant ⟹ valeur **exacte et globale**). MLP (via `TorchWrapper`) :
   produit des normes ℓ∞-opérateur (`max` des sommes absolues de lignes) des couches
   `nn.Linear` ; les activations Tanh/ReLU (1-lipschitziennes) sont ignorées ; toute
   autre architecture (LSTM, blocs résiduels) lève `ValueError` car la borne produit
   n'y est pas valide. `lipschitz_l2` : idem avec les normes spectrales
   (`torch.linalg.svdvals`), pour comparaison seulement.
2. `empirical_delta_sup(model, series, dim, lag)` — max des résidus un-pas
   `|f(x_t) − y_t|` sur états vrais ; accepte une liste de segments disjoints
   (train/val/calib) traités séparément — aucune fenêtre fictive à cheval sur une
   jonction (audit E2).
3. `certified_horizon(model, series_std, dim, lag, tolerance)` — assemble
   `G = max(1, L_∞)`, `δ`, `e_1 <= δ`, réutilise
   `horizon_from_model_bound_by_growth(G, δ, δ, τ)` (qui compte les pas **après** le
   premier, d'où le `+1` de conversion vers la convention 1-indexée des `H_w`) et
   retourne `(h_cert, G, δ)` ; `h_cert = ∞` si `δ = 0`, `h_cert = 1` si `δ >= τ`.

## Validation numérique (chiffres)

`python studies/study_certified.py` — seed 0, séries de 8000 points, dt par système
(`DEFAULT_DT`), dim=4, lag=2, split 50/15/15/20, τ = 0.4 (unités std, absolu),
étiquettes `H_w` via `build_horizon_dataset` (K=2, Hmax=400, 500 fenêtres test).
Runtime total : **16.7 s**. Résultats (`studies/study_certified_results.csv`) :

| Système | Modèle | L_∞ | L_2 | G | δ | h_cert | med(H_w) | min(H_w) | censure | **violations** | ratio h_cert/med | δ_test/δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Lorenz (dt 0.01) | LinearAR | 5.686 | 3.210 | 5.686 | 0.00456 | 4 | 20 | 10 | 0/500 | **0** | **0.200** | 0.639 |
| Lorenz (dt 0.01) | MLP (32, 15 ép.) | 25.80 | 2.371 | 25.80 | 0.1671 | 2 | 22 | 5 | 0/500 | **0** | 0.091 | 0.651 |
| Rössler (dt 0.05) | LinearAR | 5.463 | 3.086 | 5.463 | 0.01099 | 4 | 40 | 8 | 0/500 | **0** | 0.101 | 0.980 |
| Rössler (dt 0.05) | MLP (32, 15 ép.) | 26.41 | 2.416 | 26.41 | 0.1390 | 2 | 15 | 3 | 0/500 | **0** | 0.133 | 0.998 |

Variantes conservatrices (calculées systématiquement) : δ×1.5 → h_cert ∈ {2,…,4},
0 violation partout ; δ mesuré sur le seul segment calib (disjoint du train) →
h_cert inchangé (4/2/4/2), 0 violation partout.

Lecture honnête des chiffres :

- **Soundness : 0 violation sur 4/4 configurations** (2000 fenêtres au total),
  aucune censure (Hmax=400 suffisant). De plus `δ_test/δ ∈ [0.64, 1.00] <= 1` :
  le sup empirique sur train+val+calib a effectivement majoré les résidus test ici
  (avec peu de marge sur Rössler : 0.98-1.00 — la variante δ×1.5 est la sécurité
  recommandée si l'on exporte la stat sur données non stationnaires).
- **Utilité : ratio médian 0.091-0.200** ; seuil « utile » (>= 0.15) atteint sur
  1 config sur 4 (Lorenz/LinearAR). La marge min(H_w)/h_cert vaut 1.5-2.5×.
- **Source de la lâcheté** : `G` domine. Pour LinearAR, `||w||_1 ≈ 5.7` (poids de
  type extrapolation polynomiale) alors que la croissance réelle par pas est
  `e^{λ·dt} ≈ 1.009` (Lorenz) : la borne mondiale ne voit pas que les grandes
  composantes de `w` se compensent le long des trajectoires lisses. Pour le MLP,
  la borne produit ℓ∞ (≈ 26) est en outre structurellement lâche (T2) — noter
  `L_2 ≈ 2.4 << L_∞` ; et `δ` (sup, 0.14-0.17 après 15 époques) est 30× le résidu
  linéaire. h_cert exprimé en temps de Lyapunov : 0.007-0.036 T_λ.

## Bénéfice projet

- Fournit la « vraie borne (Lipschitz + bruit borné) » demandée par l'audit D2, et
  donne enfin un statut mathématique au couple (`horizon_from_model_bound_by_growth`,
  récurrence affine) : `g` y devient un majorant certifié, plus un quantile.
- Diagnostic exportable par run, quasi gratuit (0.3 s LinearAR, ~6 s MLP dont
  l'essentiel est le calcul des H_w déjà fait ailleurs ; le calcul de `h_cert` seul
  est < 0.1 s) : plancher **déterministe** `h_cert` valable pour chaque fenêtre,
  complémentaire de L(x) conforme (statistique, plus serré). Si un jour
  `L(x) < h_cert`, c'est L(x) qui est trop pessimiste ; si `H_w < h_cert` en
  production, c'est un **détecteur de sortie de distribution** (résidu > δ).
- Coût de maintenance faible : ~230 lignes pures, 8 tests (0.9 s).

## Critère de décision

Décision fixée avant l'étude : **sound (0 violation) → intégrer comme statistique
diagnostique exportée, quel que soit le ratio** ; marquer « utile » si le ratio
médian `h_cert / med(H_w)` >= 0.15 ; si violations > 0, diagnostiquer (δ sous-estimé
hors distribution) et proposer le correctif chiffré (δ×facteur ou δ sur segment
disjoint) avec verdict conditionnel.

## Verdict

**Intégrer** (comme diagnostic exporté) : 0 violation sur 2000 fenêtres × 4
configurations, y compris sous les deux variantes conservatrices. Le critère
« utile » n'est atteint que sur Lorenz/LinearAR (ratio 0.200) ; ailleurs le ratio
est 0.09-0.13 — la borne est un plancher certifié mais 5-11× plus court que
l'horizon médian réel, à cause de la borne produit sur `G` et du sup sur `δ`.
Recommandation d'intégration : exporter `h_cert`, `G`, `δ`, `δ_test/δ` dans les
stats de run (verdict « useful » réservé aux modèles linéaires) ; ne PAS remplacer
la voie conforme, qui reste la borne opérationnelle serrée. Pistes si l'on veut
resserrer plus tard : Lipschitz local (produit de bornes par région via arithmétique
d'intervalles) et δ par quantile élevé + correction extrême (mais on retomberait
dans le statistique).
