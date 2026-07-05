# Théorie unifiée — ARSAC Horizon

Document de référence consolidant la théorie du projet après la campagne d'études du
2026-07-05 (6 études : `docs/theory/*.md`, scripts reproductibles : `studies/*.py`,
benchmark comparatif : `docs/theory/eval_results.md`). Remplace toute affirmation
antérieure dispersée dans README/AGENT.MD en cas de conflit.

---

## 1. Objet et définitions

- **Série** : observable scalaire d'un système dynamique (Lorenz, Rössler,
  Mackey-Glass, logistique), échantillonnée au pas `dt` propre au système,
  standardisée (moyenne/écart-type du train). Unités : pas, unités de temps
  (u.t.) = pas × dt, temps de Lyapunov T_λ = 1/λ₁.
- **Label d'horizon `H_w`** : pour une fenêtre w, premier pas h (1-indexé) où K
  erreurs absolues consécutives |ŷ_{t+h} − y_{t+h}| dépassent la tolérance τ
  (K=2 par défaut) ; sinon `Hmax` (label **censuré à droite** : H_w ≥ Hmax).
  τ absolue en unités standardisées, ou relative = médiane des résidus un-pas
  × facteur (jamais par fenêtre sur un seul échantillon — audit C2).
- **Objectif** : une borne inférieure L(x) avec P(H_w ≥ L(x)) ≥ 1−α.
- **Références λ₁** (littérature, validées par tests physiques bloquants) :
  Lorenz 0.906/u.t., Rössler 0.071/u.t., Mackey-Glass(τ=17) ≈ 0.006/u.t.,
  logistique(r=4) ln 2 ≈ 0.693/itération.

## 2. Les trois niveaux de garantie

### (a) Certifié — `horizon_certified` (intégré, toujours actif)

**Énoncé.** Dans l'espace d'embedding muni de la norme sup, la dynamique
F(x) = (x₂,…,x_d, f(x)) vérifie ‖F(x)−F(y)‖∞ ≤ G‖x−y‖∞ avec
G = max(1, L_f^∞), L_f^∞ = borne de Lipschitz de f (exacte ‖w‖₁ pour le
linéaire, produit des normes d'opérateur ligne-somme pour le MLP). Si le résidu
un-pas est borné par δ, alors e_h ≤ δ(G^h−1)/(G−1), et
**h_cert = premier h où cette borne atteint τ** satisfait H_w ≥ h_cert pour
**chaque** fenêtre (pas 1−α d'entre elles).

**Hypothèse honnête (le maillon faible)** : δ est le sup *empirique* des résidus
sur la calibration — « certifié modulo la validité de δ sur l'attracteur ».
**Validation** : 0 violation sur 2 000 fenêtres (Lorenz/Rössler × linéaire/MLP),
y compris avec δ×1.5 et δ estimé sur segment disjoint ; utilité
h_cert/médiane(H_w) ∈ [0.09, 0.20] — diagnostic, pas borne opérationnelle.
Le ratio est piloté par la looseness de G (‖w‖₁ ≈ 5.7 vs croissance vraie
e^{λdt} ≈ 1.009). Exporté : `horizon_certified`, `lipschitz_G`, `delta_sup`.

### (b) Approximativement garanti — outils disponibles, non câblés (P1 rangé)

`src/horizon_conformal_beyond.py` fournit : quantile conforme **pondéré**
(Barber et al. 2023) avec sa borne de perte de couverture
Σ w̃ᵢ·d_TV — mesurée : perte bornée 0.0985 (poids uniformes) → 0.0115
(demi-vie n/8) — et la **calibration disjointe** (fenêtres espacées ≥ longueur
de fenêtre + Hmax, échangeabilité approximative sous mélange).

**Pourquoi non câblés** (critères pré-enregistrés non atteints, chiffres) :
la calibration disjointe réduit P(couverture seed < cible−0.02) de ×1.24
seulement — l'étude a montré via un **oracle** (vrai quantile connu) que le
plancher est 0.31 : la fluctuation résiduelle vient du côté *test* (n_eff ≈ 26)
et aucune calibration ne peut la supprimer ; le critère ×2 était structurellement
inatteignable. Le conforme pondéré gagne +2.6 à +2.8 pts sous dérive
(critère : ≥ 3). Les modules restent utilisables pour données réelles à dérive.

### (c) Empirique — la borne opérationnelle L(x)

L(x) = q̂_α(x) − c·ŝ(x), marge conforme à correction d'échantillon fini
(rang ⌈(n+1)(1−α)⌉). La couverture visée est **empirique** : les fenêtres se
chevauchent (dépendance sérielle), l'échangeabilité n'est pas satisfaite.
Mesuré (benchmark rapide, α=0.05, 5 seeds) : couverture médiane 0.967 (Lorenz),
0.970 (Rössler), 0.898 (Mackey-Glass — sous-couverture connue au profil rapide,
à traiter en Phase 3 du plan).

**Censure (P2, intégré sous flag).** Théorème de conservativité (démontré dans
`docs/theory/censored_horizons.md`) : la censure à Hmax ne peut qu'augmenter les
scores signés, donc la marge, donc abaisser L — **la couverture est préservée,
seule la tightness se perd**. La perte quantile censurée de Powell
(`--censored-quantile`, OFF par défaut) réduit le biais de q̂ de 59 % quand la
censure mord la région du quantile, avec le **score plafonné** min(q̂,Hmax)−y
appliqué de façon cohérente au score et à la borne. Toujours actif : la garde
d'identification `label_identified` (p_sat ≤ 1−α requis, sinon warning — le
quantile cible est dans la région censurée, cas Mackey-Glass historique).

## 3. Le lien chaos ↔ horizon (FTLE) — établi, non exploité (P3 rangé)

Ce que l'étude a **établi** (`docs/theory/ftle_horizons.md`) : les exposants de
Lyapunov à temps fini du vrai flot de Lorenz convergent vers λ₁ = 0.906 quand
T croît (0.53 → 0.85 pour T = 0.5 → 10) avec variance décroissante ≈ 1/T —
la distribution des FTLE prédit qualitativement les queues lourdes de slack
observées sur Lorenz. Ce qui a **échoué** : le FTLE du *modèle appris*
(produits QR de matrices compagnons) est un prédicteur de H_w plus faible
(|Spearman| ≤ 0.34) que les features existantes (jac_mean : −0.65), et
l'ajouter au modèle quantile dégrade la pinball loss (−3.6 % linéaire,
−0.55 % MLP). Même le FTLE *oracle* (vrai flot) n'apporte ≤ +0.8 %.
Conclusion : l'information locale d'expansion est déjà capturée par les
features actuelles. Module `src/horizon_ftle.py` conservé pour la recherche.

## 4. Résultats mesurés de la campagne

| Étude | Critère pré-enregistré | Mesure | Verdict |
|---|---|---|---|
| P1 conforme/dépendance | robustesse ×2 ; dérive ≥ +3 pts | ×1.24 (plancher oracle ×1.34) ; +2.6–2.8 pts | **rangé** |
| P2 censure | biais −30 %, couverture tenue | biais −59 %, couverture 0.916–0.999 | **intégré** (flag) |
| P3 FTLE | pinball +5 % ou \|ρ\| ≥ 0.3 et > jac | pinball ≤ 0 % ; ρ ≤ 0.34 < 0.65 | **rangé** |
| P4 certifié | 0 violation | 0/2000, ratio 0.09–0.20 | **intégré** (diagnostic) |
| P5 Politis-White | calibration CI plus proche du nominal | égalité (Δ ≤ 0.010 < bruit MC) | **rangé** |
| P6 embedding | ≥ 2 systèmes plus proches de λ_lit | 4/4 (Lorenz 0.194→0.034, Rössler 0.949→0.123) | **intégré** (défaut) |

Benchmark d'intégration : aucune régression (métriques conformes identiques à
10⁻⁷ flags off) ; suite 212 tests verts ; coût des ajouts ≈ +0.3 s/run.

**Découverte P5 à retenir** (au-delà du verdict) : à forte dépendance (φ=0.9),
la borne inférieure percentile du bootstrap par blocs sous-couvre (0.815 vs
0.95 nominal) **quelle que soit la longueur de bloc** — le déficit vient de
l'absence de studentisation, pas du choix de bloc. `coverage_lb` doit être lu
comme optimiste sous forte dépendance ; la studentisation est un point ouvert.

## 5. Décisions (défauts du pipeline)

| Capacité | Flag | Défaut |
|---|---|---|
| Borne certifiée exportée | — (toujours actif) | ON |
| Embedding théorique pour λ (MI + FNN) | `--lyap-dim/--lyap-lag` explicites pour bypass | ON quand None |
| Garde d'identification p_sat | — (toujours actif) | ON |
| Perte quantile censurée (Powell) | `--censored-quantile` | OFF (activer si p_sat > 0) |
| Debias post-conforme | `--debias-scale` | **0.0 — retiré** (ablation 2026-07-05 : coûtait 0.5–0.7 pt de couverture pour ≤0.009 de tightness ; il corrigeait une sur-couverture disparue avec les labels corrigés) |
| Coverage guard | `--coverage-guard-*` | conservé (protège 1.2 pt de couverture, mesuré) |
| Conforme pondéré / calibration disjointe | module `horizon_conformal_beyond` | non câblé |
| FTLE | module `horizon_ftle` | non câblé |

## 6. Limites ouvertes et prochaines étapes

1. ~~**Sous-couverture Mackey-Glass au profil rapide**~~ **résolu (2026-07-05)** :
   c'était un artefact des labels (tolérance relative + Hmax=20). Avec Hmax auto
   et τ=0.4·std (validation `studies/study_lyap_hmax.py`, 5 seeds, α=0.05) :
   Mackey-Glass 0.960 [min 0.957], Rössler 0.965 [0.938] avec tightness 0.80
   (l'anomalie tightness > 1 a également disparu), p_sat = 0 partout.
   **Diagnostic complété (même jour)** : Lorenz montrait une sous-couverture
   légère et systématique (0.941 [0.937] vs 0.95, 5/5 seeds). Le sweep
   d'amincissement (`studies/study_calib_thinning.py`, strides 1→48) exclut la
   corrélation sérielle comme cause : couverture identique à stride 12 (0.941,
   n_calib 1200→116 — les fenêtres corrélées sont redondantes mais pas
   biaisantes), puis effondrement par petit n (0.73 à n=29). Le déficit d'un
   point est un **shift calib→test** le long de la trajectoire (le verdict
   « rangé » de P1 est confirmé sur cas réel). Remède standard validé :
   calibrer à α=0.035 pour livrer 0.95 → couverture 0.951 [0.947], coût de
   tightness < 1 % (0.765→0.758). Recommandation : marge α_cal ≈ α_cible − 0.015
   pour les régimes à croissance rapide type Lorenz.
2. **Studentisation du bootstrap** de `coverage_lb` (finding P5).
3. ~~**Hmax en temps de Lyapunov**~~ **fait (2026-07-05)** : `horizon_max`
   auto-résolu à max(3 T_λ, 1.2·ln(τ/e₀)/λ₁), borné [30, 400] et par le budget
   de données (avertissement de censure sinon) ; tolérance par défaut passée à
   l'absolu 0.4·std (convention « valid time », clôt l'audit C2 : la tolérance
   relative mesurait la précision du modèle, pas le chaos). Reste le plafond de
   coût à 400 pas : Rössler (cible ~846 à dt=0.05) et les modèles quasi
   parfaits restent censurés — documenté, gate `label_identified` active.
4. Réévaluation du conforme pondéré **sur données réelles à dérive** (le
   scénario où ses +2.8 pts deviennent décisifs) — Phase 5.
5. Régénération du paper avec les pipelines corrigés (Phase 4).
