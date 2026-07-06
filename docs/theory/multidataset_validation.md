# Validation multi-datasets, multi-modèles (campagne v2)

Étude `studies/study_multidataset_validation.py` (2026-07-06). Étend la
validation industrielle 17/17 (`industrial_validation.md`) sur les deux axes
qu'elle ne couvrait pas : **de nouveaux jeux de données** (autres opérateurs,
et un domaine non-énergétique) et **de nouvelles familles de modèles**
(MLP, ridge polynomial degré 2 — l'idée de features NG-RC portée aux
données réelles).

Protocole **pré-enregistré** : les 26 vérifications et tous les seuils sont
écrits dans le docstring du script, avant tout run. Évidence brute :
`docs/theory/data/multidataset_validation.csv`, épinglée par
`tests/test_multidataset_evidence.py`. Figure auto-vérifiée :
`studies/make_multidataset_figure.py` → `assets/multidataset_validation.png`.

## Données

| Clé | Série | Source | Particularité |
|---|---|---|---|
| comed | COMED_MW (Chicago) | miroir GitHub PJM (hourly energy consumption) | opérateur non testé en v1 |
| dom | DOM_MW (Dominion, Virginie) | idem | opérateur non testé en v1 |
| traffic | traffic_volume (I-94, Minneapolis) | UCI Metro Interstate Traffic Volume | **hors énergie** ; trous horaires traités comme consécutifs (déclaré) |

3 dernières années horaires chacune (26 280 points), triées, timestamps
dédupliqués. Fichiers bruts non versionnés (re-téléchargeables, URLs dans le
script) ; seule l'évidence est versionnée.

## Familles de modèles (mêmes API et remède α_cal = 0.085 qu'en v1)

- `linear` : AR fenêtre 25 h (baseline v1)
- `naive` : persistance 24 h (BYO `lambda v: v[1]`, dim=25)
- `mlp` : MLP interne (40 epochs) — **nouvelle famille**
- `poly` : ridge polynomial degré 2 sur la fenêtre 25 (350 features,
  entraîné sur les premiers 60 % = split train de l'estimateur, zéro fuite)
  — **nouvelle famille**

## Les 26 vérifications pré-enregistrées

- **W1 (3)** — régime : comed et dom `quasi-periodic` ; traffic ≠ `chaotic`.
- **W2 (12)** — garantie opérationnelle, agnostique au modèle : pour chaque
  dataset × famille, couverture ≥ 0.88 ET borne bootstrap 95 % ≥ 0.85
  (cible 0.90, remède α fixé d'avance sur Lorenz, inchangé depuis la v1).
- **W3 (9)** — réplication de la loi d'échelle (modèle linéaire,
  τ ∈ {0.2, 0.3, 0.4, 0.6, 0.8}) : pente ∈ [1.2, 2.8] pour les charges,
  ∈ [1.0, 2.8] pour le trafic (domaine différent, prior élargi déclaré
  avant run) ; loi puissance bat la signature chaotique ; σ_eff/σ̂ ∈ [1/3, 3].
- **W4 (2)** — **invariance de l'exposant par famille de modèle** (claim
  nouveau) : sur comed, |s_poly − s_linéaire| ≤ 0.5 et loi puissance > log
  pour poly. (σ_eff peut différer — il suit l'erreur un-pas du modèle —
  seul l'exposant est contraint : il appartient au régime des données.)

## Résultats

### W1 — régimes (3/3)

| Dataset | régime mesuré | périodicité | λ résolu | prédiction |
|---|---|---|---|---|
| comed | quasi-periodic | 0.678 | non | ✓ |
| dom | quasi-periodic | 0.574 | non | ✓ |
| traffic | regular (≠ chaotic) | 0.467 | non | ✓ |

Le profileur ne déclare le chaos sur aucune des trois séries — la
prédiction théorique (cycles + bruit) tient hors du domaine énergie.

### W2 — couverture, agnostique au modèle (11/12)

| Dataset | linear | naive | mlp | poly |
|---|---|---|---|---|
| comed | 0.9019 / 0.8921 ✓ | **0.8574 / 0.8368 ✗** | 0.9074 / 0.8968 ✓ | 0.8906 / 0.8736 ✓ |
| dom | 0.9075 / 0.8964 ✓ | 0.9395 / 0.9257 ✓ | 0.9040 / 0.8878 ✓ | 0.9135 / 0.8996 ✓ |
| traffic | 0.9447 / 0.9368 ✓ | 0.9637 / 0.9530 ✓ | 0.9505 / 0.9371 ✓ | 0.9723 / 0.9648 ✓ |

(couverture / borne bootstrap 95 % ; n = 2520–2548 fenêtres par case ;
critères pré-enregistrés ≥ 0.88 / ≥ 0.85, cible 0.90, α_cal = 0.085 inchangé.)

**Le dossier de l'échec comed/naive** (post-hoc, `studies/diag_comed_naive.py`
et `diag_comed_naive_remedy.py`, tracé par tiers reproduit ci-dessous) :

1. **Cause mesurée** : basculement saisonnier en bloc entre calibration et
   test — horizon médian du modèle naïf 22 h sur la tranche de calibration,
   4 h sur la fenêtre de test (glissement −18 pas au médian) ; 41.5 % des
   fenêtres de test cassent dès h=1.
2. **Plafond mesuré** : sur cette saison, toute borne **constante** L ≥ 2
   couvre au plus 58.5 % — aucune calibration marginale ne peut délivrer
   0.90 non-trivialement (la borne marginale, uniforme, pondérée demi-vie
   30 j ou glissante 60 j, s'effondre à L = 1 trivial dans les trois
   schémas). Une borne conditionnelle parfaitement informée pourrait
   dépasser ce plafond en émettant L = 1 sur les fenêtres condamnées ;
   l'estimateur conditionnel réel a atteint 0.857 avec L médian 1.7 —
   entre le plafond constant et la garantie.
3. **L'instrument a pisté la dérive** : L médian émis par tiers de test
   4.82 → 1.43 → 1.00 (couverture ~0.85–0.86 uniforme) — la partie
   conditionnelle suit la dégradation, c'est la marge figée qui est percée.
4. **Lecture théorique** : c'est le mode de défaillance déclaré de la
   garantie (échangeabilité violée par dérive ; borne de Barber) — l'échec
   survient exactement où la théorie le prédit, et le contrôle dom/naive
   (même modèle, glissement +2 favorable) passe à 0.9395.
5. **Remède produit** (question falsifiable suivante) : recalibration
   glissante = garantie rétablie au prix d'un L honnêtement ~1 h — câbler
   le conforme pondéré/glissant en mode déploiement + alarme de couverture
   réalisée.

### W3 — la loi d'échelle réplique, y compris hors domaine (9/9)

| Dataset | pente s | R² puissance | R² chaotique | σ_eff / σ̂ | verdict |
|---|---|---|---|---|---|
| comed | 1.343 | 0.9813 | 0.2209 | 1.293 | ✓✓✓ |
| dom | 1.457 | 0.9755 | **−0.8797** | **0.986** | ✓✓✓ |
| traffic | 1.207 | 0.9694 | 0.4600 | 0.614 | ✓✓✓ |

Avec la v1 : **cinq datasets** (AEP 1.40, PJME 1.26, COMED 1.34, DOM 1.46,
trafic 1.21), deux domaines, pentes toutes dans la bande stable [1, 2],
signature chaotique rejetée partout. Sur DOM, σ_eff est prédit à **1.4 %**
par l'estimateur indépendant.

### W4 — l'exposant appartient aux données, pas au modèle (2/2)

Sur comed : s_linéaire = 1.343, s_poly = 1.347 → **|Δs| = 0.004** (critère
pré-enregistré ≤ 0.5, battu d'un facteur 100). R² poly 0.993 > log 0.42.
σ_eff diffère entre familles (0.0659 vs 0.0586 — chaque modèle a son erreur
un-pas), la pente non : c'est le régime d'accumulation des innovations de la
série qui la fixe, comme dérivé.

## Verdict global

**25/26.** Cumulé avec la v1 : **42/43 vérifications pré-enregistrées sur
5 datasets × 4 familles de modèles × 2 domaines**, l'unique échec étant
survenu dans le mode de défaillance déclaré de la garantie (dérive), avec
cause, plafond et remède mesurés.

## Notes de lecture

- Le champ `regime` des enregistrements `w2_*` est `None` (clé non exposée
  par le pipeline) — le régime fait foi dans les enregistrements `w1_*`.
- La série trafic a des heures manquantes (défaut connu du dataset UCI) ;
  le protocole les traite comme consécutives, ce qui bruite la structure
  périodique — c'est visible dans son niveau de bruit profilé (0.225 vs
  0.047/0.064 pour les charges).
- Le sweep comed/poly émet des RuntimeWarning d'overflow : le rollout du
  ridge polynomial diverge sur certaines fenêtres APRÈS le franchissement
  de la tolérance (l'erreur croît continûment, donc le label de premier
  franchissement est fixé bien avant l'overflow à ~1e154) ; censure
  mesurée 0.00 sur les 5 τ — les labels sont sains.
