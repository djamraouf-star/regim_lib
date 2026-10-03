# Contexte temporel — sessions, calendrier, événements

**Dernière mise à jour** : 2026-10-03
**Module** : `regime_lib/context/`
**Statut** : implémenté et intégré au CLI

---

## 1. Décision

Les sessions, le rollover et le jour de semaine sont calculés dans le
fuseau `reference_tz` du profil, qui vaut **New York**
(`America/New_York`) par défaut. Les événements macro du CSV sont
horodatés en heure de New York, indépendamment de `reference_tz`. Les
jours fériés sont comparés au jour UTC du début de barre.

Ainsi, le profil forex par défaut utilise NY comme référence pour les
sessions et le rollover. Les profils qui remplacent `reference_tz`
doivent tenir compte du fait que cette option ne convertit ni les
horodatages d'événements ni les dates de fériés.

La configuration est portée par le **profil YAML** de l'actif, dans une
section `context:`. Voir section 8.

## 2. Justification

### 2.1 Transitions DST

New York et Londres ne changent pas d'heure aux mêmes dates :

| Période | Décalage UTC | Décalage London/NY |
|---|---|---|
| Hiver | UTC−5 (NY), UTC+0 (LDN) | 5 h |
| Été | UTC−4 (NY), UTC+1 (LDN) | 5 h |
| **Transition mars** (~3 semaines) | UTC−4 (NY), UTC+0 (LDN) | **4 h** |
| **Transition octobre** (~3 semaines) | UTC−5 (NY), UTC+1 (LDN) | **4 h** |

Définir les sessions en heure de Londres ou en UTC décale leur position
relative à NY pendant ces deux fenêtres annuelles. Prendre NY comme
référence supprime ce problème : les bornes sont **stables** du point de
vue de la place new-yorkaise, pivot du marché forex.

### 2.2 Rollover

Le rollover forex est défini conventionnellement à **17h heure de NY**
(17:00 EST en hiver, 17:00 EDT en été). Utiliser NY évite d'avoir à
gérer deux fenêtres UTC distinctes selon la saison.

### 2.3 Convention des outils de référence

Tickstory, MetaTrader 5 et Dukascopy utilisent NY comme pivot pour la
structuration des sessions forex. S'aligner garantit la cohérence avec
les exports tick et les backtests.

### 2.4 Causalité

Le calcul de l'heure NY à partir d'un timestamp est **causal**. Aucun
lookahead n'est introduit. Le module `context` peut donc être utilisé
pour du criblage temps réel.

## 3. Définition des sessions

Bornes en heure de New York, sur une journée calendaire NY :

| Session | Début NY | Fin NY | Caractéristique |
|---|---|---|---|
| ASIE | 19:00 (J-1) | 03:00 | Liquidité réduite, spreads larges |
| LONDRES | 03:00 | 08:00 | Ouverture européenne, forte activité |
| OVERLAP | 08:00 | 12:00 | Londres + NY simultanées, pic de liquidité |
| NY | 12:00 | 17:00 | Après-midi américaine, activité décroissante |

**Règle d'affectation** : chaque barre est classée selon l'heure NY de
son **début**. Pour une barre étiquetée à droite (convention du projet),
l'heure de début vaut `étiquette − durée`.

**Cas frontière** : une barre dont le début est exactement à une borne
appartient à la session qui commence à cette borne. Convention
`[début, fin[`.

**Chevauchement minuit** : la session ASIE traverse minuit
(19:00 J-1 → 03:00 J).

## 4. Fenêtres d'événements

### 4.1 Rollover forex

| Fenêtre | Définition | Usage |
|---|---|---|
| Rollover strict | `[17:00, 18:00)` NY | Marquage par défaut |
| Rollover élargi | `[16:30, 18:30)` NY | À exclure pour la microstructure fine |

Une barre est marquée `ctx_is_rollover = True` si son **début** tombe
dans la fenêtre.

### 4.2 Événements macro

Trois familles :

| Événement | Fréquence | Heure NY | Fenêtre par défaut |
|---|---|---|---|
| **NFP** | 1er vendredi du mois | 08:30 | `[08:25, 09:30)` |
| **CPI** | Mensuel | 08:30 | `[08:25, 09:30)` |
| **FOMC** | 8 dates/an | 14:00 | `[13:55, 15:30)` |

Les dates sont fournies dans
`regime_lib/context/data/events_forex.csv`. Les fenêtres sont
configurables par type d'événement (voir section 8).

Quand les barres dépassent la date du dernier événement présent dans le
fichier, `enrichir_contexte` émet un avertissement : les événements
absents du calendrier ne peuvent pas être marqués et le fichier doit être
mis à jour.

### 4.3 Jours fériés

Liste dans `regime_lib/context/data/holidays_forex.csv` :

- Nouvel An
- Vendredi saint, Lundi de Pâques
- Early May Bank Holiday, Spring Bank Holiday, Summer Bank Holiday
- Thanksgiving US
- Noël, Boxing Day

Une barre est marquée `ctx_is_holiday = True` si le jour UTC de son
début figure dans cette liste. Les dates du CSV sont normalisées en
minuit UTC avant la comparaison.

## 5. Colonnes produites

Convention de préfixe : `ctx_` pour toutes les colonnes de contexte.

| Colonne | Type | Description |
|---|---|---|
| `ctx_session` | str | Nom de la session, ou `"HORS_SESSION"` |
| `ctx_hour_ny` | int | Heure NY du début de barre (0–23) |
| `ctx_hour_utc` | int | Heure UTC du début de barre (0–23) |
| `ctx_day_of_week` | int | 0 = lundi, 6 = dimanche (NY) |
| `ctx_is_rollover` | bool | Barre dont le début est dans la fenêtre rollover |
| `ctx_is_holiday` | bool | Barre dont le début tombe un jour férié |
| `ctx_in_event` | bool | Barre dont le début est dans une fenêtre d'événement |
| `ctx_event_type` | str | Nom de l'événement (`NFP`, `CPI`, `FOMC`), vide sinon |

Ces colonnes viennent **en complément** des régimes, jamais en
remplacement. Elles sont calculées **après** l'agrégation en barres.

## 6. Cas particulier : transitions DST

Pendant les transitions DST (mars et octobre), le décalage entre NY et
Londres passe temporairement de 5h à 4h. Les sessions définies en heure
NY restent **stables**. Les sessions exprimées en heure UTC varieraient.

**Recommandation** : ne jamais faire d'analyse conditionnelle sur
`ctx_hour_utc` pour définir une session. Utiliser `ctx_session` ou
`ctx_hour_ny`.

**Exemple concret** : la session OVERLAP (08:00–12:00 NY) correspond à
13:00–17:00 UTC en hiver, 12:00–16:00 UTC en été. La borne NY ne change
pas de sens pour un opérateur NY.

## 7. Alternatives écartées

### 7.1 Heure UTC

**Pour** : invariant, calculable sans base de fuseaux.

**Contre** : le rollover en UTC a deux définitions (21h ou 22h selon
la saison). Les sessions en UTC se décalent d'une heure deux fois par an.

**Verdict** : écarté pour les sessions. Conservé comme **colonne
d'affichage** (`ctx_hour_utc`) pour la lisibilité.

### 7.2 Heure de Londres

**Verdict** : écarté. Ne résout pas le rollover, introduit un décalage
relatif à NY pendant les transitions DST.

### 7.3 Heure locale du trader

**Verdict** : écarté. Non reproductible. La conversion est du ressort
de l'utilisateur final.

## 8. Configuration YAML

Le contexte est configuré dans la section `context:` du profil d'actif.
`load_profile` fusionne le profil actif avec `default.yaml` : les valeurs
du profil actif remplacent celles du profil par défaut, y compris les
valeurs des clés YAML définies explicitement à `null`. Si une clé reste
absente après la fusion, `ContextConfig` applique ses valeurs par défaut
codées en dur. Il n'existe pas de couche de surcharge explicite distincte
dans l'API ou le CLI.

### 8.1 Schéma

```yaml
context:
  # Fuseau de référence des sessions, du rollover et du jour de semaine.
  # Les événements restent horodatés en America/New_York ; les fériés
  # sont évalués à partir du jour UTC.
  reference_tz: "America/New_York"

  # Sessions de marché. Clés = noms libres, valeurs = bornes "HH:MM".
  # Une session peut chevaucher minuit.
  sessions:
    ASIE:    {debut: "19:00", fin: "03:00"}
    LONDRES: {debut: "03:00", fin: "08:00"}
    OVERLAP: {debut: "08:00", fin: "12:00"}
    NY:      {debut: "12:00", fin: "17:00"}

  # Fenêtre de rollover. null si absente (crypto, actions).
  rollover:
    debut: "17:00"
    fin:   "18:00"

  # Événements macro. `fichier` peut être null (aucun événement).
  events:
    fichier: "context/data/events_forex.csv"
    fenetres:
      NFP:  {avant_min: 5, apres_min: 60}
      CPI:  {avant_min: 5, apres_min: 60}
      FOMC: {avant_min: 5, apres_min: 90}

  # Jours fériés. `fichier` peut être null (pas de fériés).
  holidays:
    fichier: "context/data/holidays_forex.csv"
