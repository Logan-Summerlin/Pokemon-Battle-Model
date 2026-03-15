# Battle Decision Report — Fine-Tuning Data Analysis

*Generated from data/fine-tuning/raw — 10 battles with < 30 turns*

This report provides a comprehensive turn-by-turn account of player decisions
in Gen 9 OU battles. For each turn, we list the active Pokemon, available moves
(with type, category, base power, PP), available switches, field conditions,
and the decision actually taken by the player. This data is crucial for
understanding the decision space that the P8-Lean imitation learning model
must learn to navigate.

## Data Quality Note: Initial HP Anomalies (Corrected)

The Metamon dataset (`jakegrigsby/metamon-parsed-replays`) reports ~80% of battles
with bench Pokemon at non-100% HP on Turn 1, despite no prior moves, hazards, or
field conditions. This is a systemic data artifact. **In this report, all such
anomalous HP values have been corrected to 100% until the Pokemon first takes
verifiable damage** (HP decrease from a move, hazard, weather, or other in-battle
source). Post-damage HP values are reported as-is from the raw data.

## Table of Contents

1. [1322378-gen9ou-2387334841_Unrated_honeygather71890_vs_lockon89063_06-18-2025_WIN](#1-battle-1) — 8 turns, WIN
2. [gen9ou-2148626738_Unrated_npowerfighting1559_vs_paras64703_06-23-2024_WIN](#2-battle-2) — 13 turns, WIN
3. [gen9ou-2064798635_1558_kangaskhan20989_vs_furyattack47818_02-21-2024_LOSS](#3-battle-3) — 16 turns, LOSS
4. [gen9ou-2114115772_1986_combee67377_vs_snore48848_04-29-2024_WIN](#4-battle-4) — 19 turns, WIN
5. [gen9ou-2179884364_1736_alakazam79428_vs_uturn77356_08-14-2024_WIN](#5-battle-5) — 21 turns, WIN
6. [1520946-gen9ou-2431309839_1777_togepi73784_vs_multitype62894_08-29-2025_WIN](#6-battle-6) — 23 turns, WIN
7. [gen9ou-2154385089_1587_doubleslap89429_vs_crushgrip45765_07-03-2024_LOSS](#7-battle-7) — 24 turns, LOSS
8. [gen9ou-2233095423_Unrated_furyattack32056_vs_filter66953_10-28-2024_WIN](#8-battle-8) — 25 turns, WIN
9. [1471813-gen9ou-2419843548_1735_delibird29318_vs_absorb67344_08-11-2025_WIN](#9-battle-9) — 27 turns, WIN
10. [1553103-gen9ou-2439327985_1512_assurance64217_vs_arceusbug25750_09-10-2025_WIN](#10-battle-10) — 28 turns, WIN

## 1. Battle 1

### Battle: `1322378-gen9ou-2387334841_Unrated_honeygather71890_vs_lockon89063_06-18-2025_WIN.json.lz4`
- **Result**: WIN
- **Elo**: Unrated
- **Total Turns**: 8

#### Player's Team
1. **amoonguss** (HP: 100%) | Type: grass poison | Item: leftovers | Ability: regenerator | Status: nostatus | Tera: fire *(Lead)*
2. **leafeon** (HP: 100%) | Type: grass notype | Item: lifeorb | Ability: leafguard | Status: nostatus | Tera: fire
3. **ogerponcornerstone** (HP: 100%) | Type: grass rock | Item: cornerstonemask | Ability: sturdy | Status: nostatus | Tera: rock
4. **whimsicott** (HP: 100%) | Type: fairy grass | Item: lifeorb | Ability: chlorophyll | Status: nostatus | Tera: ghost
5. **hydrapple** (HP: 100%) | Type: dragon grass | Item: rockyhelmet | Ability: regenerator | Status: nostatus | Tera: steel
6. **meowscarada** (HP: 100%) | Type: dark grass | Item: choiceband | Ability: protean | Status: nostatus | Tera: ghost

#### Opponent's Team (from Team Preview)
1. **flamigo**
2. **ironboulder**
3. **serperior**
4. **glimmora**
5. **volcanion**
6. **kingambit**

#### Turn-by-Turn Decisions

---
**Turn 1**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **amoonguss** (HP: 100%) | Type: grass poison | Item: leftovers | Ability: regenerator | Status: nostatus | Tera: fire
**Opponent Active**: **flamigo** (HP: 100%) | Type: fighting flying | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **toxic** [poison] (status) PP:16/16
  - Slot 2: **venoshock** [poison] (special) BP:65 PP:16/16
  - Slot 3: **gigadrain** [grass] (special) BP:75 PP:16/16 <-- CHOSEN
  - Slot 4: **synthesis** [grass] (status) PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **leafeon** (HP: 100%) [nostatus]
  - Bench 2: **ogerponcornerstone** (HP: 100%) [nostatus]
  - Bench 3: **whimsicott** (HP: 100%) [nostatus]
  - Bench 4: **hydrapple** (HP: 100%) [nostatus]
  - Bench 5: **meowscarada** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **gigadrain**

---
**Turn 2**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used toxic | Opponent used nomove

**Player Active**: **amoonguss** (HP: 100%) | Type: grass poison | Item: leftovers | Ability: regenerator | Status: nostatus | Tera: fire
**Opponent Active**: **serperior** (HP: 94%) | Type: grass notype | Status: tox | Tera: notype

**Available Moves:**
  - Slot 1: **toxic** [poison] (status) PP:15/16
  - Slot 2: **venoshock** [poison] (special) BP:65 PP:16/16
  - Slot 3: **gigadrain** [grass] (special) BP:75 PP:16/16
  - Slot 4: **synthesis** [grass] (status) PP:8/8 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **leafeon** (HP: 100%) [nostatus]
  - Bench 2: **ogerponcornerstone** (HP: 100%) [nostatus]
  - Bench 3: **whimsicott** (HP: 100%) [nostatus]
  - Bench 4: **hydrapple** (HP: 100%) [nostatus]
  - Bench 5: **meowscarada** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **synthesis**

---
**Turn 3**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used venoshock | Opponent used nomove

**Player Active**: **amoonguss** (HP: 98%) | Type: grass poison | Item: leftovers | Ability: regenerator | Status: nostatus | Tera: fire
**Opponent Active**: **kingambit** (HP: 100%) | Type: dark steel | Ability: supremeoverlord | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **toxic** [poison] (status) PP:15/16
  - Slot 2: **venoshock** [poison] (special) BP:65 PP:15/16
  - Slot 3: **gigadrain** [grass] (special) BP:75 PP:16/16
  - Slot 4: **synthesis** [grass] (status) PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **leafeon** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **ogerponcornerstone** (HP: 100%) [nostatus]
  - Bench 3: **whimsicott** (HP: 100%) [nostatus]
  - Bench 4: **hydrapple** (HP: 100%) [nostatus]
  - Bench 5: **meowscarada** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **leafeon**

---
**Turn 4**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used retaliate

**Player Active**: **hydrapple** (HP: 100%) | Type: dragon grass | Item: rockyhelmet | Ability: regenerator | Status: nostatus | Tera: steel
**Opponent Active**: **kingambit** (HP: 100%) | Type: dark steel | Ability: supremeoverlord | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **earthpower** [ground] (special) BP:90 PP:16/16 <-- CHOSEN
  - Slot 2: **ficklebeam** [dragon] (special) BP:80 PP:8/8
  - Slot 3: **gigadrain** [grass] (special) BP:75 PP:16/16
  - Slot 4: **nastyplot** [dark] (status) PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **leafeon** (HP: 100%) [nostatus]
  - Bench 2: **ogerponcornerstone** (HP: 100%) [nostatus]
  - Bench 3: **amoonguss** (HP: 98%) [nostatus]
  - Bench 4: **whimsicott** (HP: 100%) [nostatus]
  - Bench 5: **meowscarada** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **earthpower**

---
**Turn 5**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used earthpower | Opponent used kowtowcleave

**Player Active**: **hydrapple** (HP: 21%) | Type: dragon grass | Item: rockyhelmet | Ability: regenerator | Status: nostatus | Tera: steel
**Opponent Active**: **kingambit** (HP: 39%) | Type: dark steel | Ability: supremeoverlord | Status: nostatus | Tera: notype | Boosts: SPD-1

**Available Moves:**
  - Slot 1: **earthpower** [ground] (special) BP:90 PP:15/16
  - Slot 2: **ficklebeam** [dragon] (special) BP:80 PP:8/8
  - Slot 3: **gigadrain** [grass] (special) BP:75 PP:16/16
  - Slot 4: **nastyplot** [dark] (status) PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **leafeon** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **ogerponcornerstone** (HP: 100%) [nostatus]
  - Bench 3: **amoonguss** (HP: 98%) [nostatus]
  - Bench 4: **whimsicott** (HP: 100%) [nostatus]
  - Bench 5: **meowscarada** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **leafeon**

---
**Turn 6**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used venoshock | Opponent used kowtowcleave

**Player Active**: **amoonguss** (HP: 62%) | Type: grass poison | Item: leftovers | Ability: regenerator | Status: nostatus | Tera: fire
**Opponent Active**: **kingambit** (HP: 39%) | Type: dark steel | Ability: supremeoverlord | Status: nostatus | Tera: notype | Boosts: SPD-1

**Available Moves:**
  - Slot 1: **toxic** [poison] (status) PP:15/16
  - Slot 2: **venoshock** [poison] (special) BP:65 PP:15/16
  - Slot 3: **gigadrain** [grass] (special) BP:75 PP:16/16
  - Slot 4: **synthesis** [grass] (status) PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **leafeon** (HP: 100%) [nostatus]
  - Bench 2: **ogerponcornerstone** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 3: **whimsicott** (HP: 100%) [nostatus]
  - Bench 4: **hydrapple** (HP: 21%) [nostatus]
  - Bench 5: **meowscarada** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 2: Switch to **ogerponcornerstone**

---
**Turn 7**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **leafeon** (HP: 100%) | Type: grass notype | Item: lifeorb | Ability: leafguard | Status: nostatus | Tera: fire
**Opponent Active**: **ironboulder** (HP: 100%) | Type: psychic rock | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:32/32
  - Slot 2: **swordsdance** [normal] (status) PP:32/32
  - Slot 3: **terablast** [normal] (special) BP:80 PP:16/16
  - Slot 4: **solarblade** [grass] (physical) BP:125 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponcornerstone** (HP: 100%) [nostatus]
  - Bench 2: **amoonguss** (HP: 62%) [nostatus]
  - Bench 3: **whimsicott** (HP: 100%) [nostatus]
  - Bench 4: **hydrapple** (HP: 21%) [nostatus]
  - Bench 5: **meowscarada** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **solarblade**

---
**Turn 8**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **leafeon** (HP: 100%) | Type: grass notype | Item: lifeorb | Ability: leafguard | Status: nostatus | Tera: fire
**Opponent Active**: **ironboulder** (HP: 100%) | Type: psychic rock | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:32/32
  - Slot 2: **swordsdance** [normal] (status) PP:32/32
  - Slot 3: **terablast** [normal] (special) BP:80 PP:16/16
  - Slot 4: **solarblade** [grass] (physical) BP:125 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponcornerstone** (HP: 100%) [nostatus]
  - Bench 2: **amoonguss** (HP: 62%) [nostatus]
  - Bench 3: **whimsicott** (HP: 100%) [nostatus]
  - Bench 4: **hydrapple** (HP: 21%) [nostatus]
  - Bench 5: **meowscarada** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **solarblade**


---

## 2. Battle 2

### Battle: `gen9ou-2148626738_Unrated_npowerfighting1559_vs_paras64703_06-23-2024_WIN.json.lz4`
- **Result**: WIN
- **Elo**: Unrated
- **Total Turns**: 13

#### Player's Team
1. **ribombee** (HP: 100%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: nostatus | Tera: bug *(Lead)*
2. **gholdengo** (HP: 100%) | Type: ghost steel | Item: airballoon | Ability: goodasgold | Status: nostatus | Tera: fairy
3. **indeedee** (HP: 100%) | Type: normal psychic | Item: choicescarf | Ability: psychicsurge | Status: nostatus | Tera: fairy
4. **ironcrown** (HP: 100%) | Type: psychic steel | Item: boosterenergy | Ability: quarkdrive | Status: nostatus | Tera: steel
5. **hoopaunbound** (HP: 100%) | Type: dark psychic | Item: lifeorb | Ability: magician | Status: nostatus | Tera: psychic
6. **blaziken** (HP: 100%) | Type: fighting fire | Item: leftovers | Ability: speedboost | Status: nostatus | Tera: fire

#### Opponent's Team (from Team Preview)
1. **vaporeon**
2. **sylveon**
3. **jolteon**
4. **flareon**
5. **espeon**
6. **leafeon**

#### Turn-by-Turn Decisions

---
**Turn 1**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **ribombee** (HP: 100%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: nostatus | Tera: bug
**Opponent Active**: **flareon** (HP: 100%) | Type: fire notype | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **uturn** [bug] (physical) BP:70 PP:32/32
  - Slot 2: **stickyweb** [bug] (status) PP:32/32
  - Slot 3: **psychic** [psychic] (special) BP:90 PP:16/16
  - Slot 4: **psychicnoise** [psychic] (special) BP:75 PP:16/16 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **gholdengo** (HP: 100%) [nostatus]
  - Bench 2: **indeedee** (HP: 100%) [nostatus]
  - Bench 3: **ironcrown** (HP: 100%) [nostatus]
  - Bench 4: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 5: **blaziken** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **psychicnoise**

---
**Turn 2** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used uturn | Opponent used nomove

**Player Active**: **ribombee** (HP: 100%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: nostatus | Tera: bug
**Opponent Active**: **espeon** (HP: 53%) | Type: notype psychic | Status: nostatus | Tera: notype

**Available Switches:**
  - Bench 1: **gholdengo** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **indeedee** (HP: 100%) [nostatus]
  - Bench 3: **ironcrown** (HP: 100%) [nostatus]
  - Bench 4: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 5: **blaziken** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **gholdengo**

---
**Turn 3**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **blaziken** (HP: 100%) | Type: fighting fire | Item: leftovers | Ability: speedboost | Status: nostatus | Tera: fire
**Opponent Active**: **espeon** (HP: 53%) | Type: notype psychic | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **swordsdance** [normal] (status) PP:32/32
  - Slot 2: **temperflare** [fire] (physical) BP:75 PP:16/16
  - Slot 3: **closecombat** [fighting] (physical) BP:120 PP:8/8 <-- CHOSEN
  - Slot 4: **flareblitz** [fire] (physical) BP:120 PP:24/24
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ribombee** (HP: 100%) [nostatus]
  - Bench 2: **gholdengo** (HP: 100%) [nostatus]
  - Bench 3: **indeedee** (HP: 100%) [nostatus]
  - Bench 4: **ironcrown** (HP: 100%) [nostatus]
  - Bench 5: **hoopaunbound** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **closecombat**

---
**Turn 4**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used swordsdance | Opponent used futuresight

**Player Active**: **blaziken** (HP: 100%) | Type: fighting fire | Item: leftovers | Ability: speedboost | Status: nostatus | Tera: fire | Boosts: ATK+2, SPE+1
**Opponent Active**: **espeon** (HP: 53%) | Type: notype psychic | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **swordsdance** [normal] (status) PP:31/32
  - Slot 2: **temperflare** [fire] (physical) BP:75 PP:16/16
  - Slot 3: **closecombat** [fighting] (physical) BP:120 PP:8/8
  - Slot 4: **flareblitz** [fire] (physical) BP:120 PP:24/24 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ribombee** (HP: 100%) [nostatus]
  - Bench 2: **gholdengo** (HP: 100%) [nostatus]
  - Bench 3: **indeedee** (HP: 100%) [nostatus]
  - Bench 4: **ironcrown** (HP: 100%) [nostatus]
  - Bench 5: **hoopaunbound** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **flareblitz**

---
**Turn 5**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used temperflare | Opponent used nomove

**Player Active**: **blaziken** (HP: 100%) | Type: fighting fire | Item: leftovers | Ability: speedboost | Status: nostatus | Tera: fire | Boosts: ATK+2, SPE+2
**Opponent Active**: **flareon** (HP: 100%) | Type: fire notype | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **swordsdance** [normal] (status) PP:31/32
  - Slot 2: **temperflare** [fire] (physical) BP:75 PP:15/16
  - Slot 3: **closecombat** [fighting] (physical) BP:120 PP:8/8
  - Slot 4: **flareblitz** [fire] (physical) BP:120 PP:24/24
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ribombee** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **gholdengo** (HP: 100%) [nostatus]
  - Bench 3: **indeedee** (HP: 100%) [nostatus]
  - Bench 4: **ironcrown** (HP: 100%) [nostatus]
  - Bench 5: **hoopaunbound** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **ribombee**

---
**Turn 6**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used quickattack

**Player Active**: **gholdengo** (HP: 100%) | Type: ghost steel | Item: airballoon | Ability: goodasgold | Status: nostatus | Tera: fairy
**Opponent Active**: **flareon** (HP: 100%) | Type: fire notype | Status: tox | Tera: notype

**Available Moves:**
  - Slot 1: **shadowball** [ghost] (special) BP:80 PP:24/24
  - Slot 2: **psyshock** [psychic] (special) BP:80 PP:16/16
  - Slot 3: **focusblast** [fighting] (special) BP:120 PP:8/8
  - Slot 4: **nastyplot** [dark] (status) PP:32/32 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ribombee** (HP: 100%) [nostatus]
  - Bench 2: **indeedee** (HP: 100%) [nostatus]
  - Bench 3: **ironcrown** (HP: 100%) [nostatus]
  - Bench 4: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 5: **blaziken** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **nastyplot**

---
**Turn 7** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used shadowball | Opponent used flareblitz

**Player Active**: **gholdengo** (HP: 0%) | Type: ghost steel | Item: airballoon | Ability: goodasgold | Status: fnt | Tera: fairy
**Opponent Active**: **flareon** (HP: 28%) | Type: fire notype | Status: tox | Tera: notype

**Available Switches:**
  - Bench 1: **ribombee** (HP: 100%) [nostatus]
  - Bench 2: **indeedee** (HP: 100%) [nostatus]
  - Bench 3: **ironcrown** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 4: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 5: **blaziken** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 3: Switch to **ironcrown**

---
**Turn 8**

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used flareblitz

**Player Active**: **indeedee** (HP: 100%) | Type: normal psychic | Item: choicescarf | Ability: psychicsurge | Status: nostatus | Tera: fairy
**Opponent Active**: **flareon** (HP: 28%) | Type: fire notype | Status: tox | Tera: notype

**Available Moves:**
  - Slot 1: **expandingforce** [psychic] (special) BP:80 PP:16/16
  - Slot 2: **healingwish** [psychic] (status) PP:16/16
  - Slot 3: **encore** [normal] (status) PP:8/8 <-- CHOSEN
  - Slot 4: **dazzlinggleam** [fairy] (special) BP:80 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ribombee** (HP: 100%) [nostatus]
  - Bench 2: **ironcrown** (HP: 100%) [nostatus]
  - Bench 3: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 4: **blaziken** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **encore**

---
**Turn 9**

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used expandingforce | Opponent used nomove

**Player Active**: **indeedee** (HP: 100%) | Type: normal psychic | Item: choicescarf | Ability: psychicsurge | Status: nostatus | Tera: fairy
**Opponent Active**: **sylveon** (HP: 100%) | Type: fairy notype | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **expandingforce** [psychic] (special) BP:80 PP:15/16
  - Slot 2: **healingwish** [psychic] (status) PP:16/16
  - Slot 3: **encore** [normal] (status) PP:8/8
  - Slot 4: **dazzlinggleam** [fairy] (special) BP:80 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ribombee** (HP: 100%) [nostatus]
  - Bench 2: **ironcrown** (HP: 100%) [nostatus]
  - Bench 3: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 4: **blaziken** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 4: Switch to **blaziken**

---
**Turn 10**

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used uturn | Opponent used hypervoice

**Player Active**: **ribombee** (HP: 1%) | Type: bug fairy | Item: noitem | Ability: shielddust | Status: nostatus | Tera: bug
**Opponent Active**: **sylveon** (HP: 100%) | Type: fairy notype | Item: noitem | Status: nostatus | Tera: fairy | Boosts: SPA+1

**Available Moves:**
  - Slot 1: **uturn** [bug] (physical) BP:70 PP:31/32
  - Slot 2: **stickyweb** [bug] (status) PP:32/32
  - Slot 3: **psychic** [psychic] (special) BP:90 PP:16/16 <-- CHOSEN
  - Slot 4: **psychicnoise** [psychic] (special) BP:75 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **indeedee** (HP: 100%) [nostatus]
  - Bench 2: **ironcrown** (HP: 100%) [nostatus]
  - Bench 3: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 4: **blaziken** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **psychic**

---
**Turn 11** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used stickyweb | Opponent used drainingkiss

**Player Active**: **ribombee** (HP: 0%) | Type: bug fairy | Item: noitem | Ability: shielddust | Status: fnt | Tera: bug
**Opponent Active**: **sylveon** (HP: 100%) | Type: fairy notype | Item: noitem | Status: nostatus | Tera: fairy | Boosts: SPA+1

**Available Switches:**
  - Bench 1: **indeedee** (HP: 100%) [nostatus]
  - Bench 2: **ironcrown** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 3: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 4: **blaziken** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 2: Switch to **ironcrown**

---
**Turn 12**

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used nomove | Opponent used drainingkiss

**Player Active**: **hoopaunbound** (HP: 100%) | Type: dark psychic | Item: lifeorb | Ability: magician | Status: nostatus | Tera: psychic
**Opponent Active**: **sylveon** (HP: 100%) | Type: fairy notype | Item: noitem | Status: nostatus | Tera: fairy | Boosts: SPA+1

**Available Moves:**
  - Slot 1: **expandingforce** [psychic] (special) BP:80 PP:16/16
  - Slot 2: **psychicnoise** [psychic] (special) BP:75 PP:16/16 <-- CHOSEN (with Tera)
  - Slot 3: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 4: **brickbreak** [fighting] (physical) BP:75 PP:24/24
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **indeedee** (HP: 100%) [nostatus]
  - Bench 2: **ironcrown** (HP: 100%) [nostatus]
  - Bench 3: **blaziken** (HP: 100%) [nostatus]

**Decision**: Tera + Move 2: Terastallize + Use **psychicnoise**

---
**Turn 13**

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used expandingforce | Opponent used drainingkiss

**Player Active**: **hoopaunbound** (HP: 91%) | Type: notype psychic | Item: lifeorb | Ability: magician | Status: nostatus | Tera: psychic
**Opponent Active**: **sylveon** (HP: 0%) | Type: fairy notype | Item: noitem | Status: fnt | Tera: fairy | Boosts: SPA+1

**Available Moves:**
  - Slot 1: **expandingforce** [psychic] (special) BP:80 PP:15/16
  - Slot 2: **psychicnoise** [psychic] (special) BP:75 PP:16/16
  - Slot 3: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 4: **brickbreak** [fighting] (physical) BP:75 PP:24/24

**Available Switches:**
  - Bench 1: **indeedee** (HP: 100%) [nostatus]
  - Bench 2: **ironcrown** (HP: 100%) [nostatus]
  - Bench 3: **blaziken** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **brickbreak**


---

## 3. Battle 3

### Battle: `gen9ou-2064798635_1558_kangaskhan20989_vs_furyattack47818_02-21-2024_LOSS.json.lz4`
- **Result**: LOSS
- **Elo**: 1558
- **Total Turns**: 16

#### Player's Team
1. **slowkinggalar** (HP: 100%) | Type: poison psychic | Item: heavydutyboots | Ability: regenerator | Status: nostatus | Tera: fairy *(Lead)*
2. **clefable** (HP: 100%) | Type: fairy notype | Item: leftovers | Ability: magicguard | Status: nostatus | Tera: steel
3. **hoopaunbound** (HP: 100%) | Type: dark psychic | Item: lifeorb | Ability: magician | Status: nostatus | Tera: fighting
4. **dragapult** (HP: 100%) | Type: dragon ghost | Item: expertbelt | Ability: clearbody | Status: nostatus | Tera: ghost
5. **dondozo** (HP: 100%) | Type: notype water | Item: leftovers | Ability: unaware | Status: nostatus | Tera: poison
6. **gliscor** (HP: 100%) | Type: flying ground | Item: toxicorb | Ability: poisonheal | Status: nostatus | Tera: water

#### Opponent's Team (from Team Preview)
1. **ribombee**
2. **ogerpon**
3. **greattusk**
4. **volcarona**
5. **ragingbolt**
6. **gholdengo**

#### Turn-by-Turn Decisions

---
**Turn 1**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **slowkinggalar** (HP: 100%) | Type: poison psychic | Item: heavydutyboots | Ability: regenerator | Status: nostatus | Tera: fairy
**Opponent Active**: **ribombee** (HP: 100%) | Type: bug fairy | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **futuresight** [psychic] (special) BP:120 PP:16/16
  - Slot 2: **sludgebomb** [poison] (special) BP:90 PP:16/16 <-- CHOSEN
  - Slot 3: **chillyreception** [ice] (status) PP:16/16
  - Slot 4: **slackoff** [normal] (status) PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **clefable** (HP: 100%) [nostatus]
  - Bench 2: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 3: **dragapult** (HP: 100%) [nostatus]
  - Bench 4: **dondozo** (HP: 100%) [nostatus]
  - Bench 5: **gliscor** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **sludgebomb**

---
**Turn 2**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used futuresight | Opponent used stickyweb

**Player Active**: **slowkinggalar** (HP: 100%) | Type: poison psychic | Item: heavydutyboots | Ability: regenerator | Status: nostatus | Tera: fairy
**Opponent Active**: **ribombee** (HP: 100%) | Type: bug fairy | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **futuresight** [psychic] (special) BP:120 PP:15/16
  - Slot 2: **sludgebomb** [poison] (special) BP:90 PP:16/16
  - Slot 3: **chillyreception** [ice] (status) PP:16/16
  - Slot 4: **slackoff** [normal] (status) PP:8/8 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **clefable** (HP: 100%) [nostatus]
  - Bench 2: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 3: **dragapult** (HP: 100%) [nostatus]
  - Bench 4: **dondozo** (HP: 100%) [nostatus]
  - Bench 5: **gliscor** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **slackoff**

---
**Turn 3**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used sludgebomb | Opponent used stunspore

**Player Active**: **slowkinggalar** (HP: 100%) | Type: poison psychic | Item: heavydutyboots | Ability: regenerator | Status: par | Tera: fairy
**Opponent Active**: **ribombee** (HP: 1%) | Type: bug fairy | Item: noitem | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **futuresight** [psychic] (special) BP:120 PP:15/16 <-- CHOSEN
  - Slot 2: **sludgebomb** [poison] (special) BP:90 PP:15/16
  - Slot 3: **chillyreception** [ice] (status) PP:16/16
  - Slot 4: **slackoff** [normal] (status) PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **clefable** (HP: 100%) [nostatus]
  - Bench 2: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 3: **dragapult** (HP: 100%) [nostatus]
  - Bench 4: **dondozo** (HP: 100%) [nostatus]
  - Bench 5: **gliscor** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **futuresight**

---
**Turn 4** *(Forced Switch)*

*Field*: Weather: snow | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used chillyreception | Opponent used skillswap

**Player Active**: **slowkinggalar** (HP: 100%) | Type: poison psychic | Item: heavydutyboots | Ability: regenerator | Status: par | Tera: fairy
**Opponent Active**: **ribombee** (HP: 1%) | Type: bug fairy | Item: noitem | Status: nostatus | Tera: notype

**Available Switches:**
  - Bench 1: **clefable** (HP: 100%) [nostatus]
  - Bench 2: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 3: **dragapult** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 4: **dondozo** (HP: 100%) [nostatus]
  - Bench 5: **gliscor** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 3: Switch to **dragapult**

---
**Turn 5**

*Field*: Weather: snow | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **dragapult** (HP: 100%) | Type: dragon ghost | Item: expertbelt | Ability: clearbody | Status: nostatus | Tera: ghost
**Opponent Active**: **ragingbolt** (HP: 100%) | Type: dragon electric | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:16/16
  - Slot 2: **willowisp** [fire] (status) PP:24/24 <-- CHOSEN
  - Slot 3: **dracometeor** [dragon] (special) BP:130 PP:8/8
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **clefable** (HP: 100%) [nostatus]
  - Bench 2: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 3: **dondozo** (HP: 100%) [nostatus]
  - Bench 4: **slowkinggalar** (HP: 100%) [par]
  - Bench 5: **gliscor** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **willowisp**

---
**Turn 6**

*Field*: Weather: snow | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used substitute | Opponent used calmmind

**Player Active**: **dragapult** (HP: 76%) | Type: dragon ghost | Item: expertbelt | Ability: clearbody | Status: nostatus | Tera: ghost
**Opponent Active**: **ragingbolt** (HP: 100%) | Type: fairy notype | Ability: protosynthesis | Status: nostatus | Tera: fairy | Boosts: SPA+1, SPD+1

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:15/16
  - Slot 2: **willowisp** [fire] (status) PP:24/24
  - Slot 3: **dracometeor** [dragon] (special) BP:130 PP:8/8
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **clefable** (HP: 100%) [nostatus]
  - Bench 2: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 3: **dondozo** (HP: 100%) [nostatus]
  - Bench 4: **slowkinggalar** (HP: 100%) [par]
  - Bench 5: **gliscor** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **uturn**

---
**Turn 7**

*Field*: Weather: snow | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used willowisp | Opponent used dragonpulse

**Player Active**: **dragapult** (HP: 76%) | Type: dragon ghost | Item: expertbelt | Ability: clearbody | Status: nostatus | Tera: ghost
**Opponent Active**: **ragingbolt** (HP: 94%) | Type: fairy notype | Ability: protosynthesis | Status: brn | Tera: fairy | Boosts: SPA+1, SPD+1

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:15/16
  - Slot 2: **willowisp** [fire] (status) PP:23/24
  - Slot 3: **dracometeor** [dragon] (special) BP:130 PP:8/8
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **clefable** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 3: **dondozo** (HP: 100%) [nostatus]
  - Bench 4: **slowkinggalar** (HP: 100%) [par]
  - Bench 5: **gliscor** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **clefable**

---
**Turn 8**

*Field*: Weather: snow | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used thunderbolt

**Player Active**: **clefable** (HP: 100%) | Type: fairy notype | Item: leftovers | Ability: magicguard | Status: nostatus | Tera: steel | Boosts: SPE-1
**Opponent Active**: **ragingbolt** (HP: 94%) | Type: fairy notype | Item: leftovers | Ability: protosynthesis | Status: brn | Tera: fairy | Boosts: SPA+1, SPD+1

**Available Moves:**
  - Slot 1: **protect** [normal] (status) Priority:4 PP:16/16
  - Slot 2: **thunderwave** [electric] (status) PP:32/32 <-- CHOSEN
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 2: **dragapult** (HP: 76%) [nostatus]
  - Bench 3: **dondozo** (HP: 100%) [nostatus]
  - Bench 4: **slowkinggalar** (HP: 100%) [par]
  - Bench 5: **gliscor** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **thunderwave**

---
**Turn 9**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used protect | Opponent used calmmind

**Player Active**: **clefable** (HP: 31%) | Type: fairy notype | Item: leftovers | Ability: magicguard | Status: nostatus | Tera: steel | Boosts: SPE-1
**Opponent Active**: **ragingbolt** (HP: 94%) | Type: fairy notype | Item: leftovers | Ability: protosynthesis | Status: brn | Tera: fairy | Boosts: SPA+2, SPD+2

**Available Moves:**
  - Slot 1: **protect** [normal] (status) Priority:4 PP:15/16
  - Slot 2: **thunderwave** [electric] (status) PP:32/32
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 2: **dragapult** (HP: 76%) [nostatus]
  - Bench 3: **dondozo** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 4: **slowkinggalar** (HP: 100%) [par]
  - Bench 5: **gliscor** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 3: Switch to **dondozo**

---
**Turn 10**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used thunderbolt

**Player Active**: **gliscor** (HP: 100%) | Type: flying ground | Item: toxicorb | Ability: poisonheal | Status: tox | Tera: water
**Opponent Active**: **ragingbolt** (HP: 94%) | Type: fairy notype | Item: leftovers | Ability: protosynthesis | Status: brn | Tera: fairy | Boosts: SPA+2, SPD+2

**Available Moves:**
  - Slot 1: **spikes** [ground] (status) PP:32/32
  - Slot 2: **stealthrock** [rock] (status) PP:32/32
  - Slot 3: **dualwingbeat** [flying] (physical) BP:40 PP:16/16
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **clefable** (HP: 31%) [nostatus] <-- CHOSEN
  - Bench 2: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 3: **dragapult** (HP: 76%) [nostatus]
  - Bench 4: **dondozo** (HP: 100%) [nostatus]
  - Bench 5: **slowkinggalar** (HP: 100%) [par]

**Decision**: Switch to bench slot 1: Switch to **clefable**

---
**Turn 11**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used protect | Opponent used dragonpulse

**Player Active**: **clefable** (HP: 37%) | Type: fairy notype | Item: leftovers | Ability: magicguard | Status: nostatus | Tera: steel | Boosts: SPE-1
**Opponent Active**: **ragingbolt** (HP: 94%) | Type: fairy notype | Item: leftovers | Ability: protosynthesis | Status: brn | Tera: fairy | Boosts: SPA+2, SPD+2

**Available Moves:**
  - Slot 1: **protect** [normal] (status) Priority:4 PP:15/16
  - Slot 2: **thunderwave** [electric] (status) PP:32/32
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 2: **dragapult** (HP: 76%) [nostatus]
  - Bench 3: **dondozo** (HP: 100%) [nostatus]
  - Bench 4: **slowkinggalar** (HP: 100%) [par]
  - Bench 5: **gliscor** (HP: 100%) [tox]

**Decision**: Action -1: Use **knockoff**

---
**Turn 12** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used protect | Opponent used thunderbolt

**Player Active**: **clefable** (HP: 0%) | Type: fairy notype | Item: leftovers | Ability: magicguard | Status: fnt | Tera: steel | Boosts: SPE-1
**Opponent Active**: **ragingbolt** (HP: 94%) | Type: fairy notype | Item: leftovers | Ability: protosynthesis | Status: brn | Tera: fairy | Boosts: SPA+2, SPD+2

**Available Switches:**
  - Bench 1: **hoopaunbound** (HP: 100%) [nostatus]
  - Bench 2: **dragapult** (HP: 76%) [nostatus]
  - Bench 3: **dondozo** (HP: 100%) [nostatus]
  - Bench 4: **slowkinggalar** (HP: 100%) [par] <-- CHOSEN
  - Bench 5: **gliscor** (HP: 100%) [tox]

**Decision**: Switch to bench slot 4: Switch to **slowkinggalar**

---
**Turn 13**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used thunderbolt

**Player Active**: **hoopaunbound** (HP: 100%) | Type: dark psychic | Item: lifeorb | Ability: magician | Status: nostatus | Tera: fighting | Boosts: SPE-1
**Opponent Active**: **ragingbolt** (HP: 94%) | Type: fairy notype | Item: leftovers | Ability: protosynthesis | Status: brn | Tera: fairy | Boosts: SPA+2, SPD+2

**Available Moves:**
  - Slot 1: **psyshock** [psychic] (special) BP:80 PP:16/16
  - Slot 2: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 3: **gunkshot** [poison] (physical) BP:120 PP:8/8
  - Slot 4: **psychicnoise** [psychic] (special) BP:75 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragapult** (HP: 76%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **slowkinggalar** (HP: 100%) [par]
  - Bench 4: **gliscor** (HP: 100%) [tox]

**Decision**: Action -1: Use **psychicnoise**

---
**Turn 14** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used thunderbolt

**Player Active**: **hoopaunbound** (HP: 0%) | Type: dark psychic | Item: lifeorb | Ability: magician | Status: fnt | Tera: fighting | Boosts: SPE-1
**Opponent Active**: **ragingbolt** (HP: 94%) | Type: fairy notype | Item: leftovers | Ability: protosynthesis | Status: brn | Tera: fairy | Boosts: SPA+2, SPD+2

**Available Switches:**
  - Bench 1: **dragapult** (HP: 76%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **slowkinggalar** (HP: 100%) [par]
  - Bench 4: **gliscor** (HP: 100%) [tox] <-- CHOSEN

**Decision**: Switch to bench slot 4: Switch to **gliscor**

---
**Turn 15**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used chillyreception | Opponent used thunderbolt

**Player Active**: **slowkinggalar** (HP: 100%) | Type: poison psychic | Item: heavydutyboots | Ability: regenerator | Status: par | Tera: fairy
**Opponent Active**: **ragingbolt** (HP: 94%) | Type: fairy notype | Item: leftovers | Ability: protosynthesis | Status: brn | Tera: fairy | Boosts: SPA+2, SPD+2

**Available Moves:**
  - Slot 1: **futuresight** [psychic] (special) BP:120 PP:15/16
  - Slot 2: **sludgebomb** [poison] (special) BP:90 PP:15/16
  - Slot 3: **chillyreception** [ice] (status) PP:15/16
  - Slot 4: **slackoff** [normal] (status) PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragapult** (HP: 76%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **gliscor** (HP: 100%) [tox]

**Decision**: Action -1: Use **slackoff**

---
**Turn 16**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stickyweb | Opponent side: noconditions

*Last turn*: Player used chillyreception | Opponent used thunderbolt

**Player Active**: **slowkinggalar** (HP: 100%) | Type: poison psychic | Item: heavydutyboots | Ability: regenerator | Status: par | Tera: fairy
**Opponent Active**: **ragingbolt** (HP: 94%) | Type: fairy notype | Item: leftovers | Ability: protosynthesis | Status: brn | Tera: fairy | Boosts: SPA+2, SPD+2

**Available Moves:**
  - Slot 1: **futuresight** [psychic] (special) BP:120 PP:15/16
  - Slot 2: **sludgebomb** [poison] (special) BP:90 PP:15/16
  - Slot 3: **chillyreception** [ice] (status) PP:15/16
  - Slot 4: **slackoff** [normal] (status) PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragapult** (HP: 76%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **gliscor** (HP: 100%) [tox]

**Decision**: Action -1: Use **slackoff**


---

## 4. Battle 4

### Battle: `gen9ou-2114115772_1986_combee67377_vs_snore48848_04-29-2024_WIN.json.lz4`
- **Result**: WIN
- **Elo**: 1986
- **Total Turns**: 19

#### Player's Team
1. **ribombee** (HP: 100%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: nostatus | Tera: steel *(Lead)*
2. **ogerponwellspring** (HP: 100%) | Type: grass water | Item: wellspringmask | Ability: waterabsorb | Status: nostatus | Tera: water
3. **greattusk** (HP: 100%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: steel
4. **ragingbolt** (HP: 100%) | Type: dragon electric | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ice
5. **gholdengo** (HP: 100%) | Type: ghost steel | Item: airballoon | Ability: goodasgold | Status: nostatus | Tera: fairy
6. **roaringmoon** (HP: 100%) | Type: dark dragon | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: flying

#### Opponent's Team (from Team Preview)
1. **indeedee**
2. **armarouge**
3. **polteageist**
4. **primarina**
5. **greattusk**
6. **roaringmoon**

#### Turn-by-Turn Decisions

---
**Turn 1**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **ribombee** (HP: 100%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: nostatus | Tera: steel
**Opponent Active**: **primarina** (HP: 100%) | Type: fairy water | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **stickyweb** [bug] (status) PP:32/32
  - Slot 2: **stunspore** [grass] (status) PP:48/48
  - Slot 3: **moonblast** [fairy] (special) BP:95 PP:24/24 <-- CHOSEN
  - Slot 4: **skillswap** [psychic] (status) PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ragingbolt** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **roaringmoon** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **moonblast**

---
**Turn 2**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used stickyweb | Opponent used moonblast

**Player Active**: **ribombee** (HP: 12%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: nostatus | Tera: steel
**Opponent Active**: **primarina** (HP: 100%) | Type: fairy water | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **stickyweb** [bug] (status) PP:31/32
  - Slot 2: **stunspore** [grass] (status) PP:48/48
  - Slot 3: **moonblast** [fairy] (special) BP:95 PP:24/24
  - Slot 4: **skillswap** [psychic] (status) PP:16/16 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ragingbolt** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **roaringmoon** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **skillswap**

---
**Turn 3** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used stunspore | Opponent used flipturn

**Player Active**: **ribombee** (HP: 0%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: fnt | Tera: steel
**Opponent Active**: **primarina** (HP: 100%) | Type: fairy water | Status: par | Tera: notype

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ragingbolt** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **roaringmoon** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 5: Switch to **roaringmoon**

---
**Turn 4**

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **roaringmoon** (HP: 100%) | Type: dark dragon | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: flying
**Opponent Active**: **indeedee** (HP: 100%) | Type: normal psychic | Ability: psychicsurge | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:32/32
  - Slot 2: **dragondance** [dragon] (status) PP:32/32
  - Slot 3: **earthquake** [ground] (physical) BP:100 PP:16/16 <-- CHOSEN
  - Slot 4: **outrage** [dragon] (physical) BP:120 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ragingbolt** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **earthquake**

---
**Turn 5** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used knockoff | Opponent used dazzlinggleam

**Player Active**: **roaringmoon** (HP: 0%) | Type: dark dragon | Item: noitem | Ability: protosynthesis | Status: fnt | Tera: flying
**Opponent Active**: **indeedee** (HP: 42%) | Type: fairy notype | Item: noitem | Ability: psychicsurge | Status: nostatus | Tera: fairy | Boosts: SPE-1

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 3: **ragingbolt** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 2: Switch to **greattusk**

---
**Turn 6**

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used nomove | Opponent used dazzlinggleam

**Player Active**: **greattusk** (HP: 100%) | Type: fighting ground | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: steel
**Opponent Active**: **indeedee** (HP: 42%) | Type: fairy notype | Item: noitem | Ability: psychicsurge | Status: nostatus | Tera: fairy | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **icespinner** [ice] (physical) BP:80 PP:24/24
  - Slot 2: **headlongrush** [ground] (physical) BP:120 PP:8/8 <-- CHOSEN
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **ragingbolt** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **headlongrush**

---
**Turn 7**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used icespinner | Opponent used nomove

**Player Active**: **greattusk** (HP: 84%) | Type: fighting ground | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: steel
**Opponent Active**: **greattusk** (HP: 43%) | Type: fighting ground | Item: rockyhelmet | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **icespinner** [ice] (physical) BP:80 PP:23/24 <-- CHOSEN
  - Slot 2: **headlongrush** [ground] (physical) BP:120 PP:8/8
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **ragingbolt** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **icespinner**

---
**Turn 8**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used headlongrush | Opponent used nomove

**Player Active**: **greattusk** (HP: 68%) | Type: fighting ground | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: steel | Boosts: DEF-1, SPD-1
**Opponent Active**: **armarouge** (HP: 100%) | Type: fire psychic | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **icespinner** [ice] (physical) BP:80 PP:23/24
  - Slot 2: **headlongrush** [ground] (physical) BP:120 PP:7/8
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **ragingbolt** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 3: Switch to **gholdengo**

---
**Turn 9**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used nomove | Opponent used endure

**Player Active**: **ragingbolt** (HP: 100%) | Type: dragon electric | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: ice
**Opponent Active**: **armarouge** (HP: 100%) | Type: fire psychic | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **calmmind** [psychic] (status) PP:32/32 <-- CHOSEN
  - Slot 2: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 3: **dragonpulse** [dragon] (special) BP:85 PP:16/16
  - Slot 4: **weatherball** [normal] (special) BP:50 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 68%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **calmmind**

---
**Turn 10**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used calmmind | Opponent used expandingforce

**Player Active**: **ragingbolt** (HP: 76%) | Type: dragon electric | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: ice | Boosts: SPA+1, SPD+1
**Opponent Active**: **armarouge** (HP: 100%) | Type: fire psychic | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **calmmind** [psychic] (status) PP:31/32
  - Slot 2: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 3: **dragonpulse** [dragon] (special) BP:85 PP:16/16 <-- CHOSEN
  - Slot 4: **weatherball** [normal] (special) BP:50 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 68%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **dragonpulse**

---
**Turn 11**

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used thunderbolt | Opponent used nomove

**Player Active**: **ragingbolt** (HP: 76%) | Type: dragon electric | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: ice | Boosts: SPA+1, SPD+1
**Opponent Active**: **polteageist** (HP: 100%) | Type: ghost notype | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **calmmind** [psychic] (status) PP:31/32
  - Slot 2: **thunderbolt** [electric] (special) BP:90 PP:23/24
  - Slot 3: **dragonpulse** [dragon] (special) BP:85 PP:16/16 <-- CHOSEN
  - Slot 4: **weatherball** [normal] (special) BP:50 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 68%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **dragonpulse**

---
**Turn 12**

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used thunderbolt | Opponent used shellsmash

**Player Active**: **ragingbolt** (HP: 76%) | Type: dragon electric | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: ice | Boosts: SPA+1, SPD+1
**Opponent Active**: **polteageist** (HP: 1%) | Type: ghost notype | Item: noitem | Status: nostatus | Tera: notype | Boosts: ATK+2, SPA+2, DEF-1, SPD-1, SPE+1

**Available Moves:**
  - Slot 1: **calmmind** [psychic] (status) PP:31/32
  - Slot 2: **thunderbolt** [electric] (special) BP:90 PP:22/24
  - Slot 3: **dragonpulse** [dragon] (special) BP:85 PP:16/16 <-- CHOSEN
  - Slot 4: **weatherball** [normal] (special) BP:50 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 68%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **dragonpulse**

---
**Turn 13**

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used thunderbolt | Opponent used nomove

**Player Active**: **ragingbolt** (HP: 6%) | Type: dragon electric | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: ice | Boosts: SPA+1, SPD+1
**Opponent Active**: **roaringmoon** (HP: 100%) | Type: dark dragon | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **calmmind** [psychic] (status) PP:31/32
  - Slot 2: **thunderbolt** [electric] (special) BP:90 PP:21/24
  - Slot 3: **dragonpulse** [dragon] (special) BP:85 PP:16/16
  - Slot 4: **weatherball** [normal] (special) BP:50 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 68%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **weatherball**

---
**Turn 14** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used thunderbolt | Opponent used acrobatics

**Player Active**: **ragingbolt** (HP: 0%) | Type: dragon electric | Item: noitem | Ability: protosynthesis | Status: fnt | Tera: ice | Boosts: SPA+1, SPD+1
**Opponent Active**: **roaringmoon** (HP: 100%) | Type: dark dragon | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Switches:**
  - Bench 1: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 68%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 3: Switch to **gholdengo**

---
**Turn 15**

*Field*: Weather: noweather | Terrain/Field: psychicterrain | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used nomove | Opponent used acrobatics

**Player Active**: **ogerponwellspring** (HP: 100%) | Type: grass water | Item: wellspringmask | Ability: waterabsorb | Status: nostatus | Tera: water
**Opponent Active**: **roaringmoon** (HP: 100%) | Type: dark dragon | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **playrough** [fairy] (physical) BP:90 PP:16/16
  - Slot 2: **ivycudgel** [grass] (physical) BP:100 PP:16/16
  - Slot 3: **trailblaze** [grass] (physical) BP:50 PP:32/32 <-- CHOSEN
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **greattusk** (HP: 68%) [nostatus]
  - Bench 2: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **trailblaze**

---
**Turn 16**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used playrough | Opponent used expandingforce

**Player Active**: **ogerponwellspring** (HP: 100%) | Type: grass water | Item: wellspringmask | Ability: waterabsorb | Status: nostatus | Tera: water
**Opponent Active**: **armarouge** (HP: 100%) | Type: fire psychic | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **playrough** [fairy] (physical) BP:90 PP:15/16 <-- CHOSEN (with Tera)
  - Slot 2: **ivycudgel** [grass] (physical) BP:100 PP:16/16
  - Slot 3: **trailblaze** [grass] (physical) BP:50 PP:32/32
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **greattusk** (HP: 68%) [nostatus]
  - Bench 2: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Tera + Move 1: Terastallize + Use **playrough**

---
**Turn 17**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used ivycudgel | Opponent used flipturn

**Player Active**: **ogerponwellspringtera** (HP: 100%) | Type: grass water | Item: wellspringmask | Ability: embodyaspectwellspring | Status: nostatus | Tera: water | Boosts: SPD+1
**Opponent Active**: **primarina** (HP: 100%) | Type: fairy water | Status: par | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **playrough** [fairy] (physical) BP:90 PP:15/16
  - Slot 2: **ivycudgel** [grass] (physical) BP:100 PP:15/16
  - Slot 3: **trailblaze** [grass] (physical) BP:50 PP:32/32
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32 <-- CHOSEN

**Available Switches:**
  - Bench 1: **greattusk** (HP: 68%) [nostatus]
  - Bench 2: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **knockoff**

---
**Turn 18**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used trailblaze | Opponent used moonblast

**Player Active**: **ogerponwellspringtera** (HP: 61%) | Type: grass water | Item: wellspringmask | Ability: embodyaspectwellspring | Status: nostatus | Tera: water | Boosts: SPD+1, SPE+1
**Opponent Active**: **primarina** (HP: 14%) | Type: fairy water | Status: par | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **playrough** [fairy] (physical) BP:90 PP:15/16 <-- CHOSEN
  - Slot 2: **ivycudgel** [grass] (physical) BP:100 PP:15/16
  - Slot 3: **trailblaze** [grass] (physical) BP:50 PP:31/32
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32

**Available Switches:**
  - Bench 1: **greattusk** (HP: 68%) [nostatus]
  - Bench 2: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **playrough**

---
**Turn 19**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stickyweb

*Last turn*: Player used ivycudgel | Opponent used moonblast

**Player Active**: **ogerponwellspringtera** (HP: 61%) | Type: grass water | Item: wellspringmask | Ability: embodyaspectwellspring | Status: nostatus | Tera: water | Boosts: SPD+1, SPE+1
**Opponent Active**: **primarina** (HP: 0%) | Type: fairy water | Status: fnt | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **playrough** [fairy] (physical) BP:90 PP:15/16
  - Slot 2: **ivycudgel** [grass] (physical) BP:100 PP:14/16
  - Slot 3: **trailblaze** [grass] (physical) BP:50 PP:31/32
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32

**Available Switches:**
  - Bench 1: **greattusk** (HP: 68%) [nostatus]
  - Bench 2: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **knockoff**


---

## 5. Battle 5

### Battle: `gen9ou-2179884364_1736_alakazam79428_vs_uturn77356_08-14-2024_WIN.json.lz4`
- **Result**: WIN
- **Elo**: 1736
- **Total Turns**: 21

#### Player's Team
1. **gholdengo** (HP: 100%) | Type: ghost steel | Item: airballoon | Ability: goodasgold | Status: nostatus | Tera: fairy *(Lead)*
2. **ironvaliant** (HP: 100%) | Type: fairy fighting | Item: boosterenergy | Ability: quarkdrive | Status: nostatus | Tera: fairy
3. **irontreads** (HP: 100%) | Type: ground steel | Item: boosterenergy | Ability: quarkdrive | Status: nostatus | Tera: flying
4. **garchomp** (HP: 100%) | Type: dragon ground | Item: rockyhelmet | Ability: roughskin | Status: nostatus | Tera: steel
5. **samurotthisui** (HP: 100%) | Type: dark water | Item: assaultvest | Ability: sharpness | Status: nostatus | Tera: dark
6. **dragonite** (HP: 100%) | Type: dragon flying | Item: choiceband | Ability: multiscale | Status: nostatus | Tera: normal

#### Opponent's Team (from Team Preview)
1. **roaringmoon**
2. **landorus**
3. **ironmoth**
4. **ironvaliant**
5. **kingambit**
6. **kyurem**

#### Turn-by-Turn Decisions

---
**Turn 1**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **gholdengo** (HP: 100%) | Type: ghost steel | Item: airballoon | Ability: goodasgold | Status: nostatus | Tera: fairy
**Opponent Active**: **kyurem** (HP: 100%) | Type: dragon ice | Ability: pressure | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **makeitrain** [steel] (special) BP:120 PP:8/8
  - Slot 2: **hex** [ghost] (special) BP:65 PP:16/16
  - Slot 3: **shadowball** [ghost] (special) BP:80 PP:24/24 <-- CHOSEN
  - Slot 4: **dazzlinggleam** [fairy] (special) BP:80 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **irontreads** (HP: 100%) [nostatus]
  - Bench 3: **garchomp** (HP: 100%) [nostatus]
  - Bench 4: **samurotthisui** (HP: 100%) [nostatus]
  - Bench 5: **dragonite** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **shadowball**

---
**Turn 2**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used makeitrain | Opponent used nomove

**Player Active**: **gholdengo** (HP: 100%) | Type: ghost steel | Item: airballoon | Ability: goodasgold | Status: nostatus | Tera: fairy | Boosts: ATK-1, SPA-1
**Opponent Active**: **landorustherian** (HP: 28%) | Type: flying ground | Ability: intimidate | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **makeitrain** [steel] (special) BP:120 PP:7/8
  - Slot 2: **hex** [ghost] (special) BP:65 PP:16/16
  - Slot 3: **shadowball** [ghost] (special) BP:80 PP:24/24
  - Slot 4: **dazzlinggleam** [fairy] (special) BP:80 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **irontreads** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 3: **garchomp** (HP: 100%) [nostatus]
  - Bench 4: **samurotthisui** (HP: 100%) [nostatus]
  - Bench 5: **dragonite** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 2: Switch to **irontreads**

---
**Turn 3**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **garchomp** (HP: 100%) | Type: dragon ground | Item: rockyhelmet | Ability: roughskin | Status: nostatus | Tera: steel
**Opponent Active**: **kyurem** (HP: 100%) | Type: dragon ice | Ability: pressure | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **ironhead** [steel] (physical) BP:80 PP:23/24
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:16/16 <-- CHOSEN (with Tera)
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **spikes** [ground] (status) PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **irontreads** (HP: 100%) [nostatus]
  - Bench 3: **samurotthisui** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **dragonite** (HP: 100%) [nostatus]

**Decision**: Tera + Move 2: Terastallize + Use **earthquake**

---
**Turn 4**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used ironhead | Opponent used icebeam

**Player Active**: **garchomp** (HP: 62%) | Type: notype steel | Item: rockyhelmet | Ability: roughskin | Status: nostatus | Tera: steel
**Opponent Active**: **kyurem** (HP: 26%) | Type: dragon ice | Ability: pressure | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **ironhead** [steel] (physical) BP:80 PP:22/24 <-- CHOSEN
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:15/16
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **spikes** [ground] (status) PP:32/32

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **irontreads** (HP: 100%) [nostatus]
  - Bench 3: **samurotthisui** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **dragonite** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **ironhead**

---
**Turn 5**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used earthquake | Opponent used nomove

**Player Active**: **garchomp** (HP: 62%) | Type: notype steel | Item: rockyhelmet | Ability: roughskin | Status: nostatus | Tera: steel
**Opponent Active**: **ironmoth** (HP: 100%) | Type: fire poison | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **ironhead** [steel] (physical) BP:80 PP:22/24
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:14/16
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **spikes** [ground] (status) PP:32/32

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **irontreads** (HP: 100%) [nostatus]
  - Bench 3: **samurotthisui** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **dragonite** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 5: Switch to **dragonite**

---
**Turn 6**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used fierydance

**Player Active**: **samurotthisui** (HP: 100%) | Type: dark water | Item: assaultvest | Ability: sharpness | Status: nostatus | Tera: dark
**Opponent Active**: **ironmoth** (HP: 100%) | Type: fire poison | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: notype | Boosts: SPA+1

**Available Moves:**
  - Slot 1: **razorshell** [water] (physical) BP:75 PP:16/16
  - Slot 2: **aquajet** [water] (physical) BP:40 Priority:1 PP:32/32
  - Slot 3: **flipturn** [water] (physical) BP:60 PP:32/32
  - Slot 4: **ceaselessedge** [dark] (physical) BP:65 PP:24/24 <-- CHOSEN

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **irontreads** (HP: 100%) [nostatus]
  - Bench 3: **garchomp** (HP: 62%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **dragonite** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **ceaselessedge**

---
**Turn 7**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used razorshell | Opponent used nomove

**Player Active**: **samurotthisui** (HP: 11%) | Type: dark water | Item: assaultvest | Ability: sharpness | Status: nostatus | Tera: dark
**Opponent Active**: **roaringmoon** (HP: 100%) | Type: dark dragon | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **razorshell** [water] (physical) BP:75 PP:15/16
  - Slot 2: **aquajet** [water] (physical) BP:40 Priority:1 PP:32/32
  - Slot 3: **flipturn** [water] (physical) BP:60 PP:32/32
  - Slot 4: **ceaselessedge** [dark] (physical) BP:65 PP:24/24

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **irontreads** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 3: **garchomp** (HP: 62%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **dragonite** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 2: Switch to **irontreads**

---
**Turn 8**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used earthquake | Opponent used nomove

**Player Active**: **garchomp** (HP: 45%) | Type: dragon ground | Item: rockyhelmet | Ability: roughskin | Status: nostatus | Tera: steel
**Opponent Active**: **ironvaliant** (HP: 100%) | Type: fairy fighting | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **ironhead** [steel] (physical) BP:80 PP:22/24
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:14/16
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **spikes** [ground] (status) PP:32/32

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **irontreads** (HP: 100%) [nostatus]
  - Bench 3: **samurotthisui** (HP: 11%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **dragonite** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 5: Switch to **dragonite**

---
**Turn 9**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used razorshell | Opponent used calmmind

**Player Active**: **samurotthisui** (HP: 11%) | Type: dark water | Item: assaultvest | Ability: sharpness | Status: nostatus | Tera: dark
**Opponent Active**: **ironvaliant** (HP: 100%) | Type: fairy fighting | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: notype | Boosts: SPA+1, SPD+1

**Available Moves:**
  - Slot 1: **razorshell** [water] (physical) BP:75 PP:15/16 <-- CHOSEN
  - Slot 2: **aquajet** [water] (physical) BP:40 Priority:1 PP:32/32
  - Slot 3: **flipturn** [water] (physical) BP:60 PP:32/32
  - Slot 4: **ceaselessedge** [dark] (physical) BP:65 PP:24/24

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **irontreads** (HP: 100%) [nostatus]
  - Bench 3: **garchomp** (HP: 45%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **dragonite** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **razorshell**

---
**Turn 10** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used aquajet | Opponent used moonblast

**Player Active**: **samurotthisui** (HP: 0%) | Type: dark water | Item: assaultvest | Ability: sharpness | Status: fnt | Tera: dark
**Opponent Active**: **ironvaliant** (HP: 75%) | Type: fairy fighting | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: notype | Boosts: SPA+1, SPD+1

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **irontreads** (HP: 100%) [nostatus]
  - Bench 3: **garchomp** (HP: 45%) [nostatus] <-- CHOSEN
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **dragonite** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 3: Switch to **garchomp**

---
**Turn 11**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used makeitrain | Opponent used moonblast

**Player Active**: **gholdengo** (HP: 100%) | Type: ghost steel | Item: airballoon | Ability: goodasgold | Status: nostatus | Tera: fairy
**Opponent Active**: **ironvaliant** (HP: 75%) | Type: fairy fighting | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: notype | Boosts: SPA+1, SPD+1

**Available Moves:**
  - Slot 1: **makeitrain** [steel] (special) BP:120 PP:7/8
  - Slot 2: **hex** [ghost] (special) BP:65 PP:16/16
  - Slot 3: **shadowball** [ghost] (special) BP:80 PP:24/24
  - Slot 4: **dazzlinggleam** [fairy] (special) BP:80 PP:16/16

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **irontreads** (HP: 100%) [nostatus]
  - Bench 3: **garchomp** (HP: 45%) [nostatus]
  - Bench 4: **dragonite** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **dazzlinggleam**

---
**Turn 12** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used makeitrain | Opponent used shadowball

**Player Active**: **gholdengo** (HP: 0%) | Type: ghost steel | Item: noitem | Ability: goodasgold | Status: fnt | Tera: fairy
**Opponent Active**: **ironvaliant** (HP: 75%) | Type: ghost notype | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: ghost | Boosts: SPA+1, SPD+1

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **irontreads** (HP: 100%) [nostatus]
  - Bench 3: **garchomp** (HP: 45%) [nostatus] <-- CHOSEN
  - Bench 4: **dragonite** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 3: Switch to **garchomp**

---
**Turn 13**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used shadowball

**Player Active**: **irontreads** (HP: 100%) | Type: ground steel | Item: boosterenergy | Ability: quarkdrive | Status: nostatus | Tera: flying
**Opponent Active**: **ironvaliant** (HP: 75%) | Type: ghost notype | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: ghost | Boosts: SPA+1, SPD+1

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:32/32
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:64/64 <-- CHOSEN
  - Slot 3: **earthquake** [ground] (physical) BP:100 PP:16/16
  - Slot 4: **voltswitch** [electric] (special) BP:70 PP:32/32

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **garchomp** (HP: 45%) [nostatus]
  - Bench 3: **dragonite** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **rapidspin**

---
**Turn 14**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used knockoff | Opponent used shadowball

**Player Active**: **irontreads** (HP: 34%) | Type: ground steel | Item: boosterenergy | Ability: quarkdrive | Status: nostatus | Tera: flying
**Opponent Active**: **ironvaliant** (HP: 19%) | Type: ghost notype | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: ghost | Boosts: SPA+1, SPD+1

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:31/32
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:64/64
  - Slot 3: **earthquake** [ground] (physical) BP:100 PP:16/16
  - Slot 4: **voltswitch** [electric] (special) BP:70 PP:32/32

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **garchomp** (HP: 45%) [nostatus]
  - Bench 3: **dragonite** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **voltswitch**

---
**Turn 15** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used knockoff | Opponent used shadowball

**Player Active**: **irontreads** (HP: 0%) | Type: ground steel | Item: boosterenergy | Ability: quarkdrive | Status: fnt | Tera: flying
**Opponent Active**: **ironvaliant** (HP: 19%) | Type: ghost notype | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: ghost | Boosts: SPA+1, SPD+1

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **garchomp** (HP: 45%) [nostatus]
  - Bench 3: **dragonite** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **ironvaliant**

---
**Turn 16**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used shadowball

**Player Active**: **dragonite** (HP: 100%) | Type: dragon flying | Item: choiceband | Ability: multiscale | Status: nostatus | Tera: normal
**Opponent Active**: **ironvaliant** (HP: 19%) | Type: ghost notype | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: ghost | Boosts: SPA+1, SPD+1

**Available Moves:**
  - Slot 1: **earthquake** [ground] (physical) BP:100 PP:16/16
  - Slot 2: **dragondance** [dragon] (status) PP:32/32 <-- CHOSEN
  - Slot 3: **outrage** [dragon] (physical) BP:120 PP:16/16
  - Slot 4: **roost** [flying] (status) PP:8/8

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **garchomp** (HP: 45%) [nostatus]

**Decision**: Move 2: Use **dragondance**

---
**Turn 17**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used earthquake | Opponent used nomove

**Player Active**: **dragonite** (HP: 23%) | Type: dragon flying | Item: choiceband | Ability: multiscale | Status: nostatus | Tera: normal | Boosts: SPA-1
**Opponent Active**: **kingambit** (HP: 100%) | Type: dark steel | Ability: supremeoverlord | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **earthquake** [ground] (physical) BP:100 PP:15/16
  - Slot 2: **dragondance** [dragon] (status) PP:32/32
  - Slot 3: **outrage** [dragon] (physical) BP:120 PP:16/16
  - Slot 4: **roost** [flying] (status) PP:8/8

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **garchomp** (HP: 45%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **ironvaliant**

---
**Turn 18**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used earthquake | Opponent used suckerpunch

**Player Active**: **garchomp** (HP: 45%) | Type: dragon ground | Item: rockyhelmet | Ability: roughskin | Status: nostatus | Tera: steel
**Opponent Active**: **kingambit** (HP: 100%) | Type: dark steel | Ability: supremeoverlord | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **ironhead** [steel] (physical) BP:80 PP:22/24
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:14/16
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **spikes** [ground] (status) PP:32/32 <-- CHOSEN

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **dragonite** (HP: 23%) [nostatus]

**Decision**: Move 4: Use **spikes**

---
**Turn 19** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used stealthrock | Opponent used kowtowcleave

**Player Active**: **garchomp** (HP: 0%) | Type: dragon ground | Item: rockyhelmet | Ability: roughskin | Status: fnt | Tera: steel
**Opponent Active**: **kingambit** (HP: 78%) | Type: dark steel | Item: leftovers | Ability: supremeoverlord | Status: nostatus | Tera: notype

**Available Switches:**
  - Bench 1: **ironvaliant** (HP: 100%) [nostatus]
  - Bench 2: **dragonite** (HP: 23%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 2: Switch to **dragonite**

---
**Turn 20**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used nomove | Opponent used kowtowcleave

**Player Active**: **ironvaliant** (HP: 100%) | Type: fairy fighting | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: fairy
**Opponent Active**: **kingambit** (HP: 78%) | Type: dark steel | Item: leftovers | Ability: supremeoverlord | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **vacuumwave** [fighting] (special) BP:40 Priority:1 PP:48/48
  - Slot 2: **calmmind** [psychic] (status) PP:32/32
  - Slot 3: **encore** [normal] (status) PP:8/8
  - Slot 4: **shadowball** [ghost] (special) BP:80 PP:24/24 <-- CHOSEN

**Available Switches:**
  - Bench 1: **dragonite** (HP: 23%) [nostatus]

**Decision**: Move 4: Use **shadowball**

---
**Turn 21**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used vacuumwave | Opponent used kowtowcleave

**Player Active**: **ironvaliant** (HP: 100%) | Type: fairy fighting | Item: noitem | Ability: quarkdrive | Status: nostatus | Tera: fairy
**Opponent Active**: **kingambit** (HP: 0%) | Type: dark steel | Item: leftovers | Ability: supremeoverlord | Status: fnt | Tera: notype

**Available Moves:**
  - Slot 1: **vacuumwave** [fighting] (special) BP:40 Priority:1 PP:47/48
  - Slot 2: **calmmind** [psychic] (status) PP:32/32
  - Slot 3: **encore** [normal] (status) PP:8/8
  - Slot 4: **shadowball** [ghost] (special) BP:80 PP:24/24

**Available Switches:**
  - Bench 1: **dragonite** (HP: 23%) [nostatus]

**Decision**: Action -1: Use **shadowball**


---

## 6. Battle 6

### Battle: `1520946-gen9ou-2431309839_1777_togepi73784_vs_multitype62894_08-29-2025_WIN.json.lz4`
- **Result**: WIN
- **Elo**: 1777
- **Total Turns**: 23

#### Player's Team
1. **dragapult** (HP: 100%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost *(Lead)*
2. **moltres** (HP: 100%) | Type: fire flying | Item: heavydutyboots | Ability: flamebody | Status: nostatus | Tera: fairy
3. **wochien** (HP: 100%) | Type: dark grass | Item: leftovers | Ability: tabletsofruin | Status: nostatus | Tera: fire
4. **greattusk** (HP: 100%) | Type: fighting ground | Item: rockyhelmet | Ability: protosynthesis | Status: nostatus | Tera: fire
5. **ogerponwellspring** (HP: 100%) | Type: grass water | Item: wellspringmask | Ability: waterabsorb | Status: nostatus | Tera: water
6. **kingambit** (HP: 100%) | Type: dark steel | Item: leftovers | Ability: supremeoverlord | Status: nostatus | Tera: flying

#### Opponent's Team (from Team Preview)
1. **ragingbolt**
2. **alomomola**
3. **slowking**
4. **greattusk**
5. **ogerpon**
6. **garganacl**

#### Turn-by-Turn Decisions

---
**Turn 1**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **dragapult** (HP: 100%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost
**Opponent Active**: **ragingbolt** (HP: 100%) | Type: dragon electric | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:16/16
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:16/16
  - Slot 3: **dragondance** [dragon] (status) PP:32/32 <-- CHOSEN
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **moltres** (HP: 100%) [nostatus]
  - Bench 2: **wochien** (HP: 100%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 5: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **dragondance**

---
**Turn 2**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used substitute | Opponent used terablast

**Player Active**: **dragapult** (HP: 82%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost
**Opponent Active**: **ragingbolt** (HP: 100%) | Type: ice notype | Ability: protosynthesis | Status: nostatus | Tera: ice

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:15/16
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:16/16
  - Slot 3: **dragondance** [dragon] (status) PP:32/32
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **moltres** (HP: 100%) [nostatus]
  - Bench 2: **wochien** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 5: **kingambit** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 2: Switch to **wochien**

---
**Turn 3**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used terablast

**Player Active**: **kingambit** (HP: 100%) | Type: dark steel | Item: leftovers | Ability: supremeoverlord | Status: nostatus | Tera: flying
**Opponent Active**: **ragingbolt** (HP: 100%) | Type: ice notype | Ability: protosynthesis | Status: nostatus | Tera: ice

**Available Moves:**
  - Slot 1: **ironhead** [steel] (physical) BP:80 PP:24/24 <-- CHOSEN
  - Slot 2: **suckerpunch** [dark] (physical) BP:70 Priority:1 PP:8/8
  - Slot 3: **kowtowcleave** [dark] (physical) BP:85 PP:16/16
  - Slot 4: **swordsdance** [normal] (status) PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **moltres** (HP: 100%) [nostatus]
  - Bench 2: **wochien** (HP: 100%) [nostatus]
  - Bench 3: **dragapult** (HP: 82%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus]
  - Bench 5: **ogerponwellspring** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **ironhead**

---
**Turn 4**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used ironhead | Opponent used nomove

**Player Active**: **kingambit** (HP: 100%) | Type: dark steel | Item: leftovers | Ability: supremeoverlord | Status: nostatus | Tera: flying
**Opponent Active**: **alomomola** (HP: 86%) | Type: notype water | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **ironhead** [steel] (physical) BP:80 PP:23/24
  - Slot 2: **suckerpunch** [dark] (physical) BP:70 Priority:1 PP:8/8
  - Slot 3: **kowtowcleave** [dark] (physical) BP:85 PP:16/16
  - Slot 4: **swordsdance** [normal] (status) PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **moltres** (HP: 100%) [nostatus]
  - Bench 2: **wochien** (HP: 100%) [nostatus]
  - Bench 3: **dragapult** (HP: 82%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 5: **ogerponwellspring** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 4: Switch to **greattusk**

---
**Turn 5**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used wish

**Player Active**: **ogerponwellspring** (HP: 100%) | Type: grass water | Item: wellspringmask | Ability: waterabsorb | Status: nostatus | Tera: water
**Opponent Active**: **alomomola** (HP: 86%) | Type: notype water | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **uturn** [bug] (physical) BP:70 PP:32/32
  - Slot 2: **ivycudgel** [grass] (physical) BP:100 PP:16/16
  - Slot 3: **trailblaze** [grass] (physical) BP:50 PP:32/32
  - Slot 4: **playrough** [fairy] (physical) BP:90 PP:16/16 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **moltres** (HP: 100%) [nostatus]
  - Bench 2: **wochien** (HP: 100%) [nostatus]
  - Bench 3: **dragapult** (HP: 82%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus]
  - Bench 5: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **playrough**

---
**Turn 6** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used uturn | Opponent used terablast

**Player Active**: **ogerponwellspring** (HP: 100%) | Type: grass water | Item: wellspringmask | Ability: waterabsorb | Status: nostatus | Tera: water
**Opponent Active**: **ragingbolt** (HP: 78%) | Type: dragon electric | Ability: protosynthesis | Status: nostatus | Tera: ice

**Available Switches:**
  - Bench 1: **moltres** (HP: 100%) [nostatus]
  - Bench 2: **wochien** (HP: 100%) [nostatus]
  - Bench 3: **dragapult** (HP: 82%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus]
  - Bench 5: **kingambit** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 5: Switch to **kingambit**

---
**Turn 7**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used terablast

**Player Active**: **wochien** (HP: 100%) | Type: dark grass | Item: leftovers | Ability: tabletsofruin | Status: nostatus | Tera: fire
**Opponent Active**: **ragingbolt** (HP: 100%) | Type: dragon electric | Ability: protosynthesis | Status: nostatus | Tera: ice

**Available Moves:**
  - Slot 1: **protect** [normal] (status) Priority:4 PP:16/16
  - Slot 2: **seedbomb** [grass] (physical) BP:80 PP:24/24 <-- CHOSEN
  - Slot 3: **stunspore** [grass] (status) PP:48/48
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **moltres** (HP: 100%) [nostatus]
  - Bench 2: **dragapult** (HP: 82%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 5: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **seedbomb**

---
**Turn 8**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used protect | Opponent used terablast

**Player Active**: **wochien** (HP: 100%) | Type: dark grass | Item: leftovers | Ability: tabletsofruin | Status: nostatus | Tera: fire
**Opponent Active**: **ragingbolt** (HP: 100%) | Type: dragon electric | Ability: protosynthesis | Status: nostatus | Tera: ice

**Available Moves:**
  - Slot 1: **protect** [normal] (status) Priority:4 PP:15/16
  - Slot 2: **seedbomb** [grass] (physical) BP:80 PP:24/24
  - Slot 3: **stunspore** [grass] (status) PP:48/48
  - Slot 4: **knockoff** [dark] (physical) BP:65 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **moltres** (HP: 100%) [nostatus]
  - Bench 2: **dragapult** (HP: 82%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **ogerponwellspring** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 5: **kingambit** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 4: Switch to **ogerponwellspring**

---
**Turn 9**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used terablast

**Player Active**: **moltres** (HP: 100%) | Type: fire flying | Item: heavydutyboots | Ability: flamebody | Status: nostatus | Tera: fairy
**Opponent Active**: **ragingbolt** (HP: 100%) | Type: dragon electric | Ability: protosynthesis | Status: nostatus | Tera: ice

**Available Moves:**
  - Slot 1: **flamethrower** [fire] (special) BP:90 PP:24/24 <-- CHOSEN
  - Slot 2: **willowisp** [fire] (status) PP:24/24
  - Slot 3: **roost** [flying] (status) PP:8/8
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **dragapult** (HP: 82%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 5: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **flamethrower**

---
**Turn 10** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used flamethrower | Opponent used terablast

**Player Active**: **moltres** (HP: 0%) | Type: fire flying | Item: heavydutyboots | Ability: flamebody | Status: fnt | Tera: fairy
**Opponent Active**: **ragingbolt** (HP: 42%) | Type: dragon electric | Ability: protosynthesis | Status: nostatus | Tera: ice

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **dragapult** (HP: 82%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 5: **kingambit** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **wochien**

---
**Turn 11**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used substitute | Opponent used terablast

**Player Active**: **dragapult** (HP: 82%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost
**Opponent Active**: **ragingbolt** (HP: 42%) | Type: dragon electric | Ability: protosynthesis | Status: nostatus | Tera: ice

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:15/16
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:16/16 <-- CHOSEN
  - Slot 3: **dragondance** [dragon] (status) PP:32/32
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **dragondarts**

---
**Turn 12**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used dragondarts | Opponent used nomove

**Player Active**: **dragapult** (HP: 88%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost
**Opponent Active**: **garganacl** (HP: 76%) | Type: notype rock | Item: leftovers | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:15/16
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:15/16
  - Slot 3: **dragondance** [dragon] (status) PP:32/32 <-- CHOSEN
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **dragondance**

---
**Turn 13**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used substitute | Opponent used saltcure

**Player Active**: **dragapult** (HP: 69%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost
**Opponent Active**: **garganacl** (HP: 82%) | Type: notype rock | Item: leftovers | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:14/16 <-- CHOSEN
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:15/16
  - Slot 3: **dragondance** [dragon] (status) PP:32/32
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **substitute**

---
**Turn 14**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used dragondance | Opponent used irondefense

**Player Active**: **dragapult** (HP: 75%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost | Boosts: ATK+1, SPE+1
**Opponent Active**: **garganacl** (HP: 88%) | Type: notype rock | Item: leftovers | Status: nostatus | Tera: notype | Boosts: DEF+2

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:14/16
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:15/16 <-- CHOSEN
  - Slot 3: **dragondance** [dragon] (status) PP:31/32
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **dragondarts**

---
**Turn 15**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used dragondarts | Opponent used irondefense

**Player Active**: **dragapult** (HP: 81%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost | Boosts: ATK+1, SPE+1
**Opponent Active**: **garganacl** (HP: 70%) | Type: notype rock | Item: leftovers | Status: nostatus | Tera: notype | Boosts: DEF+4

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:14/16 <-- CHOSEN
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:14/16
  - Slot 3: **dragondance** [dragon] (status) PP:31/32
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **substitute**

---
**Turn 16**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used dragondance | Opponent used irondefense

**Player Active**: **dragapult** (HP: 87%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost | Boosts: ATK+2, SPE+2
**Opponent Active**: **garganacl** (HP: 76%) | Type: notype rock | Item: leftovers | Status: nostatus | Tera: notype | Boosts: DEF+6

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:14/16 <-- CHOSEN
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:14/16
  - Slot 3: **dragondance** [dragon] (status) PP:30/32
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **substitute**

---
**Turn 17**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used dragondance | Opponent used saltcure

**Player Active**: **dragapult** (HP: 93%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost | Boosts: ATK+3, SPE+3
**Opponent Active**: **garganacl** (HP: 82%) | Type: notype rock | Item: leftovers | Status: nostatus | Tera: notype | Boosts: DEF+6

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:14/16
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:14/16
  - Slot 3: **dragondance** [dragon] (status) PP:29/32 <-- CHOSEN
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **dragondance**

---
**Turn 18**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used substitute | Opponent used saltcure

**Player Active**: **dragapult** (HP: 74%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost | Boosts: ATK+3, SPE+3
**Opponent Active**: **garganacl** (HP: 88%) | Type: notype rock | Item: leftovers | Status: nostatus | Tera: notype | Boosts: DEF+6

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:13/16 <-- CHOSEN
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:14/16
  - Slot 3: **dragondance** [dragon] (status) PP:29/32
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **substitute**

---
**Turn 19**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used dragondance | Opponent used saltcure

**Player Active**: **dragapult** (HP: 80%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost | Boosts: ATK+4, SPE+4
**Opponent Active**: **garganacl** (HP: 95%) | Type: notype rock | Item: leftovers | Status: nostatus | Tera: notype | Boosts: DEF+6

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:13/16
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:14/16
  - Slot 3: **dragondance** [dragon] (status) PP:28/32 <-- CHOSEN
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **dragondance**

---
**Turn 20**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used substitute | Opponent used saltcure

**Player Active**: **dragapult** (HP: 61%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost | Boosts: ATK+4, SPE+4
**Opponent Active**: **garganacl** (HP: 100%) | Type: notype rock | Item: leftovers | Status: nostatus | Tera: notype | Boosts: DEF+6

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:12/16 <-- CHOSEN
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:14/16
  - Slot 3: **dragondance** [dragon] (status) PP:28/32
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **substitute**

---
**Turn 21**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used dragondance | Opponent used saltcure

**Player Active**: **dragapult** (HP: 67%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost | Boosts: ATK+5, SPE+5
**Opponent Active**: **garganacl** (HP: 100%) | Type: notype rock | Item: leftovers | Status: nostatus | Tera: notype | Boosts: DEF+6

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:12/16
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:14/16
  - Slot 3: **dragondance** [dragon] (status) PP:27/32 <-- CHOSEN
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **dragondance**

---
**Turn 22**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used substitute | Opponent used saltcure

**Player Active**: **dragapult** (HP: 48%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost | Boosts: ATK+5, SPE+5
**Opponent Active**: **garganacl** (HP: 100%) | Type: notype rock | Item: leftovers | Status: nostatus | Tera: notype | Boosts: DEF+6

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:11/16
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:14/16 <-- CHOSEN
  - Slot 3: **dragondance** [dragon] (status) PP:27/32
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **dragondarts**

---
**Turn 23**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used dragondarts | Opponent used saltcure

**Player Active**: **dragapult** (HP: 54%) | Type: dragon ghost | Item: leftovers | Ability: infiltrator | Status: nostatus | Tera: ghost | Boosts: ATK+5, SPE+5
**Opponent Active**: **garganacl** (HP: 0%) | Type: notype rock | Item: leftovers | Status: fnt | Tera: notype | Boosts: DEF+6

**Available Moves:**
  - Slot 1: **substitute** [normal] (status) PP:11/16
  - Slot 2: **dragondarts** [dragon] (physical) BP:50 PP:13/16
  - Slot 3: **dragondance** [dragon] (status) PP:27/32
  - Slot 4: **uturn** [bug] (physical) BP:70 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **wochien** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ogerponwellspring** (HP: 100%) [nostatus]
  - Bench 4: **kingambit** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **uturn**


---

## 7. Battle 7

### Battle: `gen9ou-2154385089_1587_doubleslap89429_vs_crushgrip45765_07-03-2024_LOSS.json.lz4`
- **Result**: LOSS
- **Elo**: 1587
- **Total Turns**: 24

#### Player's Team
1. **heatran** (HP: 100%) | Type: fire steel | Item: airballoon | Ability: flashfire | Status: nostatus | Tera: flying *(Lead)*
2. **glimmora** (HP: 100%) | Type: poison rock | Item: focussash | Ability: toxicdebris | Status: nostatus | Tera: grass
3. **roaringmoon** (HP: 100%) | Type: dark dragon | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: flying
4. **kingambit** (HP: 100%) | Type: dark steel | Item: leftovers | Ability: supremeoverlord | Status: nostatus | Tera: ghost
5. **greattusk** (HP: 100%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ice
6. **ragingbolt** (HP: 100%) | Type: dragon electric | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: fairy

#### Opponent's Team (from Team Preview)
1. **corviknight**
2. **slowking**
3. **greattusk**
4. **lokix**
5. **dragapult**
6. **ragingbolt**

#### Turn-by-Turn Decisions

---
**Turn 1**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **heatran** (HP: 100%) | Type: fire steel | Item: airballoon | Ability: flashfire | Status: nostatus | Tera: flying
**Opponent Active**: **ragingbolt** (HP: 100%) | Type: dragon electric | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **magmastorm** [fire] (special) BP:100 PP:8/8
  - Slot 2: **stealthrock** [rock] (status) PP:32/32 <-- CHOSEN
  - Slot 3: **earthpower** [ground] (special) BP:90 PP:16/16
  - Slot 4: **willowisp** [fire] (status) PP:24/24
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **glimmora** (HP: 100%) [nostatus]
  - Bench 2: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 3: **kingambit** (HP: 100%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus]
  - Bench 5: **ragingbolt** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **stealthrock**

---
**Turn 2**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used magmastorm | Opponent used nomove

**Player Active**: **heatran** (HP: 42%) | Type: fire steel | Item: noitem | Ability: flashfire | Status: nostatus | Tera: flying
**Opponent Active**: **lokix** (HP: 100%) | Type: bug dark | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **magmastorm** [fire] (special) BP:100 PP:7/8
  - Slot 2: **stealthrock** [rock] (status) PP:32/32
  - Slot 3: **earthpower** [ground] (special) BP:90 PP:16/16
  - Slot 4: **willowisp** [fire] (status) PP:24/24
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **glimmora** (HP: 100%) [nostatus]
  - Bench 2: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 3: **kingambit** (HP: 100%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus]
  - Bench 5: **ragingbolt** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **willowisp**

---
**Turn 3** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used magmastorm | Opponent used knockoff

**Player Active**: **heatran** (HP: 0%) | Type: fire steel | Item: noitem | Ability: flashfire | Status: fnt | Tera: flying
**Opponent Active**: **lokix** (HP: 100%) | Type: bug dark | Status: nostatus | Tera: notype

**Available Switches:**
  - Bench 1: **glimmora** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 3: **kingambit** (HP: 100%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus]
  - Bench 5: **ragingbolt** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **glimmora**

---
**Turn 4**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used knockoff

**Player Active**: **glimmora** (HP: 100%) | Type: poison rock | Item: focussash | Ability: toxicdebris | Status: nostatus | Tera: grass
**Opponent Active**: **lokix** (HP: 100%) | Type: bug dark | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **energyball** [grass] (special) BP:90 PP:16/16 <-- CHOSEN
  - Slot 2: **mudshot** [ground] (special) BP:55 PP:24/24
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **mortalspin** [poison] (physical) BP:30 PP:24/24
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **kingambit** (HP: 100%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **ragingbolt** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **energyball**

---
**Turn 5**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: toxicspikes

*Last turn*: Player used energyball | Opponent used knockoff

**Player Active**: **glimmora** (HP: 1%) | Type: poison rock | Item: noitem | Ability: toxicdebris | Status: nostatus | Tera: grass
**Opponent Active**: **lokix** (HP: 68%) | Type: bug dark | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **energyball** [grass] (special) BP:90 PP:15/16
  - Slot 2: **mudshot** [ground] (special) BP:55 PP:24/24
  - Slot 3: **stealthrock** [rock] (status) PP:32/32 <-- CHOSEN
  - Slot 4: **mortalspin** [poison] (physical) BP:30 PP:24/24
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **kingambit** (HP: 100%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **ragingbolt** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **stealthrock**

---
**Turn 6**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used mudshot | Opponent used nomove

**Player Active**: **glimmora** (HP: 1%) | Type: poison rock | Item: noitem | Ability: toxicdebris | Status: nostatus | Tera: grass
**Opponent Active**: **slowkinggalar** (HP: 76%) | Type: poison psychic | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **energyball** [grass] (special) BP:90 PP:15/16
  - Slot 2: **mudshot** [ground] (special) BP:55 PP:23/24
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **mortalspin** [poison] (physical) BP:30 PP:24/24 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **kingambit** (HP: 100%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **ragingbolt** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **mortalspin**

---
**Turn 7** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used stealthrock | Opponent used icebeam

**Player Active**: **glimmora** (HP: 0%) | Type: poison rock | Item: noitem | Ability: toxicdebris | Status: fnt | Tera: grass
**Opponent Active**: **slowkinggalar** (HP: 76%) | Type: poison psychic | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **kingambit** (HP: 100%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **ragingbolt** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 4: Switch to **ragingbolt**

---
**Turn 8**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used nomove | Opponent used icebeam

**Player Active**: **roaringmoon** (HP: 100%) | Type: dark dragon | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: flying
**Opponent Active**: **slowkinggalar** (HP: 76%) | Type: poison psychic | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **dragondance** [dragon] (status) PP:32/32
  - Slot 2: **knockoff** [dark] (physical) BP:65 PP:32/32 <-- CHOSEN
  - Slot 3: **acrobatics** [flying] (physical) BP:55 PP:24/24
  - Slot 4: **outrage** [dragon] (physical) BP:120 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **kingambit** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ragingbolt** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **knockoff**

---
**Turn 9**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used dragondance | Opponent used nomove

**Player Active**: **roaringmoon** (HP: 100%) | Type: dark dragon | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: flying | Boosts: ATK+1, SPE+1
**Opponent Active**: **corviknight** (HP: 94%) | Type: flying steel | Item: leftovers | Ability: pressure | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **dragondance** [dragon] (status) PP:31/32
  - Slot 2: **knockoff** [dark] (physical) BP:65 PP:32/32 <-- CHOSEN (with Tera)
  - Slot 3: **acrobatics** [flying] (physical) BP:55 PP:24/24
  - Slot 4: **outrage** [dragon] (physical) BP:120 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **kingambit** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ragingbolt** (HP: 100%) [nostatus]

**Decision**: Tera + Move 2: Terastallize + Use **knockoff**

---
**Turn 10**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used dragondance | Opponent used irondefense

**Player Active**: **roaringmoon** (HP: 100%) | Type: flying notype | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: flying | Boosts: ATK+2, SPE+2
**Opponent Active**: **corviknight** (HP: 99%) | Type: flying steel | Item: leftovers | Ability: pressure | Status: nostatus | Tera: notype | Boosts: DEF+2

**Available Moves:**
  - Slot 1: **dragondance** [dragon] (status) PP:30/32
  - Slot 2: **knockoff** [dark] (physical) BP:65 PP:31/32
  - Slot 3: **acrobatics** [flying] (physical) BP:55 PP:24/24 <-- CHOSEN
  - Slot 4: **outrage** [dragon] (physical) BP:120 PP:16/16

**Available Switches:**
  - Bench 1: **kingambit** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ragingbolt** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **acrobatics**

---
**Turn 11**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used knockoff | Opponent used irondefense

**Player Active**: **roaringmoon** (HP: 100%) | Type: flying notype | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: flying | Boosts: ATK+2, SPE+2
**Opponent Active**: **corviknight** (HP: 59%) | Type: flying steel | Item: noitem | Ability: pressure | Status: nostatus | Tera: notype | Boosts: DEF+4

**Available Moves:**
  - Slot 1: **dragondance** [dragon] (status) PP:30/32
  - Slot 2: **knockoff** [dark] (physical) BP:65 PP:30/32
  - Slot 3: **acrobatics** [flying] (physical) BP:55 PP:24/24
  - Slot 4: **outrage** [dragon] (physical) BP:120 PP:16/16

**Available Switches:**
  - Bench 1: **kingambit** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **ragingbolt** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 3: Switch to **ragingbolt**

---
**Turn 12**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used nomove | Opponent used irondefense

**Player Active**: **ragingbolt** (HP: 100%) | Type: dragon electric | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: fairy
**Opponent Active**: **corviknight** (HP: 59%) | Type: flying steel | Item: noitem | Ability: pressure | Status: nostatus | Tera: notype | Boosts: DEF+6

**Available Moves:**
  - Slot 1: **calmmind** [psychic] (status) PP:32/32 <-- CHOSEN
  - Slot 2: **dracometeor** [dragon] (special) BP:130 PP:8/8
  - Slot 3: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 4: **thunderclap** [electric] (special) BP:70 Priority:1 PP:8/8

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **kingambit** (HP: 100%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **calmmind**

---
**Turn 13**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used calmmind | Opponent used icebeam

**Player Active**: **ragingbolt** (HP: 100%) | Type: dragon electric | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: fairy | Boosts: SPA+1, SPD+1
**Opponent Active**: **slowkinggalar** (HP: 100%) | Type: poison psychic | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **calmmind** [psychic] (status) PP:31/32 <-- CHOSEN
  - Slot 2: **dracometeor** [dragon] (special) BP:130 PP:8/8
  - Slot 3: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 4: **thunderclap** [electric] (special) BP:70 Priority:1 PP:8/8

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **kingambit** (HP: 100%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **calmmind**

---
**Turn 14**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used calmmind | Opponent used toxic

**Player Active**: **ragingbolt** (HP: 94%) | Type: dragon electric | Item: noitem | Ability: protosynthesis | Status: tox | Tera: fairy | Boosts: SPA+2, SPD+2
**Opponent Active**: **slowkinggalar** (HP: 100%) | Type: poison psychic | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **calmmind** [psychic] (status) PP:30/32
  - Slot 2: **dracometeor** [dragon] (special) BP:130 PP:8/8 <-- CHOSEN
  - Slot 3: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 4: **thunderclap** [electric] (special) BP:70 Priority:1 PP:8/8

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **kingambit** (HP: 100%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **dracometeor**

---
**Turn 15**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used dracometeor | Opponent used nomove

**Player Active**: **ragingbolt** (HP: 82%) | Type: dragon electric | Item: noitem | Ability: protosynthesis | Status: tox | Tera: fairy | Boosts: SPD+2
**Opponent Active**: **greattusk** (HP: 100%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **calmmind** [psychic] (status) PP:30/32
  - Slot 2: **dracometeor** [dragon] (special) BP:130 PP:7/8
  - Slot 3: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 4: **thunderclap** [electric] (special) BP:70 Priority:1 PP:8/8

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **kingambit** (HP: 100%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **thunderclap**

---
**Turn 16** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used dracometeor | Opponent used headlongrush

**Player Active**: **ragingbolt** (HP: 0%) | Type: dragon electric | Item: noitem | Ability: protosynthesis | Status: fnt | Tera: fairy | Boosts: SPD+2
**Opponent Active**: **greattusk** (HP: 100%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: DEF-1, SPD-1

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **kingambit** (HP: 100%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 3: Switch to **greattusk**

---
**Turn 17**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: stealthrock

*Last turn*: Player used knockoff | Opponent used headlongrush

**Player Active**: **roaringmoon** (HP: 100%) | Type: dark dragon | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: flying
**Opponent Active**: **greattusk** (HP: 100%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: DEF-1, SPD-1

**Available Moves:**
  - Slot 1: **dragondance** [dragon] (status) PP:30/32
  - Slot 2: **knockoff** [dark] (physical) BP:65 PP:30/32 <-- CHOSEN
  - Slot 3: **acrobatics** [flying] (physical) BP:55 PP:24/24
  - Slot 4: **outrage** [dragon] (physical) BP:120 PP:16/16

**Available Switches:**
  - Bench 1: **kingambit** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **knockoff**

---
**Turn 18**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used dragondance | Opponent used rapidspin

**Player Active**: **roaringmoon** (HP: 77%) | Type: dark dragon | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: flying | Boosts: ATK+1, SPE+1
**Opponent Active**: **greattusk** (HP: 100%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: DEF-1, SPD-1, SPE+1

**Available Moves:**
  - Slot 1: **dragondance** [dragon] (status) PP:29/32
  - Slot 2: **knockoff** [dark] (physical) BP:65 PP:30/32 <-- CHOSEN
  - Slot 3: **acrobatics** [flying] (physical) BP:55 PP:24/24
  - Slot 4: **outrage** [dragon] (physical) BP:120 PP:16/16

**Available Switches:**
  - Bench 1: **kingambit** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **knockoff**

---
**Turn 19**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used dragondance | Opponent used voltswitch

**Player Active**: **roaringmoon** (HP: 77%) | Type: dark dragon | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: flying | Boosts: ATK+2, SPE+2
**Opponent Active**: **ragingbolt** (HP: 77%) | Type: dragon electric | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **dragondance** [dragon] (status) PP:28/32
  - Slot 2: **knockoff** [dark] (physical) BP:65 PP:30/32
  - Slot 3: **acrobatics** [flying] (physical) BP:55 PP:24/24
  - Slot 4: **outrage** [dragon] (physical) BP:120 PP:16/16

**Available Switches:**
  - Bench 1: **kingambit** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **greattusk** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **kingambit**

---
**Turn 20**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used thunderbolt

**Player Active**: **greattusk** (HP: 100%) | Type: fighting ground | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: ice
**Opponent Active**: **ragingbolt** (HP: 77%) | Type: dragon electric | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **bulkup** [fighting] (status) PP:32/32 <-- CHOSEN
  - Slot 2: **icespinner** [ice] (physical) BP:80 PP:24/24
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:8/8

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 77%) [nostatus]
  - Bench 2: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **bulkup**

---
**Turn 21**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used bulkup | Opponent used irondefense

**Player Active**: **greattusk** (HP: 100%) | Type: fighting ground | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: ice | Boosts: ATK+1, DEF+1
**Opponent Active**: **corviknight** (HP: 59%) | Type: flying steel | Item: noitem | Ability: pressure | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **bulkup** [fighting] (status) PP:31/32
  - Slot 2: **icespinner** [ice] (physical) BP:80 PP:23/24
  - Slot 3: **stealthrock** [rock] (status) PP:32/32 <-- CHOSEN
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:8/8

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 77%) [nostatus]
  - Bench 2: **kingambit** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **stealthrock**

---
**Turn 22**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used icespinner | Opponent used roost

**Player Active**: **greattusk** (HP: 100%) | Type: fighting ground | Item: noitem | Ability: protosynthesis | Status: nostatus | Tera: ice | Boosts: ATK+1, DEF+1
**Opponent Active**: **corviknight** (HP: 87%) | Type: flying steel | Item: noitem | Ability: pressure | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **bulkup** [fighting] (status) PP:31/32
  - Slot 2: **icespinner** [ice] (physical) BP:80 PP:22/24
  - Slot 3: **stealthrock** [rock] (status) PP:32/32
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:8/8

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 77%) [nostatus] <-- CHOSEN
  - Bench 2: **kingambit** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **roaringmoon**

---
**Turn 23**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used irondefense

**Player Active**: **kingambit** (HP: 100%) | Type: dark steel | Item: leftovers | Ability: supremeoverlord | Status: nostatus | Tera: ghost
**Opponent Active**: **corviknight** (HP: 87%) | Type: flying steel | Item: noitem | Ability: pressure | Status: nostatus | Tera: notype | Boosts: DEF+2

**Available Moves:**
  - Slot 1: **ironhead** [steel] (physical) BP:80 PP:24/24
  - Slot 2: **kowtowcleave** [dark] (physical) BP:85 PP:16/16
  - Slot 3: **swordsdance** [normal] (status) PP:32/32
  - Slot 4: **suckerpunch** [dark] (physical) BP:70 Priority:1 PP:8/8

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 77%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **suckerpunch**

---
**Turn 24**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used irondefense

**Player Active**: **kingambit** (HP: 100%) | Type: dark steel | Item: leftovers | Ability: supremeoverlord | Status: nostatus | Tera: ghost
**Opponent Active**: **corviknight** (HP: 87%) | Type: flying steel | Item: noitem | Ability: pressure | Status: nostatus | Tera: notype | Boosts: DEF+2

**Available Moves:**
  - Slot 1: **ironhead** [steel] (physical) BP:80 PP:24/24
  - Slot 2: **kowtowcleave** [dark] (physical) BP:85 PP:16/16
  - Slot 3: **swordsdance** [normal] (status) PP:32/32
  - Slot 4: **suckerpunch** [dark] (physical) BP:70 Priority:1 PP:8/8

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 77%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **suckerpunch**


---

## 8. Battle 8

### Battle: `gen9ou-2233095423_Unrated_furyattack32056_vs_filter66953_10-28-2024_WIN.json.lz4`
- **Result**: WIN
- **Elo**: Unrated
- **Total Turns**: 25

#### Player's Team
1. **grimmsnarl** (HP: 100%) | Type: dark fairy | Item: lightclay | Ability: prankster | Status: nostatus | Tera: ghost *(Lead)*
2. **roaringmoon** (HP: 100%) | Type: dark dragon | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground
3. **greattusk** (HP: 100%) | Type: fighting ground | Item: rockyhelmet | Ability: protosynthesis | Status: nostatus | Tera: steel
4. **skeledirge** (HP: 100%) | Type: fire ghost | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy
5. **dragapult** (HP: 100%) | Type: dragon ghost | Item: choiceband | Ability: infiltrator | Status: nostatus | Tera: ghost
6. **alomomola** (HP: 100%) | Type: notype water | Item: heavydutyboots | Ability: regenerator | Status: nostatus | Tera: fairy

#### Opponent's Team (from Team Preview)
1. **dragapult**
2. **ceruledge**
3. **rillaboom**
4. **basculegion**
5. **gholdengo**
6. **hydreigon**

#### Turn-by-Turn Decisions

---
**Turn 1**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **grimmsnarl** (HP: 100%) | Type: dark fairy | Item: lightclay | Ability: prankster | Status: nostatus | Tera: ghost
**Opponent Active**: **gholdengo** (HP: 100%) | Type: ghost steel | Ability: goodasgold | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **lightscreen** [psychic] (status) PP:48/48 <-- CHOSEN
  - Slot 2: **reflect** [psychic] (status) PP:32/32
  - Slot 3: **thunderwave** [electric] (status) PP:32/32
  - Slot 4: **partingshot** [dark] (status) PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **skeledirge** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **lightscreen**

---
**Turn 2**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used lightscreen | Opponent used flashcannon

**Player Active**: **grimmsnarl** (HP: 54%) | Type: dark fairy | Item: lightclay | Ability: prankster | Status: nostatus | Tera: ghost
**Opponent Active**: **gholdengo** (HP: 100%) | Type: ghost steel | Ability: goodasgold | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **lightscreen** [psychic] (status) PP:47/48
  - Slot 2: **reflect** [psychic] (status) PP:32/32
  - Slot 3: **thunderwave** [electric] (status) PP:32/32
  - Slot 4: **partingshot** [dark] (status) PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **skeledirge** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 5: Switch to **alomomola**

---
**Turn 3**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nastyplot

**Player Active**: **skeledirge** (HP: 100%) | Type: fire ghost | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy
**Opponent Active**: **gholdengo** (HP: 100%) | Type: ghost steel | Ability: goodasgold | Status: nostatus | Tera: notype | Boosts: SPA+2

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:16/16
  - Slot 2: **willowisp** [fire] (status) PP:24/24
  - Slot 3: **slackoff** [normal] (status) PP:8/8 <-- CHOSEN
  - Slot 4: **hex** [ghost] (special) BP:65 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 54%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **slackoff**

---
**Turn 4**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used torchsong | Opponent used nomove

**Player Active**: **skeledirge** (HP: 75%) | Type: fire ghost | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+1
**Opponent Active**: **basculegion** (HP: 100%) | Type: ghost water | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:15/16
  - Slot 2: **willowisp** [fire] (status) PP:24/24
  - Slot 3: **slackoff** [normal] (status) PP:8/8
  - Slot 4: **hex** [ghost] (special) BP:65 PP:16/16 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 54%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **hex**

---
**Turn 5**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used willowisp | Opponent used phantomforce

**Player Active**: **skeledirge** (HP: 81%) | Type: fire ghost | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+1
**Opponent Active**: **basculegion** (HP: 100%) | Type: notype water | Status: nostatus | Tera: water

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:15/16
  - Slot 2: **willowisp** [fire] (status) PP:23/24
  - Slot 3: **slackoff** [normal] (status) PP:8/8
  - Slot 4: **hex** [ghost] (special) BP:65 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 54%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 5: **alomomola** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 4: Switch to **dragapult**

---
**Turn 6**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used lightscreen | Opponent used phantomforce

**Player Active**: **grimmsnarl** (HP: 33%) | Type: dark fairy | Item: lightclay | Ability: prankster | Status: nostatus | Tera: ghost
**Opponent Active**: **basculegion** (HP: 100%) | Type: notype water | Status: nostatus | Tera: water

**Available Moves:**
  - Slot 1: **lightscreen** [psychic] (status) PP:47/48
  - Slot 2: **reflect** [psychic] (status) PP:32/32
  - Slot 3: **thunderwave** [electric] (status) PP:32/32 <-- CHOSEN
  - Slot 4: **partingshot** [dark] (status) PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **skeledirge** (HP: 81%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **thunderwave**

---
**Turn 7**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: reflect | Opponent side: noconditions

*Last turn*: Player used reflect | Opponent used phantomforce

**Player Active**: **grimmsnarl** (HP: 33%) | Type: dark fairy | Item: lightclay | Ability: prankster | Status: nostatus | Tera: ghost
**Opponent Active**: **basculegion** (HP: 100%) | Type: notype water | Status: nostatus | Tera: water

**Available Moves:**
  - Slot 1: **lightscreen** [psychic] (status) PP:47/48
  - Slot 2: **reflect** [psychic] (status) PP:31/32
  - Slot 3: **thunderwave** [electric] (status) PP:32/32
  - Slot 4: **partingshot** [dark] (status) PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **skeledirge** (HP: 81%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **roaringmoon**

---
**Turn 8**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: reflect | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used phantomforce

**Player Active**: **alomomola** (HP: 100%) | Type: notype water | Item: heavydutyboots | Ability: regenerator | Status: nostatus | Tera: fairy
**Opponent Active**: **basculegion** (HP: 100%) | Type: notype water | Status: nostatus | Tera: water

**Available Moves:**
  - Slot 1: **wish** [normal] (status) PP:16/16
  - Slot 2: **mirrorcoat** [psychic] (special) Priority:-5 PP:32/32
  - Slot 3: **playrough** [fairy] (physical) BP:90 PP:16/16
  - Slot 4: **protect** [normal] (status) Priority:4 PP:16/16 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 33%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **skeledirge** (HP: 81%) [nostatus]
  - Bench 5: **dragapult** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **protect**

---
**Turn 9**

*Field*: Weather: noweather | Terrain/Field: grassyterrain | Player side: reflect | Opponent side: noconditions

*Last turn*: Player used wish | Opponent used nomove

**Player Active**: **alomomola** (HP: 88%) | Type: notype water | Item: heavydutyboots | Ability: regenerator | Status: nostatus | Tera: fairy
**Opponent Active**: **rillaboom** (HP: 100%) | Type: grass notype | Ability: grassysurge | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **wish** [normal] (status) PP:15/16
  - Slot 2: **mirrorcoat** [psychic] (special) Priority:-5 PP:32/32
  - Slot 3: **playrough** [fairy] (physical) BP:90 PP:16/16
  - Slot 4: **protect** [normal] (status) Priority:4 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 33%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 4: **skeledirge** (HP: 81%) [nostatus]
  - Bench 5: **dragapult** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 3: Switch to **greattusk**

---
**Turn 10**

*Field*: Weather: noweather | Terrain/Field: grassyterrain | Player side: reflect | Opponent side: noconditions

*Last turn*: Player used reflect | Opponent used fakeout

**Player Active**: **grimmsnarl** (HP: 99%) | Type: dark fairy | Item: lightclay | Ability: prankster | Status: nostatus | Tera: ghost
**Opponent Active**: **rillaboom** (HP: 100%) | Type: grass notype | Ability: grassysurge | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **lightscreen** [psychic] (status) PP:47/48 <-- CHOSEN
  - Slot 2: **reflect** [psychic] (status) PP:31/32
  - Slot 3: **thunderwave** [electric] (status) PP:32/32
  - Slot 4: **partingshot** [dark] (status) PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **skeledirge** (HP: 81%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Move 1: Use **lightscreen**

---
**Turn 11**

*Field*: Weather: noweather | Terrain/Field: grassyterrain | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used lightscreen | Opponent used woodhammer

**Player Active**: **grimmsnarl** (HP: 68%) | Type: dark fairy | Item: lightclay | Ability: prankster | Status: nostatus | Tera: ghost
**Opponent Active**: **rillaboom** (HP: 92%) | Type: grass notype | Ability: grassysurge | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **lightscreen** [psychic] (status) PP:46/48
  - Slot 2: **reflect** [psychic] (status) PP:31/32
  - Slot 3: **thunderwave** [electric] (status) PP:32/32
  - Slot 4: **partingshot** [dark] (status) PP:32/32 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **skeledirge** (HP: 81%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Move 4: Use **partingshot**

---
**Turn 12**

*Field*: Weather: noweather | Terrain/Field: grassyterrain | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used thunderwave | Opponent used fakeout

**Player Active**: **grimmsnarl** (HP: 74%) | Type: dark fairy | Item: lightclay | Ability: prankster | Status: nostatus | Tera: ghost
**Opponent Active**: **rillaboom** (HP: 98%) | Type: grass notype | Ability: grassysurge | Status: par | Tera: notype

**Available Moves:**
  - Slot 1: **lightscreen** [psychic] (status) PP:46/48
  - Slot 2: **reflect** [psychic] (status) PP:31/32 <-- CHOSEN
  - Slot 3: **thunderwave** [electric] (status) PP:31/32
  - Slot 4: **partingshot** [dark] (status) PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **skeledirge** (HP: 81%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Move 2: Use **reflect**

---
**Turn 13** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: grassyterrain | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used partingshot | Opponent used fakeout

**Player Active**: **grimmsnarl** (HP: 74%) | Type: dark fairy | Item: lightclay | Ability: prankster | Status: nostatus | Tera: ghost
**Opponent Active**: **rillaboom** (HP: 98%) | Type: grass notype | Ability: grassysurge | Status: par | Tera: notype | Boosts: ATK-1, SPA-1

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **greattusk** (HP: 100%) [nostatus]
  - Bench 3: **skeledirge** (HP: 81%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 5: Switch to **alomomola**

---
**Turn 14**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used willowisp | Opponent used woodhammer

**Player Active**: **skeledirge** (HP: 77%) | Type: fire ghost | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy
**Opponent Active**: **rillaboom** (HP: 98%) | Type: grass notype | Ability: grassysurge | Status: par | Tera: notype | Boosts: ATK-1, SPA-1

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:15/16
  - Slot 2: **willowisp** [fire] (status) PP:23/24
  - Slot 3: **slackoff** [normal] (status) PP:8/8 <-- CHOSEN
  - Slot 4: **hex** [ghost] (special) BP:65 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 74%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Move 3: Use **slackoff**

---
**Turn 15**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used torchsong | Opponent used nomove

**Player Active**: **skeledirge** (HP: 83%) | Type: fire ghost | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+1
**Opponent Active**: **dragapult** (HP: 81%) | Type: dragon ghost | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:14/16
  - Slot 2: **willowisp** [fire] (status) PP:23/24
  - Slot 3: **slackoff** [normal] (status) PP:8/8
  - Slot 4: **hex** [ghost] (special) BP:65 PP:16/16 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 74%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Move 4: Use **hex**

---
**Turn 16**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used willowisp | Opponent used nomove

**Player Active**: **skeledirge** (HP: 84%) | Type: fire ghost | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+1
**Opponent Active**: **hydreigon** (HP: 100%) | Type: dark dragon | Ability: levitate | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:14/16
  - Slot 2: **willowisp** [fire] (status) PP:22/24
  - Slot 3: **slackoff** [normal] (status) PP:8/8 <-- CHOSEN
  - Slot 4: **hex** [ghost] (special) BP:65 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 74%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Move 3: Use **slackoff**

---
**Turn 17**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used torchsong | Opponent used darkpulse

**Player Active**: **skeledirge** (HP: 41%) | Type: fire ghost | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+2
**Opponent Active**: **hydreigon** (HP: 80%) | Type: dark dragon | Ability: levitate | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:13/16
  - Slot 2: **willowisp** [fire] (status) PP:22/24 <-- CHOSEN (with Tera)
  - Slot 3: **slackoff** [normal] (status) PP:8/8
  - Slot 4: **hex** [ghost] (special) BP:65 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 74%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Tera + Move 2: Terastallize + Use **willowisp**

---
**Turn 18**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: lightscreen | Opponent side: noconditions

*Last turn*: Player used slackoff | Opponent used darkpulse

**Player Active**: **skeledirge** (HP: 84%) | Type: fairy notype | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+2
**Opponent Active**: **hydreigon** (HP: 80%) | Type: dark dragon | Ability: levitate | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:13/16
  - Slot 2: **willowisp** [fire] (status) PP:22/24
  - Slot 3: **slackoff** [normal] (status) PP:7/8 <-- CHOSEN
  - Slot 4: **hex** [ghost] (special) BP:65 PP:16/16

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 74%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Move 3: Use **slackoff**

---
**Turn 19**

*Field*: Weather: noweather | Terrain/Field: grassyterrain | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used torchsong | Opponent used phantomforce

**Player Active**: **skeledirge** (HP: 96%) | Type: fairy notype | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+3
**Opponent Active**: **basculegion** (HP: 100%) | Type: ghost water | Status: nostatus | Tera: water

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:12/16
  - Slot 2: **willowisp** [fire] (status) PP:22/24
  - Slot 3: **slackoff** [normal] (status) PP:7/8
  - Slot 4: **hex** [ghost] (special) BP:65 PP:16/16 <-- CHOSEN

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 74%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Move 4: Use **hex**

---
**Turn 20**

*Field*: Weather: noweather | Terrain/Field: grassyterrain | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used willowisp | Opponent used wavecrash

**Player Active**: **skeledirge** (HP: 44%) | Type: fairy notype | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+3
**Opponent Active**: **basculegion** (HP: 77%) | Type: ghost water | Status: brn | Tera: water

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:12/16
  - Slot 2: **willowisp** [fire] (status) PP:21/24 <-- CHOSEN
  - Slot 3: **slackoff** [normal] (status) PP:7/8
  - Slot 4: **hex** [ghost] (special) BP:65 PP:16/16

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 74%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Move 2: Use **willowisp**

---
**Turn 21**

*Field*: Weather: noweather | Terrain/Field: grassyterrain | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used slackoff | Opponent used wavecrash

**Player Active**: **skeledirge** (HP: 74%) | Type: fairy notype | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+3
**Opponent Active**: **basculegion** (HP: 66%) | Type: ghost water | Status: brn | Tera: water

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:12/16
  - Slot 2: **willowisp** [fire] (status) PP:21/24 <-- CHOSEN
  - Slot 3: **slackoff** [normal] (status) PP:6/8
  - Slot 4: **hex** [ghost] (special) BP:65 PP:16/16

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 74%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Move 2: Use **willowisp**

---
**Turn 22**

*Field*: Weather: noweather | Terrain/Field: grassyterrain | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used slackoff | Opponent used wavecrash

**Player Active**: **skeledirge** (HP: 100%) | Type: fairy notype | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+3
**Opponent Active**: **basculegion** (HP: 55%) | Type: ghost water | Status: brn | Tera: water

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:12/16 <-- CHOSEN
  - Slot 2: **willowisp** [fire] (status) PP:21/24
  - Slot 3: **slackoff** [normal] (status) PP:5/8
  - Slot 4: **hex** [ghost] (special) BP:65 PP:16/16

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 74%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Move 1: Use **torchsong**

---
**Turn 23**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used hex | Opponent used nomove

**Player Active**: **skeledirge** (HP: 100%) | Type: fairy notype | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+3
**Opponent Active**: **ceruledge** (HP: 100%) | Type: fire ghost | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:12/16
  - Slot 2: **willowisp** [fire] (status) PP:21/24
  - Slot 3: **slackoff** [normal] (status) PP:5/8 <-- CHOSEN
  - Slot 4: **hex** [ghost] (special) BP:65 PP:15/16

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 74%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Move 3: Use **slackoff**

---
**Turn 24**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used torchsong | Opponent used bitterblade

**Player Active**: **skeledirge** (HP: 74%) | Type: fairy notype | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+4
**Opponent Active**: **ceruledge** (HP: 56%) | Type: fire ghost | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:11/16
  - Slot 2: **willowisp** [fire] (status) PP:21/24
  - Slot 3: **slackoff** [normal] (status) PP:5/8
  - Slot 4: **hex** [ghost] (special) BP:65 PP:15/16

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 74%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Action -1: Use **hex**

---
**Turn 25**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used torchsong | Opponent used bitterblade

**Player Active**: **skeledirge** (HP: 74%) | Type: fairy notype | Item: leftovers | Ability: unaware | Status: nostatus | Tera: fairy | Boosts: SPA+4
**Opponent Active**: **ceruledge** (HP: 56%) | Type: fire ghost | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **torchsong** [fire] (special) BP:80 PP:11/16
  - Slot 2: **willowisp** [fire] (status) PP:21/24
  - Slot 3: **slackoff** [normal] (status) PP:5/8
  - Slot 4: **hex** [ghost] (special) BP:65 PP:15/16

**Available Switches:**
  - Bench 1: **roaringmoon** (HP: 100%) [nostatus]
  - Bench 2: **grimmsnarl** (HP: 74%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **dragapult** (HP: 100%) [nostatus]
  - Bench 5: **alomomola** (HP: 88%) [nostatus]

**Decision**: Action -1: Use **hex**


---

## 9. Battle 9

### Battle: `1471813-gen9ou-2419843548_1735_delibird29318_vs_absorb67344_08-11-2025_WIN.json.lz4`
- **Result**: WIN
- **Elo**: 1735
- **Total Turns**: 27

#### Player's Team
1. **darkrai** (HP: 100%) | Type: dark notype | Item: leftovers | Ability: baddreams | Status: nostatus | Tera: poison *(Lead)*
2. **dragonite** (HP: 100%) | Type: dragon flying | Item: heavydutyboots | Ability: multiscale | Status: nostatus | Tera: flying
3. **enamorus** (HP: 100%) | Type: fairy flying | Item: choicespecs | Ability: contrary | Status: nostatus | Tera: stellar
4. **gholdengo** (HP: 100%) | Type: ghost steel | Item: airballoon | Ability: goodasgold | Status: nostatus | Tera: fairy
5. **greattusk** (HP: 100%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground
6. **tinglu** (HP: 100%) | Type: dark ground | Item: leftovers | Ability: vesselofruin | Status: nostatus | Tera: poison

#### Opponent's Team (from Team Preview)
1. **jirachi**
2. **greattusk**
3. **latios**
4. **gardevoir**
5. **polteageist**
6. **crawdaunt**

#### Turn-by-Turn Decisions

---
**Turn 1**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **darkrai** (HP: 100%) | Type: dark notype | Item: leftovers | Ability: baddreams | Status: nostatus | Tera: poison
**Opponent Active**: **jirachi** (HP: 100%) | Type: psychic steel | Ability: serenegrace | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **willowisp** [fire] (status) PP:24/24
  - Slot 2: **focusblast** [fighting] (special) BP:120 PP:8/8
  - Slot 3: **icebeam** [ice] (special) BP:90 PP:16/16
  - Slot 4: **darkpulse** [dark] (special) BP:80 PP:24/24 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus]
  - Bench 5: **tinglu** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **darkpulse**

---
**Turn 2**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used willowisp | Opponent used thunderwave

**Player Active**: **darkrai** (HP: 100%) | Type: dark notype | Item: leftovers | Ability: baddreams | Status: par | Tera: poison
**Opponent Active**: **jirachi** (HP: 94%) | Type: psychic steel | Ability: serenegrace | Status: brn | Tera: notype

**Available Moves:**
  - Slot 1: **willowisp** [fire] (status) PP:23/24
  - Slot 2: **focusblast** [fighting] (special) BP:120 PP:8/8
  - Slot 3: **icebeam** [ice] (special) BP:90 PP:16/16
  - Slot 4: **darkpulse** [dark] (special) BP:80 PP:24/24 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus]
  - Bench 5: **tinglu** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **darkpulse**

---
**Turn 3**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used willowisp | Opponent used stealthrock

**Player Active**: **darkrai** (HP: 100%) | Type: dark notype | Item: leftovers | Ability: baddreams | Status: par | Tera: poison
**Opponent Active**: **jirachi** (HP: 88%) | Type: psychic steel | Ability: serenegrace | Status: brn | Tera: notype

**Available Moves:**
  - Slot 1: **willowisp** [fire] (status) PP:22/24
  - Slot 2: **focusblast** [fighting] (special) BP:120 PP:8/8
  - Slot 3: **icebeam** [ice] (special) BP:90 PP:16/16
  - Slot 4: **darkpulse** [dark] (special) BP:80 PP:24/24
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus]
  - Bench 5: **tinglu** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **darkpulse**

---
**Turn 4**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used willowisp | Opponent used ironhead

**Player Active**: **darkrai** (HP: 82%) | Type: dark notype | Item: leftovers | Ability: baddreams | Status: par | Tera: poison
**Opponent Active**: **jirachi** (HP: 82%) | Type: psychic steel | Ability: serenegrace | Status: brn | Tera: notype

**Available Moves:**
  - Slot 1: **willowisp** [fire] (status) PP:22/24
  - Slot 2: **focusblast** [fighting] (special) BP:120 PP:8/8
  - Slot 3: **icebeam** [ice] (special) BP:90 PP:16/16
  - Slot 4: **darkpulse** [dark] (special) BP:80 PP:24/24
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 5: **tinglu** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 4: Switch to **greattusk**

---
**Turn 5**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used ironhead

**Player Active**: **greattusk** (HP: 100%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground
**Opponent Active**: **jirachi** (HP: 76%) | Type: psychic steel | Ability: serenegrace | Status: brn | Tera: notype

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:32/32
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:64/64
  - Slot 3: **icespinner** [ice] (physical) BP:80 PP:24/24 <-- CHOSEN
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **darkrai** (HP: 82%) [par]
  - Bench 2: **dragonite** (HP: 100%) [nostatus]
  - Bench 3: **enamorus** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **tinglu** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **icespinner**

---
**Turn 6**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used knockoff | Opponent used nomove

**Player Active**: **greattusk** (HP: 100%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground
**Opponent Active**: **polteageist** (HP: 1%) | Type: ghost notype | Item: noitem | Ability: weakarmor | Status: nostatus | Tera: notype | Boosts: DEF-1, SPE+2

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:31/32
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:64/64
  - Slot 3: **icespinner** [ice] (physical) BP:80 PP:24/24
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **darkrai** (HP: 82%) [par]
  - Bench 2: **dragonite** (HP: 100%) [nostatus]
  - Bench 3: **enamorus** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **tinglu** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 5: Switch to **tinglu**

---
**Turn 7**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used gigadrain

**Player Active**: **tinglu** (HP: 100%) | Type: dark ground | Item: leftovers | Ability: vesselofruin | Status: nostatus | Tera: poison
**Opponent Active**: **polteageist** (HP: 25%) | Type: ghost notype | Item: noitem | Ability: weakarmor | Status: nostatus | Tera: notype | Boosts: DEF-1, SPE+2

**Available Moves:**
  - Slot 1: **stealthrock** [rock] (status) PP:32/32
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:16/16
  - Slot 3: **ruination** [dark] (special) PP:16/16 <-- CHOSEN
  - Slot 4: **whirlwind** [normal] (status) Priority:-6 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **darkrai** (HP: 82%) [par]
  - Bench 2: **dragonite** (HP: 100%) [nostatus]
  - Bench 3: **enamorus** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **greattusk** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **ruination**

---
**Turn 8**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stealthrock

*Last turn*: Player used stealthrock | Opponent used gigadrain

**Player Active**: **tinglu** (HP: 60%) | Type: dark ground | Item: leftovers | Ability: vesselofruin | Status: nostatus | Tera: poison
**Opponent Active**: **polteageist** (HP: 47%) | Type: ghost notype | Item: noitem | Ability: weakarmor | Status: nostatus | Tera: notype | Boosts: DEF-1, SPE+2

**Available Moves:**
  - Slot 1: **stealthrock** [rock] (status) PP:31/32 <-- CHOSEN
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:16/16
  - Slot 3: **ruination** [dark] (special) PP:16/16
  - Slot 4: **whirlwind** [normal] (status) Priority:-6 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **darkrai** (HP: 82%) [par]
  - Bench 2: **dragonite** (HP: 100%) [nostatus]
  - Bench 3: **enamorus** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **greattusk** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **stealthrock**

---
**Turn 9**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stealthrock

*Last turn*: Player used earthquake | Opponent used nomove

**Player Active**: **tinglu** (HP: 42%) | Type: dark ground | Item: leftovers | Ability: vesselofruin | Status: nostatus | Tera: poison
**Opponent Active**: **crawdaunt** (HP: 88%) | Type: dark water | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **stealthrock** [rock] (status) PP:31/32
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:15/16
  - Slot 3: **ruination** [dark] (special) PP:16/16
  - Slot 4: **whirlwind** [normal] (status) Priority:-6 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **darkrai** (HP: 82%) [par] <-- CHOSEN
  - Bench 2: **dragonite** (HP: 100%) [nostatus]
  - Bench 3: **enamorus** (HP: 100%) [nostatus]
  - Bench 4: **gholdengo** (HP: 100%) [nostatus]
  - Bench 5: **greattusk** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **darkrai**

---
**Turn 10** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stealthrock

*Last turn*: Player used willowisp | Opponent used crabhammer

**Player Active**: **darkrai** (HP: 0%) | Type: dark notype | Item: leftovers | Ability: baddreams | Status: fnt | Tera: poison
**Opponent Active**: **crawdaunt** (HP: 78%) | Type: dark water | Item: lifeorb | Status: nostatus | Tera: notype

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 4: **greattusk** (HP: 100%) [nostatus]
  - Bench 5: **tinglu** (HP: 42%) [nostatus]

**Decision**: Switch to bench slot 3: Switch to **gholdengo**

---
**Turn 11**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stealthrock

*Last turn*: Player used nomove | Opponent used crabhammer

**Player Active**: **gholdengo** (HP: 100%) | Type: ghost steel | Item: airballoon | Ability: goodasgold | Status: nostatus | Tera: fairy
**Opponent Active**: **crawdaunt** (HP: 78%) | Type: dark water | Item: lifeorb | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **makeitrain** [steel] (special) BP:120 PP:8/8 <-- CHOSEN
  - Slot 2: **trick** [psychic] (status) PP:16/16
  - Slot 3: **nastyplot** [dark] (status) PP:32/32
  - Slot 4: **recover** [normal] (status) PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **tinglu** (HP: 42%) [nostatus]

**Decision**: Move 1: Use **makeitrain**

---
**Turn 12**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stealthrock

*Last turn*: Player used makeitrain | Opponent used nomove

**Player Active**: **gholdengo** (HP: 100%) | Type: ghost steel | Item: airballoon | Ability: goodasgold | Status: nostatus | Tera: fairy | Boosts: SPA-1
**Opponent Active**: **greattusk** (HP: 100%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **makeitrain** [steel] (special) BP:120 PP:7/8
  - Slot 2: **trick** [psychic] (status) PP:16/16
  - Slot 3: **nastyplot** [dark] (status) PP:32/32
  - Slot 4: **recover** [normal] (status) PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **greattusk** (HP: 100%) [nostatus]
  - Bench 4: **tinglu** (HP: 42%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 4: Switch to **tinglu**

---
**Turn 13**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stealthrock

*Last turn*: Player used earthquake | Opponent used earthquake

**Player Active**: **tinglu** (HP: 15%) | Type: dark ground | Item: leftovers | Ability: vesselofruin | Status: nostatus | Tera: poison
**Opponent Active**: **greattusk** (HP: 100%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **stealthrock** [rock] (status) PP:31/32
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:15/16 <-- CHOSEN
  - Slot 3: **ruination** [dark] (special) PP:16/16
  - Slot 4: **whirlwind** [normal] (status) Priority:-6 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **earthquake**

---
**Turn 14**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used ruination | Opponent used rapidspin

**Player Active**: **tinglu** (HP: 12%) | Type: dark ground | Item: leftovers | Ability: vesselofruin | Status: nostatus | Tera: poison
**Opponent Active**: **greattusk** (HP: 51%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: SPE+1

**Available Moves:**
  - Slot 1: **stealthrock** [rock] (status) PP:31/32
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:15/16
  - Slot 3: **ruination** [dark] (special) PP:15/16
  - Slot 4: **whirlwind** [normal] (status) Priority:-6 PP:32/32
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **whirlwind**

---
**Turn 15** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used ruination | Opponent used icespinner

**Player Active**: **tinglu** (HP: 0%) | Type: dark ground | Item: leftovers | Ability: vesselofruin | Status: fnt | Tera: poison
**Opponent Active**: **greattusk** (HP: 51%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: SPE+1

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]
  - Bench 4: **greattusk** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 4: Switch to **greattusk**

---
**Turn 16**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used knockoff | Opponent used icespinner

**Player Active**: **greattusk** (HP: 100%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground
**Opponent Active**: **greattusk** (HP: 51%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: SPE+1

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:31/32
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:64/64
  - Slot 3: **icespinner** [ice] (physical) BP:80 PP:24/24
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:8/8 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **headlongrush**

---
**Turn 17**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used rapidspin | Opponent used bulkup

**Player Active**: **greattusk** (HP: 100%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground | Boosts: SPE+1
**Opponent Active**: **greattusk** (HP: 43%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: ATK+1, DEF+1, SPE+1

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:31/32
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:63/64 <-- CHOSEN
  - Slot 3: **icespinner** [ice] (physical) BP:80 PP:24/24
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **rapidspin**

---
**Turn 18**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used icespinner | Opponent used earthquake

**Player Active**: **greattusk** (HP: 47%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground | Boosts: SPE+1
**Opponent Active**: **greattusk** (HP: 19%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: ATK+1, DEF+1, SPE+1

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:31/32
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:63/64 <-- CHOSEN
  - Slot 3: **icespinner** [ice] (physical) BP:80 PP:23/24
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **rapidspin**

---
**Turn 19**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used icespinner | Opponent used nomove

**Player Active**: **greattusk** (HP: 47%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground | Boosts: SPE+1
**Opponent Active**: **latios** (HP: 100%) | Type: dragon psychic | Ability: levitate | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:31/32
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:63/64
  - Slot 3: **icespinner** [ice] (physical) BP:80 PP:22/24 <-- CHOSEN
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **icespinner**

---
**Turn 20**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: reflect

*Last turn*: Player used knockoff | Opponent used reflect

**Player Active**: **greattusk** (HP: 47%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground | Boosts: SPE+1
**Opponent Active**: **latios** (HP: 28%) | Type: dragon psychic | Item: noitem | Ability: levitate | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:30/32
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:63/64 <-- CHOSEN
  - Slot 3: **icespinner** [ice] (physical) BP:80 PP:22/24
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **rapidspin**

---
**Turn 21**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: reflect

*Last turn*: Player used icespinner | Opponent used ironhead

**Player Active**: **greattusk** (HP: 47%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground | Boosts: SPE+1
**Opponent Active**: **jirachi** (HP: 76%) | Type: psychic steel | Ability: serenegrace | Status: brn | Tera: notype

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:30/32 <-- CHOSEN
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:63/64
  - Slot 3: **icespinner** [ice] (physical) BP:80 PP:21/24
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:8/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **knockoff**

---
**Turn 22**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: reflect

*Last turn*: Player used headlongrush | Opponent used ironhead

**Player Active**: **greattusk** (HP: 30%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground | Boosts: DEF-1, SPD-1, SPE+1
**Opponent Active**: **jirachi** (HP: 18%) | Type: psychic steel | Ability: serenegrace | Status: brn | Tera: notype

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:30/32
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:63/64
  - Slot 3: **icespinner** [ice] (physical) BP:80 PP:21/24
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:7/8 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **headlongrush**

---
**Turn 23**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: reflect

*Last turn*: Player used rapidspin | Opponent used icepunch

**Player Active**: **greattusk** (HP: 9%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground | Boosts: DEF-1, SPD-1, SPE+2
**Opponent Active**: **jirachi** (HP: 8%) | Type: psychic steel | Ability: serenegrace | Status: brn | Tera: notype

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:30/32
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:62/64
  - Slot 3: **icespinner** [ice] (physical) BP:80 PP:21/24 <-- CHOSEN
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:7/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **icespinner**

---
**Turn 24**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used knockoff | Opponent used nomove

**Player Active**: **greattusk** (HP: 9%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: ground | Boosts: DEF-1, SPD-1, SPE+2
**Opponent Active**: **gardevoir** (HP: 100%) | Type: fairy psychic | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **knockoff** [dark] (physical) BP:65 PP:29/32 <-- CHOSEN
  - Slot 2: **rapidspin** [normal] (physical) BP:50 PP:62/64
  - Slot 3: **icespinner** [ice] (physical) BP:80 PP:21/24
  - Slot 4: **headlongrush** [ground] (physical) BP:120 PP:7/8
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus]
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **knockoff**

---
**Turn 25** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used headlongrush | Opponent used drainingkiss

**Player Active**: **greattusk** (HP: 0%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: fnt | Tera: ground | Boosts: DEF-2, SPD-2, SPE+2
**Opponent Active**: **gardevoir** (HP: 33%) | Type: fairy psychic | Item: noitem | Status: nostatus | Tera: notype | Boosts: SPE+1

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **enamorus** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 3: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 2: Switch to **enamorus**

---
**Turn 26**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used drainingkiss

**Player Active**: **enamorus** (HP: 100%) | Type: fairy flying | Item: choicespecs | Ability: contrary | Status: nostatus | Tera: stellar
**Opponent Active**: **gardevoir** (HP: 33%) | Type: fairy psychic | Item: noitem | Ability: contrary | Status: nostatus | Tera: notype | Boosts: SPE+1

**Available Moves:**
  - Slot 1: **moonblast** [fairy] (special) BP:95 PP:24/24
  - Slot 2: **calmmind** [psychic] (status) PP:32/32
  - Slot 3: **earthpower** [ground] (special) BP:90 PP:16/16
  - Slot 4: **healingwish** [psychic] (status) PP:16/16 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **healingwish**

---
**Turn 27**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used moonblast | Opponent used drainingkiss

**Player Active**: **enamorus** (HP: 100%) | Type: fairy flying | Item: choicespecs | Ability: contrary | Status: nostatus | Tera: stellar
**Opponent Active**: **gardevoir** (HP: 0%) | Type: fairy psychic | Item: noitem | Ability: contrary | Status: fnt | Tera: notype | Boosts: SPE+1

**Available Moves:**
  - Slot 1: **moonblast** [fairy] (special) BP:95 PP:23/24
  - Slot 2: **calmmind** [psychic] (status) PP:32/32
  - Slot 3: **earthpower** [ground] (special) BP:90 PP:16/16
  - Slot 4: **healingwish** [psychic] (status) PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **dragonite** (HP: 100%) [nostatus]
  - Bench 2: **gholdengo** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **healingwish**


---

## 10. Battle 10

### Battle: `1553103-gen9ou-2439327985_1512_assurance64217_vs_arceusbug25750_09-10-2025_WIN.json.lz4`
- **Result**: WIN
- **Elo**: 1512
- **Total Turns**: 28

#### Player's Team
1. **ribombee** (HP: 100%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: nostatus | Tera: ghost *(Lead)*
2. **greattusk** (HP: 100%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: water
3. **dondozo** (HP: 100%) | Type: notype water | Item: leftovers | Ability: unaware | Status: nostatus | Tera: dragon
4. **goodrahisui** (HP: 100%) | Type: dragon steel | Item: heavydutyboots | Ability: sapsipper | Status: nostatus | Tera: flying
5. **milotic** (HP: 100%) | Type: notype water | Item: leftovers | Ability: marvelscale | Status: nostatus | Tera: fairy
6. **salazzle** (HP: 100%) | Type: fire poison | Item: focussash | Ability: corrosion | Status: nostatus | Tera: ground

#### Opponent's Team (from Team Preview)
1. **greattusk**
2. **deoxys**
3. **samurott**
4. **sinistcha**
5. **clodsire**
6. **moltres**

#### Turn-by-Turn Decisions

---
**Turn 1**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **ribombee** (HP: 100%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: nostatus | Tera: ghost
**Opponent Active**: **samurotthisui** (HP: 100%) | Type: dark water | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **moonblast** [fairy] (special) BP:95 PP:24/24 <-- CHOSEN
  - Slot 2: **stunspore** [grass] (status) PP:48/48
  - Slot 3: **stickyweb** [bug] (status) PP:32/32
  - Slot 4: **skillswap** [psychic] (status) PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 4: **milotic** (HP: 100%) [nostatus]
  - Bench 5: **salazzle** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **moonblast**

---
**Turn 2**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used moonblast | Opponent used nomove

**Player Active**: **ribombee** (HP: 100%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: nostatus | Tera: ghost
**Opponent Active**: **clodsire** (HP: 96%) | Type: ground poison | Item: leftovers | Status: nostatus | Tera: notype | Boosts: SPA-1

**Available Moves:**
  - Slot 1: **moonblast** [fairy] (special) BP:95 PP:23/24
  - Slot 2: **stunspore** [grass] (status) PP:48/48
  - Slot 3: **stickyweb** [bug] (status) PP:32/32
  - Slot 4: **skillswap** [psychic] (status) PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 4: **milotic** (HP: 100%) [nostatus]
  - Bench 5: **salazzle** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 5: Switch to **salazzle**

---
**Turn 3**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used toxic

**Player Active**: **salazzle** (HP: 100%) | Type: fire poison | Item: focussash | Ability: corrosion | Status: nostatus | Tera: ground
**Opponent Active**: **clodsire** (HP: 100%) | Type: ground poison | Item: leftovers | Status: nostatus | Tera: notype | Boosts: SPA-1

**Available Moves:**
  - Slot 1: **toxic** [poison] (status) PP:16/16
  - Slot 2: **endeavor** [normal] (physical) PP:8/8
  - Slot 3: **fireblast** [fire] (special) BP:110 PP:8/8
  - Slot 4: **flamethrower** [fire] (special) BP:90 PP:24/24 <-- CHOSEN
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **ribombee** (HP: 100%) [nostatus]
  - Bench 4: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 5: **milotic** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **flamethrower**

---
**Turn 4**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used toxic | Opponent used earthquake

**Player Active**: **salazzle** (HP: 1%) | Type: fire poison | Item: noitem | Ability: corrosion | Status: nostatus | Tera: ground
**Opponent Active**: **clodsire** (HP: 94%) | Type: ground poison | Item: leftovers | Status: tox | Tera: notype | Boosts: SPA-1

**Available Moves:**
  - Slot 1: **toxic** [poison] (status) PP:15/16 <-- CHOSEN
  - Slot 2: **endeavor** [normal] (physical) PP:8/8
  - Slot 3: **fireblast** [fire] (special) BP:110 PP:8/8
  - Slot 4: **flamethrower** [fire] (special) BP:90 PP:24/24
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **ribombee** (HP: 100%) [nostatus]
  - Bench 4: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 5: **milotic** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **toxic**

---
**Turn 5**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used endeavor | Opponent used nomove

**Player Active**: **salazzle** (HP: 1%) | Type: fire poison | Item: noitem | Ability: corrosion | Status: nostatus | Tera: ground
**Opponent Active**: **deoxysspeed** (HP: 100%) | Type: notype psychic | Ability: pressure | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **toxic** [poison] (status) PP:15/16
  - Slot 2: **endeavor** [normal] (physical) PP:7/8
  - Slot 3: **fireblast** [fire] (special) BP:110 PP:8/8
  - Slot 4: **flamethrower** [fire] (special) BP:90 PP:24/24
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **ribombee** (HP: 100%) [nostatus]
  - Bench 4: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 5: **milotic** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **flamethrower**

---
**Turn 6** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used endeavor | Opponent used icebeam

**Player Active**: **salazzle** (HP: 0%) | Type: fire poison | Item: noitem | Ability: corrosion | Status: fnt | Tera: ground
**Opponent Active**: **deoxysspeed** (HP: 100%) | Type: notype psychic | Ability: pressure | Status: nostatus | Tera: notype

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **ribombee** (HP: 100%) [nostatus]
  - Bench 4: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 5: **milotic** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 5: Switch to **milotic**

---
**Turn 7**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used moonblast | Opponent used icebeam

**Player Active**: **ribombee** (HP: 76%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: nostatus | Tera: ghost
**Opponent Active**: **deoxysspeed** (HP: 100%) | Type: notype psychic | Ability: pressure | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **moonblast** [fairy] (special) BP:95 PP:23/24 <-- CHOSEN
  - Slot 2: **stunspore** [grass] (status) PP:48/48
  - Slot 3: **stickyweb** [bug] (status) PP:32/32
  - Slot 4: **skillswap** [psychic] (status) PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 4: **milotic** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **moonblast**

---
**Turn 8**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used moonblast | Opponent used nastyplot

**Player Active**: **ribombee** (HP: 76%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: nostatus | Tera: ghost
**Opponent Active**: **deoxysspeed** (HP: 44%) | Type: notype psychic | Ability: pressure | Status: nostatus | Tera: notype | Boosts: SPA+2

**Available Moves:**
  - Slot 1: **moonblast** [fairy] (special) BP:95 PP:21/24 <-- CHOSEN
  - Slot 2: **stunspore** [grass] (status) PP:48/48
  - Slot 3: **stickyweb** [bug] (status) PP:32/32
  - Slot 4: **skillswap** [psychic] (status) PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 4: **milotic** (HP: 100%) [nostatus]

**Decision**: Move 1: Use **moonblast**

---
**Turn 9**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used moonblast | Opponent used nomove

**Player Active**: **ribombee** (HP: 76%) | Type: bug fairy | Item: focussash | Ability: shielddust | Status: nostatus | Tera: ghost
**Opponent Active**: **moltres** (HP: 100%) | Type: fire flying | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **moonblast** [fairy] (special) BP:95 PP:19/24
  - Slot 2: **stunspore** [grass] (status) PP:48/48
  - Slot 3: **stickyweb** [bug] (status) PP:32/32
  - Slot 4: **skillswap** [psychic] (status) PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 4: **milotic** (HP: 100%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 4: Switch to **milotic**

---
**Turn 10**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used nomove

**Player Active**: **milotic** (HP: 100%) | Type: notype water | Item: leftovers | Ability: marvelscale | Status: nostatus | Tera: fairy
**Opponent Active**: **samurotthisui** (HP: 100%) | Type: dark water | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **alluringvoice** [fairy] (special) BP:80 PP:16/16 <-- CHOSEN (with Tera)
  - Slot 2: **recover** [normal] (status) PP:8/8
  - Slot 3: **drainingkiss** [fairy] (special) BP:50 PP:16/16
  - Slot 4: **icebeam** [ice] (special) BP:90 PP:16/16
  - *Terastallize available* (can use Tera + any move)

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **ribombee** (HP: 76%) [nostatus]
  - Bench 4: **goodrahisui** (HP: 100%) [nostatus]

**Decision**: Tera + Move 1: Terastallize + Use **alluringvoice**

---
**Turn 11**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used alluringvoice | Opponent used nomove

**Player Active**: **milotic** (HP: 42%) | Type: fairy notype | Item: noitem | Ability: marvelscale | Status: nostatus | Tera: fairy
**Opponent Active**: **greattusk** (HP: 100%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **alluringvoice** [fairy] (special) BP:80 PP:15/16
  - Slot 2: **recover** [normal] (status) PP:8/8
  - Slot 3: **drainingkiss** [fairy] (special) BP:50 PP:16/16
  - Slot 4: **icebeam** [ice] (special) BP:90 PP:16/16

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **ribombee** (HP: 76%) [nostatus]
  - Bench 4: **goodrahisui** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **greattusk**

---
**Turn 12**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used headlongrush

**Player Active**: **dondozo** (HP: 100%) | Type: notype water | Item: leftovers | Ability: unaware | Status: nostatus | Tera: dragon
**Opponent Active**: **greattusk** (HP: 100%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: DEF-1, SPD-1

**Available Moves:**
  - Slot 1: **liquidation** [water] (physical) BP:85 PP:16/16
  - Slot 2: **bodypress** [fighting] (physical) BP:80 PP:16/16
  - Slot 3: **sleeptalk** [normal] (status) PP:16/16
  - Slot 4: **wavecrash** [water] (physical) BP:120 PP:16/16

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **ribombee** (HP: 76%) [nostatus]
  - Bench 3: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 4: **milotic** (HP: 42%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 4: Switch to **milotic**

---
**Turn 13**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used moonblast | Opponent used knockoff

**Player Active**: **ribombee** (HP: 22%) | Type: bug fairy | Item: noitem | Ability: shielddust | Status: nostatus | Tera: ghost
**Opponent Active**: **greattusk** (HP: 100%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: DEF-1, SPD-1

**Available Moves:**
  - Slot 1: **moonblast** [fairy] (special) BP:95 PP:19/24
  - Slot 2: **stunspore** [grass] (status) PP:48/48
  - Slot 3: **stickyweb** [bug] (status) PP:32/32
  - Slot 4: **skillswap** [psychic] (status) PP:16/16 <-- CHOSEN

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 4: **milotic** (HP: 42%) [nostatus]

**Decision**: Move 4: Use **skillswap**

---
**Turn 14**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: noconditions

*Last turn*: Player used stunspore | Opponent used uturn

**Player Active**: **ribombee** (HP: 22%) | Type: bug fairy | Item: noitem | Ability: shielddust | Status: nostatus | Tera: ghost
**Opponent Active**: **moltres** (HP: 100%) | Type: fire flying | Status: par | Tera: notype

**Available Moves:**
  - Slot 1: **moonblast** [fairy] (special) BP:95 PP:19/24
  - Slot 2: **stunspore** [grass] (status) PP:47/48
  - Slot 3: **stickyweb** [bug] (status) PP:32/32 <-- CHOSEN
  - Slot 4: **skillswap** [psychic] (status) PP:16/16

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 4: **milotic** (HP: 42%) [nostatus]

**Decision**: Move 3: Use **stickyweb**

---
**Turn 15** *(Forced Switch)*

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stickyweb

*Last turn*: Player used stickyweb | Opponent used flamethrower

**Player Active**: **ribombee** (HP: 0%) | Type: bug fairy | Item: noitem | Ability: shielddust | Status: fnt | Tera: ghost
**Opponent Active**: **moltres** (HP: 100%) | Type: fire flying | Status: par | Tera: notype

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **goodrahisui** (HP: 100%) [nostatus]
  - Bench 4: **milotic** (HP: 42%) [nostatus] <-- CHOSEN

**Decision**: Switch to bench slot 4: Switch to **milotic**

---
**Turn 16**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stickyweb

*Last turn*: Player used alluringvoice | Opponent used flamethrower

**Player Active**: **milotic** (HP: 30%) | Type: notype water | Item: noitem | Ability: marvelscale | Status: nostatus | Tera: fairy
**Opponent Active**: **moltres** (HP: 100%) | Type: fire flying | Status: par | Tera: notype

**Available Moves:**
  - Slot 1: **alluringvoice** [fairy] (special) BP:80 PP:15/16
  - Slot 2: **recover** [normal] (status) PP:8/8
  - Slot 3: **drainingkiss** [fairy] (special) BP:50 PP:16/16
  - Slot 4: **icebeam** [ice] (special) BP:90 PP:16/16 <-- CHOSEN

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **goodrahisui** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **icebeam**

---
**Turn 17**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stickyweb

*Last turn*: Player used recover | Opponent used flamethrower

**Player Active**: **milotic** (HP: 51%) | Type: notype water | Item: noitem | Ability: marvelscale | Status: nostatus | Tera: fairy
**Opponent Active**: **moltres** (HP: 100%) | Type: fire flying | Status: par | Tera: notype

**Available Moves:**
  - Slot 1: **alluringvoice** [fairy] (special) BP:80 PP:15/16
  - Slot 2: **recover** [normal] (status) PP:7/8
  - Slot 3: **drainingkiss** [fairy] (special) BP:50 PP:16/16
  - Slot 4: **icebeam** [ice] (special) BP:90 PP:16/16 <-- CHOSEN

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **goodrahisui** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **icebeam**

---
**Turn 18**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stickyweb

*Last turn*: Player used recover | Opponent used nomove

**Player Active**: **milotic** (HP: 100%) | Type: notype water | Item: noitem | Ability: marvelscale | Status: nostatus | Tera: fairy
**Opponent Active**: **sinistcha** (HP: 100%) | Type: ghost grass | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **alluringvoice** [fairy] (special) BP:80 PP:15/16
  - Slot 2: **recover** [normal] (status) PP:6/8
  - Slot 3: **drainingkiss** [fairy] (special) BP:50 PP:16/16
  - Slot 4: **icebeam** [ice] (special) BP:90 PP:16/16

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 3: **goodrahisui** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 2: Switch to **dondozo**

---
**Turn 19**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stickyweb

*Last turn*: Player used nomove | Opponent used calmmind

**Player Active**: **goodrahisui** (HP: 100%) | Type: dragon steel | Item: heavydutyboots | Ability: sapsipper | Status: nostatus | Tera: flying
**Opponent Active**: **sinistcha** (HP: 100%) | Type: ghost grass | Status: nostatus | Tera: notype | Boosts: SPA+1, SPD+1

**Available Moves:**
  - Slot 1: **icebeam** [ice] (special) BP:90 PP:16/16
  - Slot 2: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 3: **heavyslam** [steel] (physical) PP:16/16 <-- CHOSEN
  - Slot 4: **dragontail** [dragon] (physical) BP:60 Priority:-6 PP:16/16

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **milotic** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **heavyslam**

---
**Turn 20**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stickyweb

*Last turn*: Player used icebeam | Opponent used shadowball

**Player Active**: **goodrahisui** (HP: 50%) | Type: dragon steel | Item: heavydutyboots | Ability: sapsipper | Status: nostatus | Tera: flying
**Opponent Active**: **sinistcha** (HP: 54%) | Type: ghost grass | Status: nostatus | Tera: notype | Boosts: SPA+1, SPD+1

**Available Moves:**
  - Slot 1: **icebeam** [ice] (special) BP:90 PP:15/16
  - Slot 2: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 3: **heavyslam** [steel] (physical) PP:16/16 <-- CHOSEN
  - Slot 4: **dragontail** [dragon] (physical) BP:60 Priority:-6 PP:16/16

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **milotic** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **heavyslam**

---
**Turn 21**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stickyweb

*Last turn*: Player used icebeam | Opponent used shadowball

**Player Active**: **goodrahisui** (HP: 6%) | Type: dragon steel | Item: heavydutyboots | Ability: sapsipper | Status: nostatus | Tera: flying
**Opponent Active**: **sinistcha** (HP: 8%) | Type: ghost grass | Status: nostatus | Tera: notype | Boosts: SPA+1, SPD+1

**Available Moves:**
  - Slot 1: **icebeam** [ice] (special) BP:90 PP:14/16
  - Slot 2: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 3: **heavyslam** [steel] (physical) PP:16/16 <-- CHOSEN
  - Slot 4: **dragontail** [dragon] (physical) BP:60 Priority:-6 PP:16/16

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus]
  - Bench 3: **milotic** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **heavyslam**

---
**Turn 22**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stickyweb

*Last turn*: Player used icebeam | Opponent used flamethrower

**Player Active**: **goodrahisui** (HP: 6%) | Type: dragon steel | Item: heavydutyboots | Ability: sapsipper | Status: nostatus | Tera: flying
**Opponent Active**: **moltres** (HP: 78%) | Type: fire flying | Status: par | Tera: notype

**Available Moves:**
  - Slot 1: **icebeam** [ice] (special) BP:90 PP:13/16
  - Slot 2: **thunderbolt** [electric] (special) BP:90 PP:24/24
  - Slot 3: **heavyslam** [steel] (physical) PP:16/16
  - Slot 4: **dragontail** [dragon] (physical) BP:60 Priority:-6 PP:16/16

**Available Switches:**
  - Bench 1: **greattusk** (HP: 100%) [nostatus]
  - Bench 2: **dondozo** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 3: **milotic** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 2: Switch to **dondozo**

---
**Turn 23**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: stealthrock | Opponent side: stickyweb

*Last turn*: Player used nomove | Opponent used knockoff

**Player Active**: **greattusk** (HP: 100%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: water
**Opponent Active**: **greattusk** (HP: 100%) | Type: fighting ground | Ability: protosynthesis | Status: nostatus | Tera: notype | Boosts: SPE-1

**Available Moves:**
  - Slot 1: **rapidspin** [normal] (physical) BP:50 PP:64/64
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:16/16
  - Slot 3: **closecombat** [fighting] (physical) BP:120 PP:8/8
  - Slot 4: **bulkup** [fighting] (status) PP:32/32 <-- CHOSEN

**Available Switches:**
  - Bench 1: **dondozo** (HP: 100%) [nostatus]
  - Bench 2: **goodrahisui** (HP: 6%) [nostatus]
  - Bench 3: **milotic** (HP: 100%) [nostatus]

**Decision**: Move 4: Use **bulkup**

---
**Turn 24**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used rapidspin | Opponent used rapidspin

**Player Active**: **greattusk** (HP: 65%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: water | Boosts: SPE+1
**Opponent Active**: **greattusk** (HP: 90%) | Type: fighting ground | Item: rockyhelmet | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **rapidspin** [normal] (physical) BP:50 PP:63/64
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:16/16
  - Slot 3: **closecombat** [fighting] (physical) BP:120 PP:8/8 <-- CHOSEN
  - Slot 4: **bulkup** [fighting] (status) PP:32/32

**Available Switches:**
  - Bench 1: **dondozo** (HP: 100%) [nostatus]
  - Bench 2: **goodrahisui** (HP: 6%) [nostatus]
  - Bench 3: **milotic** (HP: 100%) [nostatus]

**Decision**: Move 3: Use **closecombat**

---
**Turn 25**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used earthquake | Opponent used icespinner

**Player Active**: **greattusk** (HP: 30%) | Type: fighting ground | Item: boosterenergy | Ability: protosynthesis | Status: nostatus | Tera: water | Boosts: SPE+1
**Opponent Active**: **greattusk** (HP: 61%) | Type: fighting ground | Item: rockyhelmet | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **rapidspin** [normal] (physical) BP:50 PP:63/64
  - Slot 2: **earthquake** [ground] (physical) BP:100 PP:15/16
  - Slot 3: **closecombat** [fighting] (physical) BP:120 PP:8/8
  - Slot 4: **bulkup** [fighting] (status) PP:32/32

**Available Switches:**
  - Bench 1: **dondozo** (HP: 100%) [nostatus] <-- CHOSEN
  - Bench 2: **goodrahisui** (HP: 6%) [nostatus]
  - Bench 3: **milotic** (HP: 100%) [nostatus]

**Decision**: Switch to bench slot 1: Switch to **dondozo**

---
**Turn 26**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used nomove | Opponent used icespinner

**Player Active**: **dondozo** (HP: 69%) | Type: notype water | Item: leftovers | Ability: unaware | Status: nostatus | Tera: dragon
**Opponent Active**: **greattusk** (HP: 61%) | Type: fighting ground | Item: rockyhelmet | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **liquidation** [water] (physical) BP:85 PP:16/16
  - Slot 2: **bodypress** [fighting] (physical) BP:80 PP:16/16 <-- CHOSEN
  - Slot 3: **sleeptalk** [normal] (status) PP:16/16
  - Slot 4: **wavecrash** [water] (physical) BP:120 PP:16/16

**Available Switches:**
  - Bench 1: **greattusk** (HP: 30%) [nostatus]
  - Bench 2: **goodrahisui** (HP: 6%) [nostatus]
  - Bench 3: **milotic** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **bodypress**

---
**Turn 27**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used liquidation | Opponent used knockoff

**Player Active**: **dondozo** (HP: 38%) | Type: notype water | Item: noitem | Ability: unaware | Status: nostatus | Tera: dragon
**Opponent Active**: **greattusk** (HP: 14%) | Type: fighting ground | Item: rockyhelmet | Ability: protosynthesis | Status: nostatus | Tera: notype

**Available Moves:**
  - Slot 1: **liquidation** [water] (physical) BP:85 PP:15/16
  - Slot 2: **bodypress** [fighting] (physical) BP:80 PP:16/16 <-- CHOSEN
  - Slot 3: **sleeptalk** [normal] (status) PP:16/16
  - Slot 4: **wavecrash** [water] (physical) BP:120 PP:16/16

**Available Switches:**
  - Bench 1: **greattusk** (HP: 30%) [nostatus]
  - Bench 2: **goodrahisui** (HP: 6%) [nostatus]
  - Bench 3: **milotic** (HP: 100%) [nostatus]

**Decision**: Move 2: Use **bodypress**

---
**Turn 28**

*Field*: Weather: noweather | Terrain/Field: nofield | Player side: noconditions | Opponent side: noconditions

*Last turn*: Player used liquidation | Opponent used shadowball

**Player Active**: **dondozo** (HP: 38%) | Type: notype water | Item: noitem | Ability: unaware | Status: nostatus | Tera: dragon
**Opponent Active**: **sinistcha** (HP: 0%) | Type: ghost grass | Status: fnt | Tera: notype

**Available Moves:**
  - Slot 1: **liquidation** [water] (physical) BP:85 PP:14/16
  - Slot 2: **bodypress** [fighting] (physical) BP:80 PP:16/16
  - Slot 3: **sleeptalk** [normal] (status) PP:16/16
  - Slot 4: **wavecrash** [water] (physical) BP:120 PP:16/16

**Available Switches:**
  - Bench 1: **greattusk** (HP: 30%) [nostatus]
  - Bench 2: **goodrahisui** (HP: 6%) [nostatus]
  - Bench 3: **milotic** (HP: 100%) [nostatus]

**Decision**: Action -1: Use **wavecrash**


---

## Summary Statistics

- **Battles analyzed**: 10
- **Total turns**: 204
- **Average turns per battle**: 20.4
- **Wins**: 8
- **Losses**: 2
