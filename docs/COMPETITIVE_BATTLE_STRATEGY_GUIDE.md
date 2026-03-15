# Competitive Pokemon Singles Battle Strategy Guide (Gen 3 OU / ADV)

_Updated: March 2026 — Adapted for Gen 3 OU (Ruby/Sapphire/Emerald)_

A concise reference for competitive 6v6 OU singles on Pokemon Showdown,
focused on Gen 3 (ADV) mechanics, decision-making, tactics, and common pitfalls.

**Gen 3-specific mechanics:**
- **No team preview** — opponent team is entirely unknown at battle start.
- **No Terastallization** — does not exist until Gen 9.
- **Type-based physical/special split** — move category (physical/special) is determined by type, not the individual move. All Fire moves are special; all Rock moves are physical.
- **Permanent weather** — weather from abilities (Sand Stream, Drizzle, Drought) lasts indefinitely.
- **No Stealth Rock** — Spikes (up to 3 layers) is the only entry hazard. No Toxic Spikes or Sticky Web.
- **No priority priority** — Extreme Speed and Mach Punch exist but the priority move pool is smaller.

---

## Table of Contents

1. [Core Philosophy](#1-core-philosophy)
2. [Team Archetypes](#2-team-archetypes)
3. [Team Building](#3-team-building)
4. [Pokemon Roles](#4-pokemon-roles)
5. [Key Items](#5-key-items)
6. [Key Abilities](#6-key-abilities)
7. [Entry Hazards & Hazard Control](#7-entry-hazards--hazard-control)
8. [Status Conditions](#8-status-conditions)
9. [Pivoting & Momentum](#9-pivoting--momentum)
10. [Speed Control](#10-speed-control)
11. [Switching & Prediction](#11-switching--prediction)
12. [Risk & Reward](#12-risk--reward)
13. [Win Conditions & Endgame](#13-win-conditions--endgame)
14. [Gen 3-Specific Mechanics](#14-gen-3-specific-mechanics)
15. [Common Mistakes & Bad Tactics](#15-common-mistakes--bad-tactics)
16. [Advanced Heuristics & Rules of Thumb](#16-advanced-heuristics--rules-of-thumb)
17. [Sources](#17-sources)

---

## 1. Core Philosophy

The entirety of competitive Pokemon strategy revolves around **damage**: the ability
to deal it, withstand it, and avoid it. Every decision—what move to click, when to
switch, what item to hold—ultimately serves one of these three goals.

### The Three Pillars

1. **Offensive pressure**: Can your team threaten KOs and force switches?
2. **Defensive backbone**: Can your team absorb hits and pivot safely?
3. **Win condition execution**: Can you create and capitalize on a path to victory?

A competitive team must balance all three. Pure offense crumbles to bulk; pure
defense stalls out against wallbreakers; an unprotected win condition never gets
to sweep.

### The Information Game

Pokemon singles is a game of **incomplete information**. In Gen 3, there is
**no team preview** — you don't know any of your opponent's Pokemon until they
switch in. Even after seeing a Pokemon, you don't know their:

- Exact EV spreads and natures
- Held items (until revealed by effect or Knock Off)
- Full movesets (until revealed move-by-move)
- Remaining team composition (until each Pokemon switches in)
- Planned strategy and win condition

Good players extract information incrementally — through observed moves, damage
calculations, switching patterns, and item reveals — and update their strategy
accordingly. The lead matchup is critical in Gen 3 because it's your first
window into the opponent's team strategy. Information flows naturally over the
course of a game; scouting (probing with safe plays) is even more important
without team preview.

---

## 2. Team Archetypes

Teams fall along a spectrum from most aggressive to most defensive:

### TSS (Toxic / Spikes / Sandstorm) — The Signature Gen 3 Strategy

- **Goal**: Wear down the opponent through passive damage from permanent sandstorm,
  Spikes layers, and Toxic, then clean up with a sweeper.
- **Structure**: Tyranitar (Sand Stream), Skarmory (Spikes), Blissey (special wall),
  plus sweepers and a Rapid Spin blocker (Ghost-type).
- **Strengths**: The defining Gen 3 archetype. Permanent sandstorm + 3 layers of
  Spikes makes every switch devastating. Extremely hard to break through.
- **Weaknesses**: Vulnerable to Rapid Spin, dedicated wallbreakers, and teams
  with strong Spikes resistance (Flying-types, Levitate).
- **Key principle**: Every switch your opponent makes costs them HP. Patience wins.

### Hyper Offense (HO)

- **Goal**: Overwhelm the opponent with relentless pressure before they stabilize.
- **Structure**: Dragon Dance Salamence, Swords Dance Heracross, mixed Tyranitar,
  Dugtrio for trapping threats, minimal defensive investment.
- **Strengths**: Punishes passive play, forces opponent onto the back foot.
- **Weaknesses**: Fragile; one missed prediction or unlucky turn can unravel the
  whole game. Struggles against well-built TSS/stall.
- **Key principle**: Every turn you're not attacking or setting up, you're losing.

### Bulky Offense

- **Goal**: Wear down the opponent gradually while hitting hard.
- **Structure**: Salamence, Metagross, Suicune, Celebi cores with Dragon Dance
  or Calm Mind sweepers backed by sturdy pivots.
- **Strengths**: Excellent at trading blows; harder to sweep than HO.
- **Weaknesses**: Can lack the raw speed to revenge kill boosted threats.

### Balance

- **Goal**: Maintain stability while making incremental progress.
- **Structure**: Defensive core + offensive threats, Spikes setter + Rapid Spin,
  status spreaders, pivots.
- **Strengths**: The most flexible archetype; can adapt to most matchups.
- **Weaknesses**: Games tend to be longer; vulnerable to skilled wallbreakers
  that punch through the defensive core.
- **Key principle**: Patience. You don't need to win on turn 5.

### Stall

- **Goal**: Outlast the opponent through recovery, residual damage, and PP stalling.
- **Structure**: Skarmory/Blissey/Milotic/Celebi with Wish + Protect + Spikes,
  cleric support, and Toxic spreading.
- **Strengths**: Extremely hard to break for unprepared teams.
- **Weaknesses**: Taunt, Trick, dedicated wallbreakers (Choice Band Metagross),
  and Dugtrio trapping shut it down.
- **Key principle**: You win by not losing. The opponent runs out of resources
  before you do.

### Weather Teams (Rain / Sun)

- **Goal**: Exploit weather-boosted sweepers for overwhelming offense.
- **Structure**: Rain Dance + Kingdra/Ludicolo sweepers, or Sunny Day +
  Exeggutor/Houndoom. Note: ability-set weather (Drizzle/Drought) is
  banned in standard Gen 3 OU — only Tyranitar's Sand Stream is permanent.
- **Strengths**: Weather-boosted attacks can be devastating; Rain gives
  effective STAB boost to Water moves and Thunder accuracy.
- **Weaknesses**: Requires setup turns for Rain/Sun Dance; Tyranitar's
  permanent sandstorm overrides manual weather easily.

### Baton Pass Chains

- **Goal**: Chain Speed/Attack/Special Attack boosts via Baton Pass and sweep.
- **Structure**: Ninjask (Speed Boost + Baton Pass), Smeargle (Spore + BP),
  Celebi, and a receiver sweeper.
- **Strengths**: Can create unstoppable sweepers when executed well.
- **Weaknesses**: Disrupted by Roar/Whirlwind (phazing), Haze, Taunt, or
  simply hitting the chain hard enough to break it.

### Choosing Your Archetype

There is no "best" archetype. TSS is the most common and well-rounded in Gen 3,
but each has favorable and unfavorable matchups against the others.

---

## 3. Team Building

### Start with a Core

Never build a team by throwing together six individually strong Pokemon. Start
with a **core** of 2-3 Pokemon that synergize well:

- **Offensive core**: Two attackers that cover each other's checks.
  (e.g., a physical wallbreaker + a special wallbreaker that beats the physical
  wallbreaker's switch-ins)
- **Defensive core**: Two walls that cover each other's weaknesses.
  (e.g., a Water-type + a Grass-type, or a physical wall + a special wall)
- **Pivot core**: Pokemon with U-turn/Volt Switch + a wallbreaker they bring in.

### Fill Essential Roles

After your core, fill these roles (a single Pokemon can fill multiple):

1. **Spikes setter** (Spikes is the only entry hazard in Gen 3 — no Stealth Rock)
2. **Rapid Spin user** (hazard removal; or a spinblocker Ghost-type to deny opponent Spin)
3. **Speed control** (Choice Band user, Dragon Dance sweeper, or priority attacker)
4. **Wincon/sweeper** (your primary path to victory)
5. **Defensive check** to top metagame threats (Tyranitar, Salamence, Metagross)

### Type Coverage

- Ensure your team isn't walled by a single type. If three of your Pokemon lose
  to Ground, you have a problem.
- Check for common **type weaknesses**: if half your team is weak to Spikes chip
  damage, you need a reliable Rapid Spinner.
- Don't over-stack resistances at the expense of coverage. Multiple Steel-types
  resist a lot of things but share crippling weaknesses.
- Remember the **type-based physical/special split**: in Gen 3, all Fire/Water/Grass/
  Electric/Ice/Psychic/Dragon/Dark moves are special; all Normal/Fighting/Flying/
  Poison/Ground/Rock/Bug/Ghost/Steel moves are physical.

### The "What Beats My Team?" Test

After building, mentally run through the top 15-20 Pokemon in the Gen 3 OU metagame:

- Can your team handle Tyranitar? Salamence? Metagross? Suicune? Skarmory? Blissey?
- Can you deal with Dugtrio trapping one of your key Pokemon?
- You don't need a hard counter for everything, but you need at least a check
  (a Pokemon that can come in at least once and threaten it out or KO it).
- If a single common Pokemon sweeps your entire team, go back and fix it.

### Common Team Building Mistakes

- **No hazard removal**: Your team bleeds HP every time it switches.
- **No speed control**: Boosted sweepers run through you.
- **Redundant roles**: Two Choice Scarf users don't check twice as many threats.
- **Over-reliance on one Pokemon**: If your only answer to a meta threat is one
  mon, you lose when it goes down.

---

## 4. Pokemon Roles

### Sweeper

A Pokemon whose job is to clean up weakened teams, typically after boosting.
Setup moves: Swords Dance, Dragon Dance, Calm Mind, Nasty Plot, Shell Smash,
Quiver Dance, Agility/Rock Polish.

- **Do not sweep prematurely.** Your sweeper should come in *after* its checks
  and counters have been weakened or removed. Throwing your Salamence into an
  intact team is how you lose games.
- A sweeper should be able to KO at least 2-3 remaining Pokemon once it sets up.

### Wallbreaker

A Pokemon designed to punch massive holes in the opponent's team, usually with
Choice Band/Specs or Life Orb. Wallbreakers clear the way for sweepers.

- Wallbreakers don't need to sweep—they need to 2HKO (two-hit KO) walls.
- Common trait: raw power but middling speed. They come in on forced switches or
  via pivots, fire off a nuke, and often get revenge killed. That's fine—they did
  their job.

### Wall / Defensive Pivot

Pokemon that absorb hits and provide team utility (recovery, hazards, status,
phazing). Defined by high defenses, reliable recovery, and favorable typing.

- **Physical wall**: checks physical attackers. (e.g., Skarmory, Suicune, Milotic)
- **Special wall**: checks special attackers. (e.g., Blissey, Snorlax, Regice)
- **Mixed wall**: handles both somewhat. (e.g., Swampert, Celebi)

### Pivot

A Pokemon that facilitates safe switches through double switching, Baton Pass,
or simply forcing switches with threat presence. Gen 3 does not have U-turn,
Volt Switch, or Flip Turn — pivoting relies more on prediction and defensive
switching rather than dedicated pivot moves.

### Revenge Killer

A fast Pokemon (or priority user) that comes in after one of your Pokemon faints
and KOs the opponent's weakened or boosted threat. Choice Scarf is the defining
item for this role.

### Lead / Suicide Lead

Sets up entry hazards and/or disrupts the opponent's setup on turn 1. Suicide
leads often carry Taunt + hazards and are expendable—they serve their purpose
and faint.

### Cleric / Support

Provides Wish, Heal Bell/Aromatherapy, or screens (Reflect/Light Screen) to
keep the team healthy and functional.

---

## 5. Key Items

### Choice Items

| Item | Boost | Drawback |
|------|-------|----------|
| Choice Band | +50% Attack | Locked into one move |
| Choice Specs | +50% Sp. Atk | Locked into one move |
| Choice Scarf | +50% Speed | Locked into one move |

- **Choice Band**: The defining offensive item of Gen 3. Turn wallbreakers into
  nuclear weapons. Click the right move and 2HKO everything. Click the wrong move
  and your opponent gets a free setup turn. Metagross, Salamence, and Tyranitar
  are premier Choice Band users.
- **Choice Specs**: Does not exist in Gen 3 (introduced Gen 4).
- **Choice Scarf**: Does not exist in Gen 3 (introduced Gen 4).
- **Heuristic**: If a Pokemon has one clear "best move" and primarily wants to hit
  things and switch out, Choice Band is ideal. If it needs to click different
  moves each turn, don't lock it.

### Leftovers

- Recover 1/16 HP per turn. Seemingly small but adds up enormously on defensive
  Pokemon that stick around for many turns.
- **The most common item in Gen 3 OU.** Almost every defensive and bulky Pokemon
  runs Leftovers. The item pool in Gen 3 is much smaller than later gens.
- **Heuristic**: Best on Pokemon that plan to stay in for 4+ turns at a time.

### Lum Berry

- Cures any status condition once (burn, paralysis, sleep, poison, confusion).
- Extremely common in Gen 3 — protects setup sweepers from crippling status,
  especially Toxic and Thunder Wave. Dragon Dance Salamence with Lum Berry
  is a Gen 3 staple.
- **Heuristic**: Best on sweepers that need one clean setup turn.

### Life Orb / Focus Sash / Assault Vest / Heavy-Duty Boots

- **None of these items exist in Gen 3.** They were introduced in later generations.
- Gen 3's item pool is much more limited: Leftovers, Choice Band, Lum Berry,
  and type-boosting items (Charcoal, Mystic Water, etc.) dominate.

---

## 6. Key Abilities

### Sand Stream (Tyranitar)

- Summons permanent sandstorm upon switch-in.
- **The defining ability of Gen 3 OU.** Permanent sandstorm deals 1/16 HP per
  turn to all non-Rock/Ground/Steel types. This passive damage is central to
  the TSS archetype.
- Tyranitar is on roughly half of all Gen 3 OU teams because of Sand Stream alone.

### Intimidate

- Lowers the opponent's Attack by one stage upon switch-in.
- Essentially gives your team a free defensive buff every time you pivot.
- Common users: Salamence, Gyarados.
- Note: Defiant and Competitive do not exist in Gen 3.

### Arena Trap (Dugtrio) / Magnet Pull (Magneton)

- **Arena Trap** prevents grounded non-Flying opponents from switching out.
- **Magnet Pull** prevents Steel-types from switching out.
- Trapping is a Gen 3 signature strategy: Dugtrio eliminates key threats
  (Tyranitar, Blissey after Focus Punch) and Magneton removes Skarmory to
  enable physical sweepers.
- **Heuristic**: If your sweeper is walled by one specific Pokemon, consider
  a trapper to remove it.

### Levitate

- Grants Ground immunity.
- Tactically important in Gen 3 because Earthquake is the most common physical
  move. Common users: Gengar, Claydol (also a Rapid Spinner).

### Natural Cure

- Cures status conditions upon switching out.
- Extremely strong in Gen 3 where status is prevalent and Heal Bell/Aromatherapy
  are rare.
- Common users: Celebi, Blissey, Starmie.

---

## 7. Entry Hazards & Hazard Control

### Why Hazards Matter

Entry hazards are **the single most impactful passive mechanic** in singles.
They punish switching, which is the backbone of good play. Over the course of a
20-30 turn game with constant switching, hazards can deal more total damage than
any individual Pokemon.

### Spikes — The Only Entry Hazard in Gen 3

- Stacks in 3 layers: 12.5% / 16.7% / 25% per switch-in.
- **Does not affect Flying-types or Levitate users.**
- **Stealth Rock does not exist in Gen 3** (introduced Gen 4). Spikes is the
  only entry hazard. Toxic Spikes and Sticky Web also do not exist.
- Spikes is central to the Gen 3 metagame — the TSS archetype revolves around
  stacking Spikes under permanent sandstorm for devastating residual damage.
- Common setters: Skarmory (premier), Forretress, Cloyster, Smeargle.

### Hazard Removal: Rapid Spin Only

- **Defog does not exist as a hazard removal move in Gen 3** (introduced Gen 4).
- **Rapid Spin** is the sole method of removing Spikes. This makes spinners
  extremely valuable and spin-blocking extremely important.
- Common spinners: Starmie (fast, strong), Claydol (bulky, Levitate),
  Forretress (can both set and spin).

### Spin-Blocking

- Ghost-types are immune to Rapid Spin and prevent it from working.
- Keeping a Ghost-type alive to block Rapid Spin is a core Gen 3 strategy.
- Common spinblockers: Gengar (offensive), Dusclops (defensive).
- **Heuristic**: If your opponent has Spikes and a spinblocker, you need either
  to KO the spinblocker first or accept the Spikes damage.

### The Spikes Game

The Spikes game defines Gen 3 OU:
1. Your opponent sets Spikes → you try to Rapid Spin them away.
2. Their Ghost-type blocks your spin → you need to eliminate or lure the Ghost.
3. You set your own Spikes → they try to spin.
4. This back-and-forth is the strategic backbone of many Gen 3 games.

---

## 8. Status Conditions

### Burn (Will-O-Wisp, Scald)

- **Halves Attack** and deals 1/16 max HP per turn.
- The single best answer to physical sweepers. A burned physical attacker is
  almost useless.
- Scald is notoriously strong because it's a decently powerful Water-type attack
  (80 BP) with a **30% burn chance**—making it both an offensive move and a
  defensive tool.
- Fire-types are immune to burn.
- **Heuristic**: If a physical attacker is setting up on you and you can't KO it,
  burning it is often the next best option.

### Paralysis (Thunder Wave, Body Slam, Stun Spore)

- **Cuts Speed to 25% of original** and inflicts a 25% chance of full paralysis
  each turn.
- Cripples fast sweepers permanently. A paralyzed Salamence is no longer a threat
  from a speed perspective.
- Ground-types are immune to Thunder Wave (but not Body Slam or Stun Spore).
- Note: Electric-types are NOT immune to paralysis in Gen 3 (immunity added Gen 6).
- Body Slam's 30% paralysis chance makes it a staple on Snorlax and other
  Normal-types in Gen 3.
- **Heuristic**: Use Thunder Wave against fast offensive threats. Less useful
  against slow walls (they don't care about Speed loss).

### Poison / Toxic (Toxic, Toxic Spikes)

- Regular poison: 1/8 HP per turn.
- Toxic (badly poisoned): 1/16 HP escalating each turn (1/16, 2/16, 3/16...).
- Puts Pokemon on a **death timer**. Walls with recovery can outlast Toxic'd
  opponents by simply switching and stalling.
- Poison- and Steel-types are immune to poison.
- **Heuristic**: Toxic is strongest against defensive Pokemon that plan to stay in
  for many turns. Against fast attackers that switch often, paralysis or burn is
  usually better.

### Sleep (Spore, Sleep Powder, Hypnosis)

- In Gen 3, sleep lasts 1-5 turns and the counter resets if the Pokemon switches
  out and back in. This makes sleep much more punishing than in later gens.
- **Sleep Clause**: In Smogon formats, only one Pokemon per team can be asleep at
  a time (via opponent's moves). This limits sleep's abuse.
- Essentially a temporary removal of a Pokemon from the game.
- Note: Grass-types are NOT immune to powder moves in Gen 3 (immunity added Gen 6).

### Freeze

- Uncontrollable (no move inflicts freeze intentionally at high rates).
- Essentially shuts down a Pokemon until it thaws (20% chance per turn, or by
  using a Fire-type move).
- Not something you can build around, but something you must accept as variance.

### Status Immunity Notes (Gen 3)

| Status | Immune Types/Abilities |
|--------|----------------------|
| Burn | Fire-types |
| Poison | Poison-types, Steel-types |
| Paralysis | Ground-types (vs. T-Wave only), Limber |
| Sleep | Vital Spirit, Insomnia |
| All | Safeguard |

Note: Many immunities from later gens (Electric immune to paralysis, Grass immune
to powder) do not exist in Gen 3.

---

## 9. Pivoting & Momentum

### What Is Momentum?

Momentum is **having the favorable matchup on the field**. When your active Pokemon
threatens the opponent's active Pokemon, you have momentum—they are forced to
react. When the reverse is true, they have momentum.

Games are a constant tug-of-war for momentum. The player who maintains momentum
longer generally wins.

### Pivoting in Gen 3

Gen 3 does **not** have U-turn, Volt Switch, Flip Turn, or Teleport. These moves
were introduced in Gen 4 or later. Pivoting in Gen 3 is fundamentally different
from modern generations:

**How pivoting works in Gen 3:**
- **Baton Pass**: The primary "pivot move" in Gen 3, though mostly used for stat
  passing rather than pure momentum pivoting.
- **Double switching**: Predicting the opponent's switch and switching yourself
  to gain the favorable matchup. This is the main way to seize momentum.
- **Forced switches**: Using a Pokemon that threatens the opponent's active mon
  so heavily that they must switch, then predicting the switch-in and attacking
  accordingly.

### Momentum in Gen 3

Momentum is **having the favorable matchup on the field**. Without pivot moves,
momentum in Gen 3 is managed through:

1. **Prediction-based switching**: Anticipating the opponent's switch and bringing
   in the correct counter.
2. **Threat stacking**: Having multiple threats so the opponent cannot easily
   answer all of them.
3. **Residual damage**: Spikes + sandstorm chip means every switch costs HP,
   making momentum shifts more expensive for the switching player.

### The Momentum Paradox

Ironically, **KOing an opponent's Pokemon can lose you momentum**. When you score
a KO, your opponent gets a **free switch**—they choose who comes in, potentially
picking a Pokemon that threatens your attacker. Be aware of what your opponent
might bring in after a KO and position accordingly.

---

## 10. Speed Control

### Why Speed Matters

The faster Pokemon moves first. Moving first means you can KO before being hit,
apply pressure before a switch, or set up before being disrupted. Speed is the
most important stat in offensive play.

### Speed Tiers (Gen 3 OU)

The "speed tier" is where a Pokemon falls relative to the metagame's speed
benchmarks. Key Gen 3 OU thresholds:

- **Base 110+** (Starmie 115, Aerodactyl 130, Jolteon 130): Naturally fast.
- **Base 90-100** (Salamence 100, Celebi 100, Gengar 110): The competitive middle.
  Dragon Dance users often start here.
- **Base 70-80** (Tyranitar 61, Metagross 70, Heracross 85): Needs a boost to sweep.
- **Base 50 and below** (Snorlax 30, Skarmory 70, Blissey 55): Walls; not
  outspeeding anything.

### Speed Control Methods (Gen 3)

1. **Dragon Dance**: +1 Atk, +1 Spe. The premier setup move in Gen 3.
   Salamence and Tyranitar are the main users.
2. **Agility / Rock Polish**: +2 Speed. Metagross with Agility is a notable threat.
   (Note: Rock Polish does not exist in Gen 3, only Agility.)
3. **Priority moves**: Limited pool in Gen 3.
   - Extreme Speed (+2): Linoone is the notable user.
   - Mach Punch (+1): Breloom, Hitmonchan.
   - Quick Attack (+1): Generally too weak to matter.
   - Note: Sucker Punch, Bullet Punch, Ice Shard, Aqua Jet do NOT exist in Gen 3.
4. **Thunder Wave / Paralysis**: Permanent 75% speed cut. Very common in Gen 3.
5. **Choice Scarf does NOT exist in Gen 3** (introduced Gen 4). This makes speed
   control more limited — Dragon Dance and paralysis are the primary tools.
6. **Trick Room does NOT exist in Gen 3** (introduced Gen 4).

### Speed Control Heuristics

- Every team needs at least one form of speed control. Without it, a +1 Dragon
  Dance Salamence runs through you unchecked.
- Dragon Dance + paralysis support gives you two independent speed control layers.
- Know your speed tiers. Know what outspeeds what at +1 Dragon Dance.
  Miscalculating speed tiers is a common cause of lost games.

---

## 11. Switching & Prediction

### The Switching Metagame

Competitive singles involves **constant switching**. In a typical game, 20-40%
of all turns involve at least one player switching. This is fundamentally
different from in-game play where you rarely switch.

**Why switch?**
- Your current Pokemon has an unfavorable matchup.
- You want to preserve a Pokemon for later.
- You're bringing in a check to the opponent's threat.
- You're pivoting to maintain momentum.

**Cost of switching:**
- You lose your turn (no attack, no setup).
- You eat entry hazard damage.
- The incoming Pokemon takes whatever attack the opponent uses.
- You reveal information (what you switched to and possibly why).

### Types of Predictions

1. **"Safe" prediction**: Clicking the move that's best against the opponent's
   *current* Pokemon, regardless of what they might switch to. Low risk, moderate
   reward.
2. **"Read" prediction**: Clicking a move aimed at what you think the opponent
   will *switch into*. High risk, high reward.
3. **Double switch**: Switching out on the same turn you think the opponent will
   switch. You both end up with new Pokemon on the field—if you predicted right,
   you have the favorable matchup.
4. **Stay-in prediction**: You think the opponent expects *you* to switch, so you
   stay in and use a move that punishes their response (e.g., Pursuit trapping,
   or setting up as they switch to a wall).

### Prediction Heuristics

1. **Default to safe plays.** In early game, when you know little about the
   opponent, the safe play is almost always correct. You haven't earned the
   information needed to predict accurately.
2. **Predict only when the reward justifies the risk.** If hitting the switch-in
   wins you the game, go for it. If it just does chip damage, it's not worth
   risking a free turn for the opponent.
3. **Look for forced plays.** Sometimes your opponent has no good options except
   one—that's when predictions are most reliable. (e.g., they have only one
   Pokemon that resists your STAB—it's coming in.)
4. **Don't over-predict.** A common pitfall is predicting your opponent's
   prediction of your prediction. This infinite recursion is a trap. At lower
   levels of play, most opponents make the obvious play. At higher levels,
   the game becomes more about reads.
5. **Pattern recognition matters.** If an opponent has switched Skarmory into
   your physical wallbreaker three times, they'll probably do it a fourth. That's
   when you predict with a Fire-type or Electric-type coverage move.
6. **Consider the opponent's perspective.** What would *you* do in their position?
   What is the "correct" play for them? That's likely what they'll do.

### Switching Heuristics

1. **Don't switch into attacks you can't take.** This seems obvious, but many
   players switch into a "check" that can't actually survive the hit. Always know
   your damage ranges.
2. **Switch early, not late.** If a Pokemon needs to come in to check a threat,
   bring it in while it's healthy. Waiting until it's at 30% HP makes it useless
   as a check.
3. **Keep your checks healthy.** If your Skarmory is your only answer to Salamence,
   don't recklessly let it take unnecessary chip. Preserve it.
4. **Don't be afraid to sack.** Sometimes the right play is to let a less
   important Pokemon faint rather than switch and take damage on your key mon.
   This is called a **sacrifice** (or "sack"), and it's a legitimate tactic.

---

## 12. Risk & Reward

### The Risk Framework

Every turn presents a decision with varying levels of risk and reward. A
disciplined player evaluates each option through this lens:

| Situation | Recommended Approach |
|-----------|---------------------|
| Winning comfortably | Play safe. Don't give your opponent comeback opportunities. |
| Roughly even | Play mostly safe, with occasional reads on high-value turns. |
| Slightly behind | Start taking calculated risks to shift momentum. |
| Significantly behind | Go aggressive. A loss is a loss—you have nothing to lose by swinging. |

### When to Play Safe

- **Early game**: You don't know the opponent's full team or sets. Default to
  moves that are good regardless of what they do.
- **When ahead**: Protect your advantage. Don't throw the game on a prediction.
- **When your remaining Pokemon cleanly handle theirs**: Just play it out.

### When to Take Risks

- **When behind and the "safe" play leads to a slow loss**: If playing safe means
  you lose to their Tyranitar in 5 turns, take a risk now.
- **When one read wins the game**: If hitting their switch-in eliminates their
  only check to your sweeper, go for it.
- **On forced turns**: After your Pokemon faints, you get a free switch—use this
  to set up a dangerous Pokemon rather than just plugging the gap.

### The Over-Prediction Trap

Over-prediction occurs when a player tries to be "clever" by predicting the
opponent's prediction:

> "They know I'll use Fire Blast on their Corviknight switch, so they'll stay
> in with their Fire-type... so I'll use Earthquake instead."

This reasoning can go infinitely deep and is usually wrong. The reality:

- At low-to-mid level play, opponents usually make the **straightforward** play.
- At high level play, opponents randomize to make themselves harder to read.
- In both cases, over-predicting is a losing strategy most of the time.

**Rule**: If the safe play is +EV (positive expected value) across all reasonable
opponent options, just do the safe play.

---

## 13. Win Conditions & Endgame

### Identifying Your Win Condition

Throughout the game (no team preview in Gen 3), as the opponent reveals their
team, continuously ask: **"Which of my Pokemon can their revealed team least handle?"**

That Pokemon is your win condition. Everything else on your team should work
toward getting it into a position to sweep:

1. Remove or weaken its checks/counters (via wallbreakers or chip damage).
2. Set up hazards to put opponents in KO range.
3. Clear the field of priority users or Scarfers that revenge kill it.
4. Find a safe opportunity to bring it in and set up.

### Sweeping Protocol

A sweep is not improvised—it's the culmination of a plan executed over many turns.

1. **Identify what stops your sweeper**: Which opposing Pokemon can take a hit
   and KO back? Which can outspeed and revenge kill?
2. **Eliminate those threats**: Use your other Pokemon to weaken or remove them.
   Don't start your sweep until the coast is clear.
3. **Choose the right moment**: Set up when the opponent is forced to use a weak
   attack, when they're switching, or when their check has been removed.
4. **Don't get greedy**: Sometimes +1 Dragon Dance is enough to sweep. Don't go
   for +2 if it means risking a crit or a faster revenge killer.

### When Your Win Condition Fails

If your planned sweeper goes down or gets crippled, you need a **backup plan**:

- Can another Pokemon clean up late-game?
- Can you win through residual damage (hazards + status)?
- Can you stall out their remaining PP?
- Can you play the remaining turns to trade favorably 1-for-1?

Good players always have a Plan B. Great players identify when Plan A has failed
and switch to Plan B before it's too late.

### Endgame Decision Trees

In the endgame (2-3 Pokemon remaining each side), every turn is critical:

- **Count remaining Pokemon and HP**. Know exactly what can KO what.
- **Speed matters most now.** If you're faster, you win the 1v1. If not, you need
  priority or a switch play.
- **Don't throw away Pokemon unnecessarily.** Every remaining mon is precious.
  Sacking in the endgame is usually wrong unless it directly enables a win.
- **Hazards are devastating in endgame.** A Pokemon at 60% HP switches into rocks,
  drops to 35%, and is now in KO range of everything.

---

## 14. Gen 3-Specific Mechanics

### The Physical/Special Split by Type

In Gen 3, whether a move is physical or special is determined by its **type**,
not the individual move:

| Category | Types |
|----------|-------|
| **Physical** | Normal, Fighting, Flying, Poison, Ground, Rock, Bug, Ghost, Steel |
| **Special** | Fire, Water, Grass, Electric, Ice, Psychic, Dragon, Dark |

This has major implications:
- A Pokemon with high Attack but a special-type STAB (e.g., Gyarados with Water)
  cannot use its STAB moves effectively from the physical side.
- Mixed attackers are rare because type dictates category. You can't have a
  "physical Fire move" — all Fire moves are special.
- This creates unique niches: Swampert uses Earthquake (physical Ground) as its
  primary STAB despite also being Water-type, because Water moves are special
  and Swampert's Attack is higher.

### Permanent Weather

Weather from abilities lasts **indefinitely** in Gen 3 (in later gens it lasts
5 turns):
- **Sand Stream** (Tyranitar): Permanent sandstorm. The defining weather of Gen 3 OU.
  Deals 1/16 HP per turn to non-Rock/Ground/Steel types.
- **Drizzle** (Kyogre) and **Drought** (Groudon): Banned in OU (Ubers only).
- Manual weather (Rain Dance, Sunny Day) still lasts 5 turns and can override
  Sand Stream, but Tyranitar can simply switch back in to reset sandstorm.

### No Team Preview

Gen 3 has no team preview. You enter the battle knowing nothing about your
opponent's team. This means:
- **Lead selection is blind** — you pick your lead based on metagame trends,
  not specific team information.
- **Scouting is critical** — the first few turns are about gathering information.
  Safe plays early are more valuable than aggressive predictions.
- **Hidden information is maximal** — your opponent's full team, items, moves,
  and abilities are all unknown until revealed through gameplay.

### Key Moves That Don't Exist in Gen 3

Many staples of modern competitive Pokemon do not exist:
- **No Stealth Rock** (introduced Gen 4)
- **No U-turn, Volt Switch, Flip Turn** (introduced Gen 4+)
- **No Scald** (introduced Gen 5)
- **No Knock Off** as a viable damage move (only 20 BP in Gen 3)
- **No Defog** as hazard removal (introduced Gen 4)
- **No Sucker Punch, Bullet Punch, Ice Shard, Aqua Jet** (introduced Gen 4)
- **No Choice Scarf or Choice Specs** (introduced Gen 4)

---

## 15. Common Mistakes & Bad Tactics

### Team Building Errors

1. **All-offense teams**: Filling your team with sweepers sounds great until a
   single Dragon Dance Salamence at +1 sweeps your entire team because nothing
   can take a hit. You need defensive checks.
2. **No Rapid Spinner**: Spikes stack up fast. If you can't remove them, your
   Pokemon lose 12-25% HP every time they switch in. Over a 20-turn game,
   that's lethal — especially under permanent sandstorm.
3. **Type-stacking**: Three Water-types on one team means a single Grass- or
   Electric-type can check half your team. Diversify your typing.
4. **Building around a single gimmick**: Baton Pass chains or Rain Dance teams
   can be devastating when they work but crumble when the opponent has a single
   counter (Roar, Whirlwind, Tyranitar's Sand Stream). Have a Plan B.
5. **Ignoring the metagame**: Building your team in a vacuum without considering
   what's popular leads to teams that auto-lose to common threats like Tyranitar.

### In-Battle Errors

6. **Attacking when you should switch**: Your Grass-type vs. their Fire-type.
   You click Leaf Blade because "maybe it'll do enough." It won't. Switch.
7. **Switching when you should attack**: Your faster Pokemon can KO theirs, but
   you switch out of fear. You just gave them a free turn to set up or pivot.
8. **Setting up at the wrong time**: Going for Swords Dance when the opponent
   has a faster Pokemon that can KO you. Setup requires a **safe turn**—don't
   force it.
9. **Ignoring damage calcs**: "I think I survive that." Do you? Run the numbers.
   Competitive play rewards precision, not guesswork.
10. **Over-predicting** (see Section 12): Trying to be clever when the simple
    play is correct. The straightforward move is right more often than the
    galaxy-brain read.
11. **Throwing away your win condition**: Your +1 Salamence is your path to
    victory, so you recklessly leave it in against a potential Ice Beam.
    Protect your win condition at all costs.
12. **Not sacking when you should**: Sometimes a Pokemon has served its purpose.
    Letting it faint to preserve a more valuable teammate is correct. Don't
    waste recovery or switches saving a 15% HP Pokemon with no further utility.

### Mentality Errors

13. **Blaming luck**: Yes, crits and misses happen. But over hundreds of games,
    luck evens out. If you're losing consistently, the issue is your play, not
    your luck. Ask what you could have done differently.
14. **Playing on tilt**: After a bad loss or unlucky game, take a break. Tilted
    players make increasingly reckless decisions and chain losses.
15. **Autopiloting**: Clicking the same move every turn without thinking about
    what the opponent is doing. Competitive play requires active engagement
    every single turn.
16. **Copying teams without understanding them**: Pasting a top-ladder team
    without understanding its game plan or matchup spreads leads to poor play.
    Learn *why* each Pokemon and move is there.
17. **Refusing to adapt**: "I always lead with X and set up rocks." If the
    opponent has a Taunt lead, your plan needs to change. Rigid play is
    exploitable play.

---

## 16. Advanced Heuristics & Rules of Thumb

### Damage and KOs

- If you can **OHKO** the opponent's active Pokemon and it can't OHKO you, click
  the attacking move. Don't overthink.
- If you **2HKO** them and they 3HKO you, staying in and attacking twice is
  usually correct (unless they have recovery or a switch-in).
- If neither side can make progress (e.g., wall vs. wall), don't waste turns.
  Switch to a wallbreaker or use a status move.

### Switching Logic

- **Switch to your best available check**, not necessarily your "counter."
  A check needs to handle the threat once; a counter can handle it repeatedly.
  In many games, checking once is enough.
- **Rotate your checks.** Don't send the same Pokemon into the same threat
  repeatedly if it's taking heavy chip. Spread the damage across multiple
  resistances when possible.
- **Consider what happens *after* you switch.** Bringing in Skarmory against
  the Heracross is great—but what if they predict and use Hidden Power Fire?
  Think one step ahead.

### Resource Management

- **HP is a resource, not a health bar.** Taking 30% damage to deal 60% is a
  good trade. Taking 30% to accomplish nothing is a waste. Every point of HP
  should be spent on something — especially in Gen 3 where sandstorm + Spikes
  chip makes every HP point precious.
- **PP is a resource.** In stall matchups, tracking the opponent's remaining PP
  on key moves (recovery, coverage) can determine who wins. PP stalling is a
  legitimate Gen 3 strategy.
- **Your Pokemon are a resource.** You have six. Losing one to gain a massive
  positional advantage is often correct.

### Opponent Modeling

- **Play the metagame, not the opponent.** Early in a game, assume standard sets
  and standard play. Adjust as you gather information. In Gen 3 without team
  preview, this means assuming your opponent has common threats until proven
  otherwise.
- **Pay attention to what they reveal.** If their Metagross uses Agility on
  turn 3, it's not Choice Band. If their Tyranitar takes 35% from a neutral
  special hit, it's probably specially defensive.
- **Look for what's missing.** As Pokemon are revealed, note what they're missing.
  No Rapid Spinner seen yet? Spikes will be devastating. No obvious
  special wall? Your special attackers can go to work.
- **Respect coverage.** "Tyranitar is walled by my Swampert" is true until
  Tyranitar reveals Hidden Power Grass. Don't rely on a single check without
  considering the possibility of coverage moves.

### Positional Play

- **Trades favor the player with more remaining win conditions.** If you have two
  ways to win and they have one, trading Pokemon 1-for-1 is great for you.
- **Don't trade your check for their non-threat.** If your Skarmory trades with
  their Forretress but their Salamence is still alive, you just lost the game.
- **Force 50/50s when behind, avoid them when ahead.** If you're winning, don't
  give the opponent a coin-flip opportunity to equalize. If you're losing,
  50/50s are better than a guaranteed slow loss.
- **Momentum > damage.** Sometimes dealing less damage but maintaining the
  favorable matchup (via pivoting) is better than dealing more damage but
  giving the opponent a free switch.

### Pattern Recognition

- **Most opponents follow patterns.** They switch the same Pokemon into the same
  threat, they lead with the same Pokemon. Exploit these patterns when you spot them.
- **Varying your own patterns is equally important.** If you always switch
  Skarmory into their Heracross, they'll start predicting it and using a
  coverage move. Mix it up.
- **The most common play is the most common play for a reason.** It's usually
  correct. Don't be contrarian for the sake of it.

### The Art of Doing Nothing

Sometimes the best play is the least flashy:

- **Protect** (on Pokemon that have it) scouts the opponent's move risk-free.
- **Recover/Roost** when the opponent can't threaten you buys time and preserves
  your check.
- **Switching to a resist** when neither side can make progress resets the
  position without risk.

Not every turn needs to be a power move. Patient, methodical play beats
aggressive, reckless play in the long run.

---

## 17. Sources

### Gen 3 OU-Specific Resources
- [ADV OU Overview - Smogon Strategy Dex](https://www.smogon.com/dex/rs/formats/ou/)
- [Introduction to Competitive Pokemon - Smogon University](https://www.smogon.com/dp/articles/intro_comp_pokemon)
- [Getting Started with Competitive Battling - Smogon University](https://www.smogon.com/articles/getting-started)

### General Competitive Strategy
- [Teambuilding Guide - Smogon Forums](https://www.smogon.com/forums/threads/teambuilding-guide.3552468/)
- [Risk/Reward - Smogon University](https://www.smogon.com/bw/articles/bw_risk_reward)
- [Momentum - Smogon Forums](https://www.smogon.com/forums/threads/momentum.3450730/)
- [Battle Tactics: Double Switches - Smogon Forums](https://www.smogon.com/forums/threads/battle-tactics-double-switches.51714/)
- [Entry Hazards in OU - Smogon University](https://www.smogon.com/smog/issue44/entry-hazards-in-ou)
- [Status in RU - Smogon University](https://www.smogon.com/smog/issue26/ru_status)
- [Boosting Them Up: Setup Sweepers in OU - Smogon University](https://www.smogon.com/articles/ou-setup-sweepers)
- [Pokemon Battling: A Lesson in Game Theory - Smogon Forums](https://www.smogon.com/forums/threads/pokemon-battling-a-lesson-in-game-theory.3492697/)
- [Competitive Battle Tactics Guide - Pokemon Lazarus](https://pokemonlazarus.org/walkthrough/competitive-battle-tactics)
- [Predictions in Competitive Battling - PokeCommunity Daily](https://daily.pokecommunity.com/2016/06/04/predictions-competitive-battling/)
- [Common Mistakes in Competitive Pokemon - PokeBeach](https://www.pokebeach.com/2024/02/the-most-common-mistakes-in-competitive-pokemon-and-how-to-avoid-them)
- [Pokemon Roles Explained - Pokestrat](https://pokestratbuilder.com/blog/roles-explained)
- [How to Build a Competitive Pokemon Team - Pokestrat](https://pokestratbuilder.com/blog/pokemon-team)
