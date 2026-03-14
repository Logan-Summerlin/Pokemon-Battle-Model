# Competitive Pokemon Singles Battle Strategy Guide

A concise reference for competitive 6v6 OU singles on Pokemon Showdown.
Focused on decision-making, tactics, heuristics, and common pitfalls.

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
14. [Terastallization (Gen 9)](#14-terastallization-gen-9)
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

Pokemon singles is a game of **incomplete information**. You can see what species
your opponent has at Team Preview, but you don't know their:

- Exact EV spreads and natures
- Held items
- Full movesets (until revealed)
- Tera type (Gen 9)
- Planned strategy and win condition

Good players extract information incrementally—through observed moves, damage
calculations, switching patterns, and item reveals—and update their strategy
accordingly. Resist the urge to "figure everything out" on turn 1. Information
flows naturally over the course of a game.

---

## 2. Team Archetypes

Teams fall along a spectrum from most aggressive to most defensive:

### Hyper Offense (HO)

- **Goal**: Overwhelm the opponent with relentless pressure before they stabilize.
- **Structure**: Suicide lead (hazards + Taunt/Explosion), 4-5 setup sweepers
  or wallbreakers, minimal defensive investment.
- **Strengths**: Punishes passive play, forces opponent onto the back foot.
- **Weaknesses**: Fragile; one missed prediction or unlucky turn can unravel the
  whole game. Struggles against well-built stall.
- **Key principle**: Every turn you're not attacking or setting up, you're losing.

### Offense

- **Goal**: Apply strong pressure while maintaining a thin defensive safety net.
- **Structure**: Wallbreakers, sweepers, and 1-2 defensive pivots or checks.
- **Strengths**: Flexibility—can play fast or slow depending on matchup.
- **Weaknesses**: Less forgiving than balance if a pivot goes down early.

### Bulky Offense (BO)

- **Goal**: Wear down the opponent gradually while hitting hard.
- **Structure**: Tanky attackers ("tanks") that can take hits and dish them out,
  mixed with pivots and hazard support.
- **Strengths**: Excellent at trading blows; harder to sweep than HO.
- **Weaknesses**: Can lack the raw speed to revenge kill boosted threats.

### Balance

- **Goal**: Maintain stability while making incremental progress.
- **Structure**: Defensive core + offensive threats, hazard setter + hazard
  removal, status spreaders, pivots.
- **Strengths**: The most flexible archetype; can adapt to most matchups.
- **Weaknesses**: Games tend to be longer; vulnerable to skilled wallbreakers
  that punch through the defensive core.
- **Key principle**: Patience. You don't need to win on turn 5.

### Stall / Semi-Stall

- **Goal**: Outlast the opponent through recovery, residual damage, and PP stalling.
- **Structure**: Walls with reliable recovery, hazard stackers, status spreaders
  (Toxic, Will-O-Wisp), cleric support.
- **Strengths**: Extremely hard to break for unprepared teams.
- **Weaknesses**: Taunt, Trick, Encore, and dedicated wallbreakers shut it down.
  Requires deep knowledge of damage ranges and when to recover vs. switch.
- **Key principle**: You win by not losing. The opponent runs out of resources
  before you do.

### Choosing Your Archetype

Your archetype should match your understanding and comfort level:

- If you like fast-paced, read-heavy games: Offense or HO.
- If you want a forgiving, adaptable style: Balance.
- If you enjoy grinding opponents down: Stall.

There is no "best" archetype. Each has favorable and unfavorable matchups against
the others. The best players can pilot multiple styles.

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

1. **Hazard setter** (Stealth Rock is near-mandatory)
2. **Hazard removal** (Defog or Rapid Spin user)
3. **Speed control** (Choice Scarf user, priority attacker, or both)
4. **Wincon/sweeper** (your primary path to victory)
5. **Defensive check** to top metagame threats

### Type Coverage

- Ensure your team isn't walled by a single type. If three of your Pokemon lose
  to Ground, you have a problem.
- Check for common **type weaknesses**: if half your team is weak to Stealth Rock,
  you need Boots users or a reliable Defogger.
- Don't over-stack resistances at the expense of coverage. Six Steel-types resist
  a lot of things but share crippling weaknesses.

### The "What Beats My Team?" Test

After building, mentally run through the top 15-20 Pokemon in the metagame:

- Can your team handle Kingambit? Gholdengo? Great Tusk? Dragapult?
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
  and counters have been weakened or removed. Throwing your Dragonite into an
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

- **Physical wall**: checks physical attackers. (e.g., Skarmory, Toxapex)
- **Special wall**: checks special attackers. (e.g., Blissey, Clodsire)
- **Mixed wall**: handles both somewhat. (e.g., Slowbro, Corviknight)

### Pivot

A Pokemon that facilitates safe switches using U-turn, Volt Switch, Flip Turn,
or Teleport. Pivots maintain momentum by allowing you to bring in the right
Pokemon without taking unnecessary damage on a raw switch.

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

- **Choice Band/Specs**: Turn wallbreakers into nuclear weapons. Click the right
  move and 2HKO everything. Click the wrong move and your opponent gets a free
  setup turn.
- **Choice Scarf**: The premier revenge-killing item. Turns mid-speed Pokemon into
  speedsters. Pairs beautifully with U-turn/Volt Switch to avoid being trapped in
  a bad move.
- **Heuristic**: If a Pokemon has one clear "best move" and primarily wants to hit
  things and switch out, a Choice item is ideal. If it needs to click different
  moves each turn, don't lock it.

### Leftovers / Black Sludge

- Recover 1/16 HP per turn. Seemingly small but adds up enormously on defensive
  Pokemon that stick around for many turns.
- Black Sludge is Leftovers for Poison-types, with the added benefit of punishing
  Trick users.
- **Heuristic**: Best on Pokemon that plan to stay in for 4+ turns at a time.

### Heavy-Duty Boots

- Grants **immunity to all entry hazards** (Stealth Rock, Spikes, Toxic Spikes,
  Sticky Web).
- Near-mandatory on Pokemon with a Stealth Rock weakness (4x Rock-weak Pokemon
  like Volcarona, Charizard) and on pivots/walls that switch in frequently.
- **Heuristic**: If a Pokemon switches in more than 2-3 times per game and
  hazards are common in the meta, consider Boots.

### Life Orb

- +30% damage to all attacks at the cost of 10% HP per attack.
- Best on Pokemon that need the flexibility to switch moves but still want
  significant power. A middle ground between Choice items and no item.

### Assault Vest

- +50% Sp. Def, but can only use attacking moves.
- Turns physically bulky attackers into mixed tanks.
- **Heuristic**: Good on Pokemon with wide coverage and decent bulk but that
  don't need status or support moves.

### Focus Sash

- Survives any single attack from full HP with 1 HP remaining.
- Staple on suicide leads and frail sweepers that need one guaranteed setup turn.
- Useless if hazards are up on your side—another reason hazard control matters.

---

## 6. Key Abilities

### Regenerator

- Restores ~33% HP upon switching out.
- **Why it's elite**: Pairs perfectly with pivoting. A Regenerator Pokemon can
  switch in, take a hit, pivot out, and come back almost as healthy as before.
- Common users: Slowbro, Slowking, Toxapex, Tangrowth, Amoonguss.
- **The Regenerator Core**: Pairing two Regenerator Pokemon creates a defensive
  pivot loop that is extremely hard to break.

### Intimidate

- Lowers the opponent's Attack by one stage upon switch-in.
- Essentially gives your team a free defensive buff every time you pivot.
- Common users: Landorus-Therian, Gyarados, Incineroar.
- Countered by Defiant (Kingambit) and Competitive (Milotic), which boost stats
  when their stats are lowered—be aware of this before mindlessly Intimidate-cycling.

### Magic Bounce

- Reflects status moves (Stealth Rock, Toxic, Spikes, etc.) back at the opponent.
- A Magic Bounce Pokemon on the team prevents hazard setup entirely by its mere
  presence, even from the bench—opponents know they risk bouncing their own rocks.
- Common users: Hatterene, Espeon, Mega Diancie.

### Levitate

- Grants Ground immunity.
- Tactically important because it can remove what would otherwise be a key
  weakness, especially after Terastallization.

### Unaware

- Ignores the opponent's stat changes when calculating damage.
- The ultimate anti-sweeper ability. A +6 boosted attacker does normal damage.
- Common users: Clodsire, Skeledirge, Quagsire.
- **Heuristic**: If your team lacks checks to setup sweepers, an Unaware wall
  is a reliable safety valve.

---

## 7. Entry Hazards & Hazard Control

### Why Hazards Matter

Entry hazards are **the single most impactful passive mechanic** in singles.
They punish switching, which is the backbone of good play. Over the course of a
20-30 turn game with constant switching, hazards can deal more total damage than
any individual Pokemon.

### Stealth Rock

- Deals damage based on type effectiveness vs. Rock.
  - 4x weak (Fire/Flying, Bug/Flying, etc.): **50% HP** per switch-in.
  - 2x weak: 25% HP per switch-in.
  - Neutral: 12.5%.
  - 1x resistant: 6.25%.
  - 4x resistant: 3.125%.
- **Near-universal on competitive teams.** If you're not running Stealth Rock,
  you need a very good reason.
- Forces opponents to either remove hazards (costing a turn) or accept the chip.

### Spikes

- Stacks in 3 layers: 12.5% / 16.7% / 25% per switch-in.
- **Does not affect Flying-types or Levitate users.**
- Most effective on stall/semi-stall teams that force many switches.
- Stealth Rock + 2-3 layers of Spikes makes switching devastating.

### Toxic Spikes

- 1 layer: poisons grounded switch-ins. 2 layers: badly poisons them.
- Grounded Poison-types absorb and remove Toxic Spikes by switching in.
- Less universally useful than Stealth Rock/Spikes but devastating against
  teams reliant on recovery-less Pokemon.

### Sticky Web

- Lowers Speed by 1 stage for grounded Pokemon switching in.
- Enables slower offensive teams to outspeed threats they normally wouldn't.
- Niche but powerful on dedicated Sticky Web teams.

### Hazard Removal: Defog vs. Rapid Spin

| | Defog | Rapid Spin |
|---|---|---|
| **Removes** | All hazards on both sides | Hazards on your side only |
| **Blocked by** | Taunt | Ghost-types |
| **Side effect** | Removes your hazards and screens | Boosts Speed (Gen 8+) |
| **Punished by** | Defiant/Competitive (stat boost) | Ghost-type switch-ins |

- **Defog** is more universally available and can't be blocked by type immunity,
  but it removes your own hazards—painful if you invested turns setting them up.
- **Rapid Spin** preserves your hazards but is blocked entirely by Ghost-types
  (unless the spinner has Mold Breaker or a Ghost-hitting move).
- **Heuristic**: On offensive teams that rely on their own hazards, prefer Spin.
  On teams that mainly need to keep their side clean, Defog is fine.

### Hazard-Denial Strategies

- **Magic Bounce**: Reflects Stealth Rock back. Opponent can't set up.
- **Taunt**: Prevents hazard moves from being used.
- **Spin-blocking**: Keeping a Ghost-type alive to block Rapid Spin.
- **Defiant/Competitive**: Punishes Defog with an Attack/Sp.Atk boost.

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

### Paralysis (Thunder Wave, Glare, Stun Spore)

- **Cuts Speed to 25% of original** and inflicts a 25% chance of full paralysis
  each turn.
- Cripples fast sweepers permanently. A paralyzed Dragapult is no longer a threat
  from a speed perspective.
- Ground-types are immune to Thunder Wave (but not Glare or Stun Spore).
- Electric-types are immune to paralysis (Gen 6+).
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

- Prevents all action for 1-3 turns (Gen 5+) or until the Pokemon switches out.
- **Sleep Clause**: In Smogon formats, only one Pokemon per team can be asleep at
  a time (via opponent's moves). This limits sleep's abuse.
- Essentially a temporary removal of a Pokemon from the game.
- Grass-types are immune to powder moves (Gen 6+).

### Freeze

- Uncontrollable (no move inflicts freeze intentionally at high rates).
- Essentially shuts down a Pokemon until it thaws (20% chance per turn, or by
  using a Fire-type move).
- Not something you can build around, but something you must accept as variance.

### Status Immunity Notes

| Status | Immune Types/Abilities |
|--------|----------------------|
| Burn | Fire-types |
| Poison | Poison-types, Steel-types |
| Paralysis | Electric-types (Gen 6+), Ground-types (vs. T-Wave) |
| Sleep | Vital Spirit, Insomnia, Grass (vs. powder) |
| All | Misty Terrain (grounded), Safeguard |

---

## 9. Pivoting & Momentum

### What Is Momentum?

Momentum is **having the favorable matchup on the field**. When your active Pokemon
threatens the opponent's active Pokemon, you have momentum—they are forced to
react. When the reverse is true, they have momentum.

Games are a constant tug-of-war for momentum. The player who maintains momentum
longer generally wins.

### Pivoting Moves

| Move | Type | BP | Notes |
|------|------|-----|-------|
| U-turn | Bug | 70 | Physical, wide distribution |
| Volt Switch | Electric | 70 | Special, blocked by Ground-types |
| Flip Turn | Water | 60 | Physical, niche distribution |
| Teleport | Psychic | — | -6 priority, always switches, can't be blocked |

### Fast Pivots vs. Slow Pivots

- **Fast pivot** (e.g., Dragapult U-turn): Moves first, deals chip, switches out.
  You bring in a teammate *after* seeing what the opponent does. Best for scouting.
- **Slow pivot** (e.g., Slowbro Teleport): Moves last, takes a hit, then switches
  out. Your teammate comes in **without taking damage**. Best for safely bringing
  in a frail wallbreaker or sweeper.

**Critical insight**: In a Volt-Turn war (both players pivoting), the **slower**
pivot has the advantage. The slower Pokemon moves second, meaning their switch
happens last, and their teammate enters the field cleanly.

### Pivoting Heuristics

1. **Don't pivot mindlessly.** Every U-turn/Volt Switch you use is a turn you
   didn't use a stronger attack. Pivot when you need information or when the
   incoming mon can do more than your current one.
2. **Pivot into advantage.** The goal of a pivot is to get your threatening Pokemon
   in for free. If there's nothing threatening to bring in, just attack.
3. **Be wary of Ground-types vs. Volt Switch.** Ground immunizes Volt Switch,
   meaning you deal no damage and don't switch. The opponent gets a free turn.
4. **Regenerator makes pivoting almost free.** If your pivot has Regenerator,
   it heals on the switch out. You can pivot repeatedly with minimal cost.

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

### Speed Tiers

The "speed tier" is where a Pokemon falls relative to the metagame's speed
benchmarks. Key thresholds include:

- **Unboosted base 100-110**: The "crowded middle." Many common threats live here.
- **Base 120+**: Naturally fast; often doesn't need a Scarf.
- **Base 70-90**: Needs a boost (Scarf, Dragon Dance, or Agility) to sweep.
- **Base 50 and below**: Designed for Trick Room or slow pivoting, not outspeeding.

### Speed Control Methods

1. **Choice Scarf**: +50% Speed. Turns base 80-100 Pokemon into speedsters.
   The most common speed control tool.
2. **Priority moves**: Bypass Speed entirely.
   - Extreme Speed (+2), Sucker Punch (+1, fails if opponent uses status),
     Mach Punch (+1), Bullet Punch (+1), Ice Shard (+1), Aqua Jet (+1),
     Shadow Sneak (+1).
   - Priority is the ultimate answer to boosted sweepers.
3. **Setup moves**: Dragon Dance (+1 Atk, +1 Spe), Agility/Rock Polish (+2 Spe),
   Shell Smash (+2 Atk/SpA/Spe, -1 Def/SpD), Quiver Dance (+1 SpA/SpD/Spe).
4. **Tailwind**: Doubles team speed for 4 turns. More common in doubles but
   usable in singles.
5. **Trick Room**: Reverses speed order for 5 turns. Niche in 6v6 singles but
   devastating when it works.
6. **Sticky Web**: -1 Speed to grounded opponents switching in.
7. **Thunder Wave / Paralysis**: Permanent 75% speed cut.

### Speed Control Heuristics

- Every team needs at least one form of speed control. Without it, a +1 Dragon
  Dance sweeper runs through you unchecked.
- Choice Scarf + a priority user gives you two independent speed control layers.
  This redundancy is valuable.
- Know your speed tiers. Know what outspeeds what at +1, with Scarf, etc.
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
5. **Pattern recognition matters.** If an opponent has switched Corviknight into
   your wallbreaker three times, they'll probably do it a fourth. That's when
   you predict with a Fire-type coverage move.
6. **Consider the opponent's perspective.** What would *you* do in their position?
   What is the "correct" play for them? That's likely what they'll do.

### Switching Heuristics

1. **Don't switch into attacks you can't take.** This seems obvious, but many
   players switch into a "check" that can't actually survive the hit. Always know
   your damage ranges.
2. **Switch early, not late.** If a Pokemon needs to come in to check a threat,
   bring it in while it's healthy. Waiting until it's at 30% HP makes it useless
   as a check.
3. **Keep your checks healthy.** If your Toxapex is your only answer to Kingambit,
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
  you lose to their Kingambit in 5 turns, take a risk now.
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

At Team Preview (and continuously throughout the game), ask: **"Which of my
Pokemon can this opposing team least handle?"**

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
4. **Don't get greedy**: Sometimes +1 is enough to sweep. Don't go for +2 if it
   means risking a crit or a faster revenge killer.

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

## 14. Terastallization (Gen 9)

### How It Works

- Once per battle, a Pokemon can change its type to its predetermined Tera type.
- The new type persists even through switches and until the Pokemon faints.
- STAB mechanics: Original STAB moves keep 1.5x. Tera-type moves also get 1.5x.
  If Tera matches an original type, that STAB becomes 2x (Adaptability-like).

### Offensive Tera

- **Same-type Tera** (e.g., Tera Fire on a Fire-type): Maximizes damage on your
  best STAB. Turns strong hits into nuclear hits.
- **Coverage Tera** (e.g., Tera Ground on a Pokemon with Earthquake): Gives STAB
  to a coverage move, letting you break through Pokemon that would normally wall
  you.
- **Tera Blast**: A universal move (80 BP) that becomes the Tera type and uses
  the higher attacking stat. Enables surprise coverage.

### Defensive Tera

- **Removing weaknesses**: Tera Steel on a Dragon to remove Ice/Fairy weakness.
  Tera Flying on a Fighting-weak Pokemon to gain immunity to Ground.
- **Gaining immunities**: Tera Ghost for Normal/Fighting immunity. Tera Ground
  on a Levitate user for effective zero weaknesses.
- **Surviving a KO**: Tera into a type that resists the incoming attack to survive
  what would otherwise be a OHKO.

### Tera Timing Heuristics

1. **Don't Tera early without a clear reason.** Tera is a one-time resource.
   Using it on turn 2 means you can't use it for the rest of the game.
2. **Tera is strongest when it flips an expected outcome.** The opponent expects
   to KO your Pokemon or wall it—Tera changes the calculation entirely.
3. **Tera is often best on your win condition.** Using Tera to let your sweeper
   survive a revenge kill attempt or break through its last check is typically
   the highest-value use.
4. **Don't Tera reactively without thinking ahead.** Surviving one hit is nice,
   but if your new type leaves you vulnerable to everything else the opponent has,
   you've just traded one problem for another.
5. **Track whether the opponent has Tera'd.** If they haven't, factor in that
   any of their Pokemon could change type. This uncertainty is itself a weapon.

### The Tera Dilemma

The player who Tera's first often loses a strategic advantage: the opponent now
knows your Tera type, while you still don't know theirs. This creates a game of
chicken—both players want the other to Tera first.

- **Aggressive Tera** (early) is better when you can generate an overwhelming
  advantage from it (e.g., sweep immediately after Tera).
- **Reactive Tera** (defensive survival) is often lower-value because it doesn't
  generate offensive advantage.

---

## 15. Common Mistakes & Bad Tactics

### Team Building Errors

1. **All-offense teams**: Filling your team with sweepers sounds great until a
   single Dragon Dance Gyarados at +1 sweeps your entire team because nothing
   can take a hit. You need defensive checks.
2. **No hazard removal**: Stealth Rock is on nearly every team. If you can't
   remove it, your Pokemon lose 12-50% HP every time they switch in. Over a
   20-turn game, that's lethal.
3. **Type-stacking**: Three Water-types on one team means a single Grass- or
   Electric-type can check half your team. Diversify your typing.
4. **Building around a single gimmick**: Trick Room teams, dedicated weather
   teams, or Baton Pass chains can be devastating when they work but crumble
   when the opponent has a single counter. Have a Plan B.
5. **Ignoring the metagame**: Building your team in a vacuum without considering
   what's popular leads to teams that auto-lose to common threats.

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
11. **Throwing away your win condition**: Your +1 Dragonite is your path to
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
- **Consider what happens *after* you switch.** Bringing in Corviknight against
  the Rillaboom is great—but what if they predict and go Heatran? Think one
  step ahead.

### Resource Management

- **HP is a resource, not a health bar.** Taking 30% damage to deal 60% is a
  good trade. Taking 30% to accomplish nothing is a waste. Every point of HP
  should be spent on something.
- **PP is a resource.** In stall matchups, tracking the opponent's remaining PP
  on key moves (recovery, coverage) can determine who wins.
- **Tera is a resource.** (See Section 14.) Don't spend it frivolously.
- **Your Pokemon are a resource.** You have six. Losing one to gain a massive
  positional advantage is often correct.

### Opponent Modeling

- **Play the metagame, not the opponent.** Early in a game, assume standard sets
  and standard play. Adjust as you gather information.
- **Pay attention to what they reveal.** If their Garchomp uses Swords Dance on
  turn 3, it's probably not Choice Scarf. If their Heatran takes 35% from a
  neutral hit, it's probably not specially defensive.
- **Look for what's missing.** At Team Preview, they have no obvious hazard
  removal. That means hazards are extra valuable against them—prioritize getting
  Stealth Rock up and keeping it up.
- **Respect coverage.** "Kingambit is walled by my Corviknight" is true until
  Kingambit reveals Iron Head or Tera Fire. Don't rely on a single check without
  considering the possibility of coverage moves.

### Positional Play

- **Trades favor the player with more remaining win conditions.** If you have two
  ways to win and they have one, trading Pokemon 1-for-1 is great for you.
- **Don't trade your check for their non-threat.** If your Toxapex trades with
  their Amoonguss but their Kingambit is still alive, you just lost the game.
- **Force 50/50s when behind, avoid them when ahead.** If you're winning, don't
  give the opponent a coin-flip opportunity to equalize. If you're losing,
  50/50s are better than a guaranteed slow loss.
- **Momentum > damage.** Sometimes dealing less damage but maintaining the
  favorable matchup (via pivoting) is better than dealing more damage but
  giving the opponent a free switch.

### Pattern Recognition

- **Most opponents follow patterns.** They switch the same Pokemon into the same
  threat, they lead with the same Pokemon, they Tera at the same moment. Exploit
  these patterns when you spot them.
- **Varying your own patterns is equally important.** If you always switch
  Toxapex into their Volcarona, they'll start predicting it and clicking
  Psychic. Mix it up.
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

- [Introduction to Competitive Pokemon - Smogon University](https://www.smogon.com/dp/articles/intro_comp_pokemon)
- [Getting Started with Competitive Battling - Smogon University](https://www.smogon.com/articles/getting-started)
- [Teambuilding Guide - Smogon Forums](https://www.smogon.com/forums/threads/teambuilding-guide.3552468/)
- [Building Hyper Offense in OU - Smogon University](https://www.smogon.com/articles/hyper-offense-in-ou)
- [Risk/Reward - Smogon University](https://www.smogon.com/bw/articles/bw_risk_reward)
- [Pivots in SM OU - Smogon University](https://www.smogon.com/articles/pivots-sm-ou)
- [VoltTurn in OU - Smogon University](https://www.smogon.com/articles/voltturn-in-ou)
- [Momentum - Smogon Forums](https://www.smogon.com/forums/threads/momentum.3450730/)
- [Battle Tactics: Double Switches - Smogon Forums](https://www.smogon.com/forums/threads/battle-tactics-double-switches.51714/)
- [Entry Hazards in OU - Smogon University](https://www.smogon.com/smog/issue44/entry-hazards-in-ou)
- [OU: Clearing Up Some Facts With Defog - Smogon University](https://www.smogon.com/smog/issue37/ou-defog)
- [Status in RU - Smogon University](https://www.smogon.com/smog/issue26/ru_status)
- [Revenge Killers in RU - Smogon University](https://www.smogon.com/articles/revenge-killers-ru)
- [Boosting Them Up: Setup Sweepers in OU - Smogon University](https://www.smogon.com/articles/ou-setup-sweepers)
- [Setup for Success: Setup Sweepers in LC - Smogon University](https://www.smogon.com/articles/lc-setup-sweepers)
- [Choice Scarf: Role in the Meta - Smogon Forums](https://www.smogon.com/forums/threads/choice-scarf-role-in-the-meta.3500909/)
- [Pokemon Battling: A Lesson in Game Theory - Smogon Forums](https://www.smogon.com/forums/threads/pokemon-battling-a-lesson-in-game-theory.3492697/)
- [Terastallization Tiering Discussion - Smogon Forums](https://www.smogon.com/forums/threads/terastallization-tiering-discussion.3711464/)
- [Competitive Battle Tactics Guide - Pokemon Lazarus](https://pokemonlazarus.org/walkthrough/competitive-battle-tactics)
- [Predictions in Competitive Battling - PokeCommunity Daily](https://daily.pokecommunity.com/2016/06/04/predictions-competitive-battling/)
- [Common Mistakes in Competitive Pokemon - PokeBeach](https://www.pokebeach.com/2024/02/the-most-common-mistakes-in-competitive-pokemon-and-how-to-avoid-them)
- [Pokemon Roles Explained - Pokestrat](https://pokestratbuilder.com/blog/roles-explained)
- [How to Build a Competitive Pokemon Team - Pokestrat](https://pokestratbuilder.com/blog/pokemon-team)
