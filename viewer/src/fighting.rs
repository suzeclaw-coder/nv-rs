//! People and creatures noticing and fighting, as the game's combat AI
//! does (`world::combat_ai`, read from its code; this file only measures
//! and moves):
//!
//! - Each actor's detection run (every 0.3 s in combat, staggered up to
//!   5 s more out of it; none farther than 8192 units from the player)
//!   works out its detection value for the player and everyone else
//!   loaded. A value rising above −20 starts a fight where aggression and
//!   factions say so; allies join by their Assistance; the unaggressive run
//!   from those much stronger who'd attack them.
//! - Gunmen keep within their weapon's band, strafing every 2–5 s, and fire
//!   at the game's pace (semi-automatic: after the attack and a random
//!   delay; automatic: 1 s bursts, 1 s pauses), only at a target in sight
//!   within the aim arc. Melee fighters run in, fast-walk the last 64 units,
//!   and after each attack roll attack or hold by their combat style.
//! - A target unseen for 15 s is searched for; unseen for 30 s (never seen)
//!   or 60 s (and more than 4096 units away) it's given up, and they go
//!   back to their packages.
//!
//! Measured here (guesses where the game's way isn't traced): lines of
//! sight are rays through the cell's collision 60 units above the feet;
//! the spot in the band a gunman moves to is on the line to the target,
//! halfway into the band (the game searches the navmesh, `009d5000`); a
//! search walks to where the target was last seen, then to random spots
//! within the smallest search radius every `fCombatSearchAreaUpdateTime`;
//! someone fleeing runs `fCombatFleeNormalDistance` (2048) straight away
//! from the threat until they no longer notice it; paths toward a moving
//! target are made again every half second. Not done: crouching, dodging,
//! cover, blocking (no block animations are played, so the block score is 0
//! as for those without one), suppressive fire, reloads, grenades, spread
//! (every shot hits), the hit landing at the attack animation's hit key.

use std::collections::{HashMap, HashSet};

use esm::{FormId, LoadOrder};
use world::ai::NavMesh;
use world::combat::Weapon;
use world::combat_ai::{
    self, Approach, CombatStyle, Engage, EngageMove, Gait, MeleeChoice, MeleeSituation,
    RangedAttack, SettingCache, TargetMemory,
};
use world::dialogue::PLAYER_REF;
use world::scripting::{Facts, GameState, Runner, ScriptCache};

use crate::ai::{distance, step, Walker};

/// How often a path toward a moving target is made again, seconds (not
/// read from the game).
const REPATH_SECONDS: f32 = 0.5;

/// Someone others can notice, as the frame began.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Seen {
    pub reference: FormId,
    pub position: [f32; 3],
    pub moving: bool,
    pub running: bool,
    /// Attacking now (an attack under way).
    pub attacking: bool,
    /// Their collision radius.
    pub radius: f32,
}

/// What an actor fights with, read once: its combat style, a creature's
/// reach and type, its collision radius, how fast it walks and runs, and
/// how long its attack animation lasts (the weapon kind's for people, its
/// own for creatures).
#[derive(Debug, Clone)]
pub(crate) struct Kit {
    pub style: CombatStyle,
    pub creature: Option<(f32, u8)>,
    pub radius: f32,
    pub walk: f32,
    pub run: f32,
    pub attack_animation: f32,
}

impl Kit {
    /// Read from the records and the skeleton: people's radius
    /// [`combat_ai::PERSON_RADIUS`]; a creature's from its skeleton's bound
    /// (`combat_ai::creature_radius`), else the game's 25 × scale for a
    /// creature without one; walking at the game's speed
    /// (`world::animation::base_speed`: `fMoveBaseSpeed` × SpeedMult ÷
    /// 100, × the scale; the legs' condition is applied where they move)
    /// and running `fMoveRunMult` times that; the attack animation's
    /// length (1 s without one: a guess).
    pub fn read(
        order: &LoadOrder,
        state: &GameState,
        walker: &Walker,
        skeleton: &preview::cell::ActorSkeleton,
    ) -> Kit {
        let creature = world::combat::creature_reach(order, walker.reference);
        let radius = match (creature, skeleton.bound) {
            (Some(_), Some(b)) => combat_ai::creature_radius(b.half_extents, walker.scale),
            (Some(_), None) => 25.0 * walker.scale,
            (None, _) => combat_ai::PERSON_RADIUS,
        };
        let walk = world::animation::base_speed(order, state, walker.reference) * walker.scale;
        let run = walk * world::animation::run_mult(order);
        let attack_animation = skeleton
            .attack
            .as_ref()
            .map_or(1.0, |a| (a.stop - a.start).max(0.1));
        Kit {
            style: CombatStyle::of(order, walker.reference),
            creature,
            radius,
            walk,
            run,
            attack_animation,
        }
    }

    /// How long an attack lasts (`Weapon::attack_seconds`: a gun's 1 ÷
    /// its attack shots a second, a melee weapon's attack animation at its
    /// attack multiplier), else the attack animation.
    pub fn attack_seconds(&self, weapon: Option<&Weapon>) -> f32 {
        weapon.map_or(self.attack_animation, |w| {
            w.attack_seconds(self.attack_animation)
        })
    }
}

/// A move under way in a fight: its gait, whether they keep facing the
/// target meanwhile (strafing), for a chase the distance from the target
/// at which it ends, and whether it ends with the target in sight again.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Move {
    gait: Gait,
    facing: bool,
    chase: Option<f32>,
    until_seen: bool,
}

impl Move {
    /// A move to a spot.
    fn to_spot(gait: Gait, facing: bool) -> Move {
        Move {
            gait,
            facing,
            chase: None,
            until_seen: false,
        }
    }
}

/// A fight under way: what's known of the target, the gunman's and the
/// swordsman's timers, the move under way.
/// An attack windup (telegraph window) under way for a melee combatant:
/// forward momentum decelerates, committing to the lunge direction,
/// giving the player an evasion window before the strike key.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MeleeWindup {
    pub start: f32,
    pub strike_at: f32,
    pub end: f32,
    pub lunge_dir: [f32; 2],
    pub lunge_speed: f32,
    pub power: bool,
    pub weapon: Option<FormId>,
}

#[derive(Debug, Clone)]
pub(crate) struct Fight {
    pub memory: TargetMemory,
    engage: Engage,
    ranged: RangedAttack,
    /// When the attack under way ends; holding until when.
    attack_until: f32,
    /// Active attack telegraph windup window under way.
    pub(crate) windup: Option<MeleeWindup>,
    holding: bool,
    hold_until: f32,
    /// The target's last detection value and whether it was in sight.
    in_sight: bool,
    moving: Option<Move>,
    repath_at: f32,
    /// Searching: whether the last known spot has been reached, and when the
    /// next spot is due.
    searched_spot: bool,
    search_at: f32,
}

impl Fight {
    fn new(target: FormId, now: f32, at: [f32; 3]) -> Fight {
        Fight {
            memory: TargetMemory::new(target, now, at),
            engage: Engage::default(),
            ranged: RangedAttack::default(),
            attack_until: f32::NEG_INFINITY,
            windup: None,
            holding: false,
            hold_until: f32::NEG_INFINITY,
            in_sight: true,
            moving: None,
            repath_at: f32::NEG_INFINITY,
            searched_spot: false,
            search_at: f32::NEG_INFINITY,
        }
    }
}

/// Random numbers for a frame's choices, from the state's dice.
pub(crate) struct Dice(u64);

impl Dice {
    pub fn new(state: &mut GameState) -> Dice {
        Dice(state.roll().max(1))
    }

    pub fn roll(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }

    /// 0 up to (not including) 1.
    pub fn unit(&mut self) -> f32 {
        (self.roll() >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Whether nothing solid is between two people (from 60 units above their
/// feet: the game's eye points aren't traced).
pub(crate) fn clear_between(collision: &physics::Collider, from: [f32; 3], to: [f32; 3]) -> bool {
    let a = [from[0], from[1], from[2] + 60.0];
    let b = [to[0], to[1], to[2] + 60.0];
    let d = distance(a, b);
    if d < 1.0 {
        return true;
    }
    let dir = [(b[0] - a[0]) / d, (b[1] - a[1]) / d, (b[2] - a[2]) / d];
    collision
        .raycast(a, dir, d)
        .is_none_or(|(hit, _)| hit >= d - 10.0)
}

/// What noticing needs besides the state.
pub(crate) struct Noticing<'a> {
    pub order: &'a LoadOrder,
    pub settings: &'a SettingCache,
    pub collision: &'a physics::Collider,
    pub now: f32,
}

/// One actor's detection run (`008e40d0`, `008ff350`): its value for
/// everyone in `others` (the player first), what it knows of its target,
/// and, out of combat, a fight started (on noticing someone it would
/// attack, or a friend's enemy), or a flight.
pub(crate) fn detect(n: &Noticing, state: &mut GameState, walker: &mut Walker, others: &[Seen]) {
    let order = n.order;
    let me = walker.reference;
    let s = |name: &str, d: f32| n.settings.get(order, name, d);
    let max = s("fSneakMaxDistance", 1500.0)
        * if state.player_world.is_some() {
            s("fSneakExteriorDistanceMult", 2.0)
        } else {
            1.0
        };
    let mut values: Vec<(FormId, i32, bool, [f32; 3])> = Vec::new();
    {
        let facts = Facts {
            order,
            state,
            speaker: None,
        };
        for o in others {
            if o.reference == me || state.dead.contains(&o.reference) {
                continue;
            }
            let d = distance(walker.position, o.position);
            let sight = d < max && clear_between(n.collision, walker.position, o.position);
            let motion = (o.reference != PLAYER_REF).then_some((o.moving, o.running));
            if let Some(v) = combat_ai::detection_value(&facts, me, o.reference, sight, motion, &s)
            {
                values.push((o.reference, v, sight, o.position));
            }
        }
    }
    let noticed_min = s("fSneakNoticedMin", -20.0);
    let in_combat = state.combat.contains_key(&me);
    let mut start = None;
    for &(r, v, sight, at) in &values {
        if r == PLAYER_REF {
            walker.detected_player = v;
        }
        let noticed = v as f32 > noticed_min;
        let rising = noticed && walker.noticed.insert(r);
        if !noticed {
            walker.noticed.remove(&r);
            if walker.fleeing == Some(r) {
                walker.fleeing = None;
                walker.path.clear();
                walker.forget_package(n.now);
            }
        }
        if let Some(f) = walker.fight.as_mut().filter(|f| f.memory.target == r) {
            f.in_sight = sight;
            if v > 0 {
                f.memory.saw(n.now, at);
            }
        }
        if in_combat || start.is_some() {
            continue;
        }
        if rising && combat_ai::starts_combat(order, state, me, r, v, &s) {
            start = Some(r);
            continue;
        }
        if rising && walker.fleeing.is_none() && combat_ai::flees_on_sight(order, state, me, r, &s)
        {
            println!("{:.1} s: {me} runs from {r}.", n.now);
            walker.fleeing = Some(r);
            walker.path.clear();
        }
        if v > 0 {
            let value_of = |x: FormId| values.iter().find(|e| e.0 == x).map(|e| e.1);
            if let Some(enemy) = combat_ai::assists_against(order, state, me, r, v, value_of) {
                println!("{:.1} s: {me} helps {r} against {enemy}.", n.now);
                start = Some(enemy);
            }
        }
    }
    if let Some(t) = start {
        let value = values.iter().find(|e| e.0 == t).map_or(0, |e| e.1);
        println!(
            "{:.1} s: {me} notices {t} (detection {value}) and attacks.",
            n.now
        );
        state.combat.insert(me, t);
        walker.fleeing = None;
    }
}

/// What a fight needs besides the state and the fighter.
pub(crate) struct FightCtx<'a> {
    pub order: &'a LoadOrder,
    pub scripts: &'a ScriptCache,
    pub settings: &'a SettingCache,
    pub mesh: &'a NavMesh,
    pub sounds: &'a mut crate::sounds::SoundRequests,
    /// Hits made, for their sounds and effects (`hiteffects`).
    pub hits: &'a mut crate::hiteffects::HitReports,
    pub others: &'a [Seen],
    /// Turning settings (`world::movement`).
    pub moves: &'a world::movement::MoveSettings,
    pub now: f32,
    pub dt: f32,
}

/// What a fight frame did: moving (at what gait), and whether an attack
/// began.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FightFrame {
    pub gait: Option<Gait>,
    pub attacked: bool,
}

/// Reports a blow or shot that hurt `target` (`hiteffects`: its sounds,
/// heard within their distances of where it struck, and the hurt or death
/// cry). Where people's attacks meet a body isn't worked out here, so the
/// point is where the target stands.
fn report_hit(
    c: &mut FightCtx,
    state: &GameState,
    attacker: FormId,
    (target, at): (FormId, [f32; 3]),
    weapon: Option<FormId>,
    damage: f32,
) {
    c.hits.0.push(crate::hiteffects::HitReport {
        attacker,
        target: Some(target),
        weapon,
        point: at,
        havok: None,
        damage,
        killed: state.dead.contains(&target),
    });
}

/// Where someone is now.
fn position_of(order: &LoadOrder, state: &GameState, who: FormId) -> Option<[f32; 3]> {
    if who == PLAYER_REF {
        state.player_position
    } else {
        state.place(order, who).map(|p| p.2)
    }
}

/// One frame of a fight against `target`.
pub(crate) fn fight(
    c: &mut FightCtx,
    state: &mut GameState,
    walker: &mut Walker,
    kit: &Kit,
    target: FormId,
) -> FightFrame {
    let order = c.order;
    let me = walker.reference;
    let settings = c.settings;
    let s = |name: &str, d: f32| settings.get(order, name, d);
    let Some(goal) = position_of(order, state, target) else {
        return FightFrame::default();
    };
    let mut fight = match walker.fight.take() {
        Some(f) if f.memory.target == target => f,
        _ => {
            walker.path.clear();
            Fight::new(target, c.now, goal)
        }
    };
    let d = distance(walker.position, goal);
    if fight
        .memory
        .gives_up(c.now, d, state.dead.contains(&target), &s)
    {
        println!("{:.1} s: {me} gives up on {target}.", c.now);
        state.combat.remove(&me);
        end_fight(walker, c.now);
        return FightFrame::default();
    }
    let mut dice = Dice::new(state);
    let frame = if fight.memory.searching(c.now, &s) {
        search(c, state, walker, kit, &mut fight, &mut dice)
    } else {
        let weapon = world::combat::weapon_in_hand(order, state, me);
        match weapon.as_ref().filter(|w| !w.is_melee()) {
            Some(w) => ranged(
                c,
                state,
                walker,
                kit,
                &mut fight,
                (w, goal, target),
                &mut dice,
            ),
            None => melee(
                c,
                state,
                walker,
                kit,
                &mut fight,
                (weapon.as_ref(), goal, target),
                &mut dice,
            ),
        }
    };
    walker.fight = Some(fight);
    frame
}

/// A fight over: back to their packages at once.
pub(crate) fn end_fight(walker: &mut Walker, now: f32) {
    walker.fight = None;
    walker.path.clear();
    walker.next = 0;
    walker.forget_package(now);
}

/// Sets a path to `to` (over the navmesh, else `straight` allowing a
/// straight line); whether there is one.
fn go(mesh: &NavMesh, walker: &mut Walker, to: [f32; 3], straight: bool) -> bool {
    let path = match mesh.path(walker.position, to) {
        Some(p) => p,
        None if straight => vec![walker.position, to],
        None => {
            walker.clear_path();
            return false;
        }
    };
    // In a fight the walk starts at once (no turn in place first).
    walker.set_path(path, 0.0, false, &world::movement::MoveSettings::defaults());
    true
}

/// How far off to the side (radians, either way) and up or down `to` is
/// for someone at `from` facing `heading` (clockwise from north).
fn offsets(from: [f32; 3], heading: f32, to: [f32; 3]) -> (f32, f32) {
    let (dx, dy, dz) = (to[0] - from[0], to[1] - from[1], to[2] - from[2]);
    let mut yaw = (dx.atan2(dy) - heading).rem_euclid(std::f32::consts::TAU);
    if yaw > std::f32::consts::PI {
        yaw -= std::f32::consts::TAU;
    }
    (yaw, dz.atan2(dx.hypot(dy)))
}

/// Turns to face `to`: in place, at the combat rate (225°/s for people,
/// `fAICombatTurnSpeedScale`; `ai::face`).
fn face(walker: &mut Walker, to: [f32; 3], c: &FightCtx) {
    let (dx, dy) = (to[0] - walker.position[0], to[1] - walker.position[1]);
    if dx.hypot(dy) > 1.0 {
        crate::ai::face(walker, dx.atan2(dy), c.dt, true, c.moves);
    }
}

/// Walks the move under way, at `legs` × the gait's speed (crippled legs:
/// `world::body_parts::leg_speed_mult`); whether they moved.
fn walk_move(
    walker: &mut Walker,
    fight: &Fight,
    kit: &Kit,
    (goal, legs): ([f32; 3], f32),
    dt: f32,
) -> bool {
    let Some(m) = fight.moving else {
        return false;
    };
    // Strafing, they keep facing the target (stepping sideways): the
    // walking turn aims at it, not along the path (`009e4450`).
    if m.facing {
        walker.face_point = Some(goal);
    }
    let walking = step(walker, m.gait.speed(kit.walk, kit.run) * legs, dt);
    walker.face_point = None;
    walking
}

/// A gunman's frame: keeping to the band (`Engage`), and shooting
/// (`RangedAttack`).
fn ranged(
    c: &mut FightCtx,
    state: &mut GameState,
    walker: &mut Walker,
    kit: &Kit,
    fight: &mut Fight,
    (w, goal, target): (&Weapon, [f32; 3], FormId),
    dice: &mut Dice,
) -> FightFrame {
    let order = c.order;
    let settings = c.settings;
    let s = |name: &str, d: f32| settings.get(order, name, d);
    let now = c.now;
    let reach = w
        .projectile
        .and_then(|p| world::combat::projectile_reach(order, p));
    let band = combat_ai::ranged_band(Some((w, reach)), &kit.style, &s);
    let d = distance(walker.position, goal);
    // A move ends when its path does, a chase within its distance, an
    // approach once the target is in sight again.
    if let Some(m) = fight.moving {
        let done = walker.next >= walker.path.len()
            || m.chase.is_some_and(|stop| d <= stop)
            || (m.until_seen && fight.memory.unseen_for(now) <= 0.5);
        if done {
            walker.path.clear();
            fight.moving = None;
            fight.engage.arrived(now, dice.unit());
        } else if m.chase.is_some() && now >= fight.repath_at {
            fight.repath_at = now + REPATH_SECONDS;
            go(c.mesh, walker, goal, true);
        }
    }
    let unseen = fight.memory.unseen_for(now);
    let decision = fight
        .engage
        .update(now, d, &band, fight.in_sight, unseen, &mut || dice.unit());
    let toward = {
        let (dx, dy) = (goal[0] - walker.position[0], goal[1] - walker.position[1]);
        let l = dx.hypot(dy).max(1e-3);
        [dx / l, dy / l]
    };
    let at = |p: [f32; 3], along: f32, side: f32| {
        [
            p[0] + toward[0] * along + toward[1] * side,
            p[1] + toward[1] * along - toward[0] * side,
            p[2],
        ]
    };
    let started = match decision {
        EngageMove::Stay => None,
        EngageMove::Strafe { distance, left } => {
            let side = if left { -distance } else { distance };
            let spot = at(walker.position, 0.0, side);
            go(c.mesh, walker, spot, false).then_some(Move::to_spot(Gait::FastWalk, true))
        }
        EngageMove::ToBand { run } => {
            // On the line to the target, halfway into the band, else just
            // inside its nearer edge (a guess for the game's navmesh
            // search for a spot in range).
            let edge = if d < band.min {
                band.min + 16.0
            } else {
                band.optimal - 16.0
            };
            let gait = if run { Gait::Run } else { Gait::FastWalk };
            [(band.min + band.optimal) * 0.5, edge]
                .into_iter()
                .any(|want| go(c.mesh, walker, at(goal, -want, 0.0), false))
                .then_some(Move::to_spot(gait, false))
        }
        EngageMove::Step { distance, closer } => {
            let spot = at(
                walker.position,
                if closer { distance } else { -distance },
                0.0,
            );
            go(c.mesh, walker, spot, false).then_some(Move::to_spot(Gait::FastWalk, false))
        }
        EngageMove::RunAt => {
            fight.repath_at = now + REPATH_SECONDS;
            go(c.mesh, walker, goal, true).then_some(Move {
                gait: Gait::Run,
                facing: false,
                chase: Some(128.0),
                until_seen: false,
            })
        }
        EngageMove::Approach => {
            fight.repath_at = now + REPATH_SECONDS;
            go(c.mesh, walker, goal, true).then_some(Move {
                gait: Gait::FastWalk,
                facing: false,
                chase: Some(kit.radius + combat_ai::PERSON_RADIUS),
                until_seen: true,
            })
        }
    };
    if decision != EngageMove::Stay {
        println!(
            "{now:.1} s: {} at {d:.0} units (band {:.0}–{:.0}): {decision:?}{}",
            walker.reference,
            band.min,
            band.optimal,
            if started.is_some() { "" } else { " (no path)" }
        );
        match started {
            Some(m) => fight.moving = Some(m),
            // Nowhere to go: the timer starts again.
            None => fight.engage.arrived(now, dice.unit()),
        }
    }
    let legs = world::body_parts::leg_speed_mult(order, state, walker.reference);
    let walking = walk_move(walker, fight, kit, (goal, legs), c.dt);
    if !walking {
        face(walker, goal, c);
    }
    // Shooting: in sight, within the style's targeting field of view, and
    // within the aim arc (or nearer than the band's minimum).
    let (yaw, pitch) = offsets(walker.position, walker.heading, goal);
    let aimed = fight.in_sight
        && combat_ai::within_targeting_fov(&kit.style, yaw)
        && (combat_ai::within_aim_arc(w.aim_arc, yaw, pitch) || d < band.min);
    let attack = kit.attack_seconds(Some(w));
    let shoots = fight
        .ranged
        .update(now, w, &kit.style, aimed, attack, dice.unit(), &s);
    if shoots {
        if let Some(sound) = w.sound {
            c.sounds.0.push(sound);
        }
        let dealt = Runner::new(order, c.scripts, state).hit(walker.reference, target, Some(w));
        if let Some(dmg) = dealt {
            report_hit(
                c,
                state,
                walker.reference,
                (target, goal),
                Some(w.form_id),
                dmg,
            );
            let left = world::combat::health(order, state, target).unwrap_or(0.0);
            println!(
                "{now:.1} s: {} shoots {target} from {d:.0} units for {dmg:.1} ({left:.1} left).",
                walker.reference
            );
        }
    }
    FightFrame {
        gait: walking.then(|| fight.moving.map_or(Gait::FastWalk, |m| m.gait)),
        attacked: shoots,
    }
}

/// Computes a lateral flank target perpendicular to the player approach vector,
/// spreading multiple pursuing melee enemies into an encircling crescent / pincer formation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_flank_steering(
    me: FormId,
    my_pos: [f32; 3],
    my_radius: f32,
    target_pos: [f32; 3],
    target_radius: f32,
    reach: f32,
    others: &[Seen],
    combat: &HashMap<FormId, FormId>,
    target: FormId,
    dead: &HashSet<FormId>,
) -> [f32; 3] {
    let dx = target_pos[0] - my_pos[0];
    let dy = target_pos[1] - my_pos[1];
    let dist = dx.hypot(dy);
    if dist < 1e-2 {
        return target_pos;
    }
    let fwd = [dx / dist, dy / dist];
    let tangent = [-fwd[1], fwd[0]];

    // Find all active co-pursuers targeting the same target.
    let mut pursuers: Vec<(FormId, [f32; 3], f32)> = Vec::with_capacity(others.len() + 1);
    pursuers.push((me, my_pos, my_radius));
    for o in others {
        if o.reference != me
            && o.reference != target
            && !dead.contains(&o.reference)
            && combat.get(&o.reference) == Some(&target)
        {
            pursuers.push((o.reference, o.position, o.radius));
        }
    }

    if pursuers.len() <= 1 {
        return target_pos;
    }

    // Sort pursuers along their transverse projection relative to target_pos across tangent.
    pursuers.sort_by(|a, b| {
        let t_a = (a.1[0] - target_pos[0]) * tangent[0] + (a.1[1] - target_pos[1]) * tangent[1];
        let t_b = (b.1[0] - target_pos[0]) * tangent[0] + (b.1[1] - target_pos[1]) * tangent[1];
        t_a.partial_cmp(&t_b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0 .0.cmp(&b.0 .0))
    });

    let n = pursuers.len();
    let my_rank = pursuers.iter().position(|p| p.0 == me).unwrap_or(0);
    let flank_slot = my_rank as f32 - (n - 1) as f32 * 0.5;
    let max_slot = ((n - 1) as f32 * 0.5).max(1.0);
    let norm_slot = flank_slot / max_slot; // in range [-1.0, 1.0]

    // Crescent encircling angle: up to ~55 degrees (0.95 rad) on either flank.
    let angle = norm_slot * 0.95;
    let encircle_r = (reach * 0.85 + target_radius + my_radius).max(64.0);

    let sin_a = angle.sin();
    let cos_a = angle.cos();

    // Offset around target_pos forming an encircling crescent:
    let offset_x = -fwd[0] * (encircle_r * cos_a) + tangent[0] * (encircle_r * sin_a);
    let offset_y = -fwd[1] * (encircle_r * cos_a) + tangent[1] * (encircle_r * sin_a);

    [target_pos[0] + offset_x, target_pos[1] + offset_y, target_pos[2]]
}

/// When trailing closely behind another pursuer along the player approach vector,
/// applies lateral repulsion perpendicular to the approach vector to break collinear
/// alignment and force enemies into side-by-side pursuit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_boid_separation(
    me: FormId,
    my_pos: [f32; 3],
    my_radius: f32,
    fwd: [f32; 2],
    tangent: [f32; 2],
    run_speed: f32,
    others: &[Seen],
    combat: &HashMap<FormId, FormId>,
    target: FormId,
    dead: &HashSet<FormId>,
) -> [f32; 2] {
    let mut lateral_repulsion = 0.0f32;

    for o in others {
        if o.reference == me || o.reference == target || dead.contains(&o.reference) {
            continue;
        }
        if combat.get(&o.reference) != Some(&target) {
            continue;
        }

        let dx = o.position[0] - my_pos[0];
        let dy = o.position[1] - my_pos[1];
        let along = dx * fwd[0] + dy * fwd[1];
        let lateral = dx * tangent[0] + dy * tangent[1];

        let trail_distance_max = (my_radius + o.radius) * 3.5;
        let collinear_threshold = (my_radius + o.radius) * 1.25;

        // Trailing closely behind another pursuer along the approach line:
        if along > 0.0 && along < trail_distance_max && lateral.abs() < collinear_threshold {
            let side = if lateral > 1e-2 {
                -1.0
            } else if lateral < -1e-2 || (me.0 ^ o.reference.0) & 1 == 0 {
                1.0
            } else {
                -1.0
            };

            let along_factor = 1.0 - (along / trail_distance_max);
            let lateral_factor = 1.0 - (lateral.abs() / collinear_threshold);
            let strength = along_factor * lateral_factor;

            lateral_repulsion += side * (run_speed * 0.8) * strength;
        } else if along.abs() < (my_radius + o.radius) * 0.75 && lateral.abs() < collinear_threshold {
            // Side-by-side crowding buffer to keep comfortable shoulder separation
            let side = if lateral > 0.0 { -1.0 } else { 1.0 };
            let overlap = 1.0 - (lateral.abs() / collinear_threshold);
            lateral_repulsion += side * (run_speed * 0.4) * overlap;
        }
    }

    let clamped = lateral_repulsion.clamp(-run_speed, run_speed);
    [tangent[0] * clamped, tangent[1] * clamped]
}

/// A melee fighter's frame: closing in with crescent flank steering and boid separation,
/// telegraphing committed attack lunges with an evasion window, and rolling attack or hold.
fn melee(
    c: &mut FightCtx,
    state: &mut GameState,
    walker: &mut Walker,
    kit: &Kit,
    fight: &mut Fight,
    (weapon, goal, target): (Option<&Weapon>, [f32; 3], FormId),
    dice: &mut Dice,
) -> FightFrame {
    let order = c.order;
    let settings = c.settings;
    let s = |name: &str, d: f32| settings.get(order, name, d);
    let now = c.now;
    fight.moving = None;
    let creature = if weapon.is_none() { kit.creature } else { None };
    let reach = combat_ai::melee_reach(weapon, creature, walker.scale, &s);
    let them = c.others.iter().find(|o| o.reference == target);
    let their_radius = them.map_or(combat_ai::PERSON_RADIUS, |o| o.radius);

    // 1. Committed Attack Telegraphing & Evasion Windows:
    if let Some(w) = fight.windup {
        // Interruption check: attacker staggered/unconscious/dead, or target dead
        if state.dead.contains(&walker.reference) || state.unconscious.contains(&walker.reference) {
            fight.windup = None;
            fight.attack_until = now;
            return FightFrame::default();
        }
        if state.dead.contains(&target) {
            fight.windup = None;
            fight.attack_until = now;
            return FightFrame::default();
        }

        if now < w.strike_at {
            // Still in committed windup window: forward momentum decelerates smoothly to 0
            let duration = (w.strike_at - w.start).max(1e-3);
            let progress = ((now - w.start) / duration).clamp(0.0, 1.0);
            let momentum = (1.0 - progress) * w.lunge_speed;
            let step_dist = momentum * c.dt;
            walker.position[0] += w.lunge_dir[0] * step_dist;
            walker.position[1] += w.lunge_dir[1] * step_dist;
            walker.heading = w.lunge_dir[0].atan2(w.lunge_dir[1]);
            walker.path.clear();
            return FightFrame {
                gait: None,
                attacked: false,
            };
        } else {
            // Windup complete: Strike execution and evasion detection!
            fight.windup = None;
            fight.attack_until = w.end;

            // Hit detection & evasion window: check if target backstepped, dodged, or slipped out of arc
            let current_dist = distance(walker.position, goal);
            let gap = combat_ai::gap(current_dist, kit.radius, their_radius);
            let (yaw, _) = offsets(walker.position, walker.heading, goal);
            let in_range = gap <= reach;
            let in_arc = yaw.abs() <= std::f32::consts::FRAC_PI_2;

            if in_range && in_arc {
                if let Some(sound) = weapon.and_then(|w| w.sound) {
                    c.sounds.0.push(sound);
                }
                let dealt = Runner::new(order, c.scripts, state).strike(
                    walker.reference,
                    target,
                    weapon,
                    w.power,
                );
                if let Some(dmg) = dealt {
                    report_hit(
                        c,
                        state,
                        walker.reference,
                        (target, goal),
                        w.weapon,
                        dmg,
                    );
                    let left = world::combat::health(order, state, target).unwrap_or(0.0);
                    println!(
                        "{now:.1} s: {} {} {target} for {dmg:.1} ({left:.1} left).",
                        walker.reference,
                        if w.power { "power-attacks" } else { "hits" }
                    );
                }
            } else {
                println!(
                    "{now:.1} s: {target} evaded {}'s attack (gap {gap:.0} vs reach {reach:.0}, yaw {yaw:.2}).",
                    walker.reference
                );
            }
            return FightFrame {
                gait: None,
                attacked: false,
            };
        }
    }

    // 2. Approach & Steering:
    let toward_spot = if fight.memory.unseen_for(now) > 0.5 {
        fight.memory.last_known
    } else {
        goal
    };
    let gap = combat_ai::gap(distance(walker.position, toward_spot), kit.radius, their_radius);
    let gait = match combat_ai::melee_approach(gap, reach, &s) {
        Approach::Run => Some(Gait::Run),
        Approach::FastWalk => Some(Gait::FastWalk),
        Approach::InReach => None,
    };

    if let Some(g) = gait {
        let (dx, dy) = (toward_spot[0] - walker.position[0], toward_spot[1] - walker.position[1]);
        let dist = dx.hypot(dy);
        let (fwd, tangent) = if dist > 1e-3 {
            ([dx / dist, dy / dist], [-dy / dist, dx / dist])
        } else {
            ([0.0, 1.0], [-1.0, 0.0])
        };

        // Perpendicular Tangent Flank Steering:
        let flank_dest = if fight.memory.unseen_for(now) > 0.5 {
            toward_spot
        } else {
            compute_flank_steering(
                walker.reference,
                walker.position,
                kit.radius,
                toward_spot,
                their_radius,
                reach,
                c.others,
                &state.combat,
                target,
                &state.dead,
            )
        };

        // Perpendicular Boid Separation:
        let legs = world::body_parts::leg_speed_mult(order, state, walker.reference);
        let current_speed = g.speed(kit.walk, kit.run) * legs;
        let sep_vel = compute_boid_separation(
            walker.reference,
            walker.position,
            kit.radius,
            fwd,
            tangent,
            current_speed,
            c.others,
            &state.combat,
            target,
            &state.dead,
        );

        if now >= fight.repath_at || walker.next >= walker.path.len() {
            fight.repath_at = now + REPATH_SECONDS;
            if c.mesh.path(walker.position, flank_dest).is_some() {
                go(c.mesh, walker, flank_dest, true);
            } else {
                go(c.mesh, walker, toward_spot, true);
            }
        }

        let walking = step(walker, current_speed, c.dt);
        // Apply lateral boid separation to break collinear alignment:
        walker.position[0] += sep_vel[0] * c.dt;
        walker.position[1] += sep_vel[1] * c.dt;

        if !walking {
            face(walker, toward_spot, c);
        }
        return FightFrame {
            gait: walking.then_some(g),
            attacked: false,
        };
    }

    // 3. In Reach:
    walker.path.clear();
    face(walker, goal, c);
    let due = now >= fight.attack_until && (!fight.holding || now >= fight.hold_until);
    if !due {
        return FightFrame::default();
    }
    let situation = MeleeSituation {
        skill: combat_ai::melee_skill(order, state, walker.reference, weapon),
        target_attacking: them.is_some_and(|o| o.attacking),
        blocking: false,
        target_recoiling: false,
        target_unconscious: state.unconscious.contains(&target),
        unarmed: weapon.is_none(),
        can_block: false,
        target_in_combat: state.combat.contains_key(&target),
        target_is_player: target == PLAYER_REF,
        holding: fight.holding,
    };
    let scores = combat_ai::melee_scores(&kit.style, &situation, &s);
    match combat_ai::melee_choice(&scores, dice.roll(), fight.holding) {
        MeleeChoice::Attack => {
            fight.holding = false;
            let power = combat_ai::power_attack(
                &kit.style,
                situation.target_recoiling,
                situation.target_unconscious,
                1.0,
                dice.roll(),
            );
            let attack_sec = kit.attack_seconds(weapon);
            let windup_sec = (attack_sec * 0.32).clamp(0.1, (attack_sec * 0.5).max(0.1));
            let (dx, dy) = (goal[0] - walker.position[0], goal[1] - walker.position[1]);
            let dist = dx.hypot(dy).max(1e-3);
            let lunge_dir = [dx / dist, dy / dist];
            let lunge_speed = (kit.run * 0.5).max(kit.walk);

            fight.windup = Some(MeleeWindup {
                start: now,
                strike_at: now + windup_sec,
                end: now + attack_sec,
                lunge_dir,
                lunge_speed,
                power,
                weapon: weapon.map(|w| w.form_id),
            });
            fight.attack_until = now + attack_sec;

            FightFrame {
                gait: None,
                attacked: true, // Triggers attack animation start immediately!
            }
        }
        MeleeChoice::Hold => {
            fight.holding = true;
            fight.hold_until = now + combat_ai::hold_seconds(&kit.style, dice.unit());
            FightFrame::default()
        }
        MeleeChoice::Block | MeleeChoice::Nothing => {
            fight.holding = false;
            FightFrame::default()
        }
    }
}

/// Searching for a target unseen for 15 s: to where it was last seen, then
/// spots within the search radius around it (see the module notes).
fn search(
    c: &mut FightCtx,
    state: &mut GameState,
    walker: &mut Walker,
    kit: &Kit,
    fight: &mut Fight,
    dice: &mut Dice,
) -> FightFrame {
    let order = c.order;
    let settings = c.settings;
    let s = |name: &str, d: f32| settings.get(order, name, d);
    let now = c.now;
    let spot = fight.memory.last_known;
    let idle = walker.next >= walker.path.len();
    if idle {
        if !fight.searched_spot {
            fight.searched_spot = true;
            println!(
                "{now:.1} s: {} searches for {}.",
                walker.reference, fight.memory.target
            );
            go(c.mesh, walker, spot, false);
        } else if now >= fight.search_at {
            fight.search_at = now + s("fCombatSearchAreaUpdateTime", 5.0);
            let radius = combat_ai::search_radius(state.player_world.is_some(), &s);
            let angle = dice.unit() * std::f32::consts::TAU;
            let r = radius * dice.unit().sqrt();
            let to = [
                spot[0] + r * angle.sin(),
                spot[1] + r * angle.cos(),
                spot[2],
            ];
            go(c.mesh, walker, to, false);
        }
    }
    let walking = step(walker, kit.walk, c.dt);
    FightFrame {
        gait: walking.then_some(Gait::Walk),
        attacked: false,
    }
}

/// Someone running from `threat` (an unaggressive actor, see `detect`):
/// `fCombatFleeNormalDistance` (2048) straight away from it, else half
/// that, else a quarter (where the navmesh allows); whether they're
/// moving.
pub(crate) fn flee(
    order: &LoadOrder,
    settings: &SettingCache,
    mesh: &NavMesh,
    state: &GameState,
    walker: &mut Walker,
    kit: &Kit,
    dt: f32,
) -> bool {
    let Some(threat) = walker.fleeing else {
        return false;
    };
    let Some(from) = position_of(order, state, threat).filter(|_| !state.dead.contains(&threat))
    else {
        walker.fleeing = None;
        return false;
    };
    if walker.next >= walker.path.len() {
        let far = settings.get(order, "fCombatFleeNormalDistance", 2048.0);
        let (dx, dy) = (walker.position[0] - from[0], walker.position[1] - from[1]);
        let l = dx.hypot(dy).max(1e-3);
        for share in [1.0, 0.5, 0.25] {
            let to = [
                walker.position[0] + dx / l * far * share,
                walker.position[1] + dy / l * far * share,
                walker.position[2],
            ];
            if go(mesh, walker, to, false) {
                break;
            }
        }
    }
    step(walker, kit.run, dt)
}

/// Who's been noticed, kept per actor (see `detect`).
pub(crate) type Noticed = HashSet<FormId>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_are_measured_off_the_heading_both_ways() {
        use std::f32::consts::FRAC_PI_2;
        // Facing north: a target due east is a quarter turn to the right,
        // one due west a quarter to the left; one 100 up at 100 is 45° up.
        let (yaw, _) = offsets([0.0; 3], 0.0, [10.0, 0.0, 0.0]);
        assert!((yaw - FRAC_PI_2).abs() < 1e-5);
        let (yaw, _) = offsets([0.0; 3], 0.0, [-10.0, 0.0, 0.0]);
        assert!((yaw + FRAC_PI_2).abs() < 1e-5);
        // Facing east, a target just north of east is a little left.
        let (yaw, pitch) = offsets([0.0; 3], FRAC_PI_2, [100.0, 1.0, 100.0]);
        assert!(yaw < 0.0 && yaw > -0.02, "{yaw}");
        assert!((pitch - std::f32::consts::FRAC_PI_4).abs() < 1e-3);
    }

    #[test]
    fn dice_give_numbers_from_zero_to_one() {
        let mut dice = Dice(0x1234_5678_9ABC_DEF0);
        let draws: Vec<f32> = (0..1000).map(|_| dice.unit()).collect();
        assert!(draws.iter().all(|u| (0.0..1.0).contains(u)));
        // Spread over the range.
        assert!(draws.iter().any(|u| *u < 0.1) && draws.iter().any(|u| *u > 0.9));
    }

    #[test]
    fn single_pursuer_targets_player_directly() {
        let me = FormId(101);
        let target = FormId(1);
        let my_pos = [0.0, 0.0, 0.0];
        let target_pos = [0.0, 500.0, 0.0];
        let others = vec![];
        let mut combat = HashMap::new();
        combat.insert(me, target);
        let dead = HashSet::new();

        let steer = compute_flank_steering(
            me,
            my_pos,
            30.0,
            target_pos,
            30.0,
            80.0,
            &others,
            &combat,
            target,
            &dead,
        );
        assert_eq!(steer, target_pos);
    }

    #[test]
    fn multiple_pursuers_spread_laterally_into_crescent_pincer() {
        let p1 = FormId(101); // Left pursuer
        let p2 = FormId(102); // Center pursuer
        let p3 = FormId(103); // Right pursuer
        let target = FormId(1);

        let target_pos = [0.0, 500.0, 0.0];
        let pos1 = [-100.0, 100.0, 0.0];
        let pos2 = [0.0, 100.0, 0.0];
        let pos3 = [100.0, 100.0, 0.0];

        let mut combat = HashMap::new();
        combat.insert(p1, target);
        combat.insert(p2, target);
        combat.insert(p3, target);
        let dead = HashSet::new();

        let seen = vec![
            Seen {
                reference: p1,
                position: pos1,
                moving: true,
                running: true,
                attacking: false,
                radius: 30.0,
            },
            Seen {
                reference: p2,
                position: pos2,
                moving: true,
                running: true,
                attacking: false,
                radius: 30.0,
            },
            Seen {
                reference: p3,
                position: pos3,
                moving: true,
                running: true,
                attacking: false,
                radius: 30.0,
            },
        ];

        let steer1 = compute_flank_steering(
            p1,
            pos1,
            30.0,
            target_pos,
            30.0,
            80.0,
            &seen,
            &combat,
            target,
            &dead,
        );
        let steer2 = compute_flank_steering(
            p2,
            pos2,
            30.0,
            target_pos,
            30.0,
            80.0,
            &seen,
            &combat,
            target,
            &dead,
        );
        let steer3 = compute_flank_steering(
            p3,
            pos3,
            30.0,
            target_pos,
            30.0,
            80.0,
            &seen,
            &combat,
            target,
            &dead,
        );

        // p1 (left) should have negative X lateral offset relative to target
        assert!(steer1[0] < target_pos[0], "Left pursuer must flank left: {}", steer1[0]);
        // p3 (right) should have positive X lateral offset relative to target
        assert!(steer3[0] > target_pos[0], "Right pursuer must flank right: {}", steer3[0]);
        // p2 (center) should stay near center X
        assert!((steer2[0] - target_pos[0]).abs() < 5.0, "Center pursuer near center: {}", steer2[0]);

        // All crescent destinations are within encirclement reach of target
        let r1 = distance(steer1, target_pos);
        let r2 = distance(steer2, target_pos);
        let r3 = distance(steer3, target_pos);
        assert!((r1 - 128.0).abs() < 10.0, "p1 on crescent ring: {r1}");
        assert!((r2 - 128.0).abs() < 10.0, "p2 on crescent ring: {r2}");
        assert!((r3 - 128.0).abs() < 10.0, "p3 on crescent ring: {r3}");
    }

    #[test]
    fn trailing_pursuer_experiences_lateral_boid_repulsion() {
        let p_front = FormId(101);
        let p_behind = FormId(102);
        let target = FormId(1);

        let mut combat = HashMap::new();
        combat.insert(p_front, target);
        combat.insert(p_behind, target);
        let dead = HashSet::new();

        // Target at [0, 500, 0]. Approach vector fwd = [0, 1], tangent = [-1, 0] or [1, 0].
        let fwd = [0.0, 1.0];
        let tangent = [-1.0, 0.0];

        // p_front is at [2.0, 150.0, 0.0] (slightly right)
        // p_behind is at [0.0, 100.0, 0.0]
        let seen = vec![Seen {
            reference: p_front,
            position: [2.0, 150.0, 0.0],
            moving: true,
            running: true,
            attacking: false,
            radius: 30.0,
        }];

        let repulse = compute_boid_separation(
            p_behind,
            [0.0, 100.0, 0.0],
            30.0,
            fwd,
            tangent,
            200.0,
            &seen,
            &combat,
            target,
            &dead,
        );

        // Repulsion must be non-zero and push laterally along tangent
        assert!(repulse[0].abs() > 10.0, "Should apply lateral repulsion: {:?}", repulse);
        assert_eq!(repulse[1], 0.0, "Perpendicular repulsion has 0 longitudinal component");

        // When already wide apart laterally (e.g. at [150, 100, 0]), repulsion is zero
        let repulse_wide = compute_boid_separation(
            p_behind,
            [150.0, 100.0, 0.0],
            30.0,
            fwd,
            tangent,
            200.0,
            &seen,
            &combat,
            target,
            &dead,
        );
        assert_eq!(repulse_wide, [0.0, 0.0], "No repulsion when side-by-side");
    }

    #[test]
    fn attack_windup_deceleration_and_evasion_logic() {
        let windup = MeleeWindup {
            start: 10.0,
            strike_at: 10.5,
            end: 11.0,
            lunge_dir: [0.0, 1.0],
            lunge_speed: 150.0,
            power: false,
            weapon: None,
        };

        // Decelerated momentum test over time:
        let duration = windup.strike_at - windup.start;
        // At start (t = 10.0): progress = 0, full momentum
        let prog_0 = ((10.0 - windup.start) / duration).clamp(0.0, 1.0);
        let mom_0 = (1.0 - prog_0) * windup.lunge_speed;
        assert_eq!(mom_0, 150.0);

        // Halfway (t = 10.25): progress = 0.5, half momentum
        let prog_half = ((10.25 - windup.start) / duration).clamp(0.0, 1.0);
        let mom_half = (1.0 - prog_half) * windup.lunge_speed;
        assert!((mom_half - 75.0).abs() < 1e-3);

        // At strike key (t = 10.5): progress = 1.0, momentum decelerated to 0
        let prog_end = ((10.5 - windup.start) / duration).clamp(0.0, 1.0);
        let mom_end = (1.0 - prog_end) * windup.lunge_speed;
        assert_eq!(mom_end, 0.0);

        // Evasion check: player backsteps from 70 units away to 120 units away
        let reach = 80.0;
        let my_radius = 30.0;
        let player_radius = 30.0;
        let gap_close = combat_ai::gap(70.0, my_radius, player_radius);
        let gap_evaded = combat_ai::gap(150.0, my_radius, player_radius);

        assert!(gap_close <= reach, "Target in reach before backstep: {gap_close}");
        assert!(gap_evaded > reach, "Target evaded after backstep: {gap_evaded}");
    }
}
