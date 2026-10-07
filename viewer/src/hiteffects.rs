//! What hits sound like (`world::impacts`, read from the game's code): the
//! player's shots and blows (`combat`) and people's and creatures'
//! (`fighting`) report each hit here, and this plays it as the game's hit
//! handler does.
//!
//! - **On someone** (`0088e1e0`): the blow's impact sounds, each only when
//!   the camera is nearer than that sound's largest distance; their hurt
//!   line ("Hit", `0089a760`) when the hit took more than
//!   `fCombatSpeakHitThreshold` of their health or by chance, no sooner
//!   than the shared cooldown (`fDialogHitSoundCooldownMin`–`Max`, 2–4 s)
//!   and 1.5 s after any combat line (`009839b0`); on death (`0089d900`)
//!   their death cry: a creature's "death" sound, else their "Death"
//!   line. The player hit gets the `GetHit` image space modifier at
//!   strength `fGetHitPainMult` (1.5).
//! - **On the world** (shots, `009c20e0`): the weapon's impact for the
//!   struck surface's Havok material (kept with each collision triangle;
//!   land without one counts as dirt, the game's default): its two sounds.
//!
//! Not drawn (they need the game's systems run as it runs them): the
//! impacts' effect models (their controllers, billboards and particles),
//! decals on the world and on skin, blood on the player's screen. Sounds
//! aren't placed in the world (no falloff or direction, as in `sounds`).

use std::collections::HashMap;
use std::sync::Arc;

use bevy::audio::{AudioPlayer, AudioSource};
use bevy::prelude::*;
use esm::FormId;
use world::dialogue::{Speaker, PLAYER_REF};
use world::impacts::{self, CombatVoice, DeathCry, Impact, Material};

use crate::dialogue::DialogueState;
use crate::sounds::SoundRequests;
use crate::walk::game_point;
use crate::{FlyCamera, GameFiles};

/// One hit, as `combat` and `fighting` report it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HitReport {
    pub attacker: FormId,
    /// Who was hit; `None`: the world (a wall, the floor).
    pub target: Option<FormId>,
    /// The hit's weapon (`None`: fists or a creature's own attack).
    pub weapon: Option<FormId>,
    /// Where it struck (game units).
    pub point: [f32; 3],
    /// On the world: the struck surface's Havok material, if it has one.
    pub havok: Option<u32>,
    /// On someone: the health damage, and whether it killed them.
    pub damage: f32,
    pub killed: bool,
}

/// Hits reported this frame.
#[derive(Resource, Default)]
pub struct HitReports(pub Vec<HitReport>);

impl HitReports {
    /// A shot from `eye` along `dir` striking the collider's triangle `tri`
    /// `d` units away.
    pub fn shot_on_world(
        &mut self,
        collider: &physics::Collider,
        (eye, dir): ([f32; 3], [f32; 3]),
        (d, tri): (f32, u32),
        attacker: FormId,
        weapon: FormId,
    ) {
        self.0.push(HitReport {
            attacker,
            target: None,
            weapon: Some(weapon),
            point: [0, 1, 2].map(|k| eye[k] + dir[k] * d),
            havok: collider.material(tri),
            damage: 0.0,
            killed: false,
        });
    }
}

/// The combat lines' clocks, and the impact records read.
#[derive(Resource, Default)]
pub struct HitEffects {
    voice: CombatVoice,
    impacts: HashMap<FormId, Option<Impact>>,
}

pub struct HitEffectsPlugin;

impl Plugin for HitEffectsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HitReports>()
            .init_resource::<HitEffects>()
            .init_resource::<crate::grade::PhysiologicalState>()
            .add_systems(
                Update,
                play_hits
                    .after(crate::combat::player_attack)
                    .after(crate::ai::move_actors),
            );
    }
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// 0..1 from the game state's dice.
fn unit(state: &mut world::scripting::GameState) -> f32 {
    (state.roll() % 1_000_000) as f32 / 1_000_000.0
}

/// Everything playing a hit needs.
#[derive(bevy::ecs::system::SystemParam)]
pub struct HitParams<'w, 's> {
    commands: Commands<'w, 's>,
    time: Res<'w, Time>,
    game: Res<'w, GameFiles>,
    state: ResMut<'w, DialogueState>,
    reports: ResMut<'w, HitReports>,
    effects: ResMut<'w, HitEffects>,
    sounds: ResMut<'w, SoundRequests>,
    screen: ResMut<'w, crate::effects::Effects>,
    audio: ResMut<'w, Assets<AudioSource>>,
    cameras: Query<'w, 's, &'static Transform, With<FlyCamera>>,
    phys: ResMut<'w, crate::grade::PhysiologicalState>,
}

/// Plays this frame's hits.
pub fn play_hits(mut p: HitParams) {
    if p.reports.0.is_empty() {
        return;
    }
    let reports = std::mem::take(&mut p.reports.0);
    let game = p.game.0.clone();
    let now = p.time.elapsed_secs();
    let now_ms = (p.time.elapsed_secs_f64() * 1000.0) as u64;
    let eye = p
        .cameras
        .single()
        .map(|c| game_point(c.translation))
        .unwrap_or([0.0; 3]);
    for r in reports {
        match r.target {
            Some(target) => on_someone(&mut p, &game, &r, target, (eye, now, now_ms)),
            None => on_world(&mut p, &game, &r),
        }
    }
}

/// The impact record, read once.
fn impact(effects: &mut HitEffects, order: &esm::LoadOrder, id: FormId) -> Option<Impact> {
    effects
        .impacts
        .entry(id)
        .or_insert_with(|| Impact::load(order, id))
        .clone()
}

/// A shot striking the world (`009c20e0`): its impact's two sounds, at
/// any distance.
fn on_world(p: &mut HitParams, game: &cellview::Game, r: &HitReport) {
    let order = &game.order;
    let Some(weapon) = r.weapon else {
        return;
    };
    // Land and anything added without a material: the land's default,
    // dirt (`00457880`; the land texture's own material isn't looked up).
    let material = r
        .havok
        .map_or(Material::Dirt, |h| Material::from_havok(h & 0x1f));
    let Some(i) = impacts::surface_impact(order, weapon, material)
        .and_then(|id| impact(&mut p.effects, order, id))
    else {
        return;
    };
    println!(
        "  the shot strikes {} ({})",
        material.name(),
        i.editor_id.as_deref().unwrap_or("")
    );
    p.sounds.0.extend(i.sounds());
}

/// A hit on a person or creature (`0089a760` and the functions it calls).
fn on_someone(
    p: &mut HitParams,
    game: &cellview::Game,
    r: &HitReport,
    target: FormId,
    (eye, now, now_ms): ([f32; 3], f32, u64),
) {
    let order = &game.order;
    let held = world::combat::weapon_in_hand(order, &p.state.0, r.attacker).map(|w| w.form_id);
    let gore_off = game
        .settings
        .float("General", "bDisableAllGore")
        .is_some_and(|v| v != 0.0);
    let effects = {
        let state = &mut p.state.0;
        let mut dice = state.dice.max(1);
        let mut roll = || {
            dice ^= dice << 13;
            dice ^= dice >> 7;
            dice ^= dice << 17;
            dice
        };
        let e = impacts::actor_hit(
            order, state, r.attacker, target, r.weapon, held, None, gore_off, &mut roll,
        );
        state.dice = dice.max(1);
        e
    };
    // Sounds, each heard within its own distance (the player hit hears
    // them all: the game measures from the player then).
    let mut heard = Vec::new();
    for s in &effects.sounds {
        let reach = world::sound::Sound::load(order, *s).map_or(0.0, |s| s.max_distance);
        if target == PLAYER_REF || distance(eye, r.point) < reach {
            p.sounds.0.push(*s);
            heard.push(
                order
                    .get(*s)
                    .and_then(|rr| rr.editor_id().ok().flatten())
                    .unwrap_or_else(|| s.to_string()),
            );
        }
    }
    if !heard.is_empty() {
        println!(
            "  {target} hit ({}): {}",
            effects.material.name(),
            heard.join(", ")
        );
    }
    // The player hit by someone or self-inflicted:
    if target == PLAYER_REF {
        let max_hp = world::combat::max_health(order, &p.state.0, PLAYER_REF).unwrap_or(100.0) as f32;
        let damage = r.damage.max(0.0);
        let heavy_threshold = (max_hp * 0.15).max(15.0);
        let is_heavy = damage >= heavy_threshold;

        // Immediate trauma impulse scaled with hit damage
        let damage_ratio = if max_hp > 0.0 { damage / max_hp } else { 0.2 };
        let trauma_impulse = (damage_ratio * 3.0 + if is_heavy { 0.5 } else { 0.2 }).clamp(0.2, 2.0);
        p.phys.trigger_trauma(trauma_impulse, is_heavy);

        // Heavy hits on the player trigger immediate trauma impulses & boosted pain modifier
        let (modifier, base_strength) = impacts::get_hit_modifier(order);
        let strength = if is_heavy {
            base_strength * (1.0 + (damage / heavy_threshold).min(2.0))
        } else {
            base_strength
        };
        p.screen.instances.push((modifier, now, strength));
    }
    // What they say or cry out.
    let state = &mut p.state.0;
    if r.killed {
        let mut roll = || state.roll();
        match impacts::death_cry(order, target, &mut roll) {
            DeathCry::Sound(s) => p.sounds.0.push(s),
            DeathCry::Line => {
                if target != PLAYER_REF && p.effects.voice.line_allowed(now_ms) {
                    say(p, game, target, impacts::DEATH_TOPIC);
                }
            }
        }
        return;
    }
    let health = world::combat::health(order, state, target).unwrap_or(0.0) as f32;
    let t = unit(state);
    let wants = impacts::says_hurt_line(order, r.damage, health, false, t);
    let (lo, hi) = (
        game.settings
            .float("Audio", "fDialogHitSoundCooldownMin")
            .unwrap_or(2.0),
        game.settings
            .float("Audio", "fDialogHitSoundCooldownMax")
            .unwrap_or(4.0),
    );
    let mut pick = |a: u64, b: u64| a + state.roll() % (b - a + 1);
    let allowed = p.effects.voice.hurt_allowed(now_ms, lo, hi, &mut pick);
    if wants && allowed && target != PLAYER_REF && p.effects.voice.line_allowed(now_ms) {
        say(p, game, target, impacts::HIT_TOPIC);
    }
}

/// Someone says a combat topic's line (`009839b0`): the first their
/// conditions allow (`world::dialogue::pick`), in their voice, their face
/// moving with it.
fn say(p: &mut HitParams, game: &cellview::Game, who: FormId, topic: FormId) {
    let order = &game.order;
    let Some(base) = world::scripting::base_of(order, who) else {
        return;
    };
    let Some(speaker) = Speaker::load(order, who, base) else {
        return;
    };
    let Some(info) = world::dialogue::pick(order, topic, &speaker, &p.state.0) else {
        return;
    };
    let Some(response) = info.responses.first() else {
        return;
    };
    println!("  {who} says \"{}\"", response.text);
    let Some(path) = speaker
        .voice
        .and_then(|v| world::dialogue::voice_path(order, &info, response, v))
    else {
        return;
    };
    let Some(bytes) = game.assets.read(&path).ok().flatten() else {
        return;
    };
    let source = p.audio.add(AudioSource {
        bytes: Arc::from(bytes.into_boxed_slice()),
    });
    let (settings, voice) = crate::faces::voice_playback(game, &path, who);
    p.commands
        .spawn((AudioPlayer::new(source), settings, voice));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shot_on_the_world_keeps_the_struck_triangles_material() {
        let mut c = physics::Collider::new();
        c.add_with_material(
            &[[0.0, -50.0, -50.0], [0.0, 50.0, -50.0], [0.0, 0.0, 50.0]],
            &[[0, 1, 2]],
            9,
        );
        let (d, tri) = c
            .raycast([-100.0, 0.0, 0.0], [1.0, 0.0, 0.0], 500.0)
            .unwrap();
        let mut hits = HitReports::default();
        hits.shot_on_world(
            &c,
            ([-100.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            (d, tri),
            PLAYER_REF,
            FormId(0x123),
        );
        let r = hits.0[0];
        assert_eq!(r.havok, Some(9));
        assert!((r.point[0]).abs() < 1e-3);
        assert_eq!((r.target, r.weapon), (None, Some(FormId(0x123))));
    }

    #[test]
    fn player_heavy_hit_triggers_trauma_and_concussion() {
        let mut phys = crate::grade::PhysiologicalState::default();
        let max_hp = 100.0f32;
        let heavy_threshold = (max_hp * 0.15).max(15.0);

        // Light hit (5 damage)
        let light_dmg = 5.0f32;
        let is_heavy_light = light_dmg >= heavy_threshold;
        assert!(!is_heavy_light);
        let ratio_light = light_dmg / max_hp;
        let impulse_light = (ratio_light * 3.0 + if is_heavy_light { 0.5 } else { 0.2 }).clamp(0.2, 2.0);
        phys.trigger_trauma(impulse_light, is_heavy_light);
        assert!(phys.trauma > 0.0);
        assert_eq!(phys.concussion, 0.0);

        // Heavy hit (35 damage)
        let heavy_dmg = 35.0f32;
        let is_heavy = heavy_dmg >= heavy_threshold;
        assert!(is_heavy);
        let ratio_heavy = heavy_dmg / max_hp;
        let impulse_heavy = (ratio_heavy * 3.0 + if is_heavy { 0.5 } else { 0.2 }).clamp(0.2, 2.0);
        phys.trigger_trauma(impulse_heavy, is_heavy);
        assert!(phys.trauma >= 1.5);
        assert!(phys.concussion >= 0.6);
    }
}
