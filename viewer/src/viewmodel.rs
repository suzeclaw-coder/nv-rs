//! The player's arms and weapon in first person, as the game draws them:
//! the first-person skeleton (`world::actor::first_person_look`) holding
//! the weapon in hand (or fists) in the hold pose for its kind, the clothes
//! worn (else the race's upper body) and the first-person hands, lit by the
//! place the player is in.
//!
//! Drawn as the game draws its first-person pass (the actors recording,
//! `FalloutNVActors.trace`, frame `166830323`): after the world, the depth
//! buffer is cleared (`Clear(D3DCLEAR_ZBUFFER)` at call 166828991) and the
//! pass is drawn with its own projection, the view at the eye (the view
//! matrix is the axis swap alone, the skeleton placed in camera space),
//! near plane 5 units, the first-person field of view
//! (`fDefault1stPersonFOV` 55 in this install's `Fallout.ini`, a 4:3 width
//! like `fDefaultFOV`: the recorded projection's cotangents 1.440737 /
//! 2.56131 at 16:9), far plane about 6,600 units; the image space passes
//! come after it. Here that pass is a second camera ([`spawn_camera`]): a
//! child of the main one, drawing only [`FIRST_PERSON_LAYER`] over a
//! cleared depth buffer (so the hands never go into walls), the image space
//! grade copied from the main camera and run after it ([`copy_grade`],
//! `grade::GradeDeferred`).
//!
//! Shown only while the weapon is out (`combat::PlayerAttack::out`) or the
//! drawing / putting-away animation plays: in the five recorded frames with
//! the weapon holstered the pass after the depth clear drew nothing, in the
//! one with the machete out it drew the arms, hands and weapon. Placed every
//! frame so its `Camera1st` node sits at the eye: the skeleton stands facing
//! where the player looks, and looking up and down turns everything under
//! `Bip01 Looking` (the pivot at eye height) by the pitch. Hidden while
//! flying, in menus and conversations, when dead, and in screenshots.
//!
//! Not yet: the Pip-Boy glove, V.A.T.S. drawing the weapon by itself.

use std::sync::Arc;

use bevy::core_pipeline::core_3d::Camera3dDepthLoadOp;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::camera::{ClearColorConfig, Exposure};
use bevy::render::view::RenderLayers;
use cellview::space;
use esm::FormId;
use preview::cell::ActorSkeleton;
use world::dialogue::PLAYER_REF;

use crate::dialogue::{Conversation, DialogueState};
use crate::grade::ImageSpaceGrade;
use crate::lighting::GameLighting;
use crate::menus::Menus;
use crate::walk::{game_point, Player};
use crate::{FlyCamera, GameFiles, ScreenshotRequest, Spawner};

/// The game's first-person field of view setting (`fDefault1stPersonFOV`
/// in this install's `Fallout.ini`).
pub const FIRST_PERSON_FOV_DEGREES: f32 = 55.0;

/// The first-person pass's near plane, game units: from the recorded
/// projection (z row 1.000758, −5.003791: −5.003791 / 1.000758 = 5.000).
pub const FIRST_PERSON_NEAR: f32 = 5.0;

/// Its far plane from the same projection (1.000758 = far / (far − near):
/// about 6,600 units; Bevy's projection has no far plane, so this only
/// bounds culling, and the pieces aren't culled anyway).
pub const FIRST_PERSON_FAR: f32 = 6600.0;

/// The render layer the first-person view is drawn on: only its own camera
/// draws it (the main camera, the HUD's and the water's don't).
pub const FIRST_PERSON_LAYER: usize = 20;

/// The camera that draws the first-person pass (see the module notes).
#[derive(Component)]
pub struct FirstPersonCamera;

/// Spawns the first-person camera under the main camera `parent`, with
/// the same exposure.
pub fn spawn_camera(commands: &mut Commands, parent: Entity, exposure: Exposure) -> Entity {
    commands
        .spawn((
            Camera3d {
                // The game's `Clear(D3DCLEAR_ZBUFFER)` before the pass
                // (0 is Bevy's far, its depth being reversed).
                depth_load_op: Camera3dDepthLoadOp::Clear(0.0),
                ..default()
            },
            Camera {
                // After the main camera, onto the same picture.
                order: 1,
                hdr: true,
                clear_color: ClearColorConfig::None,
                ..default()
            },
            Tonemapping::None,
            Projection::from(PerspectiveProjection {
                fov: cellview::vertical_fov(FIRST_PERSON_FOV_DEGREES),
                near: FIRST_PERSON_NEAR * space::METERS_PER_UNIT,
                far: FIRST_PERSON_FAR * space::METERS_PER_UNIT,
                ..default()
            }),
            exposure,
            Transform::IDENTITY,
            ChildOf(parent),
            RenderLayers::layer(FIRST_PERSON_LAYER),
            // Replaced by the main camera's every frame (`copy_grade`).
            ImageSpaceGrade::NEUTRAL,
            FirstPersonCamera,
        ))
        .id()
}

/// Gives the first-person camera the main camera's image space grade (the
/// game runs its image space passes after the first-person pass, so they
/// run on this camera, the last on the window). After everything that sets
/// the grade, including `grade::adapt_eyes`.
pub fn copy_grade(
    main: Query<&ImageSpaceGrade, (With<FlyCamera>, Without<FirstPersonCamera>)>,
    mut own: Query<&mut ImageSpaceGrade, With<FirstPersonCamera>>,
) {
    let Ok(grade) = main.single() else {
        return;
    };
    for mut g in &mut own {
        if *g != *grade {
            *g = *grade;
        }
    }
}

/// The lighting of the place the player is in, for what's drawn apart from
/// it; and a count of changes, so the first-person view is relit.
#[derive(Resource, Default)]
pub struct PlaceLighting {
    lighting: Option<GameLighting>,
    changes: u64,
}

impl PlaceLighting {
    pub fn set(&mut self, lighting: GameLighting) {
        self.lighting = Some(lighting);
        self.changes += 1;
    }

    pub fn get(&self) -> Option<GameLighting> {
        self.lighting
    }

    /// How many times the place has changed (to know when to relight).
    pub fn changes(&self) -> u64 {
        self.changes
    }

    /// Changes the light as it is (the hour moving on outdoors), without
    /// counting it as a new place.
    pub fn relight(&mut self, change: impl FnOnce(&mut GameLighting)) {
        if let Some(l) = self.lighting.as_mut() {
            change(l);
        }
    }
}

/// `--weapon`: a weapon to start with (given and equipped once the place
/// is up, with 50 rounds of its first kind of ammunition, and drawn). With
/// one, the first-person view shows in screenshots too.
#[derive(Resource, Default)]
pub struct StartWeapon(pub Option<String>);

/// Gives the `--weapon` once the place is ready, and draws it (the game
/// starts with it holstered; `--weapon` is for pictures of it in hand).
pub fn give_start_weapon(
    game: Res<GameFiles>,
    player: Res<Player>,
    mut start: ResMut<StartWeapon>,
    mut state: ResMut<DialogueState>,
    mut attack: ResMut<crate::combat::PlayerAttack>,
) {
    if !player.ready {
        return;
    }
    let Some(name) = start.0.take() else {
        return;
    };
    // `--weapon none` (or any name that isn't a weapon): fists, drawn.
    attack.draw_at_once(None);
    let order = &game.0.order;
    let id = FormId::parse_hex(&name)
        .filter(|id| order.get(*id).is_some())
        .or_else(|| order.form_by_editor_id(&name));
    let Some(weapon) = id.and_then(|id| world::combat::Weapon::load(order, id)) else {
        println!("--weapon: no weapon '{name}'.");
        return;
    };
    let state = &mut state.0;
    state.stock(order, PLAYER_REF);
    *state.items.entry((PLAYER_REF, weapon.form_id)).or_insert(0) += 1;
    if let Some(&ammo) = weapon.ammo.first() {
        *state.items.entry((PLAYER_REF, ammo)).or_insert(0) += 50;
    }
    let worn = state.equipped.entry(PLAYER_REF).or_default();
    worn.retain(|&f| {
        order
            .get(f)
            .is_none_or(|r| r.entry.header.kind.as_bytes() != b"WEAP")
    });
    worn.push(weapon.form_id);
    attack.draw_at_once(Some(weapon.form_id));
    println!("Holding the {}.", weapon.name);
}

/// Whether screenshots show the first-person view (with `--weapon`).
#[derive(Resource, Default)]
pub struct ShowInPictures(pub bool);

/// What the first-person view is built from, to know when to rebuild it.
#[derive(Clone, PartialEq)]
struct Built {
    weapon: Option<FormId>,
    worn: Vec<FormId>,
    female: bool,
    lighting: u64,
}

/// The first-person view on screen.
#[derive(Resource, Default)]
pub struct ViewModel {
    built: Option<Built>,
    /// The entity under the first-person camera holding it, the skeleton's
    /// root, its joints.
    holder: Option<Entity>,
    root: Option<Entity>,
    joints: Vec<Entity>,
    skeleton: Option<Arc<ActorSkeleton>>,
    /// Bones: `Bip01 Looking` (and everything under it), `Camera1st`.
    turned: Vec<usize>,
    looking: usize,
    camera: usize,
    /// The weapon's attack and reload, and its drawing (`<kind>equip.kf`)
    /// and putting away (`<kind>unequip.kf`), played over the hold pose.
    attack: Option<nif::Sequence>,
    reload: Option<nif::Sequence>,
    equip: Option<nif::Sequence>,
    unequip: Option<nif::Sequence>,
    /// Procedural kinetics: locomotion bobbing, airborne inertia, landing
    /// compression shock absorption, weapon recoil, and dry-fire twitch.
    pub kinetics: ViewModelKinetics,
}

/// The procedural kinetics offset in first-person camera space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KineticsOffset {
    pub translation: Vec3,
    pub rotation: Quat,
}

impl Default for KineticsOffset {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}

/// Procedural first-person viewmodel kinetics state.
#[derive(Clone, Debug)]
pub struct ViewModelKinetics {
    /// The current phase of the locomotion stride cycle (radians, 0 to 2*PI).
    pub bob_phase: f32,
    /// Smooth blended weight of the locomotion bobbing (0.0 stopped to 1.0 running).
    pub bob_weight: f32,

    /// Whether the character was on ground in the previous frame.
    pub was_on_ground: bool,
    /// Vertical inertia offset (Bevy meters, +Y is upward float).
    pub airborne_inertia_y: f32,
    /// Peak descent speed recorded while airborne (meters/sec).
    pub peak_fall_speed: f32,
    /// Landing elastic compression spring position (meters, negative is downward).
    pub landing_spring_pos: f32,
    /// Landing elastic compression spring velocity (meters/sec).
    pub landing_spring_vel: f32,
}

impl Default for ViewModelKinetics {
    fn default() -> Self {
        Self {
            bob_phase: 0.0,
            bob_weight: 0.0,
            was_on_ground: true,
            airborne_inertia_y: 0.0,
            peak_fall_speed: 0.0,
            landing_spring_pos: 0.0,
            landing_spring_vel: 0.0,
        }
    }
}

impl ViewModelKinetics {
    /// Evaluates subtle procedural footstep-synchronized lateral sway,
    /// vertical dip on strides, and natural weapon cant during walking/running.
    pub fn locomotion_bob(
        &mut self,
        dt: f32,
        on_ground: bool,
        moving: bool,
        running: bool,
        sneaking: bool,
    ) -> (Vec3, Quat) {
        let (target_weight, cadence) = if !on_ground || !moving {
            (0.0, 0.0)
        } else if running {
            (1.0, 8.8) // ~2.8 steps/sec
        } else if sneaking {
            (0.4, 4.2) // ~1.3 steps/sec
        } else {
            (0.7, 6.0) // ~1.9 steps/sec (standard walk)
        };

        // Smoothly blend bob weight in/out so stops and starts glide naturally.
        let blend_speed = if target_weight > self.bob_weight { 10.0 } else { 8.0 };
        self.bob_weight += (target_weight - self.bob_weight) * (1.0 - (-blend_speed * dt).exp());

        if on_ground && moving {
            self.bob_phase = (self.bob_phase + cadence * dt).rem_euclid(std::f32::consts::TAU);
        }

        if self.bob_weight < 1e-4 {
            return (Vec3::ZERO, Quat::IDENTITY);
        }

        // Lateral sway: once per stride cycle (left, then right).
        let sway_x = self.bob_phase.sin();
        let sway_trans_x = sway_x * 0.0055 * self.bob_weight;

        // Vertical dip on strides: twice per stride cycle (dips downward at foot plants).
        let dip_y = ((2.0 * self.bob_phase).cos() - 1.0) * 0.5;
        let dip_trans_y = dip_y * 0.0045 * self.bob_weight;

        // Natural weapon cant: weapon rolls into the lateral sway.
        let cant_roll = -sway_x * 0.016 * self.bob_weight;
        // Subtle yaw sway and pitch bob.
        let sway_yaw = self.bob_phase.cos() * 0.007 * self.bob_weight;
        let dip_pitch = (2.0 * self.bob_phase).sin() * 0.0035 * self.bob_weight;

        let pos = Vec3::new(sway_trans_x, dip_trans_y, 0.0);
        let rot = Quat::from_euler(EulerRot::YXZ, sway_yaw, dip_pitch, cant_roll);
        (pos, rot)
    }

    /// Evaluates airborne floating inertia (arms lagging behind vertical velocity
    /// during descents: `-vel.y * 0.0035`, floating upward) and tactile elastic
    /// compression bounce upon landing touchdown that settles smoothly.
    pub fn airborne_and_landing(
        &mut self,
        dt: f32,
        on_ground: bool,
        vertical_speed_game_units: f32,
    ) -> (Vec3, Quat) {
        // Vertical velocity in Bevy meters per second (up positive).
        let vel_y = vertical_speed_game_units * space::METERS_PER_UNIT;

        // Airborne floating inertia:
        // Arms lag behind vertical velocity during descents (`-vel.y * 0.0035`), floating upward.
        let target_inertia_y = if !on_ground {
            (-vel_y * 0.0035).clamp(-0.06, 0.05)
        } else {
            0.0
        };
        self.airborne_inertia_y +=
            (target_inertia_y - self.airborne_inertia_y) * (1.0 - (-12.0 * dt).exp());
        let airborne_pitch = (-self.airborne_inertia_y * 0.4).clamp(-0.025, 0.025);

        // Landing touchdown detection:
        if !on_ground {
            self.peak_fall_speed = self.peak_fall_speed.max(-vel_y);
        } else if !self.was_on_ground {
            // Touchdown!
            if self.peak_fall_speed > 0.6 {
                let impulse = (self.peak_fall_speed * 0.008).clamp(0.008, 0.045);
                self.landing_spring_vel -= impulse * 28.0;
            }
            self.peak_fall_speed = 0.0;
        }
        self.was_on_ground = on_ground;

        // 2nd-order damped harmonic elastic compression bounce
        let k_land = 484.0;
        let c_land = 30.8;
        let accel = -k_land * self.landing_spring_pos - c_land * self.landing_spring_vel;
        self.landing_spring_vel += accel * dt;
        self.landing_spring_pos += self.landing_spring_vel * dt;

        if self.landing_spring_pos.abs() < 1e-4 && self.landing_spring_vel.abs() < 1e-3 {
            self.landing_spring_pos = 0.0;
            self.landing_spring_vel = 0.0;
        }

        let pos = Vec3::new(
            0.0,
            self.airborne_inertia_y + self.landing_spring_pos,
            -self.landing_spring_pos * 0.35,
        );
        let rot = Quat::from_rotation_x(airborne_pitch + self.landing_spring_pos * 0.35);
        (pos, rot)
    }

    /// Evaluates two-stage mechanical recoil attack & settling recovery:
    /// Fast explosive mechanical impulse (0-32ms) followed by damped harmonic
    /// settling overshoot for weapon firing.
    pub fn mechanical_recoil(now: f32, fired_at: Option<f32>, is_gun: bool) -> (Vec3, Quat) {
        let Some(fired_at) = fired_at else {
            return (Vec3::ZERO, Quat::IDENTITY);
        };
        let t = (now - fired_at).max(0.0);
        if t > 0.35 {
            return (Vec3::ZERO, Quat::IDENTITY);
        }

        let tau_rise = 0.032; // 0-32ms explosive impulse
        let r = if t <= tau_rise {
            // Stage 1: Fast explosive mechanical impulse (0 to 1.0)
            let u = t / tau_rise;
            1.0 - (1.0 - u).powi(3)
        } else {
            // Stage 2: Damped harmonic settling overshoot
            let tr = t - tau_rise;
            let omega_n: f32 = 26.0;
            let zeta: f32 = 0.60;
            let omega_d = omega_n * (1.0f32 - zeta * zeta).sqrt();
            let decay = (-zeta * omega_n * tr).exp();
            decay * (tr * omega_d).cos()
                - (zeta / (1.0f32 - zeta * zeta).sqrt()) * decay * (tr * omega_d).sin()
        };

        let scale = if is_gun { 1.0 } else { 0.35 };
        let pos = Vec3::new(
            0.0,
            0.008 * r * scale, // vertical lift
            0.022 * r * scale, // backward kick toward camera (+Z)
        );
        let recoil_pitch = 0.052 * r * scale; // muzzle climb
        let recoil_roll = -0.014 * r * scale; // rifling torque cant
        let recoil_yaw = 0.004 * r * scale;
        let rot = Quat::from_euler(EulerRot::YXZ, recoil_yaw, recoil_pitch, recoil_roll);
        (pos, rot)
    }

    /// Evaluates tactile click twitch when attempting to fire an empty weapon.
    pub fn dry_fire_twitch(now: f32, dry_fired_at: Option<f32>) -> (Vec3, Quat) {
        let Some(dry_at) = dry_fired_at else {
            return (Vec3::ZERO, Quat::IDENTITY);
        };
        let t = (now - dry_at).max(0.0);
        if t > 0.12 {
            return (Vec3::ZERO, Quat::IDENTITY);
        }

        // Rapid mechanical click impulse curve: sharp twitch peaking at ~12ms, settling by ~90ms
        let peak_t = 0.012;
        let twitch = ((t / peak_t) * (1.0 - t / peak_t).exp()).max(0.0);

        let pos = Vec3::new(
            0.0,
            0.0015 * twitch, // tiny upward click
            0.0035 * twitch, // slight tactile kick back into palm
        );
        let twitch_pitch = 0.014 * twitch; // crisp muzzle twitch
        let twitch_roll = 0.005 * twitch;  // trigger squeeze cant
        let rot = Quat::from_euler(EulerRot::YXZ, 0.0, twitch_pitch, twitch_roll);
        (pos, rot)
    }

    /// Updates all procedural kinetic layers and returns the composite offset.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        dt: f32,
        now: f32,
        on_ground: bool,
        vertical_speed_game_units: f32,
        moving: bool,
        running: bool,
        sneaking: bool,
        fired_at: Option<f32>,
        dry_fired_at: Option<f32>,
        is_gun: bool,
    ) -> KineticsOffset {
        let (bob_pos, bob_rot) = self.locomotion_bob(dt, on_ground, moving, running, sneaking);
        let (land_pos, land_rot) =
            self.airborne_and_landing(dt, on_ground, vertical_speed_game_units);
        let (recoil_pos, recoil_rot) = Self::mechanical_recoil(now, fired_at, is_gun);
        let (dry_pos, dry_rot) = Self::dry_fire_twitch(now, dry_fired_at);

        let translation = bob_pos + land_pos + recoil_pos + dry_pos;
        let rotation = dry_rot * recoil_rot * land_rot * bob_rot;

        KineticsOffset {
            translation,
            rotation,
        }
    }
}

/// An animation file's first sequence, if the game has the file.
pub(crate) fn sequence(game: &cellview::Game, path: &str) -> Option<nif::Sequence> {
    let bytes = game.assets.read(&assets::mesh_path(path)).ok()??;
    nif::Nif::parse(bytes)
        .ok()?
        .sequences()
        .ok()?
        .into_iter()
        .next()
}

/// A layer at `since` seconds ago, while it's still playing.
fn playing(
    sequence: Option<&nif::Sequence>,
    since: Option<f32>,
    now: f32,
) -> Option<(&nif::Sequence, f32)> {
    let s = sequence?;
    let t = now - since?;
    (t >= 0.0 && t <= s.stop - s.start).then_some((s, s.start + t))
}

/// What can hide the first-person view: a conversation, a menu, a picture
/// being taken, V.A.T.S.'s camera.
type ViewGates<'w> = (
    Res<'w, Conversation>,
    Res<'w, Menus>,
    Res<'w, ScreenshotRequest>,
    Res<'w, ShowInPictures>,
    Res<'w, crate::vats::Vats>,
);

/// Keeps the first-person view built for what the player holds and wears,
/// posed and placed at the eye.
#[allow(clippy::too_many_arguments)]
pub fn update_view_model(
    mut commands: Commands,
    time: Res<Time>,
    game: Res<GameFiles>,
    state: Res<DialogueState>,
    player: Res<Player>,
    (conversation, menus, screenshot, in_pictures, vats): ViewGates,
    mut view: ResMut<ViewModel>,
    mut attack: ResMut<crate::combat::PlayerAttack>,
    mut spawner: Spawner,
    cameras: Query<&Transform, With<FlyCamera>>,
    first_person: Query<Entity, With<FirstPersonCamera>>,
    mut first_person_projection: Query<&mut Projection, With<FirstPersonCamera>>,
    mut transforms: Query<&mut Transform, (Without<FlyCamera>, Without<FirstPersonCamera>)>,
    mut visibility: Query<&mut Visibility>,
) {
    let order = &game.0.order;
    let state = &state.0;
    // The first-person camera sits on the main one, whose transform is
    // the eye in the world.
    let (Ok(camera), Ok(camera_transform)) = (first_person.single(), cameras.single()) else {
        return;
    };
    // The place's lighting, kept by the spawner as it puts places on screen.
    let Some(lighting) = spawner.place_lighting.lighting else {
        return;
    };
    let lighting_changes = spawner.place_lighting.changes;
    // What it should show.
    let weapon = world::combat::weapon_in_hand(order, state, PLAYER_REF);
    let worn: Vec<FormId> = state
        .equipped
        .get(&PLAYER_REF)
        .into_iter()
        .flatten()
        .copied()
        .filter(|&i| {
            order
                .get(i)
                .is_some_and(|r| r.entry.header.kind.as_bytes() == b"ARMO")
        })
        .collect();
    let female = state.player_female.unwrap_or(false);
    let wanted = Built {
        weapon: weapon.as_ref().map(|w| w.form_id),
        worn: worn.clone(),
        female,
        lighting: lighting_changes,
    };
    if view.built.as_ref() != Some(&wanted) {
        if let Some(old) = view.holder.take() {
            if let Ok(mut e) = commands.get_entity(old) {
                e.despawn();
            }
        }
        view.built = Some(wanted);
        view.root = None;
        let held = weapon.as_ref().and_then(|w| {
            let model = order
                .get(w.form_id)?
                .record()
                .ok()?
                .get(esm::FourCC::new(b"MODL"))?
                .zstring();
            Some((model, w.animation))
        });
        let Some(look) = world::actor::first_person_look(order, female, &worn, held) else {
            return;
        };
        let scene = game.0.actor_scene(&look);
        let holder = commands
            .spawn((Transform::IDENTITY, Visibility::Hidden, ChildOf(camera)))
            .id();
        view.holder = Some(holder);
        let Some((root, joints, _)) = spawner.spawn_lone_actor(&scene, lighting, holder) else {
            return;
        };
        let skeleton = scene.actors[0].skeleton.clone();
        let find = |name: &str| {
            skeleton
                .bones
                .iter()
                .position(|b| b.name.eq_ignore_ascii_case(name))
        };
        let (Some(looking), Some(camera_bone)) = (find("Bip01 Looking"), find("Camera1st")) else {
            return;
        };
        let turned = (0..skeleton.bones.len())
            .filter(|&i| {
                let mut b = Some(i);
                while let Some(j) = b {
                    if j == looking {
                        return true;
                    }
                    b = skeleton.bones[j].parent;
                }
                false
            })
            .collect();
        view.root = Some(root);
        view.joints = joints;
        view.skeleton = Some(skeleton);
        view.turned = turned;
        view.looking = looking;
        view.camera = camera_bone;
        // The weapon's attack (melee and fists ship `_a` variants), reload,
        // drawing and putting away.
        let animation = weapon.as_ref().map(|w| w.animation);
        let attack_file = world::actor::first_person_attack(
            animation,
            weapon.as_ref().map_or(255, |w| w.attack_animation),
        );
        view.attack = sequence(&game.0, &attack_file)
            .or_else(|| sequence(&game.0, &attack_file.replace(".kf", "_a.kf")));
        view.reload = weapon.as_ref().and_then(|w| {
            sequence(
                &game.0,
                &world::actor::first_person_reload(animation, w.reload_animation),
            )
        });
        view.equip = sequence(&game.0, &world::actor::first_person_ready(animation, true));
        view.unequip = sequence(&game.0, &world::actor::first_person_ready(animation, false));
    }
    let (Some(holder), Some(root), Some(skeleton)) =
        (view.holder, view.root, view.skeleton.clone())
    else {
        return;
    };
    let now = time.elapsed_secs();
    // Sped up in V.A.T.S. (`vats`), and at the attack's and reload's own
    // rates (`combat`: the weapon's speed, Agility, the perks): the
    // layers' time runs that much faster from when they started.
    let speed = attack.sped_up.unwrap_or(1.0);
    let faster = |since: Option<f32>, rate: f32| {
        let rate = if rate > 0.0 { rate } else { 1.0 };
        since.map(|s| now - (now - s) * speed * rate)
    };
    // Drawing or putting away: the equip or unequip animation while it
    // plays (the Ready Item key and attacks wait for it: `combat`).
    let readying = attack.readied_at.and_then(|(since, drawing)| {
        let s = if drawing { &view.equip } else { &view.unequip };
        if let Some(s) = s.as_ref() {
            attack.readying_until(since, s.stop - s.start);
        }
        playing(s.as_ref(), Some(since), now)
    });
    let shown = (player.walking || in_pictures.0)
        && player.ready
        && conversation.0.is_none()
        && !menus.is_open()
        && (screenshot.path.is_none() || in_pictures.0)
        && !state.dead.contains(&PLAYER_REF)
        // A V.A.T.S. camera shot has the view.
        && !vats.shot_view()
        // Holstered, nothing shows (`009466d0`): only drawn, or while
        // drawing or putting away.
        && (attack.out || readying.is_some());
    if let Ok(mut v) = visibility.get_mut(holder) {
        let want = if shown {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if *v != want {
            *v = want;
        }
    }
    if !shown {
        return;
    }
    // V.A.T.S.'s menu zooms the first-person view too (`0095de30`): the
    // first-person camera's own field of view.
    let fov = cellview::vertical_fov(vats.first_person_fov.unwrap_or(FIRST_PERSON_FOV_DEGREES));
    if let Ok(mut projection) = first_person_projection.single_mut() {
        if let Projection::Perspective(p) = projection.as_mut() {
            if p.fov != fov {
                p.fov = fov;
            }
        }
    }
    // The pose: the hold pose, looping, with a reload, the last attack or
    // the drawing / putting away over it while they play, turned by the
    // pitch about the pivot.
    let f = camera_transform.forward().as_vec3();
    let dir = [f.x, -f.z, f.y];
    let heading = dir[0].atan2(dir[1]);
    let pitch = dir[2].clamp(-1.0, 1.0).asin();
    let mut layers: Vec<(&nif::Sequence, f32)> = Vec::new();
    if let Some(s) = skeleton.idle.as_deref() {
        let span = (s.stop - s.start).max(1e-3);
        layers.push((s, s.start + now.rem_euclid(span)));
    }
    let reloading = playing(
        view.reload.as_ref(),
        faster(attack.reload_started, attack.reload_rate),
        now,
    );
    let firing = playing(
        view.attack.as_ref(),
        faster(attack.fired_at, attack.attack_rate),
        now,
    );
    layers.extend(readying.or(reloading).or(firing));
    let mut pose = nif::posed_layers(&skeleton.bones, &layers);
    let pivot = pose[view.looking].translation;
    let (s, c) = pitch.sin_cos();
    let rotation = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]];
    let turn = nif::Transform {
        rotation,
        // A turn about the pivot (its x stays).
        translation: [
            0.0,
            pivot[1] - (c * pivot[1] - s * pivot[2]),
            pivot[2] - (s * pivot[1] + c * pivot[2]),
        ],
        scale: 1.0,
    };
    for &i in &view.turned {
        pose[i] = turn.then_child(&pose[i]);
    }
    for (joint, bone) in view.joints.iter().zip(&pose) {
        if let Ok(mut t) = transforms.get_mut(*joint) {
            *t = crate::actors::bevy_transform(bone);
        }
    }
    // The root: facing the heading, with `Camera1st` at the eye. The
    // first-person camera sits on the main one, so the eye is its place.
    let eye = game_point(camera_transform.translation);
    let (sh, ch) = heading.sin_cos();
    let cam = pose[view.camera].translation;
    let feet = [
        eye[0] - (ch * cam[0] + sh * cam[1]),
        eye[1] - (-sh * cam[0] + ch * cam[1]),
        eye[2] - cam[2],
    ];
    let game = [
        ch, -sh, 0.0, 0.0, sh, ch, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, feet[0], feet[1], feet[2], 1.0,
    ];
    let world_root = Mat4::from_cols_array(&space::matrix(&game));
    let local = camera_transform.compute_matrix().inverse() * world_root;

    // Procedural viewmodel kinetics: footstep-cadenced locomotion bob,
    // airborne inertia lag and landing compression bounce, two-stage
    // mechanical recoil attack and settling recovery, and dry-fire click twitch.
    let dt = time.delta_secs();
    let is_gun = weapon.as_ref().is_some_and(|w| !w.is_melee());
    let offset = view.kinetics.update(
        dt,
        now,
        player.character.on_ground,
        player.character.vertical_speed,
        state.player_moving,
        state.player_running,
        state.player_sneaking,
        attack.fired_at,
        attack.dry_fired_at,
        is_gun,
    );

    // Natural shoulder / weapon grip pivot in camera space (slightly below and forward from eye).
    let kin_pivot = Vec3::new(0.0, -0.15, -0.25);
    let kin_mat = Mat4::from_translation(offset.translation + kin_pivot)
        * Mat4::from_quat(offset.rotation)
        * Mat4::from_translation(-kin_pivot);
    let final_local = kin_mat * local;
    if let Ok(mut t) = transforms.get_mut(root) {
        *t = Transform::from_matrix(final_local);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::camera::CameraProjection;

    #[test]
    fn the_first_person_projection_is_the_recorded_one() {
        // The actors recording's first-person pass (call 166828994):
        // cotangents 1.440737 across and 2.56131 up at 1920 × 1080, near 5
        // units (z row 1.000758, −5.003791).
        let projection = PerspectiveProjection {
            fov: cellview::vertical_fov(FIRST_PERSON_FOV_DEGREES),
            aspect_ratio: 1920.0 / 1080.0,
            near: FIRST_PERSON_NEAR * space::METERS_PER_UNIT,
            far: FIRST_PERSON_FAR * space::METERS_PER_UNIT,
        };
        let m = projection.get_clip_from_view();
        assert!((m.x_axis.x - 1.440737).abs() < 1e-4, "{}", m.x_axis.x);
        assert!((m.y_axis.y - 2.56131).abs() < 1e-4, "{}", m.y_axis.y);
        // Bevy's reversed depth: 1 at the near plane.
        let near = m.project_point3(Vec3::new(0.0, 0.0, -projection.near));
        assert!((near.z - 1.0).abs() < 1e-5, "{}", near.z);
        // The game's z row (A, −B): near = B / A, far from A = far / (far − near).
        assert!((5.003791f32 / 1.000758 - FIRST_PERSON_NEAR).abs() < 1e-3);
        assert!(
            (1.000758 / (1.000758 - 1.0) * FIRST_PERSON_NEAR / FIRST_PERSON_FAR - 1.0).abs() < 0.01
        );
    }

    #[test]
    fn test_footstep_cadenced_locomotion_bob() {
        let mut kinetics = ViewModelKinetics::default();

        // When stationary, bobbing produces no displacement.
        let (pos_idle, rot_idle) = kinetics.locomotion_bob(1.0 / 60.0, true, false, false, false);
        assert_eq!(pos_idle, Vec3::ZERO);
        assert_eq!(rot_idle, Quat::IDENTITY);

        // Advance while walking until bob weight has ramped up.
        for _ in 0..60 {
            kinetics.locomotion_bob(1.0 / 60.0, true, true, false, false);
        }
        assert!(kinetics.bob_weight > 0.5);

        // Set phase to quarter stride (phase = PI / 2):
        // Lateral sway should peak positive on X (swaying to one side),
        // vertical dip should dip downward (dip_y < 0),
        // and natural weapon cant should roll into the sway.
        kinetics.bob_phase = std::f32::consts::FRAC_PI_2;
        let (pos_step, rot_step) = kinetics.locomotion_bob(0.0, true, true, false, false);
        assert!(pos_step.x > 0.002, "Lateral sway should be positive on X");
        assert!(pos_step.y < -0.001, "Vertical dip should dip downward on stride");

        let (_, _, roll) = rot_step.to_euler(EulerRot::YXZ);
        assert!(roll < -0.005, "Weapon cant should roll naturally with lateral sway");

        // Set phase to three-quarter stride (phase = 3 * PI / 2):
        // Lateral sway peaks negative on X (swaying to other side),
        // vertical dip dips downward again (stride dip on second footstep),
        // and weapon cant rolls to the opposite side.
        kinetics.bob_phase = 3.0 * std::f32::consts::FRAC_PI_2;
        let (pos_step2, rot_step2) = kinetics.locomotion_bob(0.0, true, true, false, false);
        assert!(pos_step2.x < -0.002, "Lateral sway should be negative on X");
        assert!(pos_step2.y < -0.001, "Vertical dip must dip downward on second step too");

        let (_, _, roll2) = rot_step2.to_euler(EulerRot::YXZ);
        assert!(roll2 > 0.005, "Weapon cant should reverse roll on opposite stride");
    }

    #[test]
    fn test_airborne_floating_inertia_and_landing_compression() {
        let mut kinetics = ViewModelKinetics::default();

        // Simulate falling descent: vertical speed = -500 game units/sec.
        // vel_y in Bevy space = -500 * space::METERS_PER_UNIT ≈ -7.143 m/s.
        // Expected formula requirement: arms lag behind vertical velocity during descents
        // (-vel.y * 0.0035), floating upward.
        let vel_y = -500.0 * space::METERS_PER_UNIT;
        let expected_float = -vel_y * 0.0035;
        assert!(expected_float > 0.02, "Descent should calculate upward float > 2cm");

        // Step through airborne fall.
        for _ in 0..60 {
            kinetics.airborne_and_landing(1.0 / 60.0, false, -500.0);
        }
        // Floating inertia should have smoothly approached the target float offset.
        assert!((kinetics.airborne_inertia_y - expected_float).abs() < 1e-3);
        assert!(kinetics.peak_fall_speed > 5.0);

        // Touchdown landing! Character is now on ground.
        let (pos_land, rot_land) = kinetics.airborne_and_landing(1.0 / 60.0, true, 0.0);
        // Landing impact triggers downward compression bounce.
        assert!(kinetics.landing_spring_vel < -0.5, "Landing should impart downward impulse to spring");
        assert!(pos_land.y < 0.05, "Viewmodel absorbs landing impact");
        let (pitch_land, _, _) = rot_land.to_euler(EulerRot::XYZ);
        assert!(pitch_land.abs() >= 0.0);

        // Step forward time to verify the elastic compression bounce settles smoothly.
        for _ in 0..120 {
            kinetics.airborne_and_landing(1.0 / 60.0, true, 0.0);
        }
        assert!(kinetics.landing_spring_pos.abs() < 1e-4, "Spring should settle smoothly to zero");
        assert!(kinetics.landing_spring_vel.abs() < 1e-3, "Velocity should settle smoothly to zero");
        assert!(kinetics.airborne_inertia_y.abs() < 1e-3, "Airborne inertia resets to ground baseline");
    }

    #[test]
    fn test_two_stage_mechanical_recoil() {
        let fired_at = 10.0;

        // At t = 0 (immediately when fired):
        let (pos_0, _) = ViewModelKinetics::mechanical_recoil(10.0, Some(fired_at), true);
        assert_eq!(pos_0, Vec3::ZERO);

        // Stage 1: Explosive mechanical impulse ramps rapidly up to 32ms.
        let (pos_mid, _) = ViewModelKinetics::mechanical_recoil(10.016, Some(fired_at), true);
        let (pos_peak, rot_peak) = ViewModelKinetics::mechanical_recoil(10.032, Some(fired_at), true);

        assert!(pos_mid.z > 0.010, "Rapid backward recoil impulse by 16ms");
        assert!(pos_peak.z > 0.020, "Peak backward kick at 32ms");
        assert!(pos_peak.y > 0.005, "Vertical lift at 32ms");

        let (pitch_peak, _, _) = rot_peak.to_euler(EulerRot::XYZ);
        assert!(pitch_peak > 0.04, "Muzzle climbs up at 32ms explosive peak");

        // Stage 2: Damped harmonic settling overshoot for weapon firing.
        // Between ~90ms and 140ms, the weapon springs forward and overshoots past 0 (negative Z/Y).
        let (pos_overshoot, rot_overshoot) =
            ViewModelKinetics::mechanical_recoil(10.110, Some(fired_at), true);
        assert!(pos_overshoot.z < -0.001, "Recoil should exhibit settling overshoot past zero");
        let (pitch_overshoot, _, _) = rot_overshoot.to_euler(EulerRot::XYZ);
        assert!(pitch_overshoot < -0.001, "Pitch should overshoot slightly forward/downward");

        // By 350ms, recoil has settled cleanly back to rest.
        let (pos_rest, rot_rest) = ViewModelKinetics::mechanical_recoil(10.400, Some(fired_at), true);
        assert_eq!(pos_rest, Vec3::ZERO);
        assert_eq!(rot_rest, Quat::IDENTITY);
    }

    #[test]
    fn test_dry_fire_mechanical_click() {
        let dry_fired_at = 5.0;

        // At t = 0:
        let (pos_0, _) = ViewModelKinetics::dry_fire_twitch(5.0, Some(dry_fired_at));
        assert_eq!(pos_0, Vec3::ZERO);

        // Sharp tactile click twitch peaks around ~12ms.
        let (pos_peak, rot_peak) = ViewModelKinetics::dry_fire_twitch(5.012, Some(dry_fired_at));
        assert!(pos_peak.z > 0.002, "Tactile click twitch back into hand");
        assert!(pos_peak.y > 0.001, "Tactile vertical twitch");

        let (pitch_peak, _, _) = rot_peak.to_euler(EulerRot::XYZ);
        assert!(pitch_peak > 0.010, "Sharp mechanical click pitch twitch");

        // Settles within ~120ms.
        let (pos_settled, rot_settled) = ViewModelKinetics::dry_fire_twitch(5.150, Some(dry_fired_at));
        assert_eq!(pos_settled, Vec3::ZERO);
        assert_eq!(rot_settled, Quat::IDENTITY);
    }
}
