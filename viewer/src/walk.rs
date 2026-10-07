//! Walking through the cell as the player, and going through load doors.
//!
//! Movement uses the game's own numbers where they're known (see
//! `MovementSettings`); the collision is the models' Havok shapes
//! (`cellview::ViewerScene::collision`) and the capsule is the `physics`
//! crate's character.

use bevy::prelude::*;
use cellview::{space, DoorData, ACTIVATE_REACH, EYE_HEIGHT};
use physics::{Character, CharacterShape, Collider};

use crate::exterior::{door_start, PendingExterior};
use crate::{FlyCamera, GameFiles, PendingScene};

/// The game's movement settings, from its code and `FalloutNV.esm`
/// (`GMST`). The game works out a speed as `SpeedMult (100) × 0.01 ×
/// fMoveBaseSpeed`, times `fMoveRunMult` when running and
/// `fMoveSneakMult` when sneaking (`00647d10` / `00647f00` in
/// FalloutNV.exe), and × 0.85 / 0.75 with one or both legs crippled
/// (`world::body_parts::leg_speed_mult`; armour penalties left out).
struct MovementSettings;

impl MovementSettings {
    /// `fMoveBaseSpeed`: 77 in FalloutNV.esm (85 built in).
    const BASE_SPEED: f32 = 77.0;
    /// `fMoveRunMult`: 4 in FalloutNV.esm.
    const RUN_MULT: f32 = 4.0;
    /// `fMoveSneakMult`: 0.57 in FalloutNV.esm.
    const SNEAK_MULT: f32 = 0.57;
    /// `fJumpHeightMin`: 64 (built in, not changed by the game's files).
    const JUMP_HEIGHT: f32 = 64.0;
}

/// The current cell's solid surfaces.
#[derive(Resource)]
pub struct CellCollision(pub Collider);

/// The current cell's load doors.
#[derive(Resource)]
pub struct Doors(pub Vec<DoorData>);

/// The line under the crosshair (a door's destination).
#[derive(Component)]
pub struct Prompt;

/// Damped harmonic oscillator simulating physical landing compression dynamics.
///
/// Tracks ground contact edges and applies an impulse proportional to airtime
/// when touching down from jumps or falls. Smoothly displaces the camera view
/// downward in translation and pitches forward in rotation with stiffness
/// `k = 175.0` and damping `c = 19.5` before settling back to equilibrium.
#[derive(Resource, Component, Debug, Clone, Copy, PartialEq)]
pub struct LandingDip {
    /// Spring stiffness constant k (s^-2).
    pub k: f32,
    /// Spring damping coefficient c (s^-1).
    pub c: f32,
    /// Vertical displacement dip in game units (positive = downwards).
    pub displacement: f32,
    /// Vertical velocity of the spring in game units per second.
    pub velocity: f32,
    /// Pitch dip offset in radians (negative = pitch forward / downward).
    pub pitch_offset: f32,
    /// Elapsed airborne duration in seconds during current jump or fall.
    pub airtime: f32,
    /// Grounded state during previous frame to detect touchdown contact edges.
    pub was_grounded: bool,
}

impl Default for LandingDip {
    fn default() -> Self {
        Self::new()
    }
}

impl LandingDip {
    pub const SPRING_K: f32 = 175.0;
    pub const SPRING_C: f32 = 19.5;

    pub fn new() -> Self {
        Self {
            k: Self::SPRING_K,
            c: Self::SPRING_C,
            displacement: 0.0,
            velocity: 0.0,
            pitch_offset: 0.0,
            airtime: 0.0,
            was_grounded: true,
        }
    }

    /// Applies an instantaneous velocity impulse to the spring.
    pub fn apply_impulse(&mut self, impulse: f32) {
        self.velocity += impulse;
    }

    /// Steps the harmonic oscillator over `dt` seconds using sub-stepped semi-implicit Euler.
    pub fn step(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        let max_substep = 0.005;
        let mut rem = dt.min(0.1);
        while rem > 0.0 {
            let step_dt = rem.min(max_substep);
            rem -= step_dt;
            // Damped harmonic oscillator: a = -k * x - c * v
            let a = -self.k * self.displacement - self.c * self.velocity;
            self.velocity += a * step_dt;
            self.displacement += self.velocity * step_dt;
        }
        if self.displacement.abs() < 1e-4 && self.velocity.abs() < 1e-4 {
            self.displacement = 0.0;
            self.velocity = 0.0;
        }
        // Pitch forward proportional to the dip displacement (approx 0.0065 rad per game unit dip)
        self.pitch_offset = -self.displacement * 0.0065;
    }

    /// Updates ground contact tracking, detects touchdown contact edges, and computes impulses.
    pub fn update(&mut self, is_grounded: bool, dt: f32) {
        if !self.was_grounded && is_grounded {
            // Touchdown contact edge detected!
            if self.airtime > 0.08 {
                // Impulse proportional to airtime (units/s)
                let impulse = (self.airtime * 260.0).clamp(15.0, 420.0);
                self.apply_impulse(impulse);
            }
            self.airtime = 0.0;
        } else if !is_grounded {
            self.airtime += dt;
        } else {
            self.airtime = 0.0;
        }
        self.was_grounded = is_grounded;

        self.step(dt);
    }

    /// Resets the spring state to equilibrium.
    pub fn reset(&mut self) {
        self.displacement = 0.0;
        self.velocity = 0.0;
        self.pitch_offset = 0.0;
        self.airtime = 0.0;
        self.was_grounded = true;
    }
}

/// Dual-rate exponential iron-sight / ADS FOV controller.
///
/// Zooms the camera FOV with snappy exponential asymptotic convergence
/// when entering aiming (rate 18.0) and smooth cinematic release when
/// lowering sights (rate 12.0), avoiding jarring instant cuts.
#[derive(Resource, Component, Debug, Clone, Copy, PartialEq)]
pub struct IronSightFov {
    /// Base vertical FOV in radians.
    pub base_fov: f32,
    /// Aimed / ADS vertical FOV in radians.
    pub ads_fov: f32,
    /// Current smoothed vertical FOV in radians.
    pub current_fov: f32,
    /// Rate constant for zooming in (s^-1).
    pub zoom_in_rate: f32,
    /// Rate constant for zooming out / lowering sights (s^-1).
    pub zoom_out_rate: f32,
    /// Whether iron-sight aiming is currently active.
    pub aiming: bool,
}

impl Default for IronSightFov {
    fn default() -> Self {
        Self::new()
    }
}

impl IronSightFov {
    pub const ZOOM_IN_RATE: f32 = 18.0;
    pub const ZOOM_OUT_RATE: f32 = 12.0;

    pub fn new() -> Self {
        let base_fov = cellview::vertical_fov(cellview::GAME_FOV_DEGREES);
        // Default Iron Sights FOV in Fallout: New Vegas (~55.0 degrees horizontal @ 4:3)
        let ads_fov = cellview::vertical_fov(55.0);
        Self {
            base_fov,
            ads_fov,
            current_fov: base_fov,
            zoom_in_rate: Self::ZOOM_IN_RATE,
            zoom_out_rate: Self::ZOOM_OUT_RATE,
            aiming: false,
        }
    }

    /// Updates FOV towards target using dual-rate exponential asymptotic smoothing.
    pub fn update(&mut self, aiming: bool, dt: f32) -> f32 {
        self.aiming = aiming;
        let target = if self.aiming {
            self.ads_fov
        } else {
            self.base_fov
        };
        let rate = if self.aiming {
            self.zoom_in_rate
        } else {
            self.zoom_out_rate
        };
        let alpha = 1.0 - (-rate * dt.max(0.0)).exp();
        self.current_fov += (target - self.current_fov) * alpha;
        self.current_fov
    }

    /// Resets FOV to base unzoomed FOV.
    pub fn reset(&mut self) {
        self.aiming = false;
        self.current_fov = self.base_fov;
    }
}

/// Camera roll banking and strafe dynamics.
///
/// Applies subtle camera roll on lateral strafes and sharp directional turns.
#[derive(Resource, Component, Debug, Clone, Copy, PartialEq)]
pub struct CameraBanking {
    /// Current camera roll angle in radians.
    pub roll: f32,
    /// Maximum allowed roll banking in radians (~2.0 degrees).
    pub max_roll: f32,
    /// Rate of convergence to target roll (s^-1).
    pub return_rate: f32,
    /// Scaling factor from lateral strafe input to target roll angle.
    pub strafe_roll_factor: f32,
    /// Scaling factor from turning angular velocity to target roll angle.
    pub turn_roll_factor: f32,
    /// Previous yaw to calculate turn rate.
    pub last_yaw: f32,
}

impl Default for CameraBanking {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraBanking {
    pub fn new() -> Self {
        Self {
            roll: 0.0,
            max_roll: 0.035, // ~2.0 degrees
            return_rate: 10.0,
            strafe_roll_factor: 0.022,
            turn_roll_factor: 0.003,
            last_yaw: 0.0,
        }
    }

    /// Updates camera banking roll based on lateral strafe wish component and yaw turn rate.
    pub fn update(&mut self, wish_strafe: f32, current_yaw: f32, dt: f32) -> f32 {
        if dt <= 0.0 {
            self.last_yaw = current_yaw;
            return self.roll;
        }
        let mut yaw_diff = current_yaw - self.last_yaw;
        while yaw_diff > std::f32::consts::PI {
            yaw_diff -= std::f32::consts::TAU;
        }
        while yaw_diff < -std::f32::consts::PI {
            yaw_diff += std::f32::consts::TAU;
        }
        let turn_rate = (yaw_diff / dt).clamp(-15.0, 15.0);
        self.last_yaw = current_yaw;

        let strafe_roll = wish_strafe * self.strafe_roll_factor;
        let turn_roll = -turn_rate * self.turn_roll_factor;
        let target_roll = (strafe_roll + turn_roll).clamp(-self.max_roll, self.max_roll);

        let alpha = 1.0 - (-self.return_rate * dt).exp();
        self.roll += (target_roll - self.roll) * alpha;
        if self.roll.abs() < 1e-5 {
            self.roll = 0.0;
        }
        self.roll
    }

    /// Resets roll to zero.
    pub fn reset(&mut self) {
        self.roll = 0.0;
    }
}

#[derive(Resource)]
pub struct Player {
    pub walking: bool,
    pub character: Character,
    /// Where the cell started the player, for R.
    start: [f32; 3],
    /// False while the ground under the player is still loading (outdoors).
    pub ready: bool,
    /// Dropped from the free camera (F): the landing does no damage.
    from_camera: bool,
    pub landing_dip: LandingDip,
    pub iron_sights: IronSightFov,
    pub banking: CameraBanking,
}

impl Player {
    pub fn new(walking: bool) -> Self {
        Self {
            walking,
            character: Character::new([0.0; 3]),
            start: [0.0; 3],
            ready: true,
            from_camera: false,
            landing_dip: LandingDip::default(),
            iron_sights: IronSightFov::default(),
            banking: CameraBanking::default(),
        }
    }

    /// Puts the player at a cell's arrival point (feet, game units).
    pub fn arrive(&mut self, feet: [f32; 3]) {
        self.start = feet;
        self.character = Character::new(feet);
        self.ready = true;
        self.landing_dip.reset();
        self.banking.reset();
    }

    /// The physical player's feet for a view at `eye` (game units).
    ///
    /// While walking, scripts and other systems may move the camera without
    /// moving the character. Reports and saves must keep using the character
    /// position in that case. Free-camera mode retains its legacy convention
    /// of treating the camera as the player and subtracting eye height.
    pub fn position_for_view(&self, eye: [f32; 3]) -> [f32; 3] {
        if self.walking {
            self.character.feet
        } else {
            [eye[0], eye[1], eye[2] - EYE_HEIGHT]
        }
    }
}

/// F switches between walking and flying.
pub fn toggle_walking(
    keys: Res<ButtonInput<KeyCode>>,
    mut player: ResMut<Player>,
    mut cameras: Query<(&Transform, Option<&mut Projection>), With<FlyCamera>>,
) {
    if !keys.just_pressed(KeyCode::KeyF) {
        return;
    }
    player.walking = !player.walking;
    if player.walking {
        // Land wherever the camera is.
        if let Ok((transform, _)) = cameras.single() {
            let [x, y, z] = game_point(transform.translation);
            player.character = Character::new([x, y, z - EYE_HEIGHT]);
            player.from_camera = true;
            player.landing_dip.reset();
            player.banking.reset();
        }
    } else {
        // Free camera mode: reset dynamic state and restore unzoomed base FOV.
        player.landing_dip.reset();
        player.banking.reset();
        player.iron_sights.reset();
        if let Ok((_, Some(mut projection))) = cameras.single_mut() {
            if let Projection::Perspective(ref mut p) = *projection {
                p.fov = player.iron_sights.base_fov;
            }
        }
    }
}

/// A point from Bevy's space (meters, y up) back to the game's (units, z
/// up).
pub fn game_point(p: Vec3) -> [f32; 3] {
    let s = 1.0 / space::METERS_PER_UNIT;
    [p.x * s, -p.z * s, p.y * s]
}

/// A direction from Bevy's space back to the game's.
fn game_direction(d: Vec3) -> [f32; 3] {
    [d.x, -d.z, d.y]
}

/// The people the player runs into: everyone alive here, upright
/// cylinders where they stand. People's controllers all have the game's
/// one size (`physics::CharacterShape::PLAYER`, 128 tall); a creature's
/// radius comes from its skeleton (`fighting::Kit`), its height taken as
/// 128 × its scale (a guess: the game sizes it from the skeleton's
/// `BSBound`). The dead don't block (their bodies are on the `DEADBIP`
/// layer, which the player's controller passes).
fn people(
    walkers: &Query<&crate::ai::Walker>,
    state: &world::scripting::GameState,
) -> Vec<physics::Person> {
    walkers
        .iter()
        .filter(|w| !state.dead.contains(&w.reference))
        .map(|w| {
            let creature = w.kit.as_ref().is_some_and(|k| k.creature.is_some());
            physics::Person {
                feet: w.position,
                radius: w
                    .kit
                    .as_ref()
                    .map_or(CharacterShape::PLAYER.radius, |k| k.radius),
                height: CharacterShape::PLAYER.height * if creature { w.scale } else { 1.0 },
            }
        })
        .collect()
}

/// Walking: the keys set the wanted speed, the character moves through the
/// cell's collision and around the people in it, and the camera sits at eye
/// height. Not while scripts have turned movement off
/// (`DisablePlayerControls`). Landing from a fall hurts as the game's falls
/// do (`world::combat::land`).
#[allow(clippy::too_many_arguments)]
pub fn walk(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut collision: ResMut<CellCollision>,
    game: Res<GameFiles>,
    mut state: ResMut<crate::dialogue::DialogueState>,
    mut player: ResMut<Player>,
    mut cameras: Query<(&mut Transform, Option<&mut Projection>, &FlyCamera)>,
    walkers: Query<&crate::ai::Walker>,
    landing_dip_res: Option<ResMut<LandingDip>>,
    iron_sights_res: Option<ResMut<IronSightFov>>,
    banking_res: Option<ResMut<CameraBanking>>,
) {
    if !player.walking || !player.ready {
        return;
    }
    collision.0.set_people(people(&walkers, &state.0));
    let locked = state.0.controls_off[world::scripting::controls::MOVEMENT]
        || state.0.dead.contains(&world::dialogue::PLAYER_REF);
    let Ok((mut transform, projection, camera)) = cameras.single_mut() else {
        return;
    };
    // Home: back to where the place started the player (R reloads).
    if keys.just_pressed(KeyCode::Home) {
        let start = player.start;
        player.character = Character::new(start);
        player.landing_dip.reset();
        player.banking.reset();
    }
    // Forward and right on the ground, in game space: yaw 0 looks north.
    let forward = [-camera.yaw.sin(), camera.yaw.cos()];
    let right = [camera.yaw.cos(), camera.yaw.sin()];
    let mut wish = [0.0f32; 2];
    for (key, dir, sign) in [
        (KeyCode::KeyW, forward, 1.0),
        (KeyCode::KeyS, forward, -1.0),
        (KeyCode::KeyD, right, 1.0),
        (KeyCode::KeyA, right, -1.0),
    ] {
        if keys.pressed(key) && !locked {
            wish[0] += dir[0] * sign;
            wish[1] += dir[1] * sign;
        }
    }
    let len = (wish[0] * wish[0] + wish[1] * wish[1]).sqrt();
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let sneak = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::KeyC);
    // Running is the game's default; Shift walks. Crippled legs and the
    // perks' "Modify Run Speed" (Travel Light) scale the whole speed.
    let mut speed = MovementSettings::BASE_SPEED
        * world::body_parts::leg_speed_mult(&game.0.order, &state.0, world::dialogue::PLAYER_REF)
        * world::combat::movement_speed_mult(&game.0.order, &state.0, world::dialogue::PLAYER_REF);
    if !shift {
        speed *= MovementSettings::RUN_MULT;
    }
    if sneak {
        speed *= MovementSettings::SNEAK_MULT;
    }
    let velocity = if len > 0.0 {
        [wish[0] / len * speed, wish[1] / len * speed]
    } else {
        [0.0, 0.0]
    };
    // How the player moves, for who notices them (`world::detection`).
    state.0.player_moving = len > 0.0;
    state.0.player_running = len > 0.0 && !shift && !sneak;
    state.0.player_sneaking = sneak;
    // The same as the game's movement flags, for `IsMoving`, `IsRunning`
    // and `IsSneaking` (`world::more_functions::movement`).
    {
        use world::more_functions::movement as m;
        let mut flags = 0;
        for (key, flag) in [
            (KeyCode::KeyW, m::FORWARD),
            (KeyCode::KeyS, m::BACK),
            (KeyCode::KeyA, m::LEFT),
            (KeyCode::KeyD, m::RIGHT),
        ] {
            if keys.pressed(key) && !locked {
                flags |= flag;
            }
        }
        if state.0.player_running {
            flags |= m::RUNNING;
        }
        if sneak {
            flags |= m::SNEAKING;
        }
        world::more_functions::report(
            &mut state.0,
            world::dialogue::PLAYER_REF,
            world::more_functions::Seen {
                movement: flags,
                ..Default::default()
            },
        );
    }
    let shape = CharacterShape::PLAYER;
    let jump = (keys.just_pressed(KeyCode::Space) && !locked)
        .then(|| (2.0 * shape.gravity * MovementSettings::JUMP_HEIGHT).sqrt());
    let dt = time.delta_secs();
    player
        .character
        .update_with_jump(&collision.0, &shape, velocity, jump, dt);
    if let Some(fell) = player.character.fell.take() {
        if !std::mem::take(&mut player.from_camera) {
            let hurt = world::combat::land(
                &game.0.order,
                &mut state.0,
                world::dialogue::PLAYER_REF,
                fell,
            );
            if hurt > 0.0 {
                println!("Fell {fell:.0} units: {hurt:.0} damage.");
            }
        }
    }

    // 1. Landing Dip: Harmonic spring tracking ground contact edges
    let is_grounded = player.character.on_ground;
    if player.from_camera {
        player.landing_dip.airtime = 0.0;
        player.landing_dip.was_grounded = is_grounded;
    } else {
        player.landing_dip.update(is_grounded, dt);
    }
    if let Some(mut dip_res) = landing_dip_res {
        *dip_res = player.landing_dip;
    }

    // 2. Camera Banking & Strafe Dynamics:
    // Projection of wish vector onto camera's right axis gives normalized lateral strafe component [-1.0, 1.0].
    let wish_strafe = if len > 0.0 {
        (wish[0] * right[0] + wish[1] * right[1]) / len
    } else {
        0.0
    };
    let roll = player.banking.update(wish_strafe, camera.yaw, dt);
    if let Some(mut bank_res) = banking_res {
        *bank_res = player.banking;
    }

    // 3. Smooth Dual-Rate Exponential Iron-Sight / ADS FOV:
    // Aiming is engaged by holding RMB or pressing Z key when controls are not locked.
    let aiming = !locked && (mouse.pressed(MouseButton::Right) || keys.pressed(KeyCode::KeyZ));
    let fov = player.iron_sights.update(aiming, dt);
    if let Some(mut ads_res) = iron_sights_res {
        *ads_res = player.iron_sights;
    }
    if let Some(mut proj) = projection {
        if let Projection::Perspective(ref mut pers) = proj.as_mut() {
            if pers.fov != fov {
                pers.fov = fov;
            }
        }
    }

    // Translation displaced downward by landing dip
    let [x, y, z] = player.character.feet;
    let dip_down = player.landing_dip.displacement;
    transform.translation = Vec3::from(space::point([x, y, z + EYE_HEIGHT - dip_down]));

    // Rotation incorporating pitch forward dip from landing and roll banking from strafe/turns
    let pitch = (camera.pitch + player.landing_dip.pitch_offset).clamp(-1.54, 1.54);
    transform.rotation = Quat::from_euler(EulerRot::YXZ, camera.yaw, pitch, roll);
}

/// The load door the view is on, within reach and not behind a wall.
pub(crate) fn door_in_view<'a>(
    doors: &'a [DoorData],
    collision: &Collider,
    eye: [f32; 3],
    dir: [f32; 3],
) -> Option<&'a DoorData> {
    let mut best: Option<(f32, &DoorData)> = None;
    for door in doors {
        // Slightly bigger than the model, so its frame counts.
        let lo = door.lo.map(|c| c - 4.0);
        let hi = door.hi.map(|c| c + 4.0);
        if let Some(t) = ray_box(eye, dir, lo, hi) {
            if t <= ACTIVATE_REACH && best.is_none_or(|(bt, _)| t < bt) {
                best = Some((t, door));
            }
        }
    }
    let (t, door) = best?;
    // A wall nearer than the door hides it (the door's own collision is
    // about as near as its box).
    if let Some((wall, _)) = collision.raycast(eye, dir, t) {
        if wall < t - 24.0 {
            return None;
        }
    }
    Some(door)
}

/// A door that opens where it stands (not a load door) under the crosshair
/// within reach. Its leaf's collision belongs to it
/// (`preview::cell::CellScene::collider`) and swings with it
/// (`doors::update_doors`), so the ray meets the leaf wherever it is; a
/// wall nearer than the door hides it.
pub(crate) fn opening_door_in_view(
    collision: &Collider,
    eye: [f32; 3],
    dir: [f32; 3],
) -> Option<esm::FormId> {
    let (_, t) = collision.raycast_including_hidden(eye, dir, ACTIVATE_REACH)?;
    let owner = collision.owner(t);
    (owner != 0).then_some(esm::FormId(owner))
}

/// Where a ray enters an axis-aligned box, if it does.
fn ray_box(origin: [f32; 3], dir: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> Option<f32> {
    let mut near = 0.0f32;
    let mut far = f32::INFINITY;
    for k in 0..3 {
        if dir[k].abs() < 1e-9 {
            if origin[k] < lo[k] || origin[k] > hi[k] {
                return None;
            }
            continue;
        }
        let a = (lo[k] - origin[k]) / dir[k];
        let b = (hi[k] - origin[k]) / dir[k];
        near = near.max(a.min(b));
        far = far.min(a.max(b));
        if near > far {
            return None;
        }
    }
    Some(near)
}

/// Load doors: the crosshair line names where the door in view leads, and
/// E goes through it, into the next interior or out to a worldspace.
#[allow(clippy::too_many_arguments)]
pub fn doors(
    keys: Res<ButtonInput<KeyCode>>,
    game: Res<GameFiles>,
    doors: Res<Doors>,
    collision: Res<CellCollision>,
    mut pending: ResMut<PendingScene>,
    mut pending_exterior: ResMut<PendingExterior>,
    cameras: Query<&Transform, With<FlyCamera>>,
    mut prompt: Query<&mut Text, With<Prompt>>,
    talk_target: Res<crate::dialogue::TalkTarget>,
    conversation: Res<crate::dialogue::Conversation>,
    activatable: Res<crate::scripts::Activatable>,
    mut activate: ResMut<crate::scripts::ActivateRequest>,
    mut state: ResMut<crate::dialogue::DialogueState>,
    scripts: Res<crate::scripts::Scripts>,
    mut sounds: ResMut<crate::sounds::SoundRequests>,
    (mut lockpicking, menus): (
        ResMut<crate::lockpick::Lockpicking>,
        Res<crate::menus::Menus>,
    ),
) {
    // A lock just picked: the player uses the door or container, as the
    // game has them do after the lockpicking menu (`00573170`): E again.
    let again = lockpicking.again.take();
    // Someone to talk to (or talking) takes the prompt and E.
    if talk_target.0.is_some() || conversation.0.is_some() {
        return;
    }
    let Ok(transform) = cameras.single() else {
        return;
    };
    let eye = game_point(transform.translation);
    let dir = game_direction(transform.forward().as_vec3());
    let door = door_in_view(&doors.0, &collision.0, eye, dir);
    // A door that swings open where it stands, when no load door is in view.
    let swing = door
        .is_none()
        .then(|| opening_door_in_view(&collision.0, eye, dir))
        .flatten();
    let line = match (door, swing, &activatable.0) {
        // Nothing while the lockpicking menu or one of the game's menus is up
        // (the roll-over is the HUD's, which their masks hide: ui::hud::parts_for_menu).
        _ if lockpicking.is_open() || menus.game_open => String::new(),
        (Some(d), _, _) => format!("E) Open door to {}", d.cell_label),
        (None, Some(d), _) => crate::doors::prompt(&game.0.order, &state.0, d).to_string(),
        // A scripted object (a machine, a switch) when no door is in view.
        (None, None, Some((_, name))) => format!("E) {name}"),
        (None, None, None) => String::new(),
    };
    for mut text in &mut prompt {
        if text.0 != line {
            text.0 = line.clone();
        }
    }
    if let Some(r) = again {
        // A container (or a door no longer in view) is used as E on it.
        let in_view = door.is_some_and(|d| d.reference == r.0) || swing == Some(r);
        if !in_view {
            activate.0 = Some(r);
            return;
        }
    }
    // The game's menus have E while they're open (game_menus).
    let pressed = !menus.game_open && (keys.just_pressed(KeyCode::KeyE) || again.is_some());
    let Some(door) = door else {
        if !pressed {
            return;
        }
        if let Some(reference) = swing {
            // Opening or closing runs the door's script, as the load
            // doors' do; then the game's door rules (`world::doors`): the
            // swing, its sound, and nothing while it's still swinging.
            if crate::scripts::door_opens(
                &game.0.order,
                &scripts.0,
                &mut state.0,
                reference,
                &mut lockpicking.request,
            ) {
                let player = world::dialogue::PLAYER_REF;
                match crate::doors::activate(
                    &game.0.order,
                    &mut state.0,
                    &mut sounds,
                    reference,
                    Some(player),
                ) {
                    world::doors::Activated::Opening => println!("The door opens."),
                    world::doors::Activated::Closing => println!("The door closes."),
                    world::doors::Activated::Busy => {}
                }
            }
        } else if let Some((reference, _)) = &activatable.0 {
            activate.0 = Some(*reference);
        }
        return;
    };
    if !pressed {
        return;
    }
    let reference = esm::FormId(door.reference);
    if !crate::scripts::door_opens(
        &game.0.order,
        &scripts.0,
        &mut state.0,
        reference,
        &mut lockpicking.request,
    ) {
        return;
    }
    // The door's opening sound (`SNAM` on its base).
    if let Some(base) = world::scripting::base_of(&game.0.order, reference) {
        if let (Some(open), _) = world::sound::door_sounds(&game.0.order, base) {
            sounds.0.push(open);
        }
    }
    let Some(cell) = door.cell else {
        return;
    };
    // Going through a door moves the player: fast travel a script turned
    // off comes back (unless it asked to keep it off).
    world::script_functions::player_moved(&mut state.0);
    if !door.interior && door.world.is_some() {
        match door_start(&game.0, door) {
            Ok(start) => pending_exterior.0 = Some(start),
            Err(e) => println!("Couldn't go out to {}: {e}", door.cell_label),
        }
        return;
    }
    println!("Loading {} ...", door.cell_label);
    let started = std::time::Instant::now();
    match game.0.load_cell_now(esm::FormId(cell), &state.0.disabled) {
        Ok(mut scene) => {
            let [x, y, z] = door.arrive;
            scene.start = cellview::Start {
                eye: [x, y, z + EYE_HEIGHT],
                heading: door.arrive_heading,
                via: "the door",
            };
            for note in &scene.notes {
                println!("  {note}");
            }
            println!(
                "Loaded {} in {:.1} s.",
                scene.cell,
                started.elapsed().as_secs_f32()
            );
            pending.0 = Some(scene);
        }
        Err(e) => println!("Couldn't load {}: {e}", door.cell_label),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rays_enter_boxes_in_front_only() {
        let lo = [10.0, -1.0, -1.0];
        let hi = [12.0, 1.0, 1.0];
        assert_eq!(ray_box([0.0; 3], [1.0, 0.0, 0.0], lo, hi), Some(10.0));
        assert_eq!(ray_box([0.0; 3], [-1.0, 0.0, 0.0], lo, hi), None);
        assert_eq!(ray_box([0.0, 5.0, 0.0], [1.0, 0.0, 0.0], lo, hi), None);
    }

    #[test]
    fn finds_the_door_in_view_open_or_shut() {
        // A door's leaf (owned by the door, 0x904) 100 units north, and a
        // wall (nobody's) 50 units east.
        let quad = |c: &mut Collider, v: [[f32; 3]; 4], owner: u32| {
            c.add_solid(&v, &[[0, 1, 2], [0, 2, 3]], 0.0, owner)
        };
        let mut c = Collider::new();
        quad(
            &mut c,
            [
                [-40.0, 100.0, 0.0],
                [40.0, 100.0, 0.0],
                [40.0, 100.0, 200.0],
                [-40.0, 100.0, 200.0],
            ],
            0x904,
        );
        quad(
            &mut c,
            [
                [50.0, -40.0, 0.0],
                [50.0, 40.0, 0.0],
                [50.0, 40.0, 200.0],
                [50.0, -40.0, 200.0],
            ],
            0,
        );
        let eye = [0.0, 0.0, 120.0];
        let north = [0.0, 1.0, 0.0];
        assert_eq!(
            opening_door_in_view(&c, eye, north),
            Some(esm::FormId(0x904))
        );
        // Swung aside (a quarter turn about its left edge), it's found
        // where it now is.
        let turn = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        c.move_owner(0x904, &turn, [-40.0 + 100.0, 100.0 + 40.0, 0.0]);
        assert_eq!(opening_door_in_view(&c, eye, north), None);
        assert_eq!(
            opening_door_in_view(&c, [-60.0, 100.0, 120.0], [1.0, 0.0, 0.0]),
            Some(esm::FormId(0x904))
        );
        // A wall isn't a door; and out of reach, nothing.
        assert_eq!(opening_door_in_view(&c, eye, [1.0, 0.0, 0.0]), None);
        assert_eq!(opening_door_in_view(&c, [0.0, -100.0, 120.0], north), None);
    }

    #[test]
    fn game_and_bevy_space_round_trip() {
        let p = [100.0, -200.0, 300.0];
        let back = game_point(Vec3::from(space::point(p)));
        assert!(back.iter().zip(p).all(|(a, b)| (a - b).abs() < 1e-3));
        let d = [0.0, 1.0, 0.0];
        assert_eq!(game_direction(Vec3::from(space::direction(d))), d);
    }

    #[test]
    fn landing_dip_harmonic_spring_tracks_contact_edges_and_settles() {
        let mut dip = LandingDip::new();
        assert_eq!(dip.k, 175.0);
        assert_eq!(dip.c, 19.5);
        assert_eq!(dip.displacement, 0.0);
        assert_eq!(dip.velocity, 0.0);
        assert_eq!(dip.pitch_offset, 0.0);

        // Grounded: airtime remains 0, no impulse
        dip.update(true, 0.016);
        assert_eq!(dip.airtime, 0.0);
        assert_eq!(dip.displacement, 0.0);

        // Jump: airborne for 0.6 seconds
        let dt = 1.0 / 60.0;
        for _ in 0..36 {
            dip.update(false, dt);
        }
        assert!((dip.airtime - 0.6).abs() < 1e-3);
        assert_eq!(dip.displacement, 0.0);

        // Touchdown contact edge: !was_grounded && is_grounded
        dip.update(true, dt);
        assert_eq!(dip.airtime, 0.0);
        // Impulse proportional to airtime should be applied, giving velocity > 0
        assert!(dip.velocity > 50.0);
        assert!(dip.displacement > 0.0);
        assert!(dip.pitch_offset < 0.0, "pitch should tilt forward (negative)");

        // Track peak displacement
        let mut peak_displacement = dip.displacement;
        for _ in 0..10 {
            dip.step(dt);
            if dip.displacement > peak_displacement {
                peak_displacement = dip.displacement;
            }
        }
        assert!(peak_displacement > 2.0, "should have noticeable dip displacement");

        // After 1 second of settling, the damped harmonic oscillator returns to equilibrium
        for _ in 0..120 {
            dip.update(true, dt);
        }
        assert!(dip.displacement.abs() < 1e-3, "displacement should settle to 0");
        assert!(dip.velocity.abs() < 1e-3, "velocity should settle to 0");
        assert!(dip.pitch_offset.abs() < 1e-4, "pitch offset should settle to 0");
    }

    #[test]
    fn smooth_dual_rate_exponential_iron_sight_fov() {
        let mut ads = IronSightFov::new();
        assert_eq!(ads.zoom_in_rate, 18.0);
        assert_eq!(ads.zoom_out_rate, 12.0);
        assert_eq!(ads.current_fov, ads.base_fov);

        // Zooming in towards ADS FOV
        let dt = 0.016;
        let initial_fov = ads.current_fov;
        let fov1 = ads.update(true, dt);
        assert!(fov1 < initial_fov, "entering sights should zoom in (decrease FOV)");

        // Verify exponential asymptotic progression
        // At rate 18.0 over 0.2s: alpha = 1 - exp(-18 * 0.2) = 1 - exp(-3.6) > 0.97
        for _ in 0..12 {
            ads.update(true, dt);
        }
        assert!((ads.current_fov - ads.ads_fov).abs() < (initial_fov - ads.ads_fov) * 0.05);

        // Release / lowering sights at rate 12.0
        let zoomed_fov = ads.current_fov;
        let fov_rel1 = ads.update(false, dt);
        assert!(fov_rel1 > zoomed_fov, "lowering sights should zoom out (increase FOV)");

        // Smooth cinematic release without instant cuts
        for _ in 0..30 {
            ads.update(false, dt);
        }
        assert!((ads.current_fov - ads.base_fov).abs() < 0.01);
    }

    #[test]
    fn camera_banking_rolls_into_strafes_and_turns() {
        let mut banking = CameraBanking::new();
        let dt = 1.0 / 60.0;

        // Strafe right: positive roll banking
        for _ in 0..20 {
            banking.update(1.0, 0.0, dt);
        }
        assert!(banking.roll > 0.01, "strafing right should bank roll positive");
        assert!(banking.roll <= banking.max_roll);

        // Stopping strafe returns roll to 0
        for _ in 0..40 {
            banking.update(0.0, 0.0, dt);
        }
        assert!(banking.roll.abs() < 1e-3, "stopping strafe should restore roll to level");

        // Strafe left: negative roll banking
        for _ in 0..20 {
            banking.update(-1.0, 0.0, dt);
        }
        assert!(banking.roll < -0.01, "strafing left should bank roll negative");
        assert!(banking.roll >= -banking.max_roll);
    }
}
