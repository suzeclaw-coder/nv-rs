//! The game's last steps on the finished picture: bloom, the HDR
//! brightness limit, and the cell's image space (saturation, a tint,
//! brightness and contrast), then the conversion to linear light for the
//! screen. Ported from the game's own passes, read from a recording of a
//! frame (apitrace) and the shaders in its package:
//!
//! 1. `ISHDRDOWN4`: the picture shrunk 4× in each direction, four times
//!    (1920×1080 → 480×270 → 120×67 → 30×16 → 7×4), each pixel the average
//!    of four bilinear samples one source texel out diagonally.
//! 2. `ISHDRDS4ADAPT`: the average brightness, a 1×1 value: the four corner
//!    pixels of the 7×4 picture averaged (the game samples outside the
//!    texture with clamping, so it reads the corners), eased toward from
//!    the last frame's average by `1 − speed^seconds` (eye adaptation: see
//!    [`adapt_eyes`]), its length clamped to [0.01, limit].
//! 3. `ISBPBLUR13`: at 480×270, a vertical 13-tap blur of
//!    `max(color − bright clamp, 0) × bright scale`; alpha holds the
//!    average's red + green + blue.
//! 4. `ISBLUR13`: the same blur horizontally, alpha carried along.
//! 5. `ISHDRBLENDINSHADERCIN`: `scene × L / max(a, L) + bloom × 0.5 /
//!    max(a, L)` (a: the bloom's alpha, L: the limit), then the cinematic
//!    step. The game then draws its HUD over the result (`hud`); its
//!    picture is laid over here, still on stored values.
//!
//! The blur weights are a Gaussian with σ = radius / 2 over taps −radius
//! … radius (radius 6 and 7 give exactly the recorded weights; the game has
//! blur shaders up to 15 taps, so larger radii are drawn as 7). All of it runs
//! on stored (gamma-encoded) values, as the game's frame buffer holds them.
//! The render-graph plumbing follows Bevy 0.16's `custom_post_processing`
//! example.

// The shader-layout derive generates checking functions the compiler
// reports as unused.
#![allow(dead_code)]

use bevy::asset::{load_internal_asset, weak_handle};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::extract_component::{
    ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
    UniformComponentPlugin,
};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{sampler, texture_2d, uniform_buffer};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::sync_world::MainEntity;
use bevy::render::texture::{CachedTexture, FallbackImageZero, GpuImage, TextureCache};
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderApp, RenderSet};
use std::collections::HashMap;

const SHADER: Handle<Shader> = weak_handle!("6d1f3a52-8c47-4e0b-9a61-2f5c7e9b1d34");

/// What a camera's final passes use, as the shader reads it. Every camera
/// needs one: the last pass also turns the picture's stored values into
/// linear light ([`ImageSpaceGrade::NEUTRAL`] when the cell has no image
/// space); cameras without one skip the passes.
#[derive(Component, Clone, Copy, Debug, PartialEq, ExtractComponent, ShaderType)]
pub struct ImageSpaceGrade {
    /// Tint color, and how much of it.
    pub tint: Vec4,
    /// Saturation, contrast, the brightness contrast spreads around, and
    /// brightness.
    pub cinematic: Vec4,
    /// Bloom: bright clamp (threshold), bright scale, blur radius in
    /// texels, and the clamp on the average brightness's length.
    pub bloom: Vec4,
    /// `x`: the final pass's brightness limit (`HDRParam.x`); `y`: 1 when
    /// the bloom and limit apply at all; `z`: the eye adaptation speed (the
    /// average pass's `HDRParam.z`); `w`: the frame's seconds as the
    /// average pass takes them (`TimingData.z`; below 0: take this frame's
    /// average as it is). See [`adapt_eyes`].
    pub hdr: Vec4,
    /// The colour the picture fades to, and how far (`Fade`, the final
    /// pass's last step: `lerp(c, Fade.rgb, Fade.w)`); set by image space
    /// modifiers.
    pub fade: Vec4,
    /// Physiological low-health feedback, crippled limb effects, and concussion/hit dynamics:
    /// `x`: tunnel vision intensity (0.0 = none, 1.0 = heavy peripheral constriction)
    /// `y`: pulse throb intensity (0.0 = none, >0.0 = arterial systolic surge)
    /// `z`: color temperature shift (-1.0 = cold/cyan shock pallor, +1.0 = warm/red arterial flush)
    /// `w`: raw trauma impulse intensity (0.0 .. 2.0)
    pub physiological: Vec4,
}

/// A camera whose final passes are left to a later camera on the same
/// window: the main camera's, since the first-person view (`viewmodel`) is
/// drawn by its own camera after it, over a cleared depth buffer, and the
/// game's image space passes come after that pass. The marked camera keeps
/// its `ImageSpaceGrade` (everything that sets the grade sets it there);
/// the later camera's is copied from it each frame.
#[derive(Component, Clone, Copy, Default, ExtractComponent)]
pub struct GradeDeferred;

impl ImageSpaceGrade {
    /// Leaves the picture as it is: no cinematic change, no bloom.
    pub const NEUTRAL: Self = Self {
        tint: Vec4::new(1.0, 1.0, 1.0, 0.0),
        cinematic: Vec4::new(1.0, 1.0, 0.0, 1.0),
        bloom: Vec4::new(1.0, 0.0, 0.0, 1.0),
        hdr: Vec4::new(1.0, 0.0, 0.0, 0.0),
        fade: Vec4::ZERO,
        physiological: Vec4::ZERO,
    };

    /// The same with image space modifiers playing (`world::modifier`):
    /// each value the passes use becomes value × multiply + add, modifier
    /// after modifier; the tints are averaged weighted by their amounts
    /// (the game's code does this: `00b8d020`), the record's own tint among
    /// them, and the largest amount kept: confirmed by the Goodsprings
    /// recording, where `NVDefaultExterior`'s tint at 0.33 and
    /// `NVWastelandIS`'s at 0.392 arrived as their amount-weighted average
    /// at 0.392. Fades apply one after another.
    pub fn with_modifiers(self, values: &[world::modifier::ModifierValues]) -> Self {
        use world::modifier::track;
        if values.is_empty() {
            return self;
        }
        let apply = |base: f32, t: usize| {
            values
                .iter()
                .fold(base, |v, m| v * m.multiply[t] + m.add[t])
        };
        let mut out = self;
        out.bloom = Vec4::new(
            apply(self.bloom.x, track::BRIGHT_CLAMP),
            apply(self.bloom.y, track::BRIGHT_SCALE),
            apply(self.bloom.z, track::BLUR_RADIUS).clamp(0.0, 7.0),
            apply(self.bloom.w, track::UPPER_LUM_CLAMP),
        );
        out.hdr.x = apply(self.hdr.x, track::TARGET_LUM).max(1e-3);
        out.hdr.z = apply(self.hdr.z, track::EYE_ADAPT_SPEED);
        out.cinematic = Vec4::new(
            apply(self.cinematic.x, track::SATURATION),
            apply(self.cinematic.y, track::CONTRAST),
            apply(self.cinematic.z, track::CONTRAST_AVERAGE),
            apply(self.cinematic.w, track::BRIGHTNESS),
        );
        let tints: Vec<Vec4> = std::iter::once(self.tint)
            .chain(values.iter().map(|m| Vec4::from_array(m.tint)))
            .filter(|t| t.w > 0.0)
            .collect();
        let weight: f32 = tints.iter().map(|t| t.w).sum();
        if weight > 0.0 {
            let color = tints.iter().map(|t| t.truncate() * t.w).sum::<Vec3>() / weight;
            let amount = tints.iter().map(|t| t.w).fold(0.0, f32::max);
            out.tint = color.extend(amount);
        }
        // mix(mix(c, f1, a1), f2, a2) as one mix(c, F, A).
        let mut fade = Vec4::ZERO;
        for m in values {
            let f = Vec4::from_array(m.fade);
            let a = fade.w + f.w - fade.w * f.w;
            if a > 0.0 {
                let rgb = (fade.truncate() * fade.w * (1.0 - f.w) + f.truncate() * f.w) / a;
                fade = rgb.extend(a);
            }
        }
        out.fade = fade;
        out
    }

    /// From the cell's image space. Without HDR values there's no bloom.
    pub fn from_cell(grade: Option<&cellview::Grade>, hdr: Option<&cellview::Hdr>) -> Self {
        let mut out = Self::NEUTRAL;
        if let Some(grade) = grade {
            out = out.with_cinematic(grade);
        }
        if let Some(h) = hdr {
            // Recorded: the bright pass gets (bright clamp, bright scale),
            // the blur reaches the blur radius, the average is clamped to
            // the upper luminance clamp and the final limit is the target
            // luminance (Doc Mitchell's house and the Mojave Outpost
            // barracks, whose values differ). The game's widest blur
            // shaders take 15 taps, so the radius stops at 7: the barracks'
            // 8 was drawn with 7.
            out.bloom = Vec4::new(
                h.bright_clamp,
                h.bright_scale,
                h.blur_radius.clamp(0.0, 7.0),
                h.upper_lum_clamp,
            );
            out.hdr = Vec4::new(h.target_lum.max(1e-3), 1.0, h.eye_adapt_speed, 0.0);
        }
        out
    }

    /// The same, with the cinematic values (saturation, tint, brightness,
    /// contrast) taken from `grade`.
    pub fn with_cinematic(self, grade: &cellview::Grade) -> Self {
        let [r, g, b] = grade.tint;
        Self {
            // As stored: the game's image space manager hands the record's
            // tint color and amount to the shader unchanged when no image
            // space modifier is active (read from FalloutNV.exe and the
            // recording, which sent (0.69, 0.561, 0.302, 0.5)).
            tint: Vec4::new(r, g, b, grade.tint_amount),
            cinematic: Vec4::new(
                grade.saturation,
                grade.contrast,
                grade.contrast_average,
                grade.brightness,
            ),
            ..self
        }
    }

    /// The same without the cinematic change (bloom and HDR kept), for
    /// comparing.
    pub fn without_cinematic(self) -> Self {
        Self {
            tint: Self::NEUTRAL.tint,
            cinematic: Self::NEUTRAL.cinematic,
            ..self
        }
    }

    /// Adjusts post-tonemapping saturation, color temperature, and contrast dynamically
    /// during low-health or pain/concussion states to convey shock and fading consciousness.
    pub fn with_physiological(self, phys: &PhysiologicalState) -> Self {
        let mut out = self;
        let sev = phys.low_health_severity;
        let trauma = phys.trauma.clamp(0.0, 2.0);
        let pulse_active = sev > 0.0 || trauma > 0.0;
        let pulse_effect = if pulse_active {
            phys.pulse * (sev * 0.85 + trauma * 0.45).min(1.2)
        } else {
            0.0
        };

        // 1. Post-tonemapping saturation: fading consciousness desaturation
        let desat = (1.0 - 0.75 * sev * (1.0 - 0.25 * phys.pulse))
            * (1.0 - 0.45 * trauma.min(1.0))
            * if phys.head_crippled { 0.75 } else { 1.0 };
        out.cinematic.x *= desat.clamp(0.05, 1.0);

        // 2. Contrast & brightness dynamics: shock & throb
        let contrast_boost = 1.0 + 0.35 * sev * phys.pulse + 0.45 * trauma;
        out.cinematic.y *= contrast_boost;

        // Brightness dims between heartbeats (fading consciousness), surging during pulse
        let brightness_mod = 1.0 - 0.35 * sev * (1.0 - 0.7 * phys.pulse) - 0.25 * trauma;
        out.cinematic.w *= brightness_mod.clamp(0.25, 1.5);

        // 3. Color temperature & arterial red pulse tint
        let cold_shock = -0.55 * sev - 0.35 * trauma;
        let warm_pulse = 0.75 * pulse_effect;
        let temp_shift = (cold_shock + warm_pulse).clamp(-1.0, 1.0);

        if pulse_effect > 0.02 {
            let red_tint = Vec4::new(1.0, 0.15, 0.08, pulse_effect * 0.35);
            let total_w = out.tint.w + red_tint.w;
            if total_w > 0.0 {
                let blended_rgb = (out.tint.truncate() * out.tint.w + red_tint.truncate() * red_tint.w) / total_w;
                let max_w = out.tint.w.max(red_tint.w);
                out.tint = blended_rgb.extend(max_w);
            }
        }

        // 4. Near-death fading consciousness blackout (< 15% HP)
        if phys.health_fraction < 0.15 {
            let fade_sev = ((0.15 - phys.health_fraction) / 0.15).powf(1.5);
            let blackout = (fade_sev * 0.8 * (1.0 - 0.4 * phys.pulse)).clamp(0.0, 0.92);
            let a = out.fade.w + blackout - out.fade.w * blackout;
            if a > 0.0 {
                let rgb = (out.fade.truncate() * out.fade.w * (1.0 - blackout)) / a;
                out.fade = rgb.extend(a);
            }
        }

        // 5. Concussion tunnel vision & physiological uniform vector
        let tunnel_base = if phys.head_crippled { 0.45 } else { 0.0 }
            + phys.concussion * 0.5
            + sev * 0.45
            + trauma * 0.35;
        let tunnel = (tunnel_base * (1.0 - 0.2 * phys.pulse)).clamp(0.0, 0.95);

        out.physiological = Vec4::new(tunnel, pulse_effect, temp_shift, trauma);
        out
    }
}

/// Heartbeat arterial waveform ("lub-dub"):
/// pulse(t) = exp(-((t - 0.07)/0.05)^2) + 0.65 * exp(-((t - 0.23)/0.055)^2)
///
/// `t` is the time within the cardiac cycle in seconds (0.0 <= t < period).
pub fn heartbeat_pulse(t: f32) -> f32 {
    let s1 = (-((t - 0.07) / 0.05).powi(2)).exp();
    let s2 = 0.65 * (-((t - 0.23) / 0.055).powi(2)).exp();
    s1 + s2
}

/// Scales pulse frequency from 72 BPM up to 138 BPM as health drops below
/// critical threshold (< 30% HP).
pub fn calculate_bpm(health_fraction: f32, trauma: f32) -> f32 {
    let base_bpm = if health_fraction < 0.30 {
        let severity = ((0.30 - health_fraction) / 0.30).clamp(0.0, 1.0);
        72.0 + (138.0 - 72.0) * severity
    } else {
        72.0
    };
    (base_bpm + trauma * 30.0).clamp(72.0, 138.0)
}

/// Physiological state tracking low-health feedback, arterial pulse,
/// crippled limb effects, and concussion/hit trauma dynamics.
#[derive(Resource, Debug, Clone)]
pub struct PhysiologicalState {
    /// Accumulated heart cycle phase (0.0 .. 1.0).
    pub phase: f32,
    /// Current heart rate in beats per minute (72.0 at 30% HP up to 138.0 at 0% HP).
    pub bpm: f32,
    /// Evaluated arterial double-pulse waveform value for this frame ("lub-dub").
    pub pulse: f32,
    /// Hit/shock trauma impulse, decaying over time (0.0 .. 2.0).
    pub trauma: f32,
    /// Concussion intensity (from head injury or heavy hit shock, 0.0 .. 1.0).
    pub concussion: f32,
    /// Low health severity: 0.0 when >= 30% HP, scaling up to 1.0 at 0% HP.
    pub low_health_severity: f32,
    /// Current health fraction (0.0 .. 1.0).
    pub health_fraction: f32,
    /// Whether the head or brain is currently crippled.
    pub head_crippled: bool,
    /// Count of other crippled limbs (arms, legs, torso).
    pub other_crippled_count: usize,
}

impl Default for PhysiologicalState {
    fn default() -> Self {
        Self {
            phase: 0.0,
            bpm: 72.0,
            pulse: 0.0,
            trauma: 0.0,
            concussion: 0.0,
            low_health_severity: 0.0,
            health_fraction: 1.0,
            head_crippled: false,
            other_crippled_count: 0,
        }
    }
}

impl PhysiologicalState {
    /// Triggers an immediate trauma impulse on the player.
    pub fn trigger_trauma(&mut self, impulse: f32, is_heavy: bool) {
        self.trauma = (self.trauma + impulse).min(2.0);
        if is_heavy {
            // Concussion shock spike
            self.concussion = (self.concussion + 0.6).min(1.0);
            // Reset cardiac phase to systolic peak ("lub") for immediate shock throb
            let period = 60.0 / self.bpm.max(1.0);
            self.phase = (0.07 / period).fract();
        }
    }
}

/// Tracks base ImageSpaceGrade before physiological modifications so they can be
/// applied frame-by-frame without compound drift.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct BaseGrade(pub ImageSpaceGrade);

/// Adjusts cameras' image space grades for physiological low-health feedback,
/// arterial heartbeat pulsation, crippled limb concussion, and hit trauma shock.
#[allow(clippy::too_many_arguments)]
pub fn apply_physiological_effects(
    time: Res<Time>,
    game: Option<Res<crate::GameFiles>>,
    state: Option<Res<crate::dialogue::DialogueState>>,
    mut phys: ResMut<PhysiologicalState>,
    mut commands: Commands,
    mut cameras: Query<(Entity, &mut ImageSpaceGrade, Option<&mut BaseGrade>)>,
) {
    let dt = time.delta_secs();

    // Extract player's live status if game files & dialogue state are loaded
    let (health_fraction, head_crippled, other_crippled_count) =
        if let (Some(game), Some(state)) = (game.as_ref(), state.as_ref()) {
            let order = &game.0.order;
            let gs = &state.0;
            let cur_h = world::combat::health(order, gs, world::dialogue::PLAYER_REF)
                .unwrap_or(100.0) as f32;
            let max_h = world::combat::max_health(order, gs, world::dialogue::PLAYER_REF)
                .unwrap_or(100.0)
                .max(1.0) as f32;
            let frac = (cur_h / max_h).clamp(0.0, 1.0);

            let head = world::body_parts::is_crippled(
                order,
                gs,
                world::dialogue::PLAYER_REF,
                world::body_parts::av::FIRST_CONDITION,
            ) || world::body_parts::is_crippled(
                order,
                gs,
                world::dialogue::PLAYER_REF,
                world::body_parts::av::LAST_CONDITION,
            );

            let mut other = 0;
            for part in 26..=30 {
                if world::body_parts::is_crippled(order, gs, world::dialogue::PLAYER_REF, part) {
                    other += 1;
                }
            }
            (frac, head, other)
        } else {
            (
                phys.health_fraction,
                phys.head_crippled,
                phys.other_crippled_count,
            )
        };

    phys.health_fraction = health_fraction;
    phys.head_crippled = head_crippled;
    phys.other_crippled_count = other_crippled_count;

    let low_health_severity = if health_fraction < 0.30 {
        ((0.30 - health_fraction) / 0.30).clamp(0.0, 1.0)
    } else {
        0.0
    };
    phys.low_health_severity = low_health_severity;

    // Heart rate scaling: 72 BPM up to 138 BPM as health drops below 30% HP
    phys.bpm = calculate_bpm(health_fraction, phys.trauma);
    let period = 60.0 / phys.bpm.max(1.0);

    // Advance cardiac cycle phase
    if dt > 0.0 && period > 0.0 {
        phys.phase = (phys.phase + dt / period).fract();
        if phys.phase < 0.0 {
            phys.phase += 1.0;
        }
    }
    let t = phys.phase * period;
    phys.pulse = heartbeat_pulse(t);

    // Exponential decay of trauma & concussion
    phys.trauma = (phys.trauma - dt * 1.5).max(0.0);
    let min_concussion = if head_crippled { 0.45 } else { 0.0 };
    phys.concussion = (phys.concussion - dt * 0.3).max(min_concussion);

    for (entity, mut grade, base) in &mut cameras {
        let base_val = if let Some(mut base_comp) = base {
            if grade.physiological == Vec4::ZERO {
                base_comp.0 = *grade;
            }
            base_comp.0
        } else {
            commands.entity(entity).insert(BaseGrade(*grade));
            *grade
        };

        *grade = base_val.with_physiological(&phys);
    }
}

/// The frame's seconds for the eye adaptation, as the game's average pass
/// gets them (`TimingData.z`): the frame's game time, so 0 while a menu is
/// open. Recorded: 0.24 at Goodsprings (the recording ran at about four
/// frames a second), and 0 in every frame with the console open (both
/// indoor recordings, and a later Goodsprings frame with the console open
/// and the same `HDRParam.z` 0.9); the game's own code for it isn't traced.
/// `menu_open`: a menu or the dialogue menu is up (the viewer's menus stop
/// its clock the same way). `starting`: a place is loading or has just
/// been entered, so the average is taken as it is (negative): what the
/// game's average holds after a loading screen isn't known.
pub fn adaptation_seconds(delta: f32, menu_open: bool, starting: bool) -> f32 {
    if starting {
        -1.0
    } else if menu_open {
        0.0
    } else {
        delta
    }
}

/// How far the average brightness moves toward this frame's in one frame
/// (`ISHDRDS4ADAPT`: `lerp(previous, current, 1 − HDRParam.z ^
/// TimingData.z)`): `1 − speed^seconds`, 0 with no time passing, 1 when
/// starting over. With `NVDefaultExterior`'s speed 0.9, a tenth of the way
/// a second.
pub fn adaptation_step(speed: f32, seconds: f32) -> f32 {
    if seconds < 0.0 {
        1.0
    } else if seconds == 0.0 {
        0.0
    } else {
        1.0 - speed.max(0.0).powf(seconds)
    }
}

/// Gives every camera's final passes the frame's seconds for the eye
/// adaptation (after the image space and its modifiers are set for the
/// frame).
#[allow(clippy::too_many_arguments)]
pub fn adapt_eyes(
    time: Res<Time>,
    menus: Option<Res<crate::menus::Menus>>,
    conversation: Option<Res<crate::dialogue::Conversation>>,
    exterior: Option<Res<crate::exterior::Exterior>>,
    grading: Option<Res<crate::Grading>>,
    mut started: Local<bool>,
    mut cameras: Query<&mut ImageSpaceGrade>,
) {
    let menu_open = menus.is_some_and(|m| m.is_open())
        || conversation.is_some_and(|c| c.0.as_ref().is_some_and(|t| !t.is_line_only()));
    let starting =
        !*started || grading.is_some_and(|g| g.is_changed()) || exterior.is_some_and(|e| e.busy());
    *started = true;
    let seconds = adaptation_seconds(time.delta_secs(), menu_open, starting);
    for mut grade in &mut cameras {
        if grade.hdr.w != seconds {
            grade.hdr.w = seconds;
        }
    }
}

pub struct GradePlugin;

impl Plugin for GradePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SHADER, "grade.wgsl", Shader::from_wgsl);
        app.add_plugins((
            ExtractComponentPlugin::<ImageSpaceGrade>::default(),
            ExtractComponentPlugin::<GradeDeferred>::default(),
            UniformComponentPlugin::<ImageSpaceGrade>::default(),
        ))
        .init_resource::<PhysiologicalState>()
        // After the image space and its modifiers are set (`Update`).
        .add_systems(
            PostUpdate,
            (apply_physiological_effects, adapt_eyes).chain(),
        );
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<EyeAverages>()
            .add_systems(
                Render,
                prepare_bloom_textures.in_set(RenderSet::PrepareResources),
            )
            .add_render_graph_node::<ViewNodeRunner<GradeNode>>(Core3d, GradeLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::Tonemapping,
                    GradeLabel,
                    Node3d::EndMainPassPostProcessing,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<GradePipeline>();
    }
}

/// The intermediate pictures of the bloom chain, per camera.
#[derive(Component)]
struct BloomTextures {
    /// The four 4× downsamples, largest first.
    levels: [CachedTexture; 4],
    /// After the bright pass and vertical blur, then the horizontal blur.
    vertical: CachedTexture,
    horizontal: CachedTexture,
}

/// The average brightness each camera keeps from frame to frame (the
/// game's `AvgLum`, which its average pass reads and rewrites): two 1×1
/// pictures taking turns, by the camera's entity.
#[derive(Resource, Default)]
struct EyeAverages(HashMap<MainEntity, Averages>);

struct Averages {
    _textures: [Texture; 2],
    views: [TextureView; 2],
    /// The one written this frame (the other holds the last frame's).
    written: usize,
}

fn prepare_bloom_textures(
    mut commands: Commands,
    mut cache: ResMut<TextureCache>,
    mut averages: ResMut<EyeAverages>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &MainEntity, &ViewTarget), With<ImageSpaceGrade>>,
) {
    let mut seen = Vec::new();
    for (_, main, _) in &views {
        seen.push(*main);
        let a = averages.0.entry(*main).or_insert_with(|| {
            let textures = ["image_space_average_a", "image_space_average_b"].map(|label| {
                device.create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: ViewTarget::TEXTURE_FORMAT_HDR,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            });
            let views = [0, 1].map(|i| textures[i].create_view(&TextureViewDescriptor::default()));
            Averages {
                _textures: textures,
                views,
                written: 0,
            }
        });
        a.written = 1 - a.written;
    }
    averages.0.retain(|main, _| seen.contains(main));
    for (entity, _, target) in &views {
        let size = target.main_texture().size();
        let mut texture = |width: u32, height: u32, label: &'static str| {
            cache.get(
                &device,
                TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: width.max(1),
                        height: height.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: ViewTarget::TEXTURE_FORMAT_HDR,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )
        };
        let (mut w, mut h) = (size.width, size.height);
        let mut level = |label| {
            w = (w / 4).max(1);
            h = (h / 4).max(1);
            (w, h, label)
        };
        let sizes = [
            level("image_space_down_1"),
            level("image_space_down_2"),
            level("image_space_down_3"),
            level("image_space_down_4"),
        ];
        let levels = sizes.map(|(w, h, label)| texture(w, h, label));
        let (bw, bh) = (sizes[0].0, sizes[0].1);
        commands.entity(entity).insert(BloomTextures {
            levels,
            vertical: texture(bw, bh, "image_space_bloom_v"),
            horizontal: texture(bw, bh, "image_space_bloom_h"),
        });
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GradeLabel;

#[derive(Default)]
struct GradeNode;

impl ViewNode for GradeNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static MainEntity,
        &'static DynamicUniformIndex<ImageSpaceGrade>,
        &'static BloomTextures,
        Option<&'static GradeDeferred>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, main, grade_index, textures, deferred): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if deferred.is_some() {
            return Ok(());
        }
        let Some(averages) = world.resource::<EyeAverages>().0.get(main) else {
            return Ok(());
        };
        let grade_pipeline = world.resource::<GradePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let ids = &grade_pipeline.pipelines;
        let Some(pipelines) = [ids.down, ids.average, ids.bright, ids.blur, ids.last]
            .map(|id| pipeline_cache.get_render_pipeline(id))
            .into_iter()
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(());
        };
        let uniforms = world.resource::<ComponentUniforms<ImageSpaceGrade>>();
        let Some(binding) = uniforms.uniforms().binding() else {
            return Ok(());
        };
        // The HUD's picture (`hud`), or nothing.
        let overlay = world
            .get_resource::<crate::hud::HudLayer>()
            .and_then(|layer| world.resource::<RenderAssets<GpuImage>>().get(&layer.0))
            .map_or(
                &world.resource::<FallbackImageZero>().texture_view,
                |image| &image.texture_view,
            );
        let pass = Pass {
            layout: &grade_pipeline.layout,
            sampler: &grade_pipeline.sampler,
            uniform: binding,
            offset: grade_index.index(),
            overlay,
        };
        let [down, average, bright, blur, last] = pipelines[..] else {
            return Ok(());
        };

        // Bloom and the average brightness, from the finished scene.
        let scene = view_target.main_texture_view();
        let avg = &averages.views[averages.written];
        let previous = &averages.views[1 - averages.written];
        let mut source = scene;
        for level in &textures.levels {
            pass.draw(render_context, down, source, avg, &level.default_view);
            source = &level.default_view;
        }
        let vertical = &textures.vertical.default_view;
        let horizontal = &textures.horizontal.default_view;
        // The average pass eases from the last frame's average.
        pass.draw(render_context, average, source, previous, avg);
        pass.draw(
            render_context,
            bright,
            &textures.levels[0].default_view,
            avg,
            vertical,
        );
        pass.draw(render_context, blur, vertical, avg, horizontal);

        // The final pass reads the current picture and writes the adjusted
        // one; the view target swaps to it afterwards.
        let post_process = view_target.post_process_write();
        pass.draw(
            render_context,
            last,
            post_process.source,
            horizontal,
            post_process.destination,
        );
        Ok(())
    }
}

/// What every pass binds besides its textures.
struct Pass<'a> {
    layout: &'a BindGroupLayout,
    sampler: &'a Sampler,
    uniform: BindingResource<'a>,
    offset: u32,
    /// The HUD's picture (read by the last pass only).
    overlay: &'a TextureView,
}

impl Pass<'_> {
    /// One full-screen pass reading `source` (and `extra`) into `target`.
    fn draw(
        &self,
        render_context: &mut RenderContext,
        pipeline: &RenderPipeline,
        source: &TextureView,
        extra: &TextureView,
        target: &TextureView,
    ) {
        let bind_group = render_context.render_device().create_bind_group(
            "image_space_bind_group",
            self.layout,
            &BindGroupEntries::sequential((
                source,
                self.sampler,
                self.uniform.clone(),
                extra,
                self.overlay,
            )),
        );
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("image_space_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[self.offset]);
        render_pass.draw(0..3, 0..1);
    }
}

struct Pipelines {
    down: CachedRenderPipelineId,
    average: CachedRenderPipelineId,
    bright: CachedRenderPipelineId,
    blur: CachedRenderPipelineId,
    last: CachedRenderPipelineId,
}

#[derive(Resource)]
struct GradePipeline {
    layout: BindGroupLayout,
    sampler: Sampler,
    pipelines: Pipelines,
}

impl FromWorld for GradePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            "image_space_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                    uniform_buffer::<ImageSpaceGrade>(true),
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    texture_2d(TextureSampleType::Float { filterable: true }),
                ),
            ),
        );
        // Bilinear and clamped, as the game samples these pictures.
        let sampler = render_device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });
        let mut queue = |entry: &'static str| {
            world
                .resource_mut::<PipelineCache>()
                .queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some(format!("image_space_{entry}").into()),
                    layout: vec![layout.clone()],
                    vertex: fullscreen_shader_vertex_state(),
                    fragment: Some(FragmentState {
                        shader: SHADER,
                        shader_defs: vec![],
                        entry_point: entry.into(),
                        // Every picture here is in the camera's HDR format.
                        targets: vec![Some(ColorTargetState {
                            format: ViewTarget::TEXTURE_FORMAT_HDR,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        })],
                    }),
                    primitive: PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    push_constant_ranges: vec![],
                    zero_initialize_workgroup_memory: false,
                })
        };
        let pipelines = Pipelines {
            down: queue("downsample"),
            average: queue("average"),
            bright: queue("bright_blur"),
            blur: queue("blur"),
            last: queue("fragment"),
        };
        Self {
            layout,
            sampler,
            pipelines,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use world::modifier::{track, ModifierValues, TRACKS};

    fn values(fade: [f32; 4]) -> ModifierValues {
        ModifierValues {
            multiply: [1.0; TRACKS],
            add: [0.0; TRACKS],
            tint: [1.0, 1.0, 1.0, 0.0],
            fade,
            blur: 0.0,
            double_vision: 0.0,
        }
    }

    #[test]
    fn modifiers_scale_values_and_fades_stack() {
        let base = ImageSpaceGrade {
            bloom: Vec4::new(0.9, 2.4, 6.0, 1.0),
            ..ImageSpaceGrade::NEUTRAL
        };
        assert_eq!(base.with_modifiers(&[]), base);
        let mut flash = values([1.0, 1.0, 1.0, 0.5]);
        flash.multiply[track::BRIGHT_SCALE] = 2.0;
        flash.add[track::BRIGHT_SCALE] = 0.2;
        let dark = values([0.0, 0.0, 0.0, 0.5]);
        let out = base.with_modifiers(&[flash, dark]);
        assert!((out.bloom.y - 5.0).abs() < 1e-5);
        assert_eq!(out.bloom.x, 0.9);
        // Half to white then half to black: 3/4 faded, a third white.
        assert!((out.fade.w - 0.75).abs() < 1e-6);
        assert!((out.fade.x - 1.0 / 3.0).abs() < 1e-6);
        // No tint anywhere: unchanged.
        assert_eq!(out.tint, base.tint);
    }

    #[test]
    fn the_eye_adapts_a_tenth_of_the_way_a_second_outdoors() {
        // Recorded at Goodsprings: speed 0.9 (`NVDefaultExterior`'s eye
        // adapt speed, `HDRParam.z`), a 0.24 s frame (`TimingData.z`).
        let hdr = cellview::Hdr {
            eye_adapt_speed: 0.9,
            blur_radius: 7.0,
            emissive_mult: 1.5,
            target_lum: 1.4,
            upper_lum_clamp: 1.0,
            bright_scale: 2.0,
            bright_clamp: 0.6,
            skin_directional: 2.0,
        };
        let grade = ImageSpaceGrade::from_cell(None, Some(&hdr));
        assert_eq!(grade.hdr.z, 0.9);
        let step = adaptation_step(grade.hdr.z, 0.24);
        assert!((step - (1.0 - 0.9f32.powf(0.24))).abs() < 1e-6);
        assert!((step - 0.025_0).abs() < 1e-3, "{step}");
        // Over a second of frames: a tenth of the way.
        let mut left = 1.0f32;
        for _ in 0..60 {
            left *= 1.0 - adaptation_step(0.9, 1.0 / 60.0);
        }
        assert!((left - 0.9).abs() < 1e-4, "{left}");
        // With the console (any menu) open the game's frame time is 0: the
        // average stays (both indoor recordings, and Goodsprings with the
        // console open).
        assert_eq!(adaptation_seconds(0.24, true, false), 0.0);
        assert_eq!(adaptation_step(0.9, 0.0), 0.0);
        // While a place loads the viewer takes the average as it is.
        assert!(adaptation_seconds(0.24, false, true) < 0.0);
        assert_eq!(adaptation_step(0.9, -1.0), 1.0);
        assert_eq!(adaptation_seconds(0.24, false, false), 0.24);
    }

    #[test]
    fn modifiers_change_the_eye_adaptation_speed() {
        let base = ImageSpaceGrade {
            hdr: Vec4::new(1.0, 1.0, 0.9, 0.0),
            ..ImageSpaceGrade::NEUTRAL
        };
        let mut slow = values([0.0; 4]);
        slow.multiply[track::EYE_ADAPT_SPEED] = 0.5;
        assert_eq!(base.with_modifiers(&[slow]).hdr.z, 0.45);
    }

    #[test]
    fn the_weathers_modifier_gives_goodsprings_its_recorded_grade() {
        // `NVDefaultExterior`: saturation 1.1, contrast 1.1 around 0.2,
        // brightness 1, tint (0.984, 0.569, 0) at 0.33; the day's weather
        // modifier `NVWastelandIS` at its first keys: brightness × 1.3, tint
        // (1, 0.737, 0.051) at 0.392.
        let record = cellview::Grade {
            saturation: 1.1,
            contrast_average: 0.2,
            contrast: 1.1,
            brightness: 1.0,
            tint: [251.0 / 255.0, 145.0 / 255.0, 0.0],
            tint_amount: 0.33,
        };
        let base = ImageSpaceGrade::NEUTRAL.with_cinematic(&record);
        let mut day = values([0.0; 4]);
        day.multiply[track::BRIGHTNESS] = 1.3;
        day.tint = [1.0, 188.0 / 255.0, 13.0 / 255.0, 100.0 / 255.0];
        let out = base.with_modifiers(&[day]);
        // Recorded: `Cinematic` (1.1, 0.2, 1.1, 1.3) and `Tint` (0.99283,
        // 0.6602, 0.02768, 0.39216): the two tints averaged by their
        // amounts, the larger amount kept.
        assert!((out.cinematic.w - 1.3).abs() < 1e-5);
        assert!((out.cinematic.y - 1.1).abs() < 1e-5);
        let want = Vec4::new(0.99283, 0.6602, 0.02768, 0.39216);
        assert!(
            (out.tint - want).abs().max_element() < 1e-4,
            "{:?}",
            out.tint
        );
    }

    #[test]
    fn physiological_heartbeat_waveform_reproduces_double_pulse() {
        let p1 = heartbeat_pulse(0.07);
        assert!((p1 - 1.0).abs() < 1e-3, "First peak (systole 'lub') should peak at ~1.0, got {p1}");

        let p2 = heartbeat_pulse(0.23);
        assert!((p2 - 0.65).abs() < 1e-3, "Second peak (closure 'dub') should peak at ~0.65, got {p2}");

        let p_diastole = heartbeat_pulse(0.45);
        assert!(p_diastole < 0.001, "Diastolic rest should approach 0, got {p_diastole}");
    }

    #[test]
    fn pulse_frequency_scales_from_72_to_138_bpm_below_critical_threshold() {
        // Above or at 30% HP: exactly 72 BPM
        assert_eq!(calculate_bpm(1.0, 0.0), 72.0);
        assert_eq!(calculate_bpm(0.5, 0.0), 72.0);
        assert_eq!(calculate_bpm(0.30, 0.0), 72.0);

        // Halfway below critical threshold (15% HP): 105 BPM
        let mid = calculate_bpm(0.15, 0.0);
        assert!((mid - 105.0).abs() < 1e-4, "Mid-threshold BPM should be 105, got {mid}");

        // At 0% HP: 138 BPM
        let max_bpm = calculate_bpm(0.0, 0.0);
        assert!((max_bpm - 138.0).abs() < 1e-4, "Critical 0% HP BPM should be 138, got {max_bpm}");

        // Trauma impulse elevates BPM
        let trauma_bpm = calculate_bpm(0.5, 1.0);
        assert_eq!(trauma_bpm, 102.0);
    }

    #[test]
    fn dynamic_color_grading_desaturates_and_tunnels_under_low_health() {
        let base = ImageSpaceGrade::NEUTRAL;
        let mut phys = PhysiologicalState::default();

        // Neutral full health produces no changes
        let neutral = base.with_physiological(&phys);
        assert_eq!(neutral.cinematic.x, 1.0);
        assert_eq!(neutral.physiological.x, 0.0); // Tunnel vision = 0

        // Low health (< 30% HP) causes desaturation and tunnel vision
        phys.health_fraction = 0.10;
        phys.low_health_severity = (0.30 - 0.10) / 0.30;
        phys.pulse = 0.0; // Between beats (diastole)
        let low = base.with_physiological(&phys);
        assert!(low.cinematic.x < 0.6, "Low health should desaturate image, got {}", low.cinematic.x);
        assert!(low.physiological.x > 0.25, "Low health should induce tunnel vision, got {}", low.physiological.x);
        assert!(low.physiological.z < 0.0, "Shock should shift color temperature cold/cyan, got {}", low.physiological.z);

        // Crippled head causes concussion tunnel vision and desaturation even at full health
        let mut head_crippled_phys = PhysiologicalState::default();
        head_crippled_phys.head_crippled = true;
        let head_grade = base.with_physiological(&head_crippled_phys);
        assert!(head_grade.physiological.x >= 0.40, "Crippled head should trigger tunnel vision");
        assert!(head_grade.cinematic.x < 1.0, "Crippled head should reduce saturation");
    }

    #[test]
    fn hit_trauma_impulse_triggers_shock_and_systolic_peak() {
        let mut phys = PhysiologicalState::default();
        assert_eq!(phys.trauma, 0.0);

        // Minor hit
        phys.trigger_trauma(0.3, false);
        assert!((phys.trauma - 0.3).abs() < 1e-4);
        assert_eq!(phys.concussion, 0.0);

        // Heavy hit
        phys.trigger_trauma(0.8, true);
        assert!((phys.trauma - 1.1).abs() < 1e-4);
        assert!(phys.concussion > 0.5);
        // Cardiac phase should be reset to systolic peak (~0.07 s)
        let period = 60.0 / phys.bpm;
        let expected_phase = (0.07 / period).fract();
        assert!((phys.phase - expected_phase).abs() < 1e-4);
    }
}
