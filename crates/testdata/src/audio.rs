//! Zero-dependency procedural DSP audio synthesis module for test fixtures,
//! benchmarks, and sound pipeline verification.
//!
//! Provides deterministic oscillators, noise sources, single- and multi-pole
//! filters, envelope generators, audio buffer operations (normalization, mixing,
//! PCM16 / WAV export), and procedural sound presets (gunshot impulse, mechanical
//! click, explosion rumble, footstep thud, heartbeat thump, test tone).

use std::f32::consts::{LN_2, PI, TAU};
use std::ops::{Deref, DerefMut};

/// Standard sample rate (44.1 kHz) matching the game's audio assets.
pub const SAMPLE_RATE: f32 = 44_100.0;

// ---------------------------------------------------------------------------
// Audio Buffer
// ---------------------------------------------------------------------------

/// Mono audio buffer of 32-bit floating point samples.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Buf(pub Vec<f32>);

impl Deref for Buf {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Buf {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Vec<f32>> for Buf {
    fn from(samples: Vec<f32>) -> Self {
        Self(samples)
    }
}

impl From<Buf> for Vec<f32> {
    fn from(buf: Buf) -> Self {
        buf.0
    }
}

impl Buf {
    /// Creates a new buffer from an existing vector of samples.
    pub fn new(samples: Vec<f32>) -> Self {
        Self(samples)
    }

    /// Creates a zero-filled buffer of length `count`.
    pub fn zeros(count: usize) -> Self {
        Self(vec![0.0; count])
    }

    /// Creates a silent buffer for the given duration in seconds.
    pub fn silence(duration_secs: f32, sample_rate: f32) -> Self {
        let count = (duration_secs.max(0.0) * sample_rate.max(1.0)).round() as usize;
        Self::zeros(count)
    }

    /// Returns the number of samples in the buffer.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if the buffer contains no samples.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the sample slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Returns the mutable sample slice.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.0
    }

    /// Scales all samples in the buffer by a constant factor.
    pub fn apply_gain(&mut self, gain: f32) {
        for s in &mut self.0 {
            *s *= gain;
        }
    }

    /// Multiplies samples in `self` element-wise by samples from `other` (e.g. envelope modulation).
    pub fn multiply(&mut self, other: &Buf) {
        for (s, &m) in self.0.iter_mut().zip(other.0.iter()) {
            *s *= m;
        }
    }

    /// Normalizes the buffer so that its peak absolute amplitude equals `target_peak`.
    ///
    /// If all samples are zero or non-finite, the buffer is left unmodified.
    pub fn normalize(&mut self, target_peak: f32) {
        let peak = self
            .0
            .iter()
            .copied()
            .filter(|s| s.is_finite())
            .fold(0.0f32, |acc, s| acc.max(s.abs()));

        if peak > 1e-9 {
            let scale = target_peak / peak;
            for s in &mut self.0 {
                *s *= scale;
            }
        }
    }

    /// Returns a new buffer normalized to `target_peak`.
    pub fn normalized(&self, target_peak: f32) -> Self {
        let mut cloned = self.clone();
        cloned.normalize(target_peak);
        cloned
    }

    /// Mixes another buffer into `self` with a gain factor, extending `self` if `other` is longer.
    pub fn mix(&mut self, other: &Buf, gain: f32) {
        self.mix_at(other, 0, gain);
    }

    /// Mixes another buffer into `self` starting at `offset_samples` with a gain factor.
    pub fn mix_at(&mut self, other: &Buf, offset_samples: usize, gain: f32) {
        let required_len = offset_samples + other.len();
        if required_len > self.0.len() {
            self.0.resize(required_len, 0.0);
        }
        for (i, &sample) in other.0.iter().enumerate() {
            self.0[offset_samples + i] += sample * gain;
        }
    }

    /// Serializes audio samples to little-endian 16-bit signed PCM byte stream.
    ///
    /// Samples are clamped to `[-1.0, 1.0]` before conversion to `[-32767, 32767]`.
    pub fn to_pcm16_le(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.0.len() * 2);
        for &s in &self.0 {
            let clamped = s.clamp(-1.0, 1.0);
            let val = (clamped * 32767.0).round() as i16;
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Encodes the buffer as a canonical 16-bit mono PCM RIFF WAVE file byte stream.
    pub fn to_wav(&self, sample_rate: u32) -> Vec<u8> {
        let pcm_data = self.to_pcm16_le();
        let data_len = pcm_data.len() as u32;
        let channels = 1u16;
        let bits_per_sample = 16u16;
        let block_align = channels * (bits_per_sample / 8);
        let byte_rate = sample_rate * u32::from(block_align);
        let riff_chunk_size = 36u32.saturating_add(data_len);

        let mut out = Vec::with_capacity(44 + pcm_data.len());
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&riff_chunk_size.to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM format tag = 1
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&bits_per_sample.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        out.extend_from_slice(&pcm_data);
        out
    }
}

// ---------------------------------------------------------------------------
// DSP Synthesis Primitives
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random white noise generator using xorshift64.
#[derive(Debug, Clone)]
pub struct Noise {
    state: u64,
}

impl Noise {
    /// Creates a deterministic noise generator with a standard default seed.
    pub fn new() -> Self {
        Self::with_seed(0x853C_49E6_748F_EA9B)
    }

    /// Creates a deterministic noise generator with a specified non-zero seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853C_49E6_748F_EA9B } else { seed },
        }
    }

    /// Returns the next pseudo-random sample uniformly distributed in `[-1.0, 1.0]`.
    pub fn next_sample(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // Take top 24 bits for mantissa to construct a uniform float in [-1.0, 1.0]
        let bits = (x >> 40) as u32;
        (bits as f32 / 8_388_607.5) - 1.0
    }

    /// Generates a buffer of `count` noise samples.
    pub fn generate(&mut self, count: usize) -> Buf {
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            v.push(self.next_sample());
        }
        Buf(v)
    }
}

impl Default for Noise {
    fn default() -> Self {
        Self::new()
    }
}

/// Sine wave oscillator with phase accumulator.
#[derive(Debug, Clone)]
pub struct Osc {
    freq: f32,
    sample_rate: f32,
    phase: f32,
}

impl Osc {
    /// Creates a new sine wave oscillator at `freq` Hz and `sample_rate` Hz.
    pub fn new(freq: f32, sample_rate: f32) -> Self {
        Self {
            freq,
            sample_rate: sample_rate.max(1.0),
            phase: 0.0,
        }
    }

    /// Sets the frequency in Hz.
    pub fn set_freq(&mut self, freq: f32) {
        self.freq = freq;
    }

    /// Returns the current frequency.
    pub fn freq(&self) -> f32 {
        self.freq
    }

    /// Resets the phase accumulator to 0.0.
    pub fn reset_phase(&mut self) {
        self.phase = 0.0;
    }

    /// Returns the next sine sample in `[-1.0, 1.0]`.
    pub fn next_sample(&mut self) -> f32 {
        let sample = (self.phase * TAU).sin();
        let step = self.freq / self.sample_rate;
        self.phase = (self.phase + step).fract();
        if self.phase < 0.0 {
            self.phase += 1.0;
        }
        sample
    }

    /// Advances the oscillator while overriding the frequency for this step.
    pub fn next_sample_with_freq(&mut self, freq: f32) -> f32 {
        self.freq = freq;
        self.next_sample()
    }

    /// Generates a buffer of `count` sine samples at the current frequency.
    pub fn generate(&mut self, count: usize) -> Buf {
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            v.push(self.next_sample());
        }
        Buf(v)
    }
}

/// One-pole low-pass filter (exponential smoothing / RC filter).
#[derive(Debug, Clone)]
pub struct Lp {
    alpha: f32,
    prev_y: f32,
}

impl Lp {
    /// Creates a new one-pole low-pass filter at `cutoff` Hz.
    pub fn new(cutoff: f32, sample_rate: f32) -> Self {
        let mut filter = Self {
            alpha: 0.0,
            prev_y: 0.0,
        };
        filter.set_cutoff(cutoff, sample_rate);
        filter
    }

    /// Updates the cutoff frequency in Hz.
    pub fn set_cutoff(&mut self, cutoff: f32, sample_rate: f32) {
        let rate = sample_rate.max(1.0);
        let normalized = (cutoff / rate).clamp(0.0, 0.499);
        self.alpha = (1.0 - (-TAU * normalized).exp()).clamp(0.0, 1.0);
    }

    /// Processes a single input sample through the low-pass filter.
    pub fn process(&mut self, sample: f32) -> f32 {
        self.prev_y += self.alpha * (sample - self.prev_y);
        self.prev_y
    }

    /// Resets internal filter state to 0.0.
    pub fn reset(&mut self) {
        self.prev_y = 0.0;
    }

    /// Filters an entire buffer in place.
    pub fn filter_buf(&mut self, buf: &mut Buf) {
        for s in &mut buf.0 {
            *s = self.process(*s);
        }
    }
}

/// One-pole high-pass filter.
#[derive(Debug, Clone)]
pub struct Hp {
    alpha: f32,
    prev_x: f32,
    prev_y: f32,
}

impl Hp {
    /// Creates a new one-pole high-pass filter at `cutoff` Hz.
    pub fn new(cutoff: f32, sample_rate: f32) -> Self {
        let mut filter = Self {
            alpha: 0.0,
            prev_x: 0.0,
            prev_y: 0.0,
        };
        filter.set_cutoff(cutoff, sample_rate);
        filter
    }

    /// Updates the cutoff frequency in Hz.
    pub fn set_cutoff(&mut self, cutoff: f32, sample_rate: f32) {
        let rate = sample_rate.max(1.0);
        let normalized = (cutoff / rate).clamp(0.0, 0.499);
        self.alpha = (-TAU * normalized).exp().clamp(0.0, 1.0);
    }

    /// Processes a single input sample through the high-pass filter.
    pub fn process(&mut self, sample: f32) -> f32 {
        let y = self.alpha * (self.prev_y + sample - self.prev_x);
        self.prev_x = sample;
        self.prev_y = y;
        y
    }

    /// Resets internal filter state to 0.0.
    pub fn reset(&mut self) {
        self.prev_x = 0.0;
        self.prev_y = 0.0;
    }

    /// Filters an entire buffer in place.
    pub fn filter_buf(&mut self, buf: &mut Buf) {
        for s in &mut buf.0 {
            *s = self.process(*s);
        }
    }
}

/// Two-pole state-variable bandpass filter (Chamberlin SVF topology with 2x internal steps).
#[derive(Debug, Clone)]
pub struct Bp {
    cutoff: f32,
    q: f32,
    sample_rate: f32,
    low: f32,
    band: f32,
}

impl Bp {
    /// Creates a 2-pole state-variable bandpass filter with given `cutoff` Hz and resonance `q`.
    pub fn new(cutoff: f32, q: f32, sample_rate: f32) -> Self {
        Self {
            cutoff: cutoff.max(1.0),
            q: q.max(0.1),
            sample_rate: sample_rate.max(1.0),
            low: 0.0,
            band: 0.0,
        }
    }

    /// Updates cutoff frequency and quality factor Q.
    pub fn set_params(&mut self, cutoff: f32, q: f32) {
        self.cutoff = cutoff.max(1.0);
        self.q = q.max(0.1);
    }

    /// Processes a single sample and returns the bandpass filtered result.
    pub fn process(&mut self, sample: f32) -> f32 {
        let f = (2.0 * (PI * self.cutoff / self.sample_rate).sin()).clamp(0.0, 0.95);
        let q_inv = 1.0 / self.q;

        // 2x internal sub-stepping maintains stability up to near Nyquist
        let f_half = f * 0.5;
        for _ in 0..2 {
            let high = sample - self.low - q_inv * self.band;
            self.band += f_half * high;
            self.low += f_half * self.band;
        }
        self.band
    }

    /// Resets internal filter state to 0.0.
    pub fn reset(&mut self) {
        self.low = 0.0;
        self.band = 0.0;
    }

    /// Filters an entire buffer in place.
    pub fn filter_buf(&mut self, buf: &mut Buf) {
        for s in &mut buf.0 {
            *s = self.process(*s);
        }
    }
}

// ---------------------------------------------------------------------------
// Envelope Generators
// ---------------------------------------------------------------------------

/// Current stage of an ADSR envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdsrStage {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// Attack-Decay-Sustain-Release envelope generator.
#[derive(Debug, Clone)]
pub struct Adsr {
    attack_rate: f32,
    decay_rate: f32,
    sustain_level: f32,
    release_rate: f32,
    current_level: f32,
    stage: AdsrStage,
}

impl Adsr {
    /// Creates an ADSR envelope with durations in seconds and sustain level `[0.0, 1.0]`.
    pub fn new(
        attack_secs: f32,
        decay_secs: f32,
        sustain: f32,
        release_secs: f32,
        sample_rate: f32,
    ) -> Self {
        let rate = sample_rate.max(1.0);
        let attack_samples = (attack_secs.max(0.0001) * rate).max(1.0);
        let decay_samples = (decay_secs.max(0.0001) * rate).max(1.0);
        let release_samples = (release_secs.max(0.0001) * rate).max(1.0);
        let sustain_clamped = sustain.clamp(0.0, 1.0);

        Self {
            attack_rate: 1.0 / attack_samples,
            decay_rate: (1.0 - sustain_clamped) / decay_samples,
            sustain_level: sustain_clamped,
            release_rate: 1.0 / release_samples,
            current_level: 0.0,
            stage: AdsrStage::Idle,
        }
    }

    /// Triggers gate on (`true`) or release (`false`).
    pub fn gate(&mut self, on: bool) {
        if on {
            self.stage = AdsrStage::Attack;
        } else if self.stage != AdsrStage::Idle {
            self.stage = AdsrStage::Release;
        }
    }

    /// Returns the current lifecycle stage of the envelope.
    pub fn stage(&self) -> AdsrStage {
        self.stage
    }

    /// Returns true if the envelope is currently sounding.
    pub fn is_active(&self) -> bool {
        self.stage != AdsrStage::Idle
    }

    /// Computes and returns the next envelope amplitude sample in `[0.0, 1.0]`.
    pub fn next_sample(&mut self) -> f32 {
        match self.stage {
            AdsrStage::Idle => 0.0,
            AdsrStage::Attack => {
                self.current_level += self.attack_rate;
                if self.current_level >= 1.0 {
                    self.current_level = 1.0;
                    self.stage = AdsrStage::Decay;
                }
                self.current_level
            }
            AdsrStage::Decay => {
                self.current_level -= self.decay_rate;
                if self.current_level <= self.sustain_level {
                    self.current_level = self.sustain_level;
                    self.stage = AdsrStage::Sustain;
                }
                self.current_level
            }
            AdsrStage::Sustain => self.sustain_level,
            AdsrStage::Release => {
                self.current_level -= self.release_rate;
                if self.current_level <= 0.0 {
                    self.current_level = 0.0;
                    self.stage = AdsrStage::Idle;
                }
                self.current_level
            }
        }
    }

    /// Generates a buffer for a gate duration followed by a release duration.
    pub fn generate_gate(&mut self, gate_samples: usize, release_samples: usize) -> Buf {
        let total = gate_samples + release_samples;
        let mut out = Vec::with_capacity(total);
        self.gate(true);
        for _ in 0..gate_samples {
            out.push(self.next_sample());
        }
        self.gate(false);
        for _ in 0..release_samples {
            out.push(self.next_sample());
        }
        Buf(out)
    }
}

/// Exponential decay envelope generator.
#[derive(Debug, Clone)]
pub struct Decay {
    coeff: f32,
    current: f32,
}

impl Decay {
    /// Creates an exponential decay with time constant tau in seconds (drops to ~36.8% at tau).
    pub fn new(tau_secs: f32, sample_rate: f32) -> Self {
        let tau = tau_secs.max(1e-5);
        let rate = sample_rate.max(1.0);
        let coeff = (-1.0 / (tau * rate)).exp();
        Self {
            coeff,
            current: 1.0,
        }
    }

    /// Creates an exponential decay specifying half-life duration in seconds (drops to 50% at half-life).
    pub fn with_half_life(half_life_secs: f32, sample_rate: f32) -> Self {
        let t_half = half_life_secs.max(1e-5);
        let rate = sample_rate.max(1.0);
        let coeff = (-LN_2 / (t_half * rate)).exp();
        Self {
            coeff,
            current: 1.0,
        }
    }

    /// Returns the next decay sample.
    pub fn next_sample(&mut self) -> f32 {
        let val = self.current;
        self.current *= self.coeff;
        val
    }

    /// Resets the current amplitude level.
    pub fn reset(&mut self, initial_level: f32) {
        self.current = initial_level;
    }

    /// Generates a buffer of `count` decay envelope samples.
    pub fn generate(&mut self, count: usize) -> Buf {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.next_sample());
        }
        Buf(out)
    }
}

// ---------------------------------------------------------------------------
// Procedural Sound Presets
// ---------------------------------------------------------------------------

/// Procedural gunshot sound effect: sharp crack transient, barrel resonance,
/// and low explosive boom.
pub fn gunshot_impulse() -> Buf {
    let sample_rate = SAMPLE_RATE;
    let total_samples = (sample_rate * 0.35) as usize;

    let mut crack_noise = Noise::with_seed(0x1001);
    let mut crack_hp = Hp::new(1200.0, sample_rate);
    let mut crack_env = Decay::new(0.015, sample_rate);

    let mut body_noise = Noise::with_seed(0x2002);
    let mut body_lp = Lp::new(350.0, sample_rate);
    let mut body_env = Decay::new(0.08, sample_rate);

    let mut punch_osc = Osc::new(260.0, sample_rate);
    let mut punch_env = Decay::new(0.045, sample_rate);

    let mut samples = Vec::with_capacity(total_samples);
    for i in 0..total_samples {
        let t = i as f32 / sample_rate;

        // Transient crack
        let crack = crack_hp.process(crack_noise.next_sample()) * crack_env.next_sample();

        // Low body explosion
        let body = body_lp.process(body_noise.next_sample()) * body_env.next_sample();

        // Pitch-dropping thump
        let punch_freq = 55.0 + 205.0 * (-t / 0.02).exp();
        let punch = punch_osc.next_sample_with_freq(punch_freq) * punch_env.next_sample();

        samples.push(0.7 * crack + 0.6 * body + 0.5 * punch);
    }

    let mut buf = Buf(samples);
    buf.normalize(0.95);
    buf
}

/// Procedural mechanical click: short high-frequency metallic impact and spring resonance.
pub fn mechanical_click() -> Buf {
    let sample_rate = SAMPLE_RATE;
    let total_samples = (sample_rate * 0.035) as usize;

    let mut click_noise = Noise::with_seed(0x3003);
    let mut click_bp = Bp::new(3800.0, 3.5, sample_rate);
    let mut click_env = Decay::new(0.003, sample_rate);

    let mut ping_osc = Osc::new(2800.0, sample_rate);
    let mut ping_env = Decay::new(0.008, sample_rate);

    let mut secondary_osc = Osc::new(4400.0, sample_rate);
    let mut secondary_env = Decay::new(0.005, sample_rate);

    let secondary_delay = (sample_rate * 0.007) as usize;

    let mut samples = Vec::with_capacity(total_samples);
    for i in 0..total_samples {
        let click = click_bp.process(click_noise.next_sample()) * click_env.next_sample();
        let ping = ping_osc.next_sample() * ping_env.next_sample();

        let secondary = if i >= secondary_delay {
            secondary_osc.next_sample() * secondary_env.next_sample() * 0.5
        } else {
            0.0
        };

        samples.push(0.8 * click + 0.5 * ping + secondary);
    }

    let mut buf = Buf(samples);
    buf.normalize(0.95);
    buf
}

/// Procedural explosion rumble: heavy sub-bass boom with extended low-frequency rumble.
pub fn explosion_rumble() -> Buf {
    let sample_rate = SAMPLE_RATE;
    let total_samples = (sample_rate * 1.4) as usize;

    let mut blast_noise = Noise::with_seed(0x4004);
    let mut blast_lp = Lp::new(450.0, sample_rate);
    let mut blast_env = Decay::new(0.18, sample_rate);

    let mut sub_osc = Osc::new(130.0, sample_rate);
    let mut sub_env = Decay::new(0.35, sample_rate);

    let mut rumble_noise = Noise::with_seed(0x5005);
    let mut rumble_bp = Bp::new(85.0, 1.8, sample_rate);
    let mut rumble_lp = Lp::new(180.0, sample_rate);
    let mut rumble_env = Decay::new(0.55, sample_rate);

    let mut samples = Vec::with_capacity(total_samples);
    for i in 0..total_samples {
        let t = i as f32 / sample_rate;

        let blast = blast_lp.process(blast_noise.next_sample()) * blast_env.next_sample();

        let sub_freq = 30.0 + 100.0 * (-t / 0.15).exp();
        let sub = sub_osc.next_sample_with_freq(sub_freq) * sub_env.next_sample();

        let rumble_raw = rumble_bp.process(rumble_noise.next_sample());
        let rumble = rumble_lp.process(rumble_raw) * rumble_env.next_sample();

        samples.push(0.6 * blast + 0.6 * sub + 0.7 * rumble);
    }

    let mut buf = Buf(samples);
    buf.normalize(0.95);
    buf
}

/// Procedural footstep thud: surface scuff followed by low-frequency ground impact.
pub fn footstep_thud() -> Buf {
    let sample_rate = SAMPLE_RATE;
    let total_samples = (sample_rate * 0.16) as usize;

    let mut scuff_noise = Noise::with_seed(0x6006);
    let mut scuff_bp = Bp::new(1400.0, 1.5, sample_rate);
    let mut scuff_env = Decay::new(0.012, sample_rate);

    let mut thud_osc = Osc::new(95.0, sample_rate);
    let mut thud_env = Decay::new(0.04, sample_rate);

    let mut body_noise = Noise::with_seed(0x7007);
    let mut body_lp = Lp::new(220.0, sample_rate);
    let mut body_env = Decay::new(0.05, sample_rate);

    let mut samples = Vec::with_capacity(total_samples);
    for i in 0..total_samples {
        let t = i as f32 / sample_rate;

        let scuff = scuff_bp.process(scuff_noise.next_sample()) * scuff_env.next_sample();

        let thud_freq = 38.0 + 57.0 * (-t / 0.02).exp();
        let thud = thud_osc.next_sample_with_freq(thud_freq) * thud_env.next_sample();

        let body = body_lp.process(body_noise.next_sample()) * body_env.next_sample();

        samples.push(0.4 * scuff + 0.8 * thud + 0.5 * body);
    }

    let mut buf = Buf(samples);
    buf.normalize(0.95);
    buf
}

/// Procedural heartbeat thump: classic double "lub-dub" low-frequency pulse.
pub fn heartbeat_thump() -> Buf {
    let sample_rate = SAMPLE_RATE;
    let total_samples = (sample_rate * 0.65) as usize;

    fn make_pulse(base_freq: f32, duration: f32, sample_rate: f32) -> Buf {
        let count = (duration * sample_rate) as usize;
        let mut osc = Osc::new(base_freq, sample_rate);
        let mut lp = Lp::new(110.0, sample_rate);
        let mut decay = Decay::new(0.045, sample_rate);
        let attack_count = (0.008 * sample_rate) as usize;

        let mut samples = Vec::with_capacity(count);
        for i in 0..count {
            let t = i as f32 / sample_rate;
            let freq = (base_freq * 0.7) + (base_freq * 0.3) * (-t / 0.03).exp();
            let raw = osc.next_sample_with_freq(freq);
            let filtered = lp.process(raw);
            let attack = if i < attack_count {
                i as f32 / attack_count as f32
            } else {
                1.0
            };
            let env = attack * decay.next_sample();
            samples.push(filtered * env);
        }
        Buf(samples)
    }

    let lub = make_pulse(55.0, 0.15, sample_rate);
    let dub = make_pulse(70.0, 0.15, sample_rate);

    let mut out = Buf::zeros(total_samples);
    out.mix_at(&lub, 0, 1.0);
    let dub_offset = (sample_rate * 0.22) as usize;
    out.mix_at(&dub, dub_offset, 0.75);

    out.normalize(0.95);
    out
}

/// Synthesizes a pure sine wave tone at the default frequency (440.0 Hz) for 0.5 seconds.
pub fn test_tone() -> Buf {
    test_tone_custom(440.0, 0.5, SAMPLE_RATE)
}

/// Synthesizes a test tone at `freq` Hz for `duration_secs` with smooth 5ms cosine windowing.
pub fn tone(freq: f32, duration_secs: f32) -> Buf {
    test_tone_custom(freq, duration_secs, SAMPLE_RATE)
}

/// Synthesizes a test tone with customizable frequency, duration, and sample rate.
pub fn test_tone_custom(freq: f32, duration_secs: f32, sample_rate: f32) -> Buf {
    let rate = sample_rate.max(1.0);
    let total_samples = (duration_secs.max(0.001) * rate).round() as usize;
    let fade_samples = ((0.005 * rate).round() as usize).min(total_samples / 2);

    let mut osc = Osc::new(freq, rate);
    let mut samples = Vec::with_capacity(total_samples);

    for i in 0..total_samples {
        let mut sample = osc.next_sample();

        // Smooth cosine fade in/out to eliminate DC clicks
        if i < fade_samples {
            let progress = i as f32 / fade_samples as f32;
            let window = 0.5 * (1.0 - (progress * PI).cos());
            sample *= window;
        } else if i >= total_samples - fade_samples {
            let progress = (total_samples - 1 - i) as f32 / fade_samples as f32;
            let window = 0.5 * (1.0 - (progress * PI).cos());
            sample *= window;
        }

        samples.push(sample);
    }

    let mut buf = Buf(samples);
    buf.normalize(0.95);
    buf
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_deterministic_bounded_and_non_zero() {
        let mut n1 = Noise::with_seed(12345);
        let mut n2 = Noise::with_seed(12345);

        let buf1 = n1.generate(100);
        let buf2 = n2.generate(100);

        assert_eq!(buf1, buf2);
        assert_eq!(buf1.len(), 100);
        for &s in buf1.as_slice() {
            assert!(s.is_finite());
            assert!((-1.0..=1.0).contains(&s));
        }

        // Noise should produce varying samples
        let distinct = buf1.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(distinct > 80);
    }

    #[test]
    fn osc_produces_periodic_finite_sine() {
        let mut osc = Osc::new(440.0, 44100.0);
        let buf = osc.generate(4410); // 100ms

        assert_eq!(buf.len(), 4410);
        let mut max_val = 0.0f32;
        let mut min_val = 0.0f32;
        for &s in buf.as_slice() {
            assert!(s.is_finite());
            assert!((-1.0..=1.0).contains(&s));
            max_val = max_val.max(s);
            min_val = min_val.min(s);
        }
        assert!(max_val > 0.95);
        assert!(min_val < -0.95);
    }

    #[test]
    fn lp_filter_passes_dc_and_smooths() {
        let mut lp = Lp::new(100.0, 44100.0);
        let mut output = 0.0;
        // Step response with DC input = 1.0
        for _ in 0..2000 {
            output = lp.process(1.0);
        }
        // At steady-state DC, output approaches input
        assert!((output - 1.0).abs() < 0.01);
    }

    #[test]
    fn hp_filter_blocks_dc() {
        let mut hp = Hp::new(500.0, 44100.0);
        let mut output = 0.0;
        // Step response with DC input = 1.0
        for _ in 0..2000 {
            output = hp.process(1.0);
        }
        // At steady-state DC, output approaches 0.0
        assert!(output.abs() < 0.01);
    }

    #[test]
    fn bp_filter_is_stable_and_selects_frequency() {
        let mut bp = Bp::new(1000.0, 5.0, 44100.0);

        // Process a sine at resonance frequency (1000 Hz)
        let mut osc_res = Osc::new(1000.0, 44100.0);
        let mut peak_res = 0.0f32;
        for _ in 0..2000 {
            let y = bp.process(osc_res.next_sample());
            assert!(y.is_finite());
            peak_res = peak_res.max(y.abs());
        }

        bp.reset();

        // Process a sine far off resonance (100 Hz)
        let mut osc_off = Osc::new(100.0, 44100.0);
        let mut peak_off = 0.0f32;
        for _ in 0..2000 {
            let y = bp.process(osc_off.next_sample());
            assert!(y.is_finite());
            peak_off = peak_off.max(y.abs());
        }

        assert!(peak_res > peak_off * 2.0);
    }

    #[test]
    fn adsr_lifecycle() {
        let mut adsr = Adsr::new(0.01, 0.01, 0.5, 0.01, 1000.0);
        // 10 samples attack, 10 decay, sustain at 0.5, 10 release
        let buf = adsr.generate_gate(30, 10);

        assert_eq!(buf.len(), 40);
        for &s in buf.as_slice() {
            assert!(s.is_finite());
            assert!((0.0..=1.0).contains(&s));
        }

        // Peak reaches 1.0 at end of attack
        let max_val = buf.as_slice().iter().copied().fold(0.0f32, f32::max);
        assert!((max_val - 1.0).abs() < 0.05);

        // During sustain it holds around 0.5
        assert!((buf[25] - 0.5).abs() < 0.05);

        // At end of release it reaches 0.0
        assert!(buf[39] <= 0.05);
    }

    #[test]
    fn decay_monotonically_drops() {
        let mut decay = Decay::new(0.01, 1000.0);
        let buf = decay.generate(50);

        assert_eq!(buf.len(), 50);
        assert!((buf[0] - 1.0).abs() < 1e-4);
        for w in buf.as_slice().windows(2) {
            assert!(w[0] >= w[1]);
            assert!(w[1].is_finite());
            assert!(w[1] >= 0.0);
        }
    }

    #[test]
    fn buf_operations_and_normalization() {
        let mut b1 = Buf::new(vec![0.1, -0.2, 0.5]);
        b1.normalize(1.0);
        assert!((b1[2] - 1.0).abs() < 1e-5);
        assert!((b1[1] - (-0.4)).abs() < 1e-5);

        let mut b2 = Buf::zeros(2);
        b2.mix(&b1, 0.5);
        assert_eq!(b2.len(), 3);
        assert!((b2[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn pcm16_and_wav_serialization() {
        let buf = Buf::new(vec![0.0, 1.0, -1.0, 0.5]);
        let pcm = buf.to_pcm16_le();

        assert_eq!(pcm.len(), 8);
        let s0 = i16::from_le_bytes([pcm[0], pcm[1]]);
        let s1 = i16::from_le_bytes([pcm[2], pcm[3]]);
        let s2 = i16::from_le_bytes([pcm[4], pcm[5]]);
        let s3 = i16::from_le_bytes([pcm[6], pcm[7]]);

        assert_eq!(s0, 0);
        assert_eq!(s1, 32767);
        assert_eq!(s2, -32767);
        assert!((s3 - 16384).abs() <= 1);

        let wav = buf.to_wav(44100);
        assert_eq!(wav.len(), 44 + 8);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 44);
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1); // PCM
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1); // 1 channel
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 44100);
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16); // 16 bits
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 8);
        assert_eq!(&wav[44..], &pcm[..]);
    }

    fn verify_sound_preset(name: &str, buf: Buf) {
        assert!(
            !buf.is_empty(),
            "Preset '{name}' returned an empty buffer"
        );
        for (i, &s) in buf.as_slice().iter().enumerate() {
            assert!(
                s.is_finite(),
                "Preset '{name}' sample at index {i} is not finite: {s}"
            );
        }

        let peak = buf
            .as_slice()
            .iter()
            .copied()
            .fold(0.0f32, |acc, s| acc.max(s.abs()));
        assert!(
            peak > 0.1,
            "Preset '{name}' has nearly silent peak: {peak}"
        );
        assert!(
            peak <= 1.0,
            "Preset '{name}' peak exceeds 1.0: {peak}"
        );

        let pcm = buf.to_pcm16_le();
        assert_eq!(pcm.len(), buf.len() * 2);
        assert!(pcm.iter().any(|&b| b != 0));

        let wav = buf.to_wav(44100);
        assert_eq!(wav.len(), 44 + buf.len() * 2);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..16], b"WAVEfmt ");
    }

    #[test]
    fn preset_gunshot_impulse() {
        verify_sound_preset("gunshot_impulse", gunshot_impulse());
    }

    #[test]
    fn preset_mechanical_click() {
        verify_sound_preset("mechanical_click", mechanical_click());
    }

    #[test]
    fn preset_explosion_rumble() {
        verify_sound_preset("explosion_rumble", explosion_rumble());
    }

    #[test]
    fn preset_footstep_thud() {
        verify_sound_preset("footstep_thud", footstep_thud());
    }

    #[test]
    fn preset_heartbeat_thump() {
        verify_sound_preset("heartbeat_thump", heartbeat_thump());
    }

    #[test]
    fn preset_test_tone() {
        verify_sound_preset("test_tone", test_tone());
        verify_sound_preset("tone_1000", tone(1000.0, 0.2));
    }
}
