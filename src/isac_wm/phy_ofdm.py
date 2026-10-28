"""A small, independently testable common-waveform MIMO-OFDM ISAC baseline.

It models a two-dimensional downlink multi-stream MIMO channel and a collocated
radar array, with the *same transmitted OFDM resource grid*. Data also illuminate
the radar target; dedicated sensing beams cost payload symbols. Frequency-domain
channels are CP-compatible and quasi-static within each OFDM symbol. This is
not a 3GPP stack, ray tracer, real measured CSI, or object classifier.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

C = 299_792_458.0


@dataclass(frozen=True)
class OFDMConfig:
    n_fft: int = 128
    cp_samples: int = 24
    guard_bins: int = 12
    subcarrier_spacing_hz: float = 500_000.0
    carrier_hz: float = 24e9
    pulse_period_s: float = 0.0005
    symbols: int = 64
    tx_antennas: int = 2
    user_rx_antennas: int = 4
    radar_rx_antennas: int = 8
    target_range_m: float = 18.75
    target_bearing_deg: float = 18.0
    target_radial_speed_mps: float = 1.25
    user_distance_m: float = 40.0
    beam_bearing_deg: float = 20.0  # Fixed sector configuration; true target defaults to 18 deg.
    target_reflectivity: float = 0.32
    clutter_reflectivity: float = 0.12
    comm_snr_db: float = 18.0
    radar_snr_db: float = 8.0

    def __post_init__(self) -> None:
        for field in (
            "n_fft",
            "cp_samples",
            "guard_bins",
            "symbols",
            "tx_antennas",
            "user_rx_antennas",
            "radar_rx_antennas",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.n_fft & (self.n_fft - 1) or not 32 <= self.n_fft <= 4096:
            raise ValueError("n_fft must be a power of two in [32,4096]")
        if not 1 <= self.cp_samples < self.n_fft // 2:
            raise ValueError("cyclic prefix must be positive and shorter than half the FFT")
        if not 1 <= self.guard_bins < self.n_fft // 4:
            raise ValueError("guard_bins must be positive and less than a quarter FFT")
        if not 16 <= self.symbols <= 4096 or self.symbols < self.tx_antennas + 4:
            raise ValueError("insufficient OFDM symbols")
        if not 1 <= self.tx_antennas <= 8 or not self.tx_antennas <= self.user_rx_antennas <= 32:
            raise ValueError(
                "use 1-8 transmitted streams with at least that many user RX antennas (maximum 32)"
            )
        if not 2 <= self.radar_rx_antennas <= 64:
            raise ValueError("radar receiver requires an array of 2-64 antennas")
        positive = (
            "subcarrier_spacing_hz",
            "carrier_hz",
            "pulse_period_s",
            "target_range_m",
            "user_distance_m",
            "target_reflectivity",
        )
        for field in positive:
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        for field in (
            "target_bearing_deg",
            "target_radial_speed_mps",
            "beam_bearing_deg",
            "comm_snr_db",
            "radar_snr_db",
            "clutter_reflectivity",
        ):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if not 0 <= self.clutter_reflectivity <= 5:
            raise ValueError("clutter_reflectivity must be within [0,5]")
        if abs(self.target_bearing_deg) > 60 or abs(self.beam_bearing_deg) > 60:
            raise ValueError("this experiment only models the forward 120-degree sector")
        if self.symbols * self.n_fft * self.user_rx_antennas * self.tx_antennas > 2_000_000:
            raise ValueError("requested communication channel grid exceeds the configured memory budget")
        if self.symbols * self.n_fft * self.radar_rx_antennas > 1_000_000:
            raise ValueError("requested radar cube exceeds the configured memory budget")
        if not -40 <= self.comm_snr_db <= 70 or not -60 <= self.radar_snr_db <= 70:
            raise ValueError("SNR values must be within bounded finite ranges")
        if self.user_distance_m / C > self.cp_samples / self.sample_rate_hz:
            raise ValueError("user direct path is outside the cyclic prefix duration")
        ue = np.array([self.user_distance_m, 0.0])
        target = self.target_range_m * np.array(
            [math.cos(math.radians(self.target_bearing_deg)), math.sin(math.radians(self.target_bearing_deg))]
        )
        reflectors = (np.array([14.0, 9.0]), np.array([28.0, -10.0]), target)
        if (
            max(np.linalg.norm(p) + np.linalg.norm(ue - p) for p in reflectors) / C
            > self.cp_samples / self.sample_rate_hz
        ):
            raise ValueError("one or more bistatic communication paths exceed cyclic prefix duration")
        if self.pulse_period_s < self.ofdm_symbol_duration_s:
            raise ValueError("pulse repetition interval cannot be shorter than the OFDM symbol")
        if self.target_range_m > self.cp_unambiguous_range_m * 0.95:
            raise ValueError("target echo round-trip delay must fit in the cyclic prefix")
        if abs(self.target_doppler_hz) >= 0.45 / self.pulse_period_s:
            raise ValueError("target Doppler outside conservatively unambiguous slow-time band")

    @property
    def sample_rate_hz(self) -> float:
        return self.n_fft * self.subcarrier_spacing_hz

    @property
    def ofdm_symbol_duration_s(self) -> float:
        return (self.n_fft + self.cp_samples) / self.sample_rate_hz

    @property
    def cp_unambiguous_range_m(self) -> float:
        return C * self.cp_samples / (2 * self.sample_rate_hz)

    @property
    def target_doppler_hz(self) -> float:
        return -2 * self.carrier_hz * self.target_radial_speed_mps / C

    @property
    def range_bin_m(self) -> float:
        return C / (2 * self.sample_rate_hz)

    @property
    def doppler_bin_mps(self) -> float:
        return C / (2 * self.carrier_hz * self.symbols * self.pulse_period_s)

    def active_bins(self) -> np.ndarray:
        # Actual FFT bin order; exclude DC and guard subcarriers near Nyquist.
        bins = np.arange(self.n_fft)
        signed = np.minimum(bins, self.n_fft - bins)
        return np.flatnonzero((signed > 0) & (signed < self.n_fft // 2 - self.guard_bins))

    def to_dict(self) -> dict:
        return asdict(self)


def ofdm_modulate(grid: np.ndarray, config: OFDMConfig) -> np.ndarray:
    """Orthonormal IFFT with a cyclic prefix, FFT dimension is second axis."""
    a = np.asarray(grid)
    if a.ndim != 3 or a.shape[1] != config.n_fft or not np.isfinite(a).all():
        raise ValueError("grid must be finite [symbols,fft,streams]")
    td = np.fft.ifft(a, axis=1, norm="ortho")
    return np.concatenate([td[:, -config.cp_samples :], td], axis=1)


def ofdm_demodulate(waveform: np.ndarray, config: OFDMConfig) -> np.ndarray:
    a = np.asarray(waveform)
    if a.ndim != 3 or a.shape[1] != config.n_fft + config.cp_samples or not np.isfinite(a).all():
        raise ValueError("waveform must be finite [symbols,fft+cp,streams]")
    return np.fft.fft(a[:, config.cp_samples :], axis=1, norm="ortho")


def steering(n: int, bearing_deg: float) -> np.ndarray:
    """ULA half-wavelength spacings, broadside is 0 degrees."""
    return np.exp(1j * np.pi * np.arange(n) * np.sin(np.deg2rad(bearing_deg)))


def qpsk(bits: np.ndarray) -> np.ndarray:
    a = np.asarray(bits)
    if a.shape[-1] != 2 or not np.isin(a, [0, 1]).all():
        raise ValueError("QPSK bit pairs must be binary")
    return ((1 - 2 * a[..., 0].astype(np.float64)) + 1j * (1 - 2 * a[..., 1].astype(np.float64))) / math.sqrt(
        2
    )


def qpsk_bits(symbols: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(symbols).real < 0, np.asarray(symbols).imag < 0], axis=-1).astype(np.uint8)


def _complex_noise(rng: np.random.Generator, shape: tuple, variance: float) -> np.ndarray:
    return math.sqrt(variance / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def tx_grid(
    config: OFDMConfig, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the transmit grid, payload slot IDs, actual data bits and probe slots.

    Every active RE carries the *same total average radiated energy*: independent
    MIMO payloads or one target-directed beam. An initial orthogonal per-Tx
    preamble is reserved across ALL fractions so channel estimation is fair.
    """
    if not math.isfinite(fraction) or fraction < 0 or fraction > 0.8:
        raise ValueError("probe fraction must be finite and in [0,0.8]")
    active = config.active_bins()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    bits_all = rng.integers(0, 2, (config.symbols, len(active), config.tx_antennas, 2), dtype=np.uint8)
    pilots = qpsk(rng.integers(0, 2, (config.symbols, len(active), 2), dtype=np.uint8))
    available = config.symbols - config.tx_antennas
    n_probe = round(available * fraction)
    probes = (
        np.rint(np.linspace(config.tx_antennas, config.symbols - 1, n_probe)).astype(int)
        if n_probe
        else np.empty(0, dtype=int)
    )
    if len(np.unique(probes)) != len(probes):
        raise RuntimeError("probe scheduler duplicated a symbol")
    payload = np.setdiff1d(np.arange(config.tx_antennas, config.symbols), probes)
    grid = np.zeros((config.symbols, config.n_fft, config.tx_antennas), dtype=np.complex128)
    # Orthonormal block preamble identifies each Tx -> all Rx channel coefficients.
    for t in range(config.tx_antennas):
        grid[t, active, t] = pilots[t]
    if len(payload):
        grid[np.ix_(payload, active, np.arange(config.tx_antennas))] = qpsk(bits_all[payload]) / math.sqrt(
            config.tx_antennas
        )
    if len(probes):
        beam = steering(config.tx_antennas, config.beam_bearing_deg) / math.sqrt(config.tx_antennas)
        grid[np.ix_(probes, active, np.arange(config.tx_antennas))] = pilots[probes, :, None] * beam
    return grid, payload, bits_all[payload], probes


def _echo_geometry(config: OFDMConfig) -> tuple[np.ndarray, float, float]:
    theta = math.radians(config.target_bearing_deg)
    point = config.target_range_m * np.array([math.cos(theta), math.sin(theta)])
    ue = np.array([config.user_distance_m, 0.0])
    unit_velocity = point / np.linalg.norm(point)
    to_target_from_ue = (point - ue) / np.linalg.norm(point - ue)
    bistatic_fd = (
        -config.carrier_hz
        / C
        * config.target_radial_speed_mps
        * float(np.dot(unit_velocity, unit_velocity + to_target_from_ue))
    )
    return point, np.linalg.norm(point - ue), bistatic_fd


def _channel(config: OFDMConfig) -> np.ndarray:
    """Communication CSI [slow-time, FFT bin, user Rx, base-station Tx].

    A shared moving scatterer appears in *both* links. LOS and two static
    reflectors give a realistic-enough rank-2 downlink for spatial multiplexing;
    constant complex gains and simple ULA steering are explicit assumptions.
    """
    f = np.fft.fftfreq(config.n_fft, d=1 / config.sample_rate_hz)
    t = np.arange(config.symbols) * config.pulse_period_s
    h = np.zeros(
        (config.symbols, config.n_fft, config.user_rx_antennas, config.tx_antennas),
        dtype=np.complex128,
    )
    ue = np.array([config.user_distance_m, 0.0])
    point, _, fd = _echo_geometry(config)
    paths = [
        (np.array([0.0, 0.0]), 0.8 + 0j, 0.0, True),
        (np.array([14.0, 9.0]), 0.48 * np.exp(0.7j), 0.0, False),
        (np.array([28.0, -10.0]), 0.32 * np.exp(-0.9j), 0.0, False),
        (point, 0.22 * np.exp(1.3j), fd, False),
    ]
    for p, alpha, doppler, direct in paths:
        if direct:
            departure, arrival, distance = 0.0, 0.0, config.user_distance_m
        else:
            departure = math.degrees(math.atan2(p[1], p[0]))
            arrival = math.degrees(math.atan2(p[1], config.user_distance_m - p[0]))
            distance = np.linalg.norm(p) + np.linalg.norm(ue - p)
        array = np.outer(
            steering(config.user_rx_antennas, arrival), steering(config.tx_antennas, departure).conj()
        )
        spectral = np.exp(-2j * np.pi * f * distance / C)
        slowtime = np.exp(2j * np.pi * doppler * t)
        h += alpha * slowtime[:, None, None, None] * spectral[None, :, None, None] * array
    return h


def _radar_noiseless(config: OFDMConfig, tx: np.ndarray) -> np.ndarray:
    """Monostatic moving point target plus one stationary clutter reflector."""
    f = np.fft.fftfreq(config.n_fft, d=1 / config.sample_rate_hz)
    slowtime = np.arange(config.symbols) * config.pulse_period_s
    theta = config.target_bearing_deg
    target_amp = config.target_reflectivity * (18.75 / config.target_range_m) ** 2
    paths = (
        (theta, config.target_range_m, config.target_doppler_hz, target_amp),
        (-28.0, 12.0, 0.0, config.clutter_reflectivity * np.exp(-0.5j)),
    )
    result = np.zeros((config.symbols, config.n_fft, config.radar_rx_antennas), dtype=np.complex128)
    for bearing, distance, doppler, alpha in paths:
        a_tx, a_rx = steering(config.tx_antennas, bearing), steering(config.radar_rx_antennas, bearing)
        projection = np.einsum("t,skt->sk", a_tx.conj(), tx, optimize=True)
        phase = (
            np.exp(-2j * np.pi * f * (2 * distance / C))[None, :]
            * np.exp(2j * np.pi * doppler * slowtime)[:, None]
        )
        result += (alpha * projection * phase)[:, :, None] * a_rx[None, None, :]
    return result


@dataclass
class FrameSignals:
    tx_grid: np.ndarray
    user_grid: np.ndarray
    radar_grid: np.ndarray
    payload_slots: np.ndarray
    bits: np.ndarray
    probe_slots: np.ndarray
    user_noise_power: float
    radar_noise_power: float


def simulate_frame(config: OFDMConfig, *, probe_fraction: float, seed: int = 2026) -> FrameSignals:
    """Generate one shared OFDM burst and both received paths.

    FFT-domain path gains approximate a CP-compatible circular channel under
    quasi-static-per-symbol assumptions (fractional delays are idealized). Noise is independent circular AWGN
    injected *in time domain* and recovered with the actual FFT receiver.
    """
    if type(seed) is not int or seed < 0 or seed >= 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    tx, payload, bits, probes = tx_grid(config, probe_fraction, seed)
    h_comm = _channel(config)
    user_signal = np.einsum("skrt,skt->skr", h_comm, tx, optimize=True)
    radar_signal = _radar_noiseless(config, tx)
    active = config.active_bins()
    # Use a FIXED receiver thermal-noise reference for every sensing schedule.
    # This averages received channel power for equal-energy independent Tx
    # streams before choosing which symbols will instead be radar probes.
    user_reference_power = float(
        np.mean(np.sum(np.abs(h_comm[0, active]) ** 2, axis=-1) / config.tx_antennas)
    )
    if user_reference_power <= 0:
        raise RuntimeError("zero communication reference energy")
    user_n0 = user_reference_power / 10 ** (config.comm_snr_db / 10)
    target_amplitude = config.target_reflectivity * (18.75 / config.target_range_m) ** 2
    radar_n0 = target_amplitude**2 / 10 ** (config.radar_snr_db / 10)
    rng_comm = np.random.default_rng(np.random.SeedSequence([seed, 22]))
    rng_radar = np.random.default_rng(np.random.SeedSequence([seed, 33]))
    user_waveform = ofdm_modulate(user_signal, config)
    radar_waveform = ofdm_modulate(radar_signal, config)
    user_waveform += _complex_noise(rng_comm, user_waveform.shape, user_n0)
    radar_waveform += _complex_noise(rng_radar, radar_waveform.shape, radar_n0)
    return FrameSignals(
        tx,
        ofdm_demodulate(user_waveform, config),
        ofdm_demodulate(radar_waveform, config),
        payload,
        bits,
        probes,
        user_n0,
        radar_n0,
    )


def decode_downlink(config: OFDMConfig, signals: FrameSignals) -> dict:
    """Independent pilot CSI estimate; MMSE spatial separation of QPSK payloads."""
    active = config.active_bins()
    h_hat = np.zeros((len(active), config.user_rx_antennas, config.tx_antennas), complex)
    for tx in range(config.tx_antennas):
        symbols = signals.tx_grid[tx, active, tx]
        if np.any(abs(symbols) < 0.99):
            raise ValueError("training pilot unexpectedly absent")
        h_hat[:, :, tx] = signals.user_grid[tx, active, :] / symbols[:, None]
    errors = 0
    for k, carrier in enumerate(active):
        h = h_hat[k]
        w = np.linalg.solve(
            h.conj().T @ h + signals.user_noise_power * config.tx_antennas * np.eye(config.tx_antennas),
            h.conj().T,
        )
        estimated = (w @ signals.user_grid[signals.payload_slots, carrier, :].T).T
        decoded = qpsk_bits(estimated)
        truth = signals.bits[:, k, :, :]
        errors += int(np.count_nonzero(decoded != truth))
    total_bits = int(signals.bits.size)
    if total_bits == 0:
        raise ValueError("frame has no payload")
    duration_s = config.symbols * config.pulse_period_s
    return {
        "uncoded_bit_errors": errors,
        "payload_bits": total_bits,
        "ber": errors / total_bits,
        "nominal_raw_payload_rate_bps": total_bits / duration_s,
        "correct_bit_rate_proxy_bps": (total_bits - errors) / duration_s,
        "training_symbols": config.tx_antennas,
        "sensing_only_symbols": int(len(signals.probe_slots)),
        "data_symbols": int(len(signals.payload_slots)),
        "note": "Uncoded QPSK detection with noisy pilot-estimated CSI; correct-bit rate is NOT reliable transport goodput.",
    }


def estimate_radar(
    config: OFDMConfig, signals: FrameSignals, *, angle_grid_deg: np.ndarray | None = None
) -> dict:
    """Matched angle / range / Doppler search using transmitted *data and probes*.

    The unknown target is NOT supplied to this estimator. The fixed angular
    search grid and CP-limited delay window are declared in the configuration.
    Stationary Doppler bin is excluded; detecting stationary targets requires
    a separate clutter model, not a misleading default result here.
    """
    grid = np.arange(-60.0, 60.01, 2.0) if angle_grid_deg is None else np.asarray(angle_grid_deg)
    if grid.ndim != 1 or not 1 <= len(grid) <= 360 or not np.isfinite(grid).all() or np.any(abs(grid) > 80):
        raise ValueError("angle_grid_deg must contain 1-360 finite angles within +-80 degrees")
    best_score, best_result = -math.inf, None
    n_symbols = config.symbols
    for bearing in grid:
        ar = steering(config.radar_rx_antennas, float(bearing))
        at = steering(config.tx_antennas, float(bearing))
        rx = np.einsum("r,skr->sk", ar.conj(), signals.radar_grid, optimize=True)
        tx = np.einsum("t,skt->sk", at.conj(), signals.tx_grid, optimize=True)
        cross = rx * tx.conj()
        spectrum = np.fft.fft(cross, axis=0)
        responses = np.fft.ifft(spectrum, axis=1)
        # Strictly exclude zero-Doppler and out-of-CP-range hypotheses.
        magnitude = np.abs(responses[:, 1 : config.cp_samples]) ** 2
        magnitude[0, :] = 0
        indices = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
        score = float(magnitude[indices])
        if score > best_score:
            doppler_bin, delay_bin = int(indices[0]), int(indices[1] + 1)
            best_score = score
            best_result = (float(bearing), delay_bin, doppler_bin)
    assert best_result is not None
    bearing, delay_bin, doppler_bin = best_result
    fd = np.fft.fftfreq(n_symbols, d=config.pulse_period_s)[doppler_bin]
    return {
        "range_m": delay_bin * config.range_bin_m,
        "bearing_deg": bearing,
        "radial_speed_mps": float(-fd * C / (2 * config.carrier_hz)),
        "delay_bin": delay_bin,
        "doppler_bin": doppler_bin,
        "matched_peak_energy": best_score,
        "range_bin_m": config.range_bin_m,
        "velocity_bin_mps": config.doppler_bin_mps,
        "note": "On-grid one-moving-target matched filter; no material classification or real-hardware calibration.",
    }


def evaluate_frame(config: OFDMConfig, *, probe_fraction: float, seed: int = 2026) -> dict:
    signals = simulate_frame(config, probe_fraction=probe_fraction, seed=seed)
    comm = decode_downlink(config, signals)
    radar = estimate_radar(config, signals)
    return {
        "schema_version": 1,
        "seed": seed,
        "probe_fraction_requested": probe_fraction,
        "probe_fraction_realized": len(signals.probe_slots) / config.symbols,
        "ofdm": config.to_dict(),
        "physical_resolution": {
            "sample_rate_hz": config.sample_rate_hz,
            "occupied_bandwidth_hz": len(config.active_bins()) * config.subcarrier_spacing_hz,
            "cp_unambiguous_monostatic_range_m": config.cp_unambiguous_range_m,
            "coherent_pulse_interval_s": config.symbols * config.pulse_period_s,
        },
        "truth_for_evaluation_only": {
            "range_m": config.target_range_m,
            "bearing_deg": config.target_bearing_deg,
            "radial_speed_mps": config.target_radial_speed_mps,
        },
        "communication": comm,
        "sensing": radar,
        "assumptions": [
            "One moving point target, fixed departure/arrival angles during short coherent burst",
            "LOS + two static reflectors + shared moving target for the downlink channel",
            "Quasi-static within OFDM symbols; ideal fractional-delay frequency response; no intrasymbol ICI",
            "Perfect frame and frequency synchronization, finite noisy CSI from a shared fixed preamble",
            "Fixed non-oracle sensing beam; total transmitted RE energy normalized across data/probe modes",
            "Known radar transmit waveform; monostatic target Doppler and zero-Doppler clutter cancellation",
            "Not real 5G NR, calibrated channel map, material classifier, or complete sensing/communication Pareto boundary",
        ],
    }
