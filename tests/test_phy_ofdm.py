"""Independent physical sanity checks for the common-waveform PHY baseline."""

from dataclasses import replace

import numpy as np
import pytest

from isac_wm.phy_ofdm import (
    C,
    OFDMConfig,
    _channel,
    _radar_noiseless,
    decode_downlink,
    estimate_radar,
    evaluate_frame,
    ofdm_demodulate,
    ofdm_modulate,
    qpsk,
    qpsk_bits,
    simulate_frame,
    tx_grid,
)


def test_ofdm_cyclic_prefix_and_frequency_roundtrip():
    cfg = OFDMConfig()
    rng = np.random.default_rng(31)
    symbols = rng.normal(size=(5, cfg.n_fft, 3)) + 1j * rng.normal(size=(5, cfg.n_fft, 3))
    returned = ofdm_demodulate(ofdm_modulate(symbols, cfg), cfg)
    assert np.allclose(returned, symbols, rtol=1e-13, atol=1e-13)


def test_ofdm_sample_delay_matches_independent_time_shifted_cp_signal():
    cfg = OFDMConfig()
    rng = np.random.default_rng(43)
    spectrum = rng.normal(size=(2, cfg.n_fft, 1)) + 1j * rng.normal(size=(2, cfg.n_fft, 1))
    tx = ofdm_modulate(spectrum, cfg)
    delay_samples = 7
    time_indices = np.arange(cfg.cp_samples, cfg.cp_samples + cfg.n_fft) - delay_samples
    delayed_data = tx[:, time_indices, :]
    observed = np.fft.fft(delayed_data, axis=1, norm="ortho")
    bins = np.arange(cfg.n_fft)
    expected = spectrum * np.exp(-2j * np.pi * bins[None, :, None] * delay_samples / cfg.n_fft)
    assert np.allclose(observed, expected, atol=1e-12, rtol=1e-12)


def test_steered_pilots_use_fixed_total_transmit_energy():
    cfg = OFDMConfig()
    mask = cfg.active_bins()
    for frac in (0.0, 0.2, 0.4):
        grid, payload, bits, probes = tx_grid(cfg, frac, 10)
        assert len(payload) + len(probes) == cfg.symbols - cfg.tx_antennas
        energies = np.sum(abs(grid[:, mask, :]) ** 2, axis=-1)
        assert np.allclose(energies, 1.0)
        assert bits.shape == (len(payload), len(mask), cfg.tx_antennas, 2)
        assert np.all(grid[:, np.setdiff1d(np.arange(cfg.n_fft), mask), :] == 0)


def test_qpsk_mapping_is_bijective():
    bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.uint8)
    assert np.array_equal(qpsk_bits(qpsk(bits)), bits)


def test_radar_uses_shared_downlink_transmit_symbols_not_oracle_state():
    cfg = replace(OFDMConfig(), radar_snr_db=45.0, clutter_reflectivity=0.0)
    signals = simulate_frame(cfg, probe_fraction=0.2, seed=18)
    radar = estimate_radar(cfg, signals)
    assert abs(radar["range_m"] - cfg.target_range_m) < cfg.range_bin_m
    assert abs(radar["bearing_deg"] - cfg.target_bearing_deg) <= 2.0
    assert abs(radar["radial_speed_mps"] - cfg.target_radial_speed_mps) <= cfg.doppler_bin_mps


def test_shared_scatterer_changes_both_links_consistently():
    base = OFDMConfig()
    shifted = replace(base, target_bearing_deg=28.0)
    s = tx_grid(base, 0.2, 18)[0]
    assert not np.allclose(_channel(base), _channel(shifted))
    assert not np.allclose(_radar_noiseless(base, s), _radar_noiseless(shifted, s))


def test_mimo_user_has_nontrivial_full_rank_channel_and_pilot_estimator():
    cfg = replace(OFDMConfig(), comm_snr_db=60.0)
    h = _channel(cfg)[0]
    # A genuine spatially multiplexed channel, not a renamed single-antenna link.
    assert np.linalg.matrix_rank(h[cfg.active_bins()[0]]) == cfg.tx_antennas
    frame = simulate_frame(cfg, probe_fraction=0.2, seed=19)
    stats = decode_downlink(cfg, frame)
    assert stats["payload_bits"] > 0
    assert 0 <= stats["ber"] < 0.1


def test_low_snr_degrades_communication_ber_on_paired_trials():
    cfg = OFDMConfig()
    low, high = [], []
    for seed in (310, 311, 312, 313):
        low.append(
            evaluate_frame(replace(cfg, comm_snr_db=2.0), probe_fraction=0.2, seed=seed)["communication"][
                "ber"
            ]
        )
        high.append(
            evaluate_frame(replace(cfg, comm_snr_db=28.0), probe_fraction=0.2, seed=seed)["communication"][
                "ber"
            ]
        )
    assert float(np.mean(low)) > float(np.mean(high)) + 0.08


def test_sensing_probe_trades_payload_for_target_energy():
    cfg = OFDMConfig()
    a = evaluate_frame(cfg, probe_fraction=0.0, seed=71)
    b = evaluate_frame(cfg, probe_fraction=0.4, seed=71)
    assert b["communication"]["payload_bits"] < a["communication"]["payload_bits"]
    assert (
        b["communication"]["nominal_raw_payload_rate_bps"]
        < a["communication"]["nominal_raw_payload_rate_bps"]
    )
    assert b["sensing"]["matched_peak_energy"] > a["sensing"]["matched_peak_energy"]


def test_wrong_probe_bearing_can_reduce_target_energy():
    cfg = OFDMConfig()
    aligned = evaluate_frame(cfg, probe_fraction=0.4, seed=71)
    miss = evaluate_frame(replace(cfg, beam_bearing_deg=-40.0), probe_fraction=0.4, seed=71)
    assert miss["sensing"]["matched_peak_energy"] < aligned["sensing"]["matched_peak_energy"]


def test_immutable_config_rejects_underspecified_or_unphysical_channel():
    cfg = OFDMConfig()
    failures = (
        {"n_fft": 96},
        {"cp_samples": 1},
        {"target_range_m": 60},
        {"user_distance_m": 300},
        {"tx_antennas": 5, "user_rx_antennas": 4},
        {"guard_bins": 99},
        {"pulse_period_s": 1e-8},
        {"target_radial_speed_mps": 100},
        {"symbols": 999999},
        {"comm_snr_db": float("nan")},
        {"target_bearing_deg": 100},
    )
    for setting in failures:
        with pytest.raises(ValueError):
            replace(cfg, **setting)


def test_reject_bad_waveform_shapes_fractions_and_invalid_bits():
    cfg = OFDMConfig()
    with pytest.raises(ValueError):
        tx_grid(cfg, float("nan"), 1)
    with pytest.raises(ValueError):
        simulate_frame(cfg, probe_fraction=0.1, seed=-1)
    with pytest.raises(ValueError):
        ofdm_modulate(np.zeros((3, 17, 1)), cfg)
    with pytest.raises(ValueError):
        ofdm_demodulate(np.zeros((3, 17, 1)), cfg)
    with pytest.raises(ValueError):
        qpsk(np.array([[2, 0]]))
    with pytest.raises(ValueError):
        estimate_radar(cfg, simulate_frame(cfg, probe_fraction=0.1, seed=1), angle_grid_deg=[float("nan")])


def test_frame_reproducibility_and_nonoracle_scattering_labels():
    cfg = OFDMConfig()
    left = evaluate_frame(cfg, probe_fraction=0.2, seed=1234)
    right = evaluate_frame(cfg, probe_fraction=0.2, seed=1234)
    assert left == right
    assert C > 299e6
    assert "material" not in left["sensing"]
    assert "object_class" not in left["sensing"]


def test_receiver_noise_power_does_not_change_when_probe_allocation_changes():
    cfg = OFDMConfig()
    a = simulate_frame(cfg, probe_fraction=0, seed=52)
    b = simulate_frame(cfg, probe_fraction=0.4, seed=52)
    assert a.user_noise_power == pytest.approx(b.user_noise_power, abs=0)
    assert a.radar_noise_power == pytest.approx(b.radar_noise_power, abs=0)
    assert cfg.beam_bearing_deg != cfg.target_bearing_deg
