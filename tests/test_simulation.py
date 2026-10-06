"""DGP normalisation, determinism and the nested fixed-permutation observation protocol."""
import numpy as np
import pytest

from src.simulation import make_beta_star, n_features, n_observed, simulate_replication

CFG = {"seed": 5205, "simulation": {"T_train": 30, "T_test": 4000, "c_true": 10.0, "b_star": 0.2,
                                    "noise_std": 1.0, "n_replications": 3}}


def test_beta_is_dense_with_exact_norm():
    b = make_beta_star(500, 0.2, np.random.default_rng(1))
    assert b @ b == pytest.approx(0.2, abs=1e-12)
    assert np.count_nonzero(b) == 500


def test_dimensions_and_P():
    d = simulate_replication(CFG, 0)
    assert n_features(CFG) == 300 and d.S_train.shape == (30, 300) and d.S_test.shape == (4000, 300)


def test_second_moment_of_returns_is_one_plus_bstar():
    d = simulate_replication(CFG, 0)
    assert np.mean(d.y_test ** 2) == pytest.approx(1.2, abs=0.08)


def test_replication_is_deterministic_and_independent_of_requested_count():
    a, b = simulate_replication(CFG, 1), simulate_replication(CFG, 1)
    assert a.fingerprint() == b.fingerprint()
    cfg2 = {**CFG, "simulation": {**CFG["simulation"], "n_replications": 50}}
    assert simulate_replication(cfg2, 1).fingerprint() == a.fingerprint()
    assert simulate_replication(CFG, 2).fingerprint() != a.fingerprint()


def test_observed_feature_sets_are_nested_and_use_one_fixed_permutation():
    d = simulate_replication(CFG, 0)
    P, T = n_features(CFG), CFG["simulation"]["T_train"]
    sets = [set(d.observed_columns(n_observed(cq, T, P)).tolist()) for cq in (0.5, 1.02, 3.0, 10.0)]
    for small, big in zip(sets, sets[1:]):
        assert small < big                                   # strictly nested
    assert sets[-1] == set(range(P))                         # c_q = c reveals everything
    assert sorted(d.perm.tolist()) == list(range(P))         # a genuine permutation
    np.testing.assert_array_equal(d.observed_columns(50), d.perm[:50])   # prefix of the same permutation


def test_n_observed_rounding_and_cap():
    assert n_observed(0.5, 300, 3000) == 150 and n_observed(1.02, 300, 3000) == 306
    assert n_observed(99, 300, 3000) == 3000
