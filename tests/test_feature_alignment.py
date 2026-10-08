"""The Task 3 loader must fail loudly on every kind of train/test misalignment."""
import numpy as np
import pandas as pd
import pytest

from src.data import DataValidationError, align_train_test, load_task_data


def make_frames(T=30, P=4, T_te=10, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"feature{i + 1}" for i in range(P)]
    tr = pd.DataFrame(rng.standard_normal((T, P)), columns=cols)
    tr.insert(0, "t", [str(i) for i in range(T)])
    tr["return"] = rng.standard_normal(T)
    te = pd.DataFrame(rng.standard_normal((T_te, P)), columns=cols)
    te.insert(0, "t", [str(T + i) for i in range(T_te)])
    return tr, te


def test_valid_pair_loads_and_reorders_test_features():
    tr, te = make_frames()
    shuffled = te[["t", "feature3", "feature1", "feature4", "feature2"]]
    d = align_train_test(tr, shuffled, "X")
    assert d.feature_names == ["feature1", "feature2", "feature3", "feature4"]
    np.testing.assert_array_equal(d.X_test, te[d.feature_names].to_numpy())     # re-ordered to training order
    assert d.t_test == te["t"].tolist()                                          # test t order preserved


def test_missing_feature_fails():
    tr, te = make_frames()
    with pytest.raises(DataValidationError, match="missing"):
        align_train_test(tr, te.drop(columns="feature2"), "X")


def test_extra_feature_fails():
    tr, te = make_frames()
    te["feature5"] = 0.0
    with pytest.raises(DataValidationError, match="extra"):
        align_train_test(tr, te, "X")


def test_wrong_feature_name_fails():
    tr, te = make_frames()
    with pytest.raises(DataValidationError):
        align_train_test(tr, te.rename(columns={"feature2": "feat2"}), "X")


def test_duplicate_t_fails_in_train_and_test():
    tr, te = make_frames()
    tr2 = tr.copy(); tr2.loc[5, "t"] = tr2.loc[4, "t"]
    with pytest.raises(DataValidationError, match="duplicate"):
        align_train_test(tr2, te, "X")
    te2 = te.copy(); te2.loc[3, "t"] = te2.loc[2, "t"]
    with pytest.raises(DataValidationError, match="duplicate"):
        align_train_test(tr, te2, "X")


def test_nan_fails_in_features_target_and_t():
    tr, te = make_frames()
    for frame_name in ("train_feature", "train_target", "test_feature"):
        a, b = tr.copy(), te.copy()
        if frame_name == "train_feature":
            a.loc[2, "feature1"] = np.nan
        elif frame_name == "train_target":
            a.loc[2, "return"] = np.nan
        else:
            b.loc[1, "feature3"] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            align_train_test(a, b, "X")
    c = te.copy(); c.loc[0, "t"] = np.nan
    with pytest.raises(DataValidationError):
        align_train_test(tr, c, "X")


def test_non_increasing_t_fails():
    tr, te = make_frames()
    te2 = te.iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataValidationError, match="increasing"):
        align_train_test(tr, te2, "X")


def test_missing_target_column_fails():
    tr, te = make_frames()
    with pytest.raises(DataValidationError, match="return"):
        align_train_test(tr.drop(columns="return"), te, "X")


def test_load_from_csv_files_preserves_raw_t_strings(tmp_path):
    tr, te = make_frames()
    tr["t"] = [f"2020-01-{i + 1:02d}" for i in range(len(tr))]
    te["t"] = [f"2020-02-{i + 1:02d}" for i in range(len(te))]
    tr.to_csv(tmp_path / "pairA_train.csv", index=False)
    te.to_csv(tmp_path / "pairA_test_features.csv", index=False)
    d = load_task_data(tmp_path, "A")
    assert d.t_test == te["t"].tolist()
    with pytest.raises(FileNotFoundError):
        load_task_data(tmp_path, "B")
