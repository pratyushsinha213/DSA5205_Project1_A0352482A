"""Chronological validation helpers never look into the future of the validation block."""
import numpy as np
import pytest

from src.validation import forward_splits, holdout_split


def test_holdout_is_chronological_and_disjoint():
    fit, val = holdout_split(100, 0.25)
    assert len(fit) == 75 and len(val) == 25 and fit.max() < val.min()


def test_forward_splits_expanding_and_strictly_later():
    splits = list(forward_splits(120, 5))
    assert len(splits) == 5
    prev = 0
    for tr, va in splits:
        assert tr.min() == 0 and tr.max() < va.min()            # validation strictly after training
        assert len(tr) > prev
        prev = len(tr)
    assert splits[-1][1].max() == 119                            # last block reaches the end


def test_forward_splits_reject_too_short_series():
    with pytest.raises(AssertionError):
        list(forward_splits(5, 5))
