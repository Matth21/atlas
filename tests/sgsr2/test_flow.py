from atlas.core.sgsr2_flow import BUDGET_SAFETY_BITS, budget_gb_to_bits


def test_budget_conversion_qwen_scale():
    # 4.3 GB su 7.6B parametri ≈ 4.85 bit/w
    b = budget_gb_to_bits(4.3, 7_615_616_512)
    assert 4.7 < b < 4.9


def test_budget_conversion_roundtrip():
    n = 1_100_048_384
    gb = budget_gb_to_bits(0.6, n) * n / 8 / 1024**3
    assert abs(gb - 0.6) < 1e-9


def test_safety_margin_positive():
    assert 0 < BUDGET_SAFETY_BITS < 0.5
