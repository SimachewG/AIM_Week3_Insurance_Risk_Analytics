import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.hypothesis_testing import calculate_claim_metrics, t_test_metric, chi_squared_risk

def test_metrics_computation():
    sample = pd.DataFrame({
        'Group': ['A', 'A', 'B', 'B'],
        'TotalClaims': [0, 1000, 0, 2000],
        'TotalPremium': [5000, 5000, 5000, 5000]
    })
    freq, sev, margin = calculate_claim_metrics(sample)
    assert freq.loc['A'] == 0.5
    assert freq.loc['B'] == 0.5
    assert round(sev.loc['A'], 2) == 1000
    assert round(margin.loc['A'], 2) == 4500

def test_t_test_metric():
    df = pd.DataFrame({
        'Group': ['A'] * 10 + ['B'] * 10,
        'TotalClaims': [100] * 10 + [200] * 10
    })
    stat, p = t_test_metric(df, 'TotalClaims')
    assert p < 0.05

def test_chi_squared_risk():
    df = pd.DataFrame({
        'Group': ['A'] * 5 + ['B'] * 5,
        'TotalClaims': [0, 100, 0, 0, 0, 500, 0, 100, 0, 0]
    })
    df['HasClaim'] = df['TotalClaims'] > 0
    chi2, p = chi_squared_risk(df)
    assert 0 <= p <= 1
