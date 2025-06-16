import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

def calculate_claim_metrics(data):
    data = data.copy()
    data['HasClaim'] = data['TotalClaims'] > 0
    claim_freq = data.groupby('Group')['HasClaim'].mean()
    claim_severity = data[data['HasClaim']].groupby('Group')['TotalClaims'].mean()
    margin = data.groupby('Group').apply(lambda x: (x['TotalPremium'] - x['TotalClaims']).mean())
    return claim_freq, claim_severity, margin

def t_test_metric(data, metric):
    group_a = data[data['Group'] == 'A'][metric]
    group_b = data[data['Group'] == 'B'][metric]
    stat, p_value = ttest_ind(group_a, group_b, equal_var=False)
    return stat, p_value

def chi_squared_risk(data):
    contingency = pd.crosstab(data['Group'], data['HasClaim'])
    chi2, p, _, _ = chi2_contingency(contingency)
    return chi2, p

def interpret_result(p1, p2, p3):
    results = []
    if p1 < 0.05:
        results.append("→ Risk frequency difference is significant.")
    if p2 < 0.05:
        results.append("→ Claim severity difference is significant.")
    if p3 < 0.05:
        results.append("→ Margin difference is significant.")
    if not results:
        return "Fail to reject null hypothesis. No significant differences found."
    return " ".join(results)


def hypothesis_test_pipeline(data, group_col, hypothesis_name):
    data = data.copy()

    # Drop NA values in necessary columns
    #data = data.dropna(subset=['TotalClaims', 'TotalPremium', group_col])

    # Select top 2 categories for grouping
    top_two = data[group_col].value_counts().index[:2]
    data = data[data[group_col].isin(top_two)]
    data['Group'] = data[group_col].apply(lambda x: 'A' if x == top_two[0] else 'B')

    # Add HasClaim here so it's available globally
    data['HasClaim'] = data['TotalClaims'] > 0

    print(f"\n--- Hypothesis: {hypothesis_name} ---")
    claim_freq, claim_severity, margin = calculate_claim_metrics(data)

    # Chi-squared on claim frequency
    chi2, p1 = chi_squared_risk(data)
    print(f"Chi-squared test (Claim Frequency): p = {p1:.4f}")

    # T-test on severity
    _, p2 = t_test_metric(data[data['HasClaim']], 'TotalClaims')
    print(f"T-test (Claim Severity): p = {p2:.4f}")

    # T-test on margin
    data['Margin'] = data['TotalPremium'] - data['TotalClaims']
    _, p3 = t_test_metric(data, 'Margin')
    print(f"T-test (Margin): p = {p3:.4f}")

    return {
        "hypothesis": hypothesis_name,
        "chi2_p": p1,
        "severity_p": p2,
        "margin_p": p3,
        "recommendation": interpret_result(p1, p2, p3)
    }

