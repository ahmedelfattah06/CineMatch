"""
Apriori Association Rules Algorithm
Groups movies by director — each director's filmography is an "itemset" of genres.
Finds genre co-occurrence patterns using the Apriori algorithm.
Results are cached after the first run.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

_cached_rules = None


def run_apriori(df: pd.DataFrame, min_support: float = 0.05, min_confidence: float = 0.3, min_lift: float = 1.0) -> list:
    """
    Build association rules from genre co-occurrence within director filmographies.
    Returns top 15 rules sorted by lift as a list of dicts.
    Results are cached after the first run.
    """
    global _cached_rules
    if _cached_rules is not None:
        return _cached_rules

    # Group movies by director: each director -> set of genres they worked in
    director_genres = df.groupby('director')['genre'].apply(list).reset_index()

    # Build transactions: each director's genre list is one transaction
    transactions = director_genres['genre'].tolist()
    # Convert to sets to remove duplicates within same director
    transactions = [list(set(genres)) for genres in transactions if len(genres) > 1]

    if not transactions:
        _cached_rules = []
        return []

    # Encode transactions as a binary matrix
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    # Run Apriori to find frequent itemsets
    try:
        frequent_itemsets = apriori(te_df, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            # Relax support if no patterns found
            frequent_itemsets = apriori(te_df, min_support=0.03, use_colnames=True)
    except Exception:
        _cached_rules = []
        return []

    if frequent_itemsets.empty:
        _cached_rules = []
        return []

    # Generate association rules
    try:
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
        rules = rules[rules['lift'] >= min_lift]
    except Exception:
        _cached_rules = []
        return []

    if rules.empty:
        _cached_rules = []
        return []

    # Sort by lift descending, take top 15
    rules = rules.sort_values('lift', ascending=False).head(15)

    result = []
    for _, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        result.append({
            'antecedent': ', '.join(antecedents),
            'consequent': ', '.join(consequents),
            'support': round(float(row['support']), 4),
            'confidence': round(float(row['confidence']), 4),
            'lift': round(float(row['lift']), 4),
            'high_lift': float(row['lift']) > 1.5
        })

    _cached_rules = result
    return result
