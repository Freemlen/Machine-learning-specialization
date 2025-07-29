import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

# Synthetic dataset generation
np.random.seed(42)
N = 200
is_ca = np.random.binomial(1, 0.5, size=N)
frmt_id = np.random.choice([1,2], size=N)
region_id = np.random.choice([1,2], size=N)
age = np.random.normal(35, 10, size=N)
spend_prev = np.random.normal(1000, 300, size=N)
avg_txn_prev = np.random.normal(200, 50, size=N)
cnt_trn_prev = np.random.randint(1, 10, size=N)
# treatment effect
spend_actn = spend_prev + is_ca*50 + np.random.normal(0, 100, size=N)

# Build dataframe
cols = ['age','spend_prev','avg_txn_prev','cnt_trn_prev']
df = pd.DataFrame({
    'is_ca': is_ca,
    'frmt_id': frmt_id,
    'region_id': region_id,
    'age': age,
    'spend_prev': spend_prev,
    'avg_txn_prev': avg_txn_prev,
    'cnt_trn_prev': cnt_trn_prev,
    'spend_actn': spend_actn
})

# Propensity score model
TREATMENT = 'is_ca'
pipe = Pipeline([
    ('minmax', MinMaxScaler()),
    ('scaler', StandardScaler()),
    ('logistic_classifier', LogisticRegression(random_state=123))
])
pipe.fit(df[cols], df[TREATMENT])

df['proba'] = pipe.predict_proba(df[cols])[:,1]
# avoid values of 1
ind = df[df.proba == 1].index
if len(ind) > 0:
    df.loc[ind, 'proba'] = 0.9999
df['logit'] = np.log(df['proba']/(1-df['proba']))

# Matching
caliper = np.std(df.proba) * 0.25
knn = NearestNeighbors(n_neighbors=5)
knn.fit(df[['logit', 'spend_prev','avg_txn_prev','cnt_trn_prev']])
indexes = knn.kneighbors(df[['logit','spend_prev','avg_txn_prev','cnt_trn_prev']], return_distance=False)

# function
import numpy as np
def perform_matching(indexes, is_ca_array, frmt_id_array, region_id_array):
    n = len(is_ca_array)
    matched_element = np.full(n, np.nan, dtype=np.float32)
    used_kg_mask = np.zeros(n, dtype=bool)
    treated_indices = np.where(is_ca_array == 1)[0]
    for current_index in treated_indices:
        candidates = indexes[current_index,:]
        mask_not_self = (candidates != current_index)
        mask_ca = (is_ca_array[candidates] == 0)
        mask_frmt = (frmt_id_array[candidates] == frmt_id_array[current_index])
        mask_region = (region_id_array[candidates] == region_id_array[current_index])
        mask_unused = ~used_kg_mask[candidates]
        valid_mask = mask_not_self & mask_ca & mask_frmt & mask_region & mask_unused
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            chosen_idx = candidates[valid_indices[0]]
            matched_element[current_index] = chosen_idx
            used_kg_mask[chosen_idx] = True
    return matched_element

is_ca_array = df['is_ca'].values
frmt_id_array = df['frmt_id'].values
region_id_array = df['region_id'].values
matched_element_array = perform_matching(indexes, is_ca_array, frmt_id_array, region_id_array)
df['matched_element'] = matched_element_array

matched = df.dropna(subset=['matched_element']).copy()
matched['matched_element'] = matched['matched_element'].astype(int)
untreated = df.loc[matched['matched_element'], :].reset_index(drop=True)
matched = matched.reset_index(drop=True)

print('Matched pairs:', len(matched))
if len(matched) > 0:
    diff = matched['spend_actn'].values - untreated['spend_actn'].values
    print('Average treatment effect:', diff.mean())
else:
    print('No matches found')
