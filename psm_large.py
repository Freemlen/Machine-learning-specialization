import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)
N_CA = 100000
N_KG = 500000
N = N_CA + N_KG

is_ca = np.concatenate([
    np.ones(N_CA),
    np.zeros(N_KG)
])
frmt_id = np.random.choice([1,2,3], size=N)
region_id = np.random.choice([1,2,3,4], size=N)
age = np.random.normal(35, 10, size=N)
spend_prev = np.random.normal(1000, 300, size=N)
avg_txn_prev = np.random.normal(200, 50, size=N)
cnt_trn_prev = np.random.randint(1, 10, size=N)
spend_actn = spend_prev + is_ca*50 + np.random.normal(0, 100, size=N)

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

pipe = Pipeline([
    ('minmax', MinMaxScaler()),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=100))
])
pipe.fit(df[cols], df['is_ca'])
df['proba'] = pipe.predict_proba(df[cols])[:,1]
ind = df[df.proba == 1].index
if len(ind) > 0:
    df.loc[ind,'proba'] = 0.9999
df['logit'] = np.log(df.proba/(1 - df.proba))

knn = NearestNeighbors(n_neighbors=5)
knn.fit(df[['logit','spend_prev','avg_txn_prev','cnt_trn_prev']])
idxs = knn.kneighbors(df[['logit','spend_prev','avg_txn_prev','cnt_trn_prev']], return_distance=False)

def perform_match(idxs, is_ca_arr, frmt_arr, reg_arr):
    n = len(is_ca_arr)
    match = np.full(n, np.nan)
    used = np.zeros(n, dtype=bool)
    treated = np.where(is_ca_arr==1)[0]
    for i in treated:
        cand = idxs[i,:]
        mask = (
            (cand != i) &
            (is_ca_arr[cand]==0) &
            (frmt_arr[cand]==frmt_arr[i]) &
            (reg_arr[cand]==reg_arr[i]) &
            (~used[cand])
        )
        if mask.any():
            j = cand[np.where(mask)[0][0]]
            match[i] = j
            used[j] = True
    return match

is_ca_arr = df.is_ca.values
frmt_arr = df.frmt_id.values
reg_arr = df.region_id.values
match_arr = perform_match(idxs,is_ca_arr,frmt_arr,reg_arr)
df['match'] = match_arr
matched = df.dropna(subset=['match']).copy()
matched['match'] = matched['match'].astype(int)
untreat = df.loc[matched['match'],:].reset_index(drop=True)
matched = matched.reset_index(drop=True)

def smd(a,b):
    m1 = np.mean(a)
    m0 = np.mean(b)
    v1 = np.var(a, ddof=1)
    v0 = np.var(b, ddof=1)
    return abs(m1-m0)/np.sqrt((v1+v0)/2)

for c in cols:
    before = smd(df[df.is_ca==1][c], df[df.is_ca==0][c])
    after = smd(matched[c], untreat[c])
    print(f'{c}: before {before:.3f}, after {after:.3f}')

print('Matched pairs:', len(matched))
if len(matched)>0:
    diff = matched.spend_actn.values - untreat.spend_actn.values
    print('ATE:', diff.mean())
