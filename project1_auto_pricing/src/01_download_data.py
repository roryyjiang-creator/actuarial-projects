from sklearn.datasets import fetch_openml

freq = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
sev = fetch_openml(data_id=41215, as_frame=True, parser="auto").frame

freq["IDpol"] = freq["IDpol"].astype(int)
sev["IDpol"] = sev["IDpol"].astype(int)

freq.to_csv("data/freq.csv", index=False)
sev.to_csv("data/sev.csv", index=False)

print(freq.shape, sev.shape)
print(freq.head())