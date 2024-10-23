import pandas as pd    

df = pd.read_csv("/home/New_move/DomainGraph/storage/external/Domain_projectKB_csv_version.csv")

dfv = df[df['vul'] == 1]
dfnv = df[df['vul'] == 0]

n = 250
sdfv = dfv.sample(n, random_state= 0)
sdfnv = dfnv.sample(n, random_state=0)

sdf = pd.concat([sdfv, sdfnv], ignore_index=True)
sdf.to_csv("/home/New_move/DomainGraph/storage/external/sample_Domain_projectKB_csv.csv", index=False)
print(f"The sample data consists of {n} vul function and {n} non-vul funtion")