from graphextractor import *

import os
import pickle as pkl
import sys
import pandas as pd
import numpy as np
# from run_AST import *

# SETUP
NUM_JOBS = 1# 1
JOB_ARRAY_NUMBER = 0 #if "ipykernel" in sys.argv[0] else int(sys.argv[1]) - 1

# Read Data
df = bigvul()
# df = df[df['vul'] == 1] # this was to check whether they are funnction with enough change so that we can keep certain processing defined by linevd

df = df.iloc[::-1]
#df = df.transpose
splits = np.array_split(df, NUM_JOBS)


def preprocess(row):
    """Parallelise svdj functions.

    Example:
    df = svdd.bigvul()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    savedir_before = get_dir(processed_dir() / row["dataset"] / "before")
    savedir_after = get_dir(processed_dir() / row["dataset"] / "after")
    
    # Add the directory where to save code descriptions into txt file
    savedir_description_CVE = get_dir(processed_dir() / row['dataset'] / "CVEdescription")
    savedir_description_CWE = get_dir(processed_dir() / row['dataset'] / "CWEdescription")
    savedir_sample_func = get_dir(processed_dir() / row['dataset'] / "CWE_Samples")

    # Write C Files
    fpath1 = savedir_before / f"{row['id']}.c"
    with open(fpath1, "w") as f:
        f.write(row["before"])
    fpath2 = savedir_after / f"{row['id']}.c"
    if len(row["diff"]) > 0:
        with open(fpath2, "w") as f:
            f.write(row["after"])
            
    # add code to write vulnerability descriptions that will be used as domain informations       
    fpath3 = savedir_description_CVE / f"{row['id']}.txt"
    if not os.path.exists(f"{fpath3}.txt") and len(row['CVE_vuldescription']) > 5:
        with open(fpath3, 'w') as f:
            f.write(row['CVE_vuldescription'])       
    # add code to write description from the Mitre 1000 of the CWE
    fpath4 = savedir_description_CWE / f"{row['id']}.txt"
    if not os.path.exists(f"{fpath4}.txt") and len(row['CWE_vuldescription'])>4:
        with open(fpath4, 'w') as f:
            f.write(row['CWE_vuldescription'])
    # add the code to get sample from the mitre save as text file as well
    fpath5 = savedir_sample_func / f"{row['id']}.txt"
    if not os.path.exists(f"{fpath5}.txt") and len(row["CWE_Sample"])> 5:
        with open(fpath5, 'w') as f:
            f.write(row['CWE_Sample'])
    # Run Joern on "before" code
    if not os.path.exists(f"{fpath1}.edges.json"):
        full_run_joern(fpath1, verbose=3)

    # Run Joern on "after" code
    if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
        full_run_joern(fpath2, verbose=3)
    
        

if __name__ == "__main__":
    dfmp(splits[JOB_ARRAY_NUMBER], preprocess, ordr=False, workers=8)
