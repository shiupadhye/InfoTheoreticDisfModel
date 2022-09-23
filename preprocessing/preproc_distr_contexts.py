"""
Script for preparing causal and masked modeling inputs
"""

import re
import numpy as np
import pandas as pd


# prepare left, right, and bidirectional contexts
alldists = pd.read_csv("allDists.csv")
df = pd.read_csv("SWBD_substitutions.csv")
df = pd.merge(df,alldists,on=["repair","reprn"])
N = df.shape[0]
df["uniqueID"] = np.arange(N) + 1
keep_left_IDs = []
left_contexts = []
keep_right_IDs = []
right_contexts = []
for idx, row in df.iterrows():
    masked_context = row["masked_context"]
    tokens = masked_context.split()
    mask_idx = tokens.index("<mask>")
    left_tokens = tokens[:mask_idx]
    if len(left_tokens) >= 1:
        keep_left_IDs.append(row["uniqueID"])
        left_contexts.append(" ".join(left_tokens))
    right_tokens = tokens[mask_idx+1:]
    if len(right_tokens) >= 1:
        keep_right_IDs.append(row["uniqueID"])
        right_contexts.append(" ".join(right_tokens[::-1])) 


# combine
lc = pd.DataFrame(data={'uniqueID':keep_left_IDs,'left_context':left_contexts})
df = pd.merge(df,lc,on='uniqueID')
rc = pd.DataFrame(data={'uniqueID':keep_right_IDs,'right_context':right_contexts})
df = pd.merge(df,rc,on='uniqueID')
masked = df[["uniqueID","rID","tID","masked_context","repair","reprn","cReprn","phonREL_dist","semREL_dist"]]
left = df[["uniqueID","rID","tID","left_context","repair","reprn","cReprn",
           "phonREL_dist","semREL_dist"]]
right = df[["uniqueID","rID","tID","right_context","repair","reprn","cReprn",
           "phonREL_dist","semREL_dist"]]

# save to file
masked.to_csv("SWBD_allDists_masked_contexts.csv",index=False)
left.to_csv("SWBD_allDists_left_contexts.csv",index=False)
right.to_csv("SWBD_allDists_right_contexts.csv",index=False)




