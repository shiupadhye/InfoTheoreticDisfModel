"""
Script for extracting utterances with lexical substitutions from Switchboard
"""
import re
import numpy as np
import pandas as pd


def correct_contractions(w):
    find = r"[a-z]+ [a-z]*'[a-z]+"
    if re.findall(find,w):
        return "".join(w.split())
    else:
        return w

def correct_abbr(w):
    find = r"^([a-z]{1}\s)+[a-z]$"
    if re.findall(find,w):
        return "".join(w.split()).upper()
    else:
        return w
    
def preproc_word(w):
    w = correct_contractions(w)
    w = correct_abbr(w)
    return w

def split_into_chunks(x):
    split_idx = np.where(np.diff(x) > 1)[0]
    return np.array_split(x,split_idx+1)



def main():
    df = pd.read_csv("SWBD_data.csv")
    df.head()
    # remove filled pauses
    df = df[df["POS"] != "UH"]
    # extract utterances
    uttrIDs = []
    orig_uttrs = []
    ann_uttrs = []
    repair_IDs = []
    reprn_IDs = []
    repair_words = []
    reprn_words = []
    reprn_POS = []
    repair_POS = []
    type_subs = []
    ctr = 0
    uID = np.unique(df["uttrID"].values)
    for idx in uID:
        df_uttr = df[df["uttrID"] == idx]
        M,N = df_uttr.shape
        fluency = df_uttr["fluency"].values
        words = df_uttr["word"].values
        POS = df_uttr["POS"].values
        reprn_idx = np.array([i for i,f in enumerate(fluency) if f == "reparandum"])
        repair_idx = np.array([i for i,f in enumerate(fluency) if f == "repair"])
        # immediate repairs
        if len(reprn_idx) == len(repair_idx) and len(reprn_idx) > 0:
            D = repair_idx - reprn_idx
            singles_idx = np.where(D == 1)[0]
            phrase_idx = np.where(D > 1)[0]
            if len(singles_idx) > 0:
                for s in singles_idx:
                    reprn_ID = reprn_idx[s]
                    repair_ID = repair_idx[s]
                    reprn_word = words[reprn_ID]
                    repair_word = words[repair_ID]
                    reprn_pos = POS[reprn_ID]
                    repair_pos = POS[repair_ID]
                    if reprn_word != repair_word and "-" not in reprn_word:
                        if reprn_ID > 0 and repair_ID < len(words) - 1:
                            orig_uttr = " ".join([preproc_word(w) for w in words])
                            ann_uttr = ""
                            for j,w in enumerate(words):
                                if fluency[j] == "fluent" and words[j-1] != words[j]:
                                    ann_uttr += preproc_word(w) + " "
                                elif fluency[j] == "reparandum" and j == reprn_ID:
                                    ann_uttr += "[reprn]" + " "
                                elif fluency[j] == "repair" and j == repair_ID:
                                    ann_uttr += "[repair]" + " "
                                elif fluency[j] == "repair" and j != repair_ID:
                                    ann_uttr += preproc_word(w) + " " 
                            uttrIDs.append(idx)
                            orig_uttrs.append(orig_uttr)
                            ann_uttrs.append(ann_uttr)
                            reprn_IDs.append(reprn_ID)
                            repair_IDs.append(repair_ID)
                            reprn_POS.append(reprn_pos)
                            repair_POS.append(repair_pos)
                            reprn_words.append(preproc_word(reprn_word))
                            repair_words.append(preproc_word(repair_word)) 
                            type_subs.append("single")
            # substitutions are within repeated material
            if len(phrase_idx) > 0:
                sReprn_idx = reprn_idx[phrase_idx]
                sRepair_idx = repair_idx[phrase_idx]
                reprn_chunks = split_into_chunks(sReprn_idx)
                repair_chunks = split_into_chunks(sRepair_idx)
                if len(repair_chunks) == len(reprn_chunks):
                    N = len(repair_chunks)
                    for k in range(N):
                        reprn_chunk = reprn_chunks[k]
                        repair_chunk = repair_chunks[k]
                        reprn_phrase = " ".join(words[reprn_chunk])
                        repair_phrase = " ".join(words[repair_chunk])
                        if reprn_phrase != repair_phrase and len(reprn_chunk) == len(repair_chunk):
                            M = len(reprn_chunk)
                            subs_idx = []
                            for m in range(M):
                                if words[reprn_chunk[m]] != words[repair_chunk[m]]:
                                    subs_idx.append(m)
                            if len(subs_idx) == 1:
                                s = subs_idx[0]
                                reprn_ID = reprn_chunk[s]
                                repair_ID = repair_chunk[s]
                                reprn_word = words[reprn_ID]
                                repair_word = words[repair_ID]
                                reprn_pos = POS[reprn_ID]
                                repair_pos = POS[repair_ID]
                                if "-" not in reprn_word:
                                    if reprn_ID > 0 and repair_ID < len(words) - 1:
                                        orig_uttr = " ".join([preproc_word(w) for w in words])
                                        ann_uttr = ""
                                        for j,w in enumerate(words):
                                            if fluency[j] == "fluent" and words[j-1] != words[j]:
                                                ann_uttr += preproc_word(w) + " "
                                            elif fluency[j] == "repair" and j == repair_ID:
                                                ann_uttr += "[repair]" + " " 
                                            elif fluency[j] == "repair" and j != repair_ID:
                                                ann_uttr += preproc_word(w) + " "  
                                        uttrIDs.append(idx)
                                        orig_uttrs.append(orig_uttr)
                                        ann_uttrs.append(ann_uttr)
                                        reprn_IDs.append(reprn_ID)
                                        repair_IDs.append(repair_ID)
                                        reprn_POS.append(reprn_pos)
                                        repair_POS.append(repair_pos)
                                        reprn_words.append(preproc_word(reprn_word))
                                        repair_words.append(preproc_word(repair_word)) 
                                        type_subs.append("within-phrase")
             

    all_subs = pd.DataFrame(data={'uttrID':uttrIDs,'orig_uttr':orig_uttrs,'ann_uttr':ann_uttrs,'reprn':reprn_words,'repair':repair_words,'reprn_POS':reprn_POS,'repair_POS':repair_POS,'reprn_id':reprn_IDs,'repair_id':repair_IDs,"type_sub":type_subs})
    uIDs = np.unique(all_subs["uttrID"].values)
    masked_contexts = []
    for idx, row in all_subs.iterrows():
        ann_uttr = row["ann_uttr"]
        rm_reprn = re.sub(r"\[reprn\]","",ann_uttr)
        masked_context = re.sub(r"\[repair\]","<mask>",rm_reprn)
        masked_contexts.append(masked_context)
        all_subs["masked_context"] = masked_contexts
    all_subs.to_csv("SWBD_substitutions.csv",index=False)

if __name__ == "__main__":
    main()




