"""
Script for extracting distractors
"""
import re
import torch
import panphon.distance
import torchtext.vocab
import numpy as np
import pandas as pd
import eng_to_ipa as ipa
from pyphonetics import RefinedSoundex
from torchtext.vocab import GloVe


embeddings = GloVe()
def phonDist(w1,w2,weighted=False):
    rs = RefinedSoundex()
    return rs.distance(w1,w2)

def get_vector(embeddings, word):
    clean_word = re.sub(r"[-|']","",word)
    if clean_word in embeddings.stoi:
        return embeddings.vectors[embeddings.stoi[clean_word]]
    else:
        return embeddings.vectors['unk']

def semDist(w1,w2):
    x = embeddings[re.sub(r"['|-]","",w1)].unsqueeze(0)
    y = embeddings[re.sub(r"['|-]","",w2)].unsqueeze(0)
    cosSim = torch.cosine_similarity(x,y)
    return 1-cosSim


def main():
    vocab = pd.read_csv("vocab.csv")
    vocab = vocab[~vocab["word"].str.contains(" ")]


def main():
    df = pd.read_csv("SWBD_substitutions.csv")
    # clean
    df = df[~df["repair"].str.contains(" ")]
    df = df[~df["repair"].str.contains("#NAME")]
    corrections = df[["reprn","repair"]]
    corrections = corrections.drop_duplicates()
    repairs = np.unique(df["repair"].values)
    M = corrections.shape[0]
    corrections["cID"] = np.arange(M) + 1
    repairs = np.unique(df["repair"].values)
    target_df = pd.DataFrame(data={'tID':np.arange(len(repairs))+1,'repair':repairs})
    merged = pd.merge(corrections,target_df,on='repair') 
    vocab_words = np.unique(vocab["word"].values)
    # select distractors
    tIDs =[]
    cIDs = []
    targets = []
    truDists = []
    dists = []
    SemD_scores = []
    PhonD_scores = []
    for idx, row in merged.iterrows():
        print(idx)
        cID = row["cID"]
        tID = row["tID"]
        repair = row["repair"]
        reprn = row["reprn"]
        words = []
        SD_scores = []
        PD_scores = []
        for i,word in enumerate(vocab_words):
            if word != repair:
                SDscore = semDist(repair,word)
                PDscore = phonDist(repair,word)
                words.append(word)
                SD_scores.append(SDscore.item())
                PD_scores.append(PDscore)
        # add reprn
        if reprn not in words:
            words.append(reprn)
            SD_scores.append(semDist(repair,reprn).item())
            PD_scores.append(phonDist(repair,reprn))
        # to array
        words = np.array(words)
        SD_scores = np.array(SD_scores)
        PD_scores = np.array(PD_scores)
        SD_asc_idx = np.argsort(SD_scores)[:50]
        PD_asc_idx = np.argsort(PD_scores)[:50]
        dist_idx = np.array(np.argwhere(words == reprn)[0])
        idx = np.unique(np.concatenate((SD_asc_idx,PD_asc_idx,dist_idx)))
        keep_words = words[idx]
        keep_SD_scores = SD_scores[idx]
        keep_PD_scores = PD_scores[idx]
        M = len(keep_words)
        for j in range(M):
            tIDs.append(tID)
            cIDs.append(cID)
            targets.append(repair)
            truDists.append(reprn)
            dists.append(keep_words[j])
            SemD_scores.append(keep_SD_scores[j].item())
            PhonD_scores.append(keep_PD_scores[j]


    alldists = pd.DataFrame(data={'tID':tIDs,'rID':cIDs,'target':targets,'dist':dists,'trueDist':truDists,'semREL_dist':SemD_scores,'phonREL_dist':PhonD_scores})
    alldists.to_csv("allDists.csv",index=False)


if __name__ == "__main__":
    main()



