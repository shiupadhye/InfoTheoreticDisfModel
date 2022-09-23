import torch
import numpy as np
import pandas as pd
from transformers import XLNetTokenizer, XLNetLMHeadModel

tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")
model = XLNetLMHeadModel.from_pretrained("xlnet-large-cased")

def predict_masked(text,candidate):
    # Given an input text, find the masked probability of a candidate 
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False)).unsqueeze(0).to("cuda:0")
    predict_token = torch.tensor(tokenizer.encode(candidate,add_special_tokens = False))
    mask_idx = tokenizer.encode("<mask>",add_special_tokens=False)[0]
    mask_pos = (input_ids == mask_idx).nonzero()[0][-1].item()
    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float).to("cuda:0")
    perm_mask[:, :, mask_pos] = 1.0 
    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float).to("cuda:0")
    target_mapping[0, 0, mask_pos] = 1.0
    model.to("cuda:0")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    predict_logits = outputs[0]
    pToken = torch.sum(torch.nn.functional.softmax(predict_logits[0,0])[predict_token]).item()
    return pToken


def predict_causal(text,candidate):
    # Given an input text, find the transitional probability of a candidate conditioned on the text
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False)).unsqueeze(0).to("cuda:0")
    predict_token = torch.tensor(tokenizer.encode(candidate,add_special_tokens = False)).unsqueeze(0).to("cuda:0")
    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float).to("cuda:0")
    perm_mask[:, :, -1] = 1.0 
    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float).to("cuda:0")
    target_mapping[0, 0, -1] = 1.0
    model.to("cuda:0")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    predict_logits = outputs.logits
    predict_token = predict_token.flatten()
    pToken = torch.sum(torch.nn.functional.softmax(predict_logits[0,0])[predict_token]).item()
    return pToken


def main():
    filename = ""
    df = pd.read_csv(filename)
    reprn_probs = []
    repair_probs = []
    disfRepair_probs = []
    context_type = "Fw"
    for idx, row in df.iterrows():
        # causal
        uttr = "transcript: " + row["context"] + " " + "<mask>"
        # masked
        # uttr = "transcript: " + row["context"]
        reprn = row["reprn"]
        repair = row["repair"]
        pReprn = predict_causal(uttr,reprn)
        pRepair = predict_causal(uttr,repair)
        reprn_probs.append(pReprn)
        repair_probs.append(pRepair)

    reprn_str = "pReprn%s" % context_type
    repair_str = "pRepair%s" % context_type
    df[reprn_str] = reprn_probs
    df[repair_str] = repair_probs

    # output to file
    outfile = "SWBD_%s_scores" % context_type
    df.to_csv(outfile, index=False)

if __name__ == "__main__":
    main()
