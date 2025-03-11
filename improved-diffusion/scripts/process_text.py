from mydatasets import get_dataloader,ChEBIdataset
import torch
import transformers
from mytokenizers import SimpleSmilesTokenizer,regexTokenizer
from transformers import AutoModel
from transformers import AutoTokenizer
import argparse
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",required=True)
args = parser.parse_args()
split = args.input
smtokenizer = regexTokenizer()
train_dataset = ChEBIdataset(
        dir='../../datasets/SMILES/',
        smi_tokenizer=smtokenizer,
        split=split,
        replace_desc=False,
        load_state=False
        # pre = pre
    )
model = AutoModel.from_pretrained('../../scibert/scibert_scivocab_uncased')
tokz = AutoTokenizer.from_pretrained('../../scibert/scibert_scivocab_uncased')

volume = {}

if torch.cuda.is_available():
    model = model.cuda()
    # alllen = []
model.eval()
with torch.no_grad():
    for i in trange(len(train_dataset)):
        id = train_dataset[i]['cid']
        desc =train_dataset[i]['desc']
        tok_op = tokz(
            desc,max_length=216, truncation=True, padding='max_length'
            )
        toked_desc = torch.tensor(tok_op['input_ids']).unsqueeze(0)
        toked_desc_attentionmask = torch.tensor(tok_op['attention_mask']).unsqueeze(0)
        assert(toked_desc.shape[1] == 216)
        if torch.cuda.is_available():
            toked_desc = toked_desc.cuda()
        lh = model(toked_desc).last_hidden_state
        volume[id] = {'states':lh.to('cpu'),'mask':toked_desc_attentionmask}



torch.save(volume,'../../datasets/SMILES/'+split+'_desc_states.pt')
