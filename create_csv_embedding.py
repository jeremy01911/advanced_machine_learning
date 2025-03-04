from transformers import BertTokenizer, BertModel
from transformers import LongformerTokenizer, LongformerModel
import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

df_balanced = pd.read_csv('./data/train_v2_drcat_02.csv')

def vectoriser_model(text, model):
    if model == "BERT":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model= BertModel.from_pretrained("bert-base-uncased")
    elif model =="longform":
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
    else :
        raise Exception("entrer BERT ou longform")
    
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    output = model(**encoded_input)
    vector_cls = output.pooler_output.squeeze().detach().numpy()
    
    return vector_cls

if __name__ == "__main__":

    #df_balanced['vect'] = df_balanced['text'].progress_apply(vectoriser_model("BERT"))
    #prend en chage que des fonctions à un argument donc on passe par lambda fonction !
    df_balanced['vect'] = df_balanced['text'].progress_apply(lambda x: vectoriser_model(x, "BERT"))

    df_balanced.to_csv("./df_balanced_vectorized.csv")
    


