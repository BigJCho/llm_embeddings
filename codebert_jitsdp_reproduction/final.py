import argparse
from comparison_model_in_place import TheMotherload, CommitDataset
import csv
import pandas as pd
from pathlib import Path
import pickle
from scipy.stats import wilcoxon
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

# We read the csv file associated with command line arguments
# We sort the dataframe by loss ascending
# Select the first entry
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['openstack','qt'], help='Dataset to use: openstack or qt')
parser.add_argument('weights', type=float, help='The pos_weight coefficient that each model was trained with')
args = parser.parse_args()
pos_name = int(args.weights*100)

bert_dir = f'codebert{args.dataset}{pos_name}.csv'
llama_dir = f'llama{args.dataset}{pos_name}.csv'

# Get the best bert model according to loss
df_bert = pd.read_csv(bert_dir)
df_bert_sorted = df_bert.sort_values(by='Test_loss', ascending=True)
best_bert = df_bert_sorted.iloc[0]['Model']
best_bert_path = f'{best_bert}.pth'

# Get the best llama model according to loss
df_llama = pd.read_csv(llama_dir)
df_llama_sorted = df_llama.sort_values(by='Test_loss', ascending=True)
best_llama = df_llama_sorted.iloc[0]['Model']
best_llama_path = f'{best_llama}.pth'

print('Models found...')

# Load the test dataset
base_data_dir = Path(__file__).parent.resolve() / 'data/data_and_model/data+model/data/jit'
if args.dataset == 'qt':
    test_path = base_data_dir / 'qt_test.pkl'
else:
    test_path = base_data_dir / 'openstack_test.pkl'
with open(test_path, 'rb') as f:
    test_data = pickle.load(f)

test_commit_id, test_label, test_msg, test_code = test_data

# Generate logits one model at a time

# CodeBERT
# Load everything we need to evaluate
bert_model_name = 'microsoft/codebert-base'
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_embedder = AutoModel.from_pretrained(bert_model_name).to(device)
bert_dataset = CommitDataset(test_msg, test_code, test_label, bert_tokenizer, bert_embedder, device, bert_model_name)
bertloader = DataLoader(bert_dataset, batch_size=64, shuffle=False)
bert_trained_model = TheMotherload(embedding_dim=768, output_channels=64, kernel_sizes=[1, 2 ,3], dropout=0.5, model_name='codebert')
checkpoint = torch.load(best_bert_path)
bert_trained_model.load_state_dict(checkpoint['model_state_dict'])
bert_trained_model = bert_trained_model.to(device)
bert_trained_model.eval()


print('Bert loaded...')

# Eval loop
all_labels, bert_probs, bert_logits = [], [], []
with torch.no_grad():
    for batch_idx, (code_embeddings, msg_embeddings, labels_tensor) in enumerate(bertloader):
        code_embeddings, msg_embeddings, labels = code_embeddings.to(device), msg_embeddings.to(device), labels_tensor.to(device)
        outputs = bert_trained_model(code_embeddings, msg_embeddings)
        probs = torch.sigmoid(outputs)

        # Save the bert logits and the labels
        bert_logits.extend(outputs.squeeze().cpu().numpy()) 
        bert_probs.extend(probs.squeeze().cpu().numpy())   
        true_labels = labels.long()
        all_labels.extend(true_labels.cpu().numpy())

        print(f"Batch [{batch_idx}/{len(bertloader)}]")

df_bert = pd.DataFrame({
    "logit": bert_logits,
    "probability": bert_probs,
    "label": all_labels
})

bert_filename = f"bert_logits_and_probs{args.dataset}.csv"
df_bert.to_csv(bert_filename, index=False)

# Ensure model and memory is dumped
del bert_trained_model
del bertloader
del bert_dataset
del bert_embedder
del bert_tokenizer
del checkpoint
torch.cuda.empty_cache()

print('Bert dropped...')

# Llama
# Load everything we need to evaluate

# This is a gated model, please save your Hugging Face read-only token in token.txt
with open('token.txt', 'r') as file:
    auth_token = file.read().strip()

llama_model_name = 'meta-llama/Llama-3.2-1B'
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_auth_token=auth_token)
llama_embedder = AutoModel.from_pretrained(llama_model_name, use_auth_token=auth_token).to(device)
llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
llama_embedder.resize_token_embeddings(len(llama_tokenizer))
llama_dataset = CommitDataset(test_msg, test_code, test_label, llama_tokenizer, llama_embedder, device, llama_model_name)
llamaloader = DataLoader(llama_dataset, batch_size=64, shuffle=False)
llama_trained_model = TheMotherload(embedding_dim=2048, output_channels=64, kernel_sizes=[1, 2 ,3], dropout=0.5, model_name='llama')
checkpoint = torch.load(best_llama_path)
llama_trained_model.load_state_dict(checkpoint['model_state_dict'])
llama_trained_model = llama_trained_model.to(device)
llama_trained_model.eval()

print('Llama loaded...')

# Eval loop
llama_logits, llama_probs = [], []
with torch.no_grad():
    for batch_idx, (code_embeddings, msg_embeddings, labels_tensor) in enumerate(llamaloader):
        code_embeddings, msg_embeddings, labels = code_embeddings.to(device), msg_embeddings.to(device), labels_tensor.to(device)
        outputs = llama_trained_model(code_embeddings, msg_embeddings)
        probs = torch.sigmoid(outputs)
        # Save the llama logits
        llama_logits.extend(outputs.squeeze().cpu().numpy())
        llama_probs.extend(probs.squeeze().cpu().numpy())   

        print(f"Batch [{batch_idx}/{len(llamaloader)}]")

df_llama = pd.DataFrame({
    "logit": llama_logits,
    "probability": llama_probs,
    "label": all_labels
})

llama_filename = f"llama_logits_and_probs{args.dataset}.csv"
df_llama.to_csv(llama_filename, index=False)

# Ensure model and memory is dumped
del llama_trained_model
del llamaloader
del llama_dataset
del llama_embedder
del llama_tokenizer
torch.cuda.empty_cache()

print('Llama dropped...')
print('Running Wilcoxon...')

# Separate our probabilities based on model and label
llama_0 = [prob for prob, label in zip(llama_probs, all_labels) if label == 0]
llama_1 = [prob for prob, label in zip(llama_probs, all_labels) if label == 1]

bert_0 = [prob for prob, label in zip(bert_probs, all_labels) if label == 0]
bert_1 = [prob for prob, label in zip(bert_probs, all_labels) if label == 1]

# Save all values where models agree on an output
bert_0_agree, llama_0_agree, bert_1_agree, llama_1_agree = [],[],[],[]
for bl0, ll0 in zip(bert_0,llama_0):
    if bl0 < 0.5 and ll0 < 0.5:
        bert_0_agree.append(bl0)
        llama_0_agree.append(ll0)
for bl1, ll1 in zip(bert_1,llama_1):
    if bl1 >= 0.5 and ll1 >= 0.5:
        bert_1_agree.append(bl1)
        llama_1_agree.append(ll1)

# Run Wilcoxon and save the result
# On a given label, we predict that llama would predict with more confidence in 0 or 1. Thus we want to see if Llama > BERT is statistically significant.
stat_0, p_0 = wilcoxon(llama_0_agree, bert_0_agree, alternative='greater')
stat_1, p_1 = wilcoxon(llama_1_agree, bert_1_agree, alternative='greater')

# Write model llama name, model bert name, wilcoxon results into csv
row = [best_bert, best_llama, stat_0, p_0, stat_1, p_1]
results_filename = f'results{args.dataset}.csv'
try:
    with open('results.csv', mode='x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Bert_model','Llama_model','stat_0','p_0','stat_1','p_1'])
except FileExistsError:
    pass
with open('results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(row)

print('Done!')
