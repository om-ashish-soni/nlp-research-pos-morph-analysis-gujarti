import torch.nn as nn
import numpy as np
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
import streamlit as st
from random import randint
import tempfile
import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import pandas as pd
import traceback

def main():
    st.set_page_config(
        page_title="NLP Gujarati POS Tagging & Morph Analyzer",
        page_icon="✨",   
    )
    
if __name__ == "__main__":
    main()

NA='NA'
MAX_LENGTH=120
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_checkpoint="l3cube-pune/gujarati-bert"


all_feature_values={
    'pos':[
      NA,'DM_DMI', 'CC_CCD', 'PSP', 'PR_PRQ', 'RP_RPD', 'PR_PRP', 'DM_DMQ', 'RB', 'QT_QTO', 'JJ', 'RD_ECH', 'PR_PRF', 'N_NNP', 'N_NN', 'RP_CL', 'V_VM', 'DM_DMD', 'RP_INTF', 'QT_QTC', 'RP_INJ', 'PR_PRC', 'V_VAUX_VNP', 'RD_PUNC', 'PR_PRI', 'PR_PRL', 'DM_DMR', 'CC_CCS_UT', 'RD_RDF', 'N_NST', 'RP_NEG', 'RD_SYM', 'V_VAUX', 'QT_QTF', 'CC_CCS', 'Value'
    ],
    'gender':[
        NA,'MASC','FEM','NEUT'
    ],
    'number':[
        NA,'SG','PL'
    ],
    'type':[
        NA,'LGSPEC02','LGSPEC01','LGSPEC03'
    ],
    'person':[
        NA,'1','2','3'
    ],
    'tense':[
        NA,'PST','FUT'
        # is present tense is not there?
    ],
    'case':[
        NA,'ERG', 'GEN', 'NOM', 'DAT', 'LOC','ABL'
    ],
    'aspect':[
        NA,'NFIN'
    ],
    # what is NFIN
}

feature_values={
    'pos':[
      NA,'DM_DMI', 'CC_CCD', 'PSP', 'PR_PRQ', 'RP_RPD', 'PR_PRP', 'DM_DMQ', 'RB', 'QT_QTO', 'JJ', 'RD_ECH', 'PR_PRF', 'N_NNP', 'N_NN', 'RP_CL', 'V_VM', 'DM_DMD', 'RP_INTF', 'QT_QTC', 'RP_INJ', 'PR_PRC', 'V_VAUX_VNP', 'RD_PUNC', 'PR_PRI', 'PR_PRL', 'DM_DMR', 'CC_CCS_UT', 'RD_RDF', 'N_NST', 'RP_NEG', 'RD_SYM', 'V_VAUX', 'QT_QTF', 'CC_CCS', 'Value'
    ],
    'gender':[
        NA,'MASC','FEM','NEUT'
    ],
    'number':[
        NA,'SG','PL'
    ],
    'type':[
        NA,'LGSPEC02','LGSPEC01','LGSPEC03'
    ],
    'person':[
        NA,'1','2','3'
    ],
    'tense':[
        NA,'PST','FUT'
        # is present tense is not there?
    ],
    'case':[
        NA,'ERG', 'GEN', 'NOM', 'DAT', 'LOC','ABL'
    ],
    'aspect':[
        NA,'NFIN'
    ],
    # what is NFIN
}

feature_meanings = {
    'NA': 'Not Available',
    'DM_DMI': 'Demonstrative',
    'CC_CCD': 'Coordinating Conjunction',
    'PSP': 'Postposition',
    'PR_PRQ': 'Pronoun - Interrogative',
    'RP_RPD': 'Particle',
    'PR_PRP': 'Personal Pronoun',
    'DM_DMQ': 'Demonstrative - Interrogative',
    'RB': 'Adverb',
    'QT_QTO': 'Quantifier',
    'JJ': 'Adjective',
    'RD_ECH': 'Echo Word',
    'PR_PRF': 'Pronoun - Reflexive',
    'N_NNP': 'Noun - Proper Singular',
    'N_NN': 'Noun - Common',
    'RP_CL': 'Clitic',
    'V_VM': 'Verb - Main',
    'DM_DMD': 'Demonstrative - Determiner',
    'RP_INTF': 'Intensifier',
    'QT_QTC': 'Cardinal Numeral',
    'RP_INJ': 'Interjection',
    'PR_PRC': 'Pronoun - Relative',
    'V_VAUX_VNP': 'Verb - Auxiliary (Negative Polarity)',
    'RD_PUNC': 'Punctuation',
    'PR_PRI': 'Pronoun - Indefinite',
    'PR_PRL': 'Pronoun - Relative Locative',
    'DM_DMR': 'Demonstrative - Relative',
    'CC_CCS_UT': 'Coordinating Conjunction - Subordinating',
    'RD_RDF': 'Reduplicator',
    'N_NST': 'Noun - Honorific',
    'RP_NEG': 'Negation',
    'RD_SYM': 'Symbol',
    'V_VAUX': 'Verb - Auxiliary',
    'QT_QTF': 'Quantifier - Fraction',
    'CC_CCS': 'Coordinating Conjunction - Subordinating',
    'Value': 'Value',
    'MASC': 'Masculine',
    'FEM': 'Feminine',
    'NEUT': 'Neutral',
    'SG': 'Singular',
    'PL': 'Plural',
    'LGSPEC02': 'Language Specific 02',
    'LGSPEC01': 'Language Specific 01',
    'LGSPEC03': 'Language Specific 03',
    '1': 'First Person',
    '2': 'Second Person',
    '3': 'Third Person',
    'PST': 'Past Tense',
    'FUT': 'Future Tense',
    'ERG': 'Ergative',
    'GEN': 'Genitive',
    'NOM': 'Nominative',
    'DAT': 'Dative',
    'LOC': 'Locative',
    'ABL': 'Ablative',
    'NFIN': 'Non-Finite'
}

BOOTSTRAP_COLORS = [
    "#007BFF",
    "#6C757D",
    "#28A745",
    "#DC3545",
    "#FFC107",
    "#17A2B8",
    "#343A40",
]

feature_seq=list(feature_values.keys())
EXTRA_TOKEN=[-100]*len(feature_seq)

total_number_of_features=0
feature_value2id={}
feature_id2value={}
feature_start_range={}

start_range=0
for key,values in feature_values.items():
  feature_value2id[key]={}
  feature_start_range[key]=start_range
  for i,value in enumerate(values):
    feature_value2id[key][value]=i+start_range
  feature_id2value[key]={(y-start_range):x for x,y in feature_value2id[key].items()}
  start_range+=len(values)
  total_number_of_features+=len(values)


number_of_labels=total_number_of_features

class CustomTokenClassificationModel(nn.Module):
    def __init__(self, bert_model, feature_seq):
        super(CustomTokenClassificationModel, self).__init__()
        self.bert_model = bert_model

        self.module_list = nn.ModuleList()

        for key in feature_seq:
          num_classes = len(feature_values[key])
          module = nn.Linear(number_of_labels, num_classes)
          self.module_list.append(module)

    def forward(self, input_ids, attention_mask):
        sequence_output = self.bert_model(
              input_ids,
              attention_mask=attention_mask
        )

        # Initialize an empty list to store the logits for each attribute
        logits_list = []

        # Pass the logits through each linear layer
        for module in self.module_list:
            attribute_logits = module(sequence_output.logits)
            logits_list.append(attribute_logits)

        return logits_list

class PosMorphAnalysisModelWrapper:
    def __init__(self, tokenizer, inference_model, feature_seq, feature_id2value, max_length,NA):
        self.tokenizer = tokenizer
        self.inference_model = inference_model
        self.feature_seq = feature_seq
        self.feature_id2value = feature_id2value
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NA = NA

    def prepare_mask(self, word_ids):
        mask = []
        last = None
        for i in word_ids:
            if i is None or i == last:
                mask.append(0)
            else:
                mask.append(1)
            last = i
        return mask

    def tokenize_sentence(self, sentence, splitted=False):
        if not splitted:
            tokens = sentence.split(' ')
        else:
            tokens = sentence

        tokenized_inputs = self.tokenizer(
            tokens,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
        )
        mask = self.prepare_mask(tokenized_inputs.word_ids(0))
        sample = {
            "tokens": tokens,
            "mask": mask,
            "input_ids": tokenized_inputs['input_ids'],
            "attention_mask": tokenized_inputs['attention_mask']
        }
        return sample

    def prepare_output(self, sample):
        try:
            tokens = sample['tokens']
            output = []

            for i, token in enumerate(tokens):
                features = {}
                for feat in self.feature_seq:
                    if(i>=len(sample[feat])):
                        continue
                    feat_val = sample[feat][i]
                    print(feat,feat_val)
                    if feat_val != self.NA:
                        features[feat] = feat_val
                # print("features",features)
                output.append((token, features))
            print(output)
            return output
        except Exception as e:
            print("Error while prepare output ", str(e))
            traceback.print_exc()

    def get_value(self,key):
        if key not in self.feature_id2value:
            return 'NOT FOUND'
        return self.feature_id2value[key]
        pass

    def infer(self, sentence):
        batch = self.tokenize_sentence(sentence)

        input_ids = torch.tensor([batch["input_ids"]]).to(self.device)
        attention_mask = torch.tensor([batch["attention_mask"]]).to(self.device)
        mask = torch.tensor([batch['mask']]).to(self.device)

        logits_list = self.inference_model(input_ids, attention_mask=attention_mask)

        print("sentence",sentence)

        curr_sample = {
            "tokens": batch["tokens"],
        }

        for i, logits in enumerate(logits_list):
            key = self.feature_seq[i]
            curr_mask = (mask != 0)
            valid_logits = logits[curr_mask]
            probabilities = F.softmax(valid_logits, dim=-1)
            valid_predicted_labels = torch.argmax(probabilities, dim=-1)
            
            print("key",key)
            print("valid_predicted_labels : ",valid_predicted_labels.tolist())
            curr_id2value_map = self.get_value(key)
            curr_sample[key] = [curr_id2value_map[x] for x in valid_predicted_labels.tolist()]
            for x in valid_predicted_labels.tolist():
                print("x",curr_id2value_map[x])


        print("curr_sample",curr_sample)
        output = self.prepare_output(curr_sample)
        return output

def get_dir_path():
    temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

@st.cache_resource
def download_file():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    repo_file_name="HfApiUploaded_GUJ_SPLIT_POS_MORPH_ANAYLISIS-v6.0-model.pth"
    temp_dir=get_dir_path()
    model_filepath=os.path.join(temp_dir,repo_file_name)
    hf_hub_download(
        repo_id="om-ashish-soni/research-pos-morph-gujarati-6.0",
        filename=repo_file_name,
        local_dir = temp_dir,
        token=HF_TOKEN
    )
    return model_filepath

@st.cache_resource
def download_file_optimistic(repo_id,repo_file_name):
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    temp_dir=get_dir_path()
    model_filepath=os.path.join(temp_dir,repo_file_name)
    if os.path.exists(model_filepath):
        pass
    else:
        hf_hub_download(
            repo_id=repo_id,
            filename=repo_file_name,
            local_dir = temp_dir,
            token=HF_TOKEN
        )
    return model_filepath

@st.cache_resource
def load_tokenizer():
  return AutoTokenizer.from_pretrained(model_checkpoint)

@st.cache_resource
def load_inference_model(inference_checkpoint_path):
    inference_model=torch.load(inference_checkpoint_path,map_location=device)
    inference_model.eval()
    inference_model.to(device)
    return inference_model

@st.cache_resource
def load_inference_wrapper_model(_tokenizer,_inference_model):
  return PosMorphAnalysisModelWrapper(_tokenizer, _inference_model, feature_seq, feature_id2value, MAX_LENGTH,NA)

def get_badge_color(index):
   return BOOTSTRAP_COLORS[index%len(BOOTSTRAP_COLORS)]
   
def display_word_features(word_features):
    df = pd.DataFrame()
    try:
        output=[]
        for word,features in word_features:
            modified_features={}
            for feature_key in features:
                feature_value=str(features[feature_key]) if feature_key=='pos' else str(feature_key)+' : '+str(features[feature_key])
                
                modified_feature_key=feature_key if feature_key == 'pos' else ''
                if modified_feature_key in modified_features:
                    modified_features[modified_feature_key]+='<br/>'+feature_value
                else:
                    modified_features[modified_feature_key]=feature_value
            pass
            output.append((word,modified_features))
        
        # Extracting all unique feature keys
        feature_keys = set()
        for _, features in output:
            feature_keys.update(features.keys())

        # Creating an empty DataFrame with columns as words
        df = pd.DataFrame(columns=['Feature']+[word for word, _ in output])

        # Populating the DataFrame with feature values
        feature_keys=['pos','']
        for feature_key in feature_keys:
            feature_values = [feature_key]
            for _, features in output:
                feature_values.append(features.get(feature_key, ""))
            
            # df_key=feature_key if feature_key == 'pos' else 'morph'
            df.loc[feature_key] = feature_values
        
        print(df.columns)

        # Convert DataFrame to HTML table
        df_html = df.to_html(index=False, escape=False)

        # Display the HTML table
        st.write(df_html, unsafe_allow_html=True)

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()
    # Displaying the DataFrame
    
    

def generate_single_badge(content,color=None):
    if color == None:
        color=get_badge_color(randint(0,100))
    text_color = "#ffffff" if is_dark_color(color) else "#000000"  # Adjust text color based on background
    return (
        f'<span style="background-color: {color}; color: {text_color}; '
        f'padding: 5px; margin-right: 5px; border-radius: 5px;">{content}</span>'
    )

def generate_badges(features):
    badges = ""
    
    for feature, value in features.items():
        badges+=generate_single_badge(f'{feature}: {value}')
        
    return badges

def is_dark_color(color):
    r, g, b = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness < 128




inference_checkpoint_path=download_file_optimistic("om-ashish-soni/research-pos-morph-gujarati-6.0","HfApiUploaded_GUJ_SPLIT_POS_MORPH_ANAYLISIS-v6.0-model.pth")
# print(inference_checkpoint_path)

# input()
tokenizer = load_tokenizer()
inference_model=load_inference_model(inference_checkpoint_path)
inference_model_wrapper=load_inference_wrapper_model(tokenizer,inference_model)



st.title("Gujarati POS Tagging & Morph Analyzer")

# Your main app content goes here


# st.markdown(
#         """
#         <style>
            
#             .footer {
#                 bottom:0
#                 background-color: #f8f9fa;
#                 padding: 20px 0;
#                 color: #495057;
#                 text-align: center;
#                 border-top: 1px solid #dee2e6;
#             }
#             .footer a {
#                 color: #007bff;
#                 text-decoration: none;
#             }
#             .footer a:hover {
#                 color: #0056b3;
#                 text-decoration: underline;
#             }
#         </style>
#         <div class="content">
#             <!-- Your main app content goes here -->
#         </div>
#         <div class="footer">
#             <h3 class="mb-0">NLP Gujarati POS Tagging & Morph Analyzer</h3>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

query = st.text_input("Enter the sentence in Gujarati here....")

if st.button('Query'):
    word_features=inference_model_wrapper.infer(query)
    display_word_features(word_features)
    
st.markdown(
        """
        <style>
            
            .footer {
                bottom:0
                background-color: #f8f9fa;
                padding: 20px 0;
                color: #495057;
                text-align: center;
                border-top: 1px solid #dee2e6;
            }
            .footer a {
                color: #007bff;
                text-decoration: none;
            }
            .footer a:hover {
                color: #0056b3;
                text-decoration: underline;
            }
        </style>
        <div class="content">
            <!-- Your main app content goes here -->
        </div>
        <div class="footer">
            <p class="mb-0">Research with ❤️ design & training by Prof. Brijesh Bhatt, Prof. Jatayu Baxi, Om Ashishkumar Soni</p>
        </div>
        """,
        unsafe_allow_html=True,
    )