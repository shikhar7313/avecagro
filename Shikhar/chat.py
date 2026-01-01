import random
import json
import torch
import nltk
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import os
from fuzzywuzzy import process
import string
from transformers import AutoTokenizer, AutoModelForCausalLM
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
import nltk
nltk.download('punkt_tab')
 
# LOAD ALL INTENTS FILES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
directory_path = os.path.join(BASE_DIR, 'intentsbruh')
 
json_files = []
if not os.path.isdir(directory_path):
    raise FileNotFoundError(f"Intents folder not found: {directory_path}")

for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        json_files.append(os.path.join(directory_path, filename))
 
intents = {"intents": []}
intent_sources = {}
 
for json_file in json_files:
    if not os.path.exists(json_file):
        print(f"Warning: {json_file} does not exist.")
        continue
 
    try:
        with open(json_file, 'r') as json_data:
            data = json.load(json_data)
            if 'intents' in data:
                for intent in data['intents']:
                    intents['intents'].append(intent)
                    intent_sources[intent['tag']] = json_file
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
 
# LOAD TRAINED INTENT CLASSIFICATION MODEL
models = []
model_files = [
    os.path.join(BASE_DIR, 'data22.pth')
]
for file in model_files:
    if os.path.exists(file):
        data = torch.load(file)
        print(f"Loaded model from {file}")
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data["all_words"]
        tags = data["tags"]
 
        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(data["model_state"])
        model.eval()
        models.append((model, all_words, tags))
    else:
        print(f"Warning: {file} does not exist.")
 
bot_name = "Avec Agro"
 
# ------------------------------
# REPLACE DialoGPT WITH LLaMA-3
# ------------------------------
model_name = "Qwen/Qwen2.5-1.5B-Instruct"   # BEST

tokenizer = AutoTokenizer.from_pretrained(model_name)
preferred_dtype = torch.float16 if device.type == 'cuda' else torch.float32
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=preferred_dtype
)
llm_model.to(device)
 
# ------------------------------
# FUNCTIONS
# ------------------------------
 
def preprocess_input(msg):
    msg = msg.lower()
    msg = msg.translate(str.maketrans('', '', string.punctuation))
    return msg
 
def generate_fallback_llama3(msg):
    """Generate natural fallback response using Llama-3."""
    prompt = f"You are a friendly assistant. Respond casually.\nUser: {msg}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    output = llm_model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
 
def get_response(msg, models):
    if msg is None:
        return ["I do not understand..."]
 
    responses = []
    clean_msg = preprocess_input(msg)
 
    # ---- INTENT CLASSIFIER ----
    for model, all_words, tags in models:
        sentence = tokenize(clean_msg)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
 
        output = model(X)
        _, predicted = torch.max(output, dim=1)
 
        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
 
        print(f"Predicted tag: {tag}, Probability: {prob.item()}")
 
        if prob.item() > 0.85:
            intent_tags = [intent['tag'] for intent in intents['intents']]
            best_match, score = process.extractOne(tag, intent_tags)
 
            if score >= 55:
                for intent in intents['intents']:
                    if best_match == intent["tag"]:
                        response = random.choice(intent['responses'])
                        responses.append(response)
                        break
            else:
                responses.append("I do not understand...")
        else:
            responses.append("I do not understand...")
 
    # ---- FALLBACK: GENERATIVE MODEL ----
    if not responses or "I do not understand..." in responses:
        fallback_resp = generate_fallback_llama3(msg)
        if "I do not understand..." in responses:
            responses[responses.index("I do not understand...")] = fallback_resp
 
    return responses