import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import os
import numpy as np
from scipy.spatial import cKDTree

QWEN_MODEL = "Qwen/Qwen1.5-1.8B-Chat"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DATA_FILE = "laspad_texts.jsonl"
K_RETRIEVED_DOCS = 2

def load_and_preprocess_data(file_path):
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                full_text = f"Titre: {data['title']}. Description: {data['meta']['description']}. Contenu: {data['text']}"
                texts.append(full_text)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'a pas été trouvé. Assurez-vous qu'il se trouve dans le même répertoire.")
        return None
    return texts


def create_scipy_index(texts):
    """Crée des embeddings et un index cKDTree de SciPy."""
    print("Génération des embeddings et création de l'index SciPy...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    index = cKDTree(embeddings.cpu().numpy())
    
    print("Index SciPy créé avec succès.")
    return index, embedding_model, embeddings


def load_qwen_model():
    print(f"Chargement du modèle Qwen : {QWEN_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype="auto",
        device_map="auto"
    )
    print("Modèle Qwen chargé avec succès.")
    return model, tokenizer


def chatbot_response(query, model, tokenizer, scipy_index, embedding_model, all_embeddings, texts, k=K_RETRIEVED_DOCS):
    """
    Exécute la logique RAG pour générer une réponse en utilisant un index SciPy.
    """
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    
    distances, indices = scipy_index.query(query_embedding, k=k)

    context = ""
    for idx in indices[0]:
        context += f"Document : {texts[idx]}\n\n"

    prompt = f"""
    Basé uniquement sur les informations fournies du LASPAD :
    {context}
    
    Question : {query}
    Réponse :
    """

    messages = [
        {"role": "system", "content": "Vous êtes un assistant IA utile qui fournit des informations précises sur le LASPAD. Répondez uniquement en utilisant le contexte fourni. Si la réponse n'est pas dans le contexte, dites que vous ne la connaissez pas."},
        {"role": "user", "content": prompt}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    model_inputs = tokenizer([input_text], return_tensors="pt", padding=True).to(model.device)
    
    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response = response.split("Réponse :")[-1].strip()
    return response


if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"Veuillez créer le fichier {DATA_FILE} avec les données JSONL fournies dans l'énoncé.")
    else:
        laspad_texts = load_and_preprocess_data(DATA_FILE)
        if laspad_texts:
            scipy_index, embedding_model, all_embeddings = create_scipy_index(laspad_texts)
            qwen_model, qwen_tokenizer = load_qwen_model()
            
            print("\nChatbot du LASPAD prêt ! Posez vos questions (tapez 'quitter' pour arrêter).")
            
            while True:
                user_query = input("Vous : ")
                if user_query.lower() in ['quitter', 'exit', 'stop']:
                    print("Chatbot : Au revoir !")
                    break
                
                response = chatbot_response(user_query, qwen_model, qwen_tokenizer, scipy_index, embedding_model, all_embeddings, laspad_texts)
                print(f"Chatbot : {response}")
