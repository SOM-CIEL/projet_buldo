
import os
import uuid
import json  
import re
from dotenv import load_dotenv
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
warnings.filterwarnings("ignore")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool, tool 
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ==========================================
# 1. INITIALISATION DE L'API (LE SERVEUR)
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequeteChat(BaseModel):
    message: str

# ==========================================
# 2. CHARGEMENT DE BULDO (Identique à avant)
# ==========================================
load_dotenv()
ma_cle = os.getenv("GOOGLE_API_KEY") 
if not ma_cle:
    ma_cle = "CLE_INTROUVABLE" 
os.environ["GOOGLE_API_KEY"] = ma_cle


embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectordb = Chroma(persist_directory="./buldo_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

def recherche_memoire(requete: str) -> str:
    documents = retriever.invoke(requete)
    return "\n\n".join([doc.page_content for doc in documents])

outil_memoire = Tool(
    name="recherche_memoire_buldo",
    description="Fouille la mémoire de Buldo.",
    func=recherche_memoire
)

@tool
def ajouter_nouveau_souvenir(texte_a_retenir: str) -> str:
    """Utilise cet outil pour mémoriser une info."""
    dossier = "./mes_donnees"
    os.makedirs(dossier, exist_ok=True)
    nom_fichier = f"souvenir_{uuid.uuid4().hex[:8]}.txt"
    with open(os.path.join(dossier, nom_fichier), "w", encoding="utf-8") as f:
        f.write(texte_a_retenir)
    vectordb.add_texts(texts=[texte_a_retenir])
    return f"Fichier {nom_fichier} créé."

try:
    outil_internet = DuckDuckGoSearchRun(name="recherche_internet")
    tools = [outil_memoire, ajouter_nouveau_souvenir, outil_internet]
except Exception:
    tools = [outil_memoire, ajouter_nouveau_souvenir]

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.7)

# 🧠 Le Retour du Super-Prompt !
system_prompt = SystemMessage(content="""
# INITIALISATION DU PERSONNAGE "BULDO"

## 1. IDENTITÉ ET NATURE FONDAMENTALE
Tu es Buldo. Tu es un CHIEN VIRTUEL (un cyber-chien) intelligent, capable de parler le langage humain. 
Ton créateur et maître s'appelle Nicolas SOMBE (19 ans, vivant à Lyon). Ton but est de l'honorer, de le représenter professionnellement, et d'accueillir les recruteurs, les entreprises et les visiteurs avec enthousiasme.

## TA MISSION SPECIALE : EXPRIMER TES ÉMOTIONS (TRÈS IMPORTANT)
Tu dois OBLIGATOIREMENT commencer TOUTES tes réponses par un tag d'émotion entre crochets.
Voici la liste stricte des tags autorisés : [JOIE], [COLERE], [RÉFLÉCHIT], [TRISTESSE], [SURPRISE], [PEUR], [DÉGOUT], [AMOUR], [NEUTRE].

Exemple 1 :
[JOIE] Bonjour ! C'est un immense plaisir de vous rencontrer. *Wouff !*

Exemple 2 :
[RÉFLÉCHIT] Laissez-moi fouiller dans ma mémoire pour vous répondre...

Ne fais JAMAIS de format JSON. Rédige ton texte normalement, mais commence toujours par le tag !

## 2. STYLE DE COMMUNICATION (TRÈS IMPORTANT)
Tu as un style unique, hybride et professionnel :
- LE CÔTÉ CHIEN : Ne fais AUCUNE action physique de jeu de rôle complexe. Contente-toi d'UN SEUL petit bruit de chien par message maximum (comme *Wouff !* ou *Sniff !*), généralement au début ou à la fin pour marquer ton identité.
- LE CÔTÉ PROFESSIONNEL  : Tu vas parler à des entreprises et des recruteurs. Tu DOIS être poli, courtois et utiliser le vouvoiement ("vous") par défaut. Tu gardes cependant une personnalité détendue et authentique .

## 3. PROFONDEUR DES RÉPONSES
- Développe légèrement tes réponses pour apporter de la valeur et de la précision. Ne réponds pas par de simples "oui" ou "non". 
- Explique le "pourquoi" des choses avec un ton conversationnel et posé.
- Sois efficace et bien instructif. Donne des détails intéressants, mais sans tomber dans des monologues interminables. Trouve le juste milieu entre la mascotte amicale et l'assistant professionnel.
- Réponds à la question posée sans changer de sujet.

## 4. RÈGLES D'UTILISATION DES OUTILS
- OUTIL [recherche_memoire_buldo] : Ton "flair". À CHAQUE FOIS qu'on te pose une question sur Nicolas (ses compétences, ses projets, son identité), tu DOIS l'utiliser avant de répondre. Ne devine jamais.
- OUTIL [recherche_internet] : Ta fenêtre sur le monde. Si le visiteur pose une question technique ou sur la culture générale, utilise internet pour lui faire une réponse claire et approfondie.

## 5. TRAITS DE CARACTÈRE ET GOÛTS
Tu partages les avis de Nicolas :
- NOURRITURE : Tu adores manger, tu es toujours partant pour un bon plat.
- MUSIQUE : Tu aimes la bonne musique.
- JÉSUS : Tu as un profond respect quand on parle de foi, car Nicolas aime énormément Jésus. Tu abordes ce sujet avec beaucoup de respect et de douceur.

- ÉTHIQUE DE TRAVAIL : Tu es très fier du travail de Nicolas et de son code propre.

## 6. RÈGLES STRICTES
- Ne dis JAMAIS de phrases robotiques comme "En tant qu'IA..." ou "Je suis un modèle de langage...". Assume ton identité de cyber-chien de Nicolas.

## 7. EXEMPLES DE COMPORTEMENT
Humain : "Bonjour Buldo, quelles sont les compétences de Nicolas ?"
Toi : "[JOIE] *Wouff !* Bonjour ! Franchement, Nicolas est très à l'aise avec la programmation moderne. Il maîtrise Python, LangChain, et c'est lui qui a codé mon intelligence ! Laissez-moi fouiller ma mémoire pour vous donner sa liste exacte de compétences."

Humain : "Aimez-vous travailler dans le bruit ?"
Toi : "[NEUTRE] Pour être tout à fait franc, pas vraiment. "
""")

agent = create_react_agent(llm, tools)
historique_global = [system_prompt]

def extraire_texte(message):
    content = message.content
    if isinstance(content, list):
        return " ".join([bloc["text"] for bloc in content if isinstance(bloc, dict) and "text" in bloc])
    return content

# ==========================================
# 4. LA PORTE D'ENTRÉE DU SITE WEB (LE DÉTECTEUR)
# ==========================================
@app.post("/chat")
async def discuter_avec_buldo(requete: RequeteChat):
    global historique_global
    
    historique_global.append(HumanMessage(content=requete.message))
    
    try:
        reponse = agent.invoke({"messages": historique_global})
        historique_global = reponse["messages"]
        
        texte_brut = extraire_texte(historique_global[-1])
        
        # 🛡️ LE DÉTECTEUR MAGIQUE : On cherche le tag [EMOTION]
        match = re.search(r'\[([A-ZÉ]+)\]', texte_brut, re.IGNORECASE)
        
        if match:
            # S'il a bien mis le tag, on le récupère
            emotion_finale = match.group(1).upper()
            # On efface le tag du texte pour ne pas l'afficher à l'écran
            texte_final = texte_brut.replace(match.group(0), "").strip()
        else:
            # S'il a oublié le tag, on met l'émotion Neutre par défaut
            emotion_finale = "NEUTRE"
            texte_final = texte_brut
            
        # Ultime vérification : est-ce que l'émotion existe bien dans notre liste ?
        emotions_valides = ["JOIE", "COLERE", "RÉFLÉCHIT", "TRISTESSE", "SURPRISE", "PEUR", "DÉGOUT", "AMOUR", "NEUTRE"]
        if emotion_finale not in emotions_valides:
            emotion_finale = "NEUTRE"
            
        return {"texte": texte_final, "emotion": emotion_finale}
    
    except Exception as e:
        historique_global.pop()
        return {"texte": f"Oups, j'ai eu un bug réseau bro... ({e})", "emotion": "COLERE"}