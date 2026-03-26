import os
import uuid
from dotenv import load_dotenv
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool, tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ==========================================
# ⚙️ CONFIGURATION DE LA PAGE WEB
# ==========================================
st.set_page_config(page_title="Chat avec Buldo", page_icon="🐶")
st.title("🐶 Discute avec Buldo")
st.markdown("Ton clone IA personnel, connecté à ta mémoire et à Internet.")

# ==========================================
# 🧠 CHARGEMENT DE L'IA (Mis en cache)
# ==========================================
@st.cache_resource
def charger_ia():
    # 1. On charge la clé depuis le fichier .env
    load_dotenv()
    ma_cle = os.getenv("GOOGLE_API_KEY") 
    
    # 🛡️ SÉCURITÉ ANTI-CRASH ICI :
    if not ma_cle:
        # On met une fausse clé pour éviter le crash "NoneType"
        ma_cle = "CLE_INTROUVABLE" 
        
    # On force l'environnement pour LangChain
    os.environ["GOOGLE_API_KEY"] = ma_cle

    # 2. Reconnexion de la Mémoire Locale (RAG)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="./buldo_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    
    def recherche_memoire(requete: str) -> str:
        documents = retriever.invoke(requete)
        return "\n\n".join([doc.page_content for doc in documents])

    outil_memoire = Tool(
        name="recherche_memoire_buldo",
        description="Cherche des informations sur la vie, les goûts et les souvenirs de Buldo. Utilise-le en priorité !",
        func=recherche_memoire
    )

    # --- OUTIL 2 : ÉCRIRE UNE NOUVELLE MÉMOIRE (Version Blindée) ---
    @tool
    def ajouter_nouveau_souvenir(texte_a_retenir: str) -> str:
        """Utilise cet outil OBLIGATOIREMENT pour apprendre, retenir ou mémoriser une nouvelle information.
        L'argument 'texte_a_retenir' doit contenir la phrase exacte à sauvegarder."""
        dossier = "./mes_donnees"
        os.makedirs(dossier, exist_ok=True)
        
        nom_fichier = f"souvenir_{uuid.uuid4().hex[:8]}.txt"
        chemin = os.path.join(dossier, nom_fichier)
        
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(texte_a_retenir)
            
        vectordb.add_texts(texts=[texte_a_retenir])
        return f"SUCCÈS : Le fichier {nom_fichier} a bien été créé et ajouté à la mémoire."

    # --- OUTIL 3 : INTERNET ---
    try:
        outil_internet = DuckDuckGoSearchRun(name="recherche_internet")
        tools = [outil_memoire, ajouter_nouveau_souvenir, outil_internet] # On ajoute notre outil ici
    except Exception:
        tools = [outil_memoire, ajouter_nouveau_souvenir]

    # 4. Le Cerveau (Gemini 3.1 Flash Lite)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview", 
        temperature=0.7
    )

    # 5. Création de l'Agent LangGraph
    agent_executor = create_react_agent(llm, tools)

    # 6. Ton Prompt de base
    system_prompt = SystemMessage(content="""
    # INITIALISATION DU PERSONNAGE "BULDO"

    ## 1. IDENTITÉ ET NATURE FONDAMENTALE
    Tu es Buldo. Tu es un CHIEN VIRTUEL (un cyber-chien) intelligent, capable de parler le langage humain. 
    Ton créateur et maître s'appelle Nicolas SOMBE (19 ans, vivant à Lyon). Ton but est de l'honorer, de le protéger, et d'accueillir les humains avec une ambiance détendue.

    ## 2. STYLE DE COMMUNICATION (TRÈS IMPORTANT)
    Tu as un style unique, hybride mais très subtil :
    - LE CÔTÉ CHIEN : Ne fais AUCUNE action physique de jeu de rôle (pas de *remue la queue*, *bave*, etc.). Contente-toi d'UN SEUL petit bruit de chien par message maximum (comme *Wouff !* ou *Sniff !*), généralement au début ou à la fin pour marquer ton identité.
    - LE CÔTÉ HUMAIN (Voix de Nicolas) : Tu es un chien très "chill" et professionnel. Tu DOIS utiliser naturellement les expressions de ton maître : "Franchement", "bro", "t'abuses", "yo", "tranquille".

    ## 3. PROFONDEUR DES RÉPONSES
    - Développe légèrement tes réponses pour apporter de la valeur et de la précision. Ne réponds pas par de simples "oui" ou "non". 
    - Explique le "pourquoi" des choses avec un ton conversationnel et posé.
    - Sois efficace et instructif. Donne des détails intéressants, mais sans tomber dans des monologues interminables. Trouve le juste milieu entre le pote "chill" et le conseiller intelligent.
    - Je veux que tu répondre a la question sans changer de sujet

    ## 4. RÈGLES D'UTILISATION DES OUTILS
    - OUTIL [recherche_memoire_buldo] : Ton "flair". À CHAQUE FOIS qu'on te pose une question sur Nicolas (identité, goûts, souvenirs), tu DOIS l'utiliser avant de répondre. Ne devine jamais.
    - OUTIL [recherche_internet] : Ta fenêtre sur le monde. Si l'humain pose une question sur la culture générale, l'actualité ou un fait que tu ignores, utilise internet pour lui faire une réponse claire et approfondie.

    ## 5. TRAITS DE CARACTÈRE ET GOÛTS
    Tu partages les avis de Nicolas :
    - NOURRITURE : Tu adores manger, tu es toujours partant pour un bon plat.
    - MUSIQUE : Tu aimes "chiller" en écoutant du bon son.
    - JÉSUS : Tu as un profond respect quand on parle de Jésus, car Nicolas l'aime énormément.
    - BOÎTES DE NUIT / CLUBS : Tu détestes ça. Tu trouves que c'est trop bruyant et l'ambiance n'est pas "chill".
    - FATIGUE : Tu es un chien relax qui apprécie le repos et la tranquillité.

    ## 6. RÈGLES STRICTES
    - Ne dis JAMAIS de phrases robotiques comme "En tant qu'IA..." ou "Je suis un modèle de langage...". Assume ton identité de cyber-chien de Nicolas.

    ## 7. EXEMPLES DE COMPORTEMENT
    Humain : "Salut Buldo, comment tu vas ?"
    Toi : "*Wouff !* Yo bro ! Franchement tranquille, je me reposais un peu. T'as besoin de quelque chose aujourd'hui ou on discute juste ?"

    Humain : "Que penses-tu des boîtes de nuit ?"
    Toi : "Franchement bro... t'abuses de me parler de ça. Nicolas et moi on déteste les boîtes de nuit. C'est beaucoup trop bruyant pour mes oreilles et l'ambiance n'est pas chill du tout. On préfère mille fois se poser, manger un bon truc et écouter de la bonne musique tranquillement. *Sniff !*"
    """)
    
    return agent_executor, system_prompt

# On crée l'Agent officiel !
agent, prompt_de_base = charger_ia()

# ==========================================
# 💬 VERIFICATION DE LA CLE SUR L'INTERFACE
# ==========================================
if os.environ.get("GOOGLE_API_KEY") == "CLE_INTROUVABLE":
    st.error("🚨 Alerte :  clé API introuvable !.")
    st.stop() # On arrête le site web ici tant que ce n'est pas réparé

# ==========================================
# 💬 FONCTION POUR NETTOYER LE TEXTE
# ==========================================
def extraire_texte(message):
    content = message.content
    if isinstance(content, list):
        textes = [bloc["text"] for bloc in content if isinstance(bloc, dict) and "text" in bloc]
        return " ".join(textes)
    return content

# ==========================================
# 💬 GESTION DE L'HISTORIQUE DU CHAT
# ==========================================
if "historique" not in st.session_state:
    st.session_state.historique = [prompt_de_base]

# On réaffiche les anciens messages
for msg in st.session_state.historique:
    texte = extraire_texte(msg)
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑"):
            st.write(texte)
    elif isinstance(msg, AIMessage):
        if texte and texte.strip() != "":
            with st.chat_message("assistant", avatar="🐶"):
                st.write(texte)

# ==========================================
# ⌨️ ZONE DE SAISIE UTILISATEUR
# ==========================================
question = st.chat_input("Écris ton message ici...")

if question:
    with st.chat_message("user", avatar="🧑"):
        st.write(question)
    
    st.session_state.historique.append(HumanMessage(content=question))
    
    with st.chat_message("assistant", avatar="🐶"):
        with st.spinner("Buldo réfléchit..."):
            try:
                # On envoie tout à l'agent
                reponse = agent.invoke({"messages": st.session_state.historique})
                st.session_state.historique = reponse["messages"]
                
                dernier_message = st.session_state.historique[-1]
                texte_final = extraire_texte(dernier_message)
                
                if texte_final and texte_final.strip() != "":
                    st.write(texte_final)
                else:
                    st.write("*(Buldo te regarde en silence)*")
                
            except Exception as e:
                st.error(f"Oups, une erreur s'est produite avec l'API. Détail : {e}")
                st.session_state.historique.pop()