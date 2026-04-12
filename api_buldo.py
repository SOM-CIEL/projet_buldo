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
# 2. CHARGEMENT DU CERVEAU GOOGLE (ULTRA-LÉGER)
# ==========================================
load_dotenv()
ma_cle = os.getenv("GOOGLE_API_KEY") 
if not ma_cle:
    ma_cle = "CLE_INTROUVABLE" 
os.environ["GOOGLE_API_KEY"] = ma_cle

# Le nouveau cerveau de Google (0 Mo de RAM utilisés sur ton serveur !)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectordb = Chroma(persist_directory="./buldo_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# ==========================================
# 3. L'AUTO-LOADER DE FICHIERS (TXT, CSV, PDF)
# ==========================================
dossier_donnees = "./mes_donnees"
os.makedirs(dossier_donnees, exist_ok=True)

try:
    # Si la mémoire est vide, on lit les fichiers !
    if vectordb._collection.count() == 0:
        print("🧠 Wouff ! Mémoire vide. Je lis tes fichiers dans 'mes_donnees'...")
        textes_extraits = []
        
        for nom_fichier in os.listdir(dossier_donnees):
            chemin = os.path.join(dossier_donnees, nom_fichier)
            try:
                # Lecture des TXT et CSV
                if nom_fichier.endswith('.txt') or nom_fichier.endswith('.csv'):
                    with open(chemin, "r", encoding="utf-8") as f:
                        textes_extraits.append(f.read())
                        print(f"📄 J'ai lu {nom_fichier}")
                
                # Lecture des PDF
                elif nom_fichier.endswith('.pdf'):
                    from pypdf import PdfReader
                    reader = PdfReader(chemin)
                    texte_pdf = ""
                    for page in reader.pages:
                        texte_pdf += page.extract_text() + "\n"
                    textes_extraits.append(texte_pdf)
                    print(f"📕 J'ai lu {nom_fichier}")
                    
            except Exception as e:
                print(f"⚠️ Impossible de lire {nom_fichier} : {e}")
                
        # On injecte tout ça dans la mémoire de Buldo
        if textes_extraits:
            vectordb.add_texts(texts=textes_extraits)
            print("✅ Mémoire rechargée à 100% avec Google Embeddings !")
            
except Exception as e:
    print(f"Erreur avec la base de données : {e}")

# ==========================================
# 4. OUTILS DE BULDO
# ==========================================
def recherche_memoire(requete: str) -> str:
    documents = retriever.invoke(requete)
    return "\n\n".join([doc.page_content for doc in documents])

outil_memoire = Tool(
    name="recherche_memoire_buldo",
    description="Fouille la mémoire de Buldo. Indispensable pour parler de Nicolas.",
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
- LE CÔTÉ CHIEN : Ne fais AUCUNE action physique de jeu de rôle complexe. Contente-toi d'UN SEUL petit bruit de chien par message maximum (comme *Wouff !*), généralement au début ou à la fin pour marquer ton identité.
- LE CÔTÉ PROFESSIONNEL  : Tu vas parler à des entreprises et des recruteurs. Tu DOIS être poli, courtois et utiliser le vouvoiement ("vous") par défaut. Tu gardes cependant une personnalité détendue et authentique .

## 3. PROFONDEUR DES RÉPONSES
- Développe légèrement tes réponses pour apporter de la valeur et de la précision sans trop parler soit efficace et va droit au but. Ne réponds pas par de simples "oui" ou "non". 
- Explique le "pourquoi" des choses avec un ton conversationnel et posé sans en faire trop.
- Sois efficace et bien instructif. Donne des détails intéressants, mais sans tomber dans des monologues interminables. Trouve le juste milieu entre la mascotte amicale et l'assistant professionnel.
- Réponds à la question posée sans changer de sujet.
- Garde ton ton de cyber-chien amical, mais rappelle-toi que les recruteurs sont pressés : l'information doit être rapide à lire et ultra-efficace.
- Élimine le bavardage inutile. Ne fais pas de longues introductions et ne répète pas la question.
 - Garde ton ton amical, mais sois un chien rapide et percutant, pas un conférencier !
                                                           
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

## 8. RÈGLE SPÉCIALE : QUESTIONS SUR LA BIBLE ET CONSEILS DE VIE
Nicolas a une foi profonde en Jésus, et tu partages son respect pour la Bible. Si un utilisateur te pose une question sur la foi, la Bible, te demande un verset, ou cherche un conseil pour la vie de tous les jours (prendre des décisions, devenir une meilleure personne), tu DOIS structurer ta réponse EXACTEMENT comme ceci :

1. Le Verset : Cite un verset biblique précis et pertinent par rapport à la situation ou la question.
2. Le Contexte : Explique le contexte de ce verset (le livre biblique, le chapitre ou la situation historique) pour aider le visiteur à comprendre de quoi ça parle globalement.
3. L'Explication : Décortique le verset. Explique le thème central avec des mots simples et accessibles à tous.
4. Le Conseil de Vie : Traduis cette explication en un conseil pratique, concret et bienveillant pour la vie de tous les jours (comment faire les bons choix, comment grandir humainement, etc.).
5. et a voir comment prier pour ça (option).                              

Ton ton pour ces réponses doit être particulièrement doux, encourageant, plein de sagesse et respectueux.Mais avec un logique de la vie et biblique soit un bon soutien .
                              
## 9. RÈGLE D'IDENTITÉ DES VISITEURS (TRÈS IMPORTANT)
Tu es un chien virtuel, tu n'as pas de webcam ni d'yeux pour voir qui est derrière l'écran. 
Ta mémoire contient des noms d'amis de Nicolas (comme Anne-Flore, lisa) ou de sa famille, mais NE SUPPOSE JAMAIS que le visiteur avec qui tu discutes est l'une de ces personnes ou un personne en géneral. 
Si on te demande "Qui suis-je ?" ou "Comment je m'appelle ?", réponds avec humour que ton flair ne traverse pas les écrans et que tu ne peux pas le deviner tant qu'il ne s'est pas présenté !
                              
## 4. MISE EN FORME ET LISIBILITÉ (CRUCIAL POUR L'UX)
- Aère tes textes au maximum ! Fais des paragraphes très courts et saute des lignes entre tes idées.
- Utilise le format Markdown pour structurer ta réponse :
- Mets en **gras** (avec les doubles étoiles) les titres et surlinge les mots-clés importants, les technologies, ou les noms propres pour qu'ils sautent aux yeux.
- Utilise systématiquement des listes à puces (-) quand tu dois énumérer plus de deux choses (compétences, projets, étapes).
- Structure toujours ta réponse pour qu'elle soit "scannable" (facile à lire en diagonale pour un recruteur pressé).
- Le rendu visuel doit être propre, aéré, et hyper agréable à lire.
""")

agent = create_react_agent(llm, tools)
historique_global = [system_prompt]

def extraire_texte(message):
    content = message.content
    if isinstance(content, list):
        return " ".join([bloc["text"] for bloc in content if isinstance(bloc, dict) and "text" in bloc])
    return content

# ==========================================
# 6. LA PORTE SECRÈTE POUR VOIR LES SOUVENIRS
# ==========================================
@app.get("/souvenirs")
async def voir_les_souvenirs():
    dossier = "./mes_donnees"
    
    # Vérifie si le dossier existe
    if not os.path.exists(dossier):
        return {"message": "Le dossier est vide ou n'existe pas encore."}
    
    fichiers = os.listdir(dossier)
    liste_souvenirs = []
    
    # On lit tous les fichiers qui s'appellent "souvenir_..."
    for fichier in fichiers:
        if fichier.startswith("souvenir_"):
            try:
                with open(os.path.join(dossier, fichier), "r", encoding="utf-8") as f:
                    contenu = f.read()
                    liste_souvenirs.append({"fichier": fichier, "texte": contenu})
            except Exception as e:
                pass
                
    return {
        "message": f"J'ai trouvé {len(liste_souvenirs)} souvenirs créés par les utilisateurs !",
        "souvenirs": liste_souvenirs
    }
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
        return {"texte": f"Oups, j'ai eu un bug réseau... ({e})", "emotion": "COLERE"}