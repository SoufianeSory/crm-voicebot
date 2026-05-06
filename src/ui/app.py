"""
Interface Streamlit — Chatbot RAG Support Client
Lance : cd src/ui && streamlit run app.py
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import streamlit as st
import requests

# ─── Config page ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Support Client — Chatbot IA",
    page_icon="💬",
    layout="centered",
)

API_URL = "http://localhost:8000"

# ─── CSS personnalisé ─────────────────────────────────────────────────────────

st.markdown("""
<style>
.user-bubble {
    background-color: #0084ff;
    color: white;
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0;
    max-width: 75%;
    margin-left: auto;
    text-align: right;
}
.bot-bubble {
    background-color: #f0f0f0;
    color: #222;
    padding: 10px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 0;
    max-width: 75%;
}
.escalade-bubble {
    background-color: #ffe0e0;
    color: #cc0000;
    padding: 10px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 0;
    max-width: 75%;
    border-left: 4px solid #cc0000;
}
.meta-info {
    font-size: 11px;
    color: #999;
    margin: 2px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("## 💬 Support Client — Chatbot IA")
st.markdown("Posez vos questions. Le chatbot répond automatiquement ou vous transfère à un agent.")
st.divider()

# ─── Initialisation session ───────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_meta" not in st.session_state:
    st.session_state.show_meta = False

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Options")
    st.session_state.show_meta = st.toggle("Afficher métadonnées (intent, score)", value=False)

    st.divider()

    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        try:
            requests.post(f"{API_URL}/reset", timeout=3)
        except:
            pass
        st.rerun()

    st.divider()
    st.markdown("### 📊 Statistiques session")
    nb_user  = sum(1 for m in st.session_state.messages if m["role"] == "user")
    nb_auto  = sum(1 for m in st.session_state.messages if m.get("action") == "respond")
    nb_escal = sum(1 for m in st.session_state.messages if m.get("action") == "escalate")

    col1, col2 = st.columns(2)
    col1.metric("Questions", nb_user)
    col2.metric("Réponses auto", nb_auto)
    if nb_escal > 0:
        st.metric("Escalades", nb_escal, delta=f"-{nb_escal}", delta_color="inverse")

    st.divider()
    st.markdown("**Stack technique**")
    st.markdown("🧠 Llama 3.3-70B (Groq)")
    st.markdown("🔍 Qdrant (vector store)")
    st.markdown("⚙️ LangGraph (agents)")

# ─── Affichage historique ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">👤 {msg["content"]}</div>', unsafe_allow_html=True)

    elif msg["role"] == "assistant":
        action = msg.get("action", "respond")
        if action == "escalate":
            st.markdown(f'<div class="escalade-bubble">🔴 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        if st.session_state.show_meta:
            intent     = msg.get("intent", "—")
            confidence = msg.get("confidence", 0)
            nb_docs    = msg.get("nb_docs", 0)
            rewritten  = msg.get("rewritten", "—")
            st.markdown(
                f'<div class="meta-info">'
                f'intent: <b>{intent}</b> · '
                f'confiance: <b>{confidence:.2f}</b> · '
                f'docs: <b>{nb_docs}</b> · '
                f'action: <b>{action}</b><br>'
                f'reformulé: <i>{rewritten}</i>'
                f'</div>',
                unsafe_allow_html=True
            )

# ─── Input utilisateur ────────────────────────────────────────────────────────

question = st.chat_input("Posez votre question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Le chatbot réfléchit..."):
        try:
            resp = requests.post(
                f"{API_URL}/chat",
                json={"question": question},
                timeout=300,
            )
            data = resp.json()

            st.session_state.messages.append({
                "role"      : "assistant",
                "content"   : data["response"],
                "intent"    : data["intent"],
                "confidence": data["confidence"],
                "action"    : data["action"],
                "reason"    : data["reason"],
                "nb_docs"   : data.get("nb_docs", 0),
                "rewritten" : data.get("rewritten", ""),
            })

        except requests.exceptions.ConnectionError:
            st.session_state.messages.append({
                "role"   : "assistant",
                "content": "❌ Impossible de joindre l'API. Lance d'abord : python main.py dans src/api",
                "action" : "error",
            })
        except Exception as e:
            st.session_state.messages.append({
                "role"   : "assistant",
                "content": f"❌ Erreur : {str(e)}",
                "action" : "error",
            })

    st.rerun()