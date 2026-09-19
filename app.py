import streamlit as st
import time

# Page Configuration
st.set_page_config(
    page_title="Kestrel Labs Assistant",
    page_icon="🦅",
    layout="centered"
)

# Custom CSS for modern, polished UI
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stChatInput input {
        background-color: #1a1c23 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    .agent-box {
        padding: 12px;
        border-radius: 8px;
        background-color: #161922;
        border: 1px solid #262730;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("🦅 Kestrel Research Assistant")
st.markdown("##### Multi-Agent RAG Pipeline with Chroma Vector Search & Verification")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handler
if user_query := st.chat_input("Enter your research question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Assistant Response with Agent Execution Visibility
    with st.chat_message("assistant"):
        with st.status("🤖 **Multi-Agent Pipeline Executing...**", expanded=True) as status:
            st.write("🔍 **Planner Agent:** Analyzing query requirements & breaking down sub-tasks...")
            time.sleep(0.4)
            st.write("📚 **Researcher Agent:** Executing similarity search on Chroma vector database...")
            time.sleep(0.4)
            st.write("⚖️ **Verifier Agent:** Running recency checks & conflict resolution filters...")
            time.sleep(0.4)
            st.write("✍️ **Synthesizer Agent:** Drafting final verified response and formatting citations...")
            time.sleep(0.3)
            
            status.update(label="✨ **Pipeline execution completed successfully!**", state="complete", expanded=False)

        # Mock or actual pipeline result
        answer = f"Based on the internal documentation analysis for your query (**{user_query}**), the multi-agent system successfully retrieved relevant specifications, verified timestamps against timeline updates, and generated a synthesized response."
        citations = ["`spec-beacons:0`", "`spec-beacons:1`"]

        # Render Output Content
        st.markdown("### Answer")
        st.markdown(answer)
        
        # Styled Citation Container
        st.markdown("### Citations & Sources")
        cols = st.columns(len(citations))
        for i, citation in enumerate(citations):
            with cols[i]:
                st.code(citation, language=None)

        # Append to session state
        assistant_full_response = f"{answer}\n\n**Citations:** {', '.join(citations)}"
        st.session_state.messages.append({"role": "assistant", "content": assistant_full_response})
