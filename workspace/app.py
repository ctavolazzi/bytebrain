import streamlit as st
import ollama

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

def get_ollama_response(message):
    """Get response from Ollama model"""
    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[{
                'role': 'user',
                'content': message
            }]
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def main():
    # Set up the Streamlit page
    st.title("🤖 Ollama Chat Interface")
    st.write("Chat with the Llama 3.2 model using Ollama")

    # Chat input
    user_message = st.text_input("Your message:", key="user_input")

    # Send button
    if st.button("Send"):
        if user_message:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_message})

            # Get bot response
            bot_response = get_ollama_response(user_message)
            if bot_response:
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Display chat history
    st.write("### Chat History")
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write("You: " + message["content"])
        else:
            st.write("Bot: " + message["content"])
        st.write("---")

if __name__ == "__main__":
    main()