from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from autogen import AssistantAgent, UserProxyAgent

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return HTMLResponse("""
    <div id="output"></div>
    <input id="msg"><button onclick="send()">Send</button>
    <script>
        let text = document.getElementById('output');
        async function send() {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: document.getElementById('msg').value})
            });
            const reader = response.body.getReader();
            while (true) {
                const {done, value} = await reader.read();
                if (done) break;
                let chunk = new TextDecoder().decode(value);
                console.log('Got chunk:', chunk);
                text.innerText += chunk;
            }
        }
    </script>
    """)

@app.post("/chat")
async def chat(request: ChatRequest):
    async def generate():
        assistant = AssistantAgent(
            name="assistant",
            llm_config={
                "config_list": [{
                    "model": "llama3.2",
                    "base_url": "http://localhost:11434/v1",
                    "api_key": "ollama"
                }]
            }
        )

        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config=False
        )

        chat_result = user_proxy.initiate_chat(
            assistant,
            message=request.message
        )

        for msg in chat_result.chat_history:
            if msg['role'] == 'assistant':
                yield msg['content']

    return StreamingResponse(generate())