import openai
from utils import ModelServer

server = ModelServer()

def gpt(prompt, model="gpt-3.5-turbo", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    def call_openai_api(messages, model, temperature, max_tokens, n, stop):
        outputs = []
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            res = server.get_completion_or_embedding("8", messages)
            outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        return outputs
    
    messages = [{"role": "user", "content": prompt}]
    return call_openai_api(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

