import openai
from utils import ModelServer
import time

import re

def split_and_keep_prefixes(s, delimiters):
    # 构造一个包含所有分隔符的正则表达式，使用捕获组以保留分隔符
    regex_pattern = f"({'|'.join(map(re.escape, delimiters))})"
    # 使用正则表达式分割，保留分隔符
    parts = re.split(regex_pattern, s)
    # 重新组合分割后的字符串，使每部分以分隔符开头
    result = [parts[0]]
    for i in range(1, len(parts), 2):
        result.append(parts[i] + (parts[i+1] if i+1 < len(parts) else ''))
    return result

server = ModelServer()

def online_embed(traj):
    return server.get_completion_or_embedding(
                        "7",
                        message=traj,
                        get_embedding=True,
                    )

def gpt(prompt, model="gpt-3.5-turbo", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    def call_openai_api(messages, model, temperature, max_tokens, n, stop):
        outputs = []
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            res = server.get_completion_or_embedding("8", messages)
            outputs.extend([re.sub(r'^Thought \d+: ', '', choice.message.content) for choice in res.choices])
        return outputs
    
    #messages = [{"role": "user", "content": prompt}]
    messages=[]
    # 使用正则表达式来分割字符串
    # 正则表达式解释：'Thought \d+:|Observation \d+:' 匹配 'Thought ' 后跟一个或多个数字（\d+）和冒号
    # 或者 'Observation ' 后跟一个或多个数字和冒号
    parts = re.split(r'(Thought \d+:|Observation \d+:|Question:)', prompt)

    # 过滤空字符串
    #parts = [part.strip() for part in parts if part.strip()]
    result = [parts[0].strip()]
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            result.append(parts[i] + " "+parts[i + 1].strip())
    #print(result)
    #time.sleep(30)
    result.pop()
    #print(result)
    #print("\n\n\n")
    for msg in result:
        if msg.startswith("Solve"):
            messages.append({"role": "user", "content": msg})
        if msg.startswith("Thought"):
            messages.append({"role": "assistant", "content": msg})
        if msg.startswith("Observation"):
            messages.append({"role": "user", "content": msg})
        if msg.startswith("Question"):
            messages.append({"role": "user", "content": msg})
    #print(messages)
    #time.sleep(30)
    return call_openai_api(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

