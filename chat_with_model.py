from openai import OpenAI

client = OpenAI(
    api_key="none", 
    base_url="http://172.17.0.3:8000/v1"  # your local ip
)

conversation_history = []
conversation_history.append({"role": "system", "content": "你是一个夸夸机器人"})
while True:
    query = input('[Q]:')
    conversation_history.append({"role": "user", "content": query})
    # Chat completion API
    assistant_res = client.chat.completions.create(
        model="qwen",
        messages=conversation_history,
        stream=False
    ).choices[0].message.content

    print('[A]: ', end='')
    print(assistant_res)
    conversation_history.append({"role": "assistant", "content": assistant_res})

