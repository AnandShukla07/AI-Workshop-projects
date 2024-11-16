import openai

# Define the chatbot function
def chat_with_openai(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or use "gpt-4" if you have access
        messages=messages,
        max_tokens=100,          # Adjust token limit based on your needs
        temperature=0.7          # Set for creativity; lower for more deterministic responses
    )
    return response['choices'][0]['message']['content']

# Initial system message (optional but helpful to set the chatbot's behavior)
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("Chatbot: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye!")
        break
    
    # Append user message
    messages.append({"role": "user", "content": user_input})
    
    # Get response from the model
    chatbot_response = chat_with_openai(messages)
    
    # Append assistant's response to conversation history
    messages.append({"role": "assistant", "content": chatbot_response})
    
    print(f"Chatbot: {chatbot_response}")
