def chatbot_response(user_input):
    user_input = user_input.lower()

    # Define some simple rules and responses
    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I assist you today?"

    elif "how are you" in user_input:
        return "I'm good thank you, How are you? And How can I help you?"

    elif "what's your name?" in user_input or "who are you?" in user_input:
        return "I am Rishabh, your friendly rule-based chatbot."

    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a great day!"

    else:
        return "I'm sorry, I don't understand. Can you please rephrase or ask another question?"

# Main loop for user interaction
def main():
    print("Chatbot: Hi! I am Rishabh, your friendly chatbot. How can I assist you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye! Have a great day!")
            break
        else:
            response = chatbot_response(user_input)
            print("Chatbot:", response)

if __name__ == "__main__":
    main()