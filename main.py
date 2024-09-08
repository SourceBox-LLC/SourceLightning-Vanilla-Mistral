from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Function to save the API key to a .env file
def save_api_key_to_env(api_key, key_name="HUGGINGFACEHUB_API_TOKEN"):
    if os.path.exists(".env"):
        os.remove(".env")  # Remove existing .env file if it exists
    with open(".env", "w") as env_file:
        env_file.write(f"{key_name}={api_key}\n")
    print(".env file created with the API key.")

def get_huggingface_api_key():
    # Check if API key is present in environment variables
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        # Prompt the user to enter the API key if not found
        manual_api_key = input("Enter your HuggingFace API key: ")
        save_api_key_to_env(manual_api_key)
        # Reload the environment with the new API key
        load_dotenv()
        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return api_key

# Load environment variables
load_dotenv()

# Get the HuggingFace API key
api_key = get_huggingface_api_key()

# Initialize the HuggingFace Inference Client with the API key
client = InferenceClient(
    model="mistralai/Mistral-Nemo-Instruct-2407",
    token=api_key
)

# List to store conversation history
conversation_history = []

def huggingface_response(user_input):
    print("Generating response...\n")
    """
    Generate a response from the HuggingFace model based on user input and conversation history.
    """
    try:
        # Add the user's input to the conversation history
        conversation_history.append({"role": "user", "content": user_input})

        # Concatenate the entire conversation history into a single prompt
        conversation_string = ""
        for message in conversation_history:
            conversation_string += f"{message['role']}: {message['content']}\n"

        # Send the concatenated conversation to the HuggingFace model
        response = client.chat_completion(
            messages=[{"role": "user", "content": conversation_string}],
            max_tokens=500,
            stream=False  # Change to True if you want streaming responses
        )

        # Extract the assistant's reply
        assistant_reply = response.choices[0].message.content

        # Add the assistant's reply to the conversation history
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

    except Exception as e:
        return f"Error: Unable to communicate with HuggingFace API: {e}"

def run_chatbot():
    print("\nWelcome to the HuggingFace Command Line Chatbot!\n")
    print("Type 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Generate a response and print it
        response = huggingface_response(user_input)
        print(f"HuggingFace: {response}\n")

if __name__ == "__main__":
    run_chatbot()
