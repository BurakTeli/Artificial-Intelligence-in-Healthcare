import openai

openai.api_key = "api_key"

def chat_with_gpt(prompt, history):
    
   responce = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role":"user",
                     "content": f"Sen bir doktorsun yani aile hekimi. Geçmiş konuşmalarımız: {history}, yeni soru: {prompt}"}]
   )
   
   return responce.choices[0].message.content.strip()

if __name__ == "__main__":
    history = []
    
    while True:
        
        user_input = input("Mesajınız nedir?")
        
        if user_input.lower() in ["exit", ""]:
            print("Gorusme tamamlandi")
            break
        
        history.append(user_input)
        responce = chat_with_gpt(user_input, history)
        print("Chatbot: ",responce )
    
    