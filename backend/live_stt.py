# pip install pyaudio speechrecognition


import speech_recognition as sr

def speech_to_text():
    """
    Converts live speech to text using Google's Speech Recognition API.
    """
    recognizer = sr.Recognizer()

    # Initialize microphone
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        
        while True:
            try:
                # Listen for audio
                audio = recognizer.listen(source)
                
                # Recognize speech using Google's STT
                text = recognizer.recognize_google(audio)
                print("You said:", text)
                
            except sr.UnknownValueError:
                # Handle cases where speech is unintelligible
                print("Sorry, I did not understand that.")
            
            except sr.RequestError as e:
                # Handle API request errors
                print(f"Sorry, there was an error with the request; {e}")
                break

def main():
    print("Live Speech-to-Text Application")
    print("Say something into the microphone. Type 'exit' to quit.")
    
    try:
        speech_to_text()
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
