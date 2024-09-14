import cv2
import pytesseract
import pyttsx3
from PIL import Image

# Configure the path to the Tesseract executable
# Update the path based on your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the TTS engine
engine = pyttsx3.init()

def perform_ocr(image):
    """
    Perform OCR on the given image and return the extracted text.
    """
    # Convert the image to a format suitable for pytesseract
    pil_image = Image.fromarray(image)
    # Perform OCR with the language set to English
    text = pytesseract.image_to_string(pil_image, lang='eng')
    return text.strip()

def speak_text(text):
    """
    Convert text to speech using pyttsx3.
    """
    engine.say(text)
    engine.runAndWait()

def main():
    print("Live OCR Application")
    print("Press 'q' to quit.")
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Perform OCR on the captured frame
        text = perform_ocr(frame)
        
        # Display the captured frame
        cv2.imshow('Live OCR', frame)
        
        # Overlay the extracted text on the video feed
        if text:
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Speak the recognized text
            speak_text(text)
        
        # Display the frame with overlaid text
        cv2.imshow('Live OCR', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
