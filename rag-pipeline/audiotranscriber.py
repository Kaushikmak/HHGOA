import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

# Load environment variables from a local .env file into os.environ
load_dotenv()

def transcribe_audio(file_name):
    # Retrieve the API key stored in the .env file
    api_key = os.getenv("SARVAM_KEY")
    
    # Check if the API key exists; if not, print an error and exit the function early
    if not api_key:
        print("Error: SARVAM_API_KEY not found in .env file.")
        return None

    # Initialize the Sarvam AI client using the retrieved API key
    client = SarvamAI(
        api_subscription_key=api_key,
    )

    print(f"Sending '{file_name}' to Sarvam AI...")
    
    try:
        # Open the target audio file in binary read mode ("rb")
        with open(file_name, "rb") as audio_file:
            # Send the audio file to the Sarvam API for transcription
            response = client.speech_to_text.transcribe(
                file=audio_file,
                model="saaras:v3", # Specify the speech-to-text model version
                mode="transcribe"  # Set the mode to transcription
            )
            
        # Extract the resulting text from the API response
        transcript_text = response.transcript
        
        # Print the transcript clearly to the console
        print("\n--- Transcription Result ---")
        print(transcript_text)
        print("----------------------------\n")
        
        return transcript_text

    # Handle the specific error where the provided file path is incorrect or missing
    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_name}'. Ensure the extension (e.g., .wav, .mp3) is included.")
        return None
        
    # Catch any other potential errors (like network timeouts or API failures)
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None

# Standard Python idiom to ensure this block only runs if the script is executed directly
if __name__ == "__main__":
    # Define the audio file to be transcribed
    AUDIO_FILE = "test1.wav" 
    
    # Call the transcription function
    transcript = transcribe_audio(AUDIO_FILE)