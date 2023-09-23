import vosk
import speech_recognition as sr
import wave
import json
from Utilities import convertphonemes, util
import ASR
# from pocketsphinx import LiveSpeech

# Define your Vosk ASR model path
vosk_model_path = "E:/SIH/vosk-model-small-hi-0.22"

# Initialize Vosk ASR model
vosk_model = vosk.Model(vosk_model_path)

# Import the necessary functions from indicnlp
from indicnlp.transliterate.unicode_transliterate import ItransTransliterator

# Define a function to perform phoneme recognition
def phoneme_recognition(audio_data):
    recognizer = vosk.KaldiRecognizer(vosk_model, 48000)
    recognizer.AcceptWaveform(audio_data)
    result = recognizer.Result()
    return result


# Initialize the SpeechRecognition recognizer and microphone
recognizer = sr.Recognizer()
microphone = sr.Microphone()

print("Say आप कैसे हैं in Hindi:")

with microphone as source:
    audio = recognizer.listen(source)

# Perform phoneme recognition
audio_data = audio.frame_data
recognized_text = phoneme_recognition(audio_data)
print("Hiii",recognized_text)

og_text = "आप कैसे हैं"
print("Og Text:", og_text)
og_phenome = convertphonemes.words_to_phonemes(util.tokenize(og_text))
print(og_phenome)

print("Recognized Text:", recognized_text)
data = json.loads(recognized_text)
rec_phenome = convertphonemes.words_to_phonemes(util.tokenize(data["text"]))
print(rec_phenome)

# og_text = "आप कैसे हैं"
# print("Og Text:", og_text)
# og_phenome = ASR.text_to_phonemes(util.tokenize(og_text))
# print(og_phenome)

# print("Recognized Text:", recognized_text)
# data = json.loads(recognized_text)
# rec_phenome = ASR.text_to_phonemes(util.tokenize(data["text"]))
# print(rec_phenome)

# Save the captured audio as a .wav file
with wave.open("captured_audio.wav", 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(48000)
    wf.writeframes(audio_data)


