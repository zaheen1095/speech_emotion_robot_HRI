#  this code to check the speaker, first one is for google
# from gtts import gTTS
# from playsound import playsound
# import os

# text = "This is a Google TTS test. If you hear this, everything is working."

# # Save the speech to a file
# tts = gTTS(text=text, lang='en')
# tts.save("test_output.mp3")

# # Play the file
# playsound("test_output.mp3")

# # Optional: delete the file afterward
# os.remove("test_output.mp3")


#  to check the speaker with the inbuit actors voice
# import pyttsx3

# engine = pyttsx3.init()
# engine.say("Hello, this is a test.")
# engine.runAndWait()
