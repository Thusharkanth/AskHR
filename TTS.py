from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine()  # replace with your TTS engine
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
