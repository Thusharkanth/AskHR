from RealtimeTTS import (
    TextToAudioStream,
    OpenAIEngine,
    SystemEngine,
    AzureEngine,
    ElevenlabsEngine,
)

engine = SystemEngine()
engine.set_voice_parameters(rate=180)


# replace with your TTS engine
stream = TextToAudioStream(engine)
stream.feed(
    "Hello I'm Mint G P T, your AI assistant. Mint HRM's vision is to help companies Develop your people, scale your business. The company provides a cloud-based Human Resources Information System (HRIS) designed to enhance people enablement processes and create a joyful and efficient HR experience for businesses"
)
stream.play_async()


# change the voice
# modify the speed #finetune the stt to understand our english
