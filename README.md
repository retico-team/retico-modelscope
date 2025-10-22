# retico-modelscope
    
Retico modules for ModelScope.

### Installation and requirements ###

The package can be installed using the following:  
`pip install git+https://github.com/retico-team/retico-modelscope`

### Example Runner ###
```python
from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_whisperasr.whisperasr import WhisperASRModule
from retico_modelscope import ChatbotModule


mic = MicrophoneModule()
asr = WhisperASRModule(language='english')

checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"
lm = ChatbotModule(checkpoint)

mic.subscribe(asr)
asr.subscribe(lm)

mic.run()
asr.run()
lm.run()

print(f"ModelScope Model: {checkpoint}")
input()

mic.stop()
asr.stop()
lm.stop()

```

