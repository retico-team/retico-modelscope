# retico-modelscope
    
Retico modules for ModelScope.

### Installation and requirements ###

The package can be installed using the following:  
`pip install git+https://github.com/retico-team/retico-modelscope`

### Example ###
```python
from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_whisperasr.whisperasr import WhisperASRModule
from retico_modelscope import ChatbotModule


mic = MicrophoneModule()
asr = WhisperASRModule(language='english')
debug = DebugModule(print_payload_only=True)

checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"
lm = ChatbotModule(checkpoint)

mic.subscribe(asr)
asr.subscribe(debug)
asr.subscribe(lm)

mic.run()
asr.run()
debug.run()
print(f"ModelScope Model: {checkpoint}")
lm.run()

input()

mic.stop()
asr.stop()
lm.stop()
debug.stop()

```
