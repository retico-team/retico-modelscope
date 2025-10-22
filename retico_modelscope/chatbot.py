import torch
from modelscope.hub.snapshot_download import snapshot_download
from retico_core.abstract import AbstractModule, UpdateType, UpdateMessage
from retico_core.text import SpeechRecognitionIU, TextIU
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


class ChatbotModule(AbstractModule):

    @staticmethod
    def name():
        return "ChatbotModule"

    @staticmethod
    def description():
        return "A chatbot module using modelscope."

    @staticmethod
    def input_ius():
        return [SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(
            self,
            model_path,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            **kwargs
    ):
        super().__init__(**kwargs)

        model_dir = snapshot_download(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        )
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.messages = [
            {
                'role': 'system',
                'content': 'You are a friendly chatbot who responds to questions.'
            }
        ]

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == UpdateType.ADD:
                self.current_output.append(iu)
            elif ut == UpdateType.REVOKE:
                self.revoke(iu)
            elif ut == UpdateType.COMMIT:
                last_commit_sentence = ""
                for unit in self.current_output:
                    last_commit_sentence += f"{unit.text} "
                self.current_output = []
                last_commit_sentence = last_commit_sentence.strip()

                if len(last_commit_sentence) > 0:
                    self.process_prompt(last_commit_sentence, iu)

    def process_prompt(self, last_commit_sentence, iu):
        self.messages.append({
            'role': 'user',
            'content': last_commit_sentence
        })

        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                streamer=self.streamer,
            )

        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        self.messages.append({
            'role':'assistant',
            'content': response
        })
        words = response.split()

        new_iu = None
        for word in words:
            new_iu = self.create_iu(iu)
            new_iu.payload = word
            um = UpdateMessage.from_iu(new_iu, UpdateType.ADD)
            self.append(um)

        if new_iu is not None:
            um = UpdateMessage.from_iu(new_iu, UpdateType.COMMIT)
            self.append(um)
