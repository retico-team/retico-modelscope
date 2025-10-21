from modelscope.pipelines import pipeline
from retico_core.abstract import AbstractModule, UpdateType, UpdateMessage
from retico_core.text import SpeechRecognitionIU, TextIU


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
            temperature=0.7,
            repetition_penalty=1.1,
            top_p=0.9,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.chatbot = pipeline(
            'chat',
            model=model_path
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p

    def process_update(self, update_message):
        send_prompt = False
        for iu, ut in update_message:
            if ut == UpdateType.ADD:
                self.current_output.append(iu)
            elif ut == UpdateType.REVOKE:
                self.revoke(iu)
            elif ut == UpdateType.COMMIT:
                send_prompt = True

        if send_prompt:
            send_prompt = False
            last_commit_sentence = ""
            for unit in self.current_output:
                last_commit_sentence += f"{unit.text} "

            self.current_output = []

            if len(last_commit_sentence) > 0:
                self.process_prompt(last_commit_sentence, iu)

    def process_prompt(self, last_commit_sentence, iu):
        response = self.chatbot(
            last_commit_sentence.strip(),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            top_p=self.top_p
        )
        words = response['text'].split(' ')
        for idx, word in enumerate(words):
            new_iu = self.create_iu()
            new_iu.payload = word
            if idx == len(words)-1:
                print(word, end='\n', flush=True)
                um = UpdateMessage.from_iu(new_iu, UpdateType.COMMIT)
            else:
                print(word, end=' ', flush=True)
                um = UpdateMessage.from_iu(iu, UpdateType.ADD)
            self.append(um)

