from abc import abstractmethod


class LanguageModel():
    '''
    def __init__(self, tokenizer=None, reconstructor=None, rec_kwargs=None, *args, **kwargs):
        super(LanguageModel, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer  
        self.reconstructor = reconstructor
        self.rec_kwargs = rec_kwargs
    '''

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_reconstructor(self, reconstructor, tmd_layer=-1, **kwargs):
        self.reconstructor = reconstructor
        self.tmd_layer = tmd_layer
        self.rec_kwargs = kwargs

    @abstractmethod
    def text2token(self, texts):
        # Return a list of tokenized sentences
        pass

    @abstractmethod
    def token2emb(self, tokens):
        # Return the sentence embeddings of a list of sentences
        pass

    @abstractmethod
    def text2emb(self, texts):
        # Return the sentence embeddings of a list of sentences
        pass

    @abstractmethod
    def classify_embs(self, embs):
        # Classify a given list of embeddings
        pass

    @abstractmethod
    def classify_texts(self, texts):
        # Classify a given list of sentences
        pass

    def generate_from_embedding(self, embeddings, max_length=50, temperature=1.0, top_k=50):
        # Generate text from embeddings - default implementation uses nearest neighbor
        return self.find_nearest_text_from_embedding(embeddings, max_length)
    
    def find_nearest_text_from_embedding(self, embeddings, max_length=50):
        # Fallback method - should be overridden by subclasses for proper generation
        if hasattr(self, '_reference_texts') and self._reference_texts:
            return [f"[Generated from embedding - dim {embeddings.shape[-1]}]"] * len(embeddings)
        return ["[Text generation not implemented]"] * len(embeddings)

    def forward(self, x):
        raise NotImplementedError

    @property
    def emb_dim(self):
        raise NotImplementedError
