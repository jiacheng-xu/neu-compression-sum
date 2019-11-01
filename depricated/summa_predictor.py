import json
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model


@Predictor.register('Seq2IdxSum')
class Seq2IdxPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, jsonline: str) -> JsonDict:
        return self.predict_json(json.loads(jsonline))
