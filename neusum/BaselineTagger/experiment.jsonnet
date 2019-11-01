local stacked_self_attention_encoder = {
        type:"stacked_self_attention",
        input_dim:200,
        hidden_dim:400,
        projection_dim:200,
        feedforward_hidden_dim:400,
        num_layers:3,
        num_attention_heads:5,
        use_positional_encoding:true
};
local lstm_encoder = {
 "type": "lstm",
        "input_size": 200,
        "hidden_size": 200,
        "num_layers": 2
};
{
    "dataset_reader":{"type":"pos-tutorial"},
    "train_data_path": '/backup3/jcxu/data/compression-train.tsv',
    "validation_data_path": '/backup3/jcxu/data/compression-test.tsv',
    "model": {
//      "type": "simple_tagger",
      "type": "lstm-tagger",

      "word_embeddings": {
        "token_embedders": {
            "tokens": {
            "type": "embedding",
            "projection_dim": 200,
            "pretrained_file": "/home/jcxu/allennlp/allennlp/tests/fixtures/embeddings/glove.6B.100d.sample.txt.gz",
            "embedding_dim": 100,
            "trainable": true
            }
        },

      },
//      calculate_span_f1:true,
//        label_encoding:"BIO",
      "encoder": lstm_encoder,
      "sec_encoder":stacked_self_attention_encoder,
//      "regularizer": [
//        ["weight$", {"type": "l2", "alpha": 10}],
//        ["bias$", {"type": "l1", "alpha": 5}]
//      ]
    },
    "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "num_tokens"]],
      "padding_noise": 0.0,
      "batch_size" : 30
    },
    "trainer": {
      "num_epochs": 30,
      "grad_norm": 1.0,
      "patience": 30,
      "cuda_device": 0,
      "optimizer": {
        "type": "adam",
        "lr": 0.0001,
//        "rho": 0.95
      }
    }
  }