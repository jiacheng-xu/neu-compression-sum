local COMMON = import 'eve_config.jsonnet';
local embedding_dim = 200;
local hidden_dim =14;
local lazy=true;
local num_epochs = 1;
local patience = 10;
local batch_size = 32;
local learning_rate = 0.001;
local debug=true;
local GPUID=0;
local NUM_GPUS = 2;
local NUM_THREADS = 4;
local maximum_samples_per_batch=1999;
local instances_per_epoch = 20000;
local data_root = "/scratch/cluster/jcxu/data/SegSum/";
local data_name = "voa";
local train_data_path= data_root+ data_name+"/train/";
local validation_data_path= data_root+ data_name+"/dev/";
local test_data_path= data_root+ data_name+"/test/";



local elmo_token_embedders={
            "elmo": {
                "type": "elmo_token_embedder",
                "dropout": 0.2,
                "options_file": COMMON['elmo_options_file'],
                "weight_file": COMMON['elmo_weight_file'],
                "do_layer_norm": false
            }
};

local buck_iterator = {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["doc", "num_fields"]],
        "maximum_samples_per_batch": ["num_fields", maximum_samples_per_batch],
        instances_per_epoch:instances_per_epoch
    };

local mul_iterator = {"iterator": {
"type": "multiprocess",
"base_iterator":buck_iterator,
"num_workers": 2,
"output_queue_size": 1000
}};

//local MULTI_ITERATOR ={
//    "type": "multiprocess",
//    "base_iterator": BASE_ITERATOR,
//    "num_workers": NUM_THREADS,
//    "sorting_keys":[["doc", "num_tokens"]],
//    // The multiprocess dataset reader and iterator use many file descriptors,
//    // so we need to increase the ulimit depending on the size of this queue.
//    // See https://pytorch.org/docs/stable/multiprocessing.html#file-descriptor-file-descriptor
//    // for a description of the underlying issue. `ulimit -n 8192` has sufficed,
//    // but that number could use tuning.
//    "output_queue_size": 400
//            };

//local BASE_ITERATOR = {
//  "type": "bucket",
//  "max_instances_in_memory": 70 * NUM_GPUS,
//  // Larger than we really desire for a batch. Since we set
//  // maximum_samples_per_batch below we will pack approximately that many
//  // samples in every batch.
//  "batch_size": batch_size * NUM_GPUS,
//  "sorting_keys": [["doc", "num_fields"]],
////  "instances_per_epoch":200
////  "maximum_samples_per_batch": ["num_tokens", 2000]
//};


{
//    "train_data_path": '/backup3/jcxu/data/dataset_abc/train.json',
//    "validation_data_path": '/backup3/jcxu/data/dataset_abc/dev.json',
    "train_data_path":train_data_path,
    "validation_data_path":validation_data_path,
//    "datasets_for_vocab_creation": [validation_data_path],
//     "vocabulary":
//            {
//            "directory_path": '/backup3/jcxu/NeuSegSum/experiments/vocabulary',
//            "extend": false,
//            },
    "dataset_reader": {
            "type": "seg_sum2",
            "debug":debug,
            "lazy":lazy,
            "max_sent_num":150,
            "max_word_num":50,
            "token_indexers": {
                    "tokens": {
                      "type": "single_id",
                      "lowercase_tokens": true
                    },
                    "elmo": {
                      "type": "elmo_characters"
                    }
                  }
    },
    "model": {
        "type": "Seq2IdxSum",
        "text_field_embedder": {
             "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
            "trainable": true
                        },
                 "elmo": {
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": false,
            "dropout": 0.5
                        }
          },        # end of text_field_embedder


        "encoder": {
            "type": "lstm",
            "input_size": hidden_dim,
            "hidden_size": hidden_dim,
            "batch_first":true,
            "bidirectional":true,
            "dropout":0.3
        },
        "word_encoder":{
            "type":"cnn",
            "embedding_dim": embedding_dim,
            "num_filters":5,
            "output_dim": hidden_dim
        },
//        "attention_function":"dot_product",
        "attention": {
      "type": "bilinear",
      "vector_dim": hidden_dim*2,
      "matrix_dim": hidden_dim*2
                },
    },

     "iterator":buck_iterator,
    "trainer": {
        "num_epochs": num_epochs,
        "cuda_device" : if NUM_GPUS > 1 then std.range(0, GPUID - 1) else GPUID,
        "optimizer": {
            "type": "adam",
            "lr": learning_rate
        },
//        "validation_metric":"+f1",
        "patience": patience,
        "validation_metric":validation_metric,
        serialization_dir:serialization_dir
    },
    evaluate_on_test:true,
}