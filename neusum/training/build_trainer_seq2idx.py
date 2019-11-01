import torch.optim as optim
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.training import Trainer

import logging


def build_trainer(train_data,
                  dev_data,
                  test_data,
                  vocab,
                  device_id: int,
                  model,
                  optim_option: str,
                  serialization_dir: str,
                  batch_size: int = 30,
                  eval_batch_size: int = 30,
                  lr: float = 0.0001,
                  patience: int = 30,
                  nepo: int = 30,
                  grad_clipping: float = 0.5,
                  validation_metric: str = "+cp-1",
                  validation_interval: int = 200
                  ):
    if optim_option.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_option.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError

    logging.info('Model Parameters: ' + model.__repr__())
    training_iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("text", "num_fields")])
    # training_iterator = BasicIterator(batch_size=batch_size)
    training_iterator.index_with(vocab)

    eval_iterator = BasicIterator(batch_size=eval_batch_size)
    eval_iterator.index_with(vocab)
    logging.info("validation_metric: {}".format(validation_metric))
    try:
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          serialization_dir=serialization_dir,
                          num_serialized_models_to_keep=3,
                          iterator=training_iterator,
                          validation_iterator=eval_iterator,
                          train_dataset=train_data,
                          validation_dataset=dev_data,
                          test_dataset=test_data,
                          patience=patience,
                          num_epochs=nepo,
                          cuda_device=device_id,
                          grad_clipping=grad_clipping,
                          validation_metric=validation_metric,
                          validation_interval=validation_interval,
                          )
    except:
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          serialization_dir=serialization_dir,
                          num_serialized_models_to_keep=3,
                          iterator=training_iterator,
                          validation_iterator=eval_iterator,
                          train_dataset=train_data,
                          validation_dataset=test_data,
                          patience=patience,
                          num_epochs=nepo,
                          cuda_device=device_id,
                          grad_clipping=grad_clipping,
                          validation_metric=validation_metric,
                          )
    return trainer
