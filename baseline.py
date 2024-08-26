from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed
import os
import argparse
import logging

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True)


def get_data(train_path, dev_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(dev_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    return train_df, val_df, test_df

def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model)     
    model = AutoModelForSequenceClassification.from_pretrained(
       model, num_labels=len(label2id), id2label=id2label, label2id=label2id 
    )
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # create Trainer 
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    

    trainer.save_model(best_model_path)


def test(test_df, model_path, id2label, label2id):
    
    # load tokenizer from saved model 
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load best model
    model = AutoModelForSequenceClassification.from_pretrained(
       model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
            
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    # return dictionary of classification report
    return results, preds

# python baseline.py --train_file_path en_train.jsonl --dev_file_path en_dev.jsonl --test_file_path en_devtest.jsonl --model roberta-base --prediction_file_path en_prediction.jsonl
# python baseline.py --train_file_path data/multilingual_train.jsonl --dev_file_path data/multilingual_dev.jsonl --test_file_path data/multilingual_devtest.jsonl --model xlm-roberta-base --prediction_file data/multilingual_prediction.jsonl
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--dev_file_path", "-d", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)
    parser.add_argument("--random_seed", "-rs", required=False, help="Random seed.", type=int, default=0)

    args = parser.parse_args()

    random_seed = args.random_seed
    
    train_path = args.train_file_path # For example 'en_train.jsonl'
    dev_path = args.dev_file_path # For example 'en_dev.jsonl'
    test_path = args.test_file_path # For example 'en_devtest.jsonl'
    model = args.model # For example 'xlm-roberta-base'
    prediction_path = args.prediction_file_path # For example en_predictions.jsonl

    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    
    if not os.path.exists(dev_path):
        logging.error("File doesnt exists: {}".format(dev_path))
        raise ValueError("File doesnt exists: {}".format(dev_path))

    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(test_path))
        raise ValueError("File doesnt exists: {}".format(test_path))
    

    id2label = {0: "human", 1: "machine"}
    label2id = {"human": 0, "machine": 1}

    set_seed(random_seed)

    #get data for train/dev/test sets
    train_df, valid_df, test_df = get_data(train_path, dev_path, test_path, random_seed)
    
    # train detector model
    fine_tune(train_df, valid_df, f"{model}/{random_seed}", id2label, label2id, model)

    # test detector model
    results, predictions = test(test_df, f"{model}/{random_seed}/best/", id2label, label2id)
    
    logging.info(results)
    predictions_df = pd.DataFrame({'id': test_df.index, 'label': predictions})
    predictions_df.to_json(prediction_path, lines=True, orient='records')