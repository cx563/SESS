import argparse
import torch

def train_argparser():

    dataset_files = {
        '14lap': {
            'train': './data/14lap/train_dep_triple_polarity_result.json',
            'valid': './data/14lap/dev_dep_triple_polarity_result.json',
            'test': './data/14lap/test_dep_triple_polarity_result.json',
            'types_path': './data/types.json',
        },
        '14res': {
            'train': './data/14res/train_dep_triple_polarity_result.json',
            'valid': './data/14res/dev_dep_triple_polarity_result.json',
            'test': './data/14res/test_dep_triple_polarity_result.json',
            'types_path': './data/types.json',
        },
        '15res': {
            'train': './data/15res/train_dep_triple_polarity_result.json',
            'valid': './data/15res/dev_dep_triple_polarity_result.json',
            'test': './data/15res/test_dep_triple_polarity_result.json',
            'types_path': './data/types.json',
        },
        '16res': {
            'train': './data/16res/train_dep_triple_polarity_result.json',
            'valid': './data/16res/dev_dep_triple_polarity_result.json',
            'test': './data/16res/test_dep_triple_polarity_result.json',
            'types_path': './data/types.json',
        },
        'debug': {
            'train': './data/14lap/train_dep_triple_polarity_result.json',
            'test': './data/16res/test_debug.json',
            'types_path': './data/types.json',
        }

    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='14res', type=str)
    parser.add_argument('--model_type', default='SESS', type=str)

    parser.add_argument('--drop_out_rate', type=float, default=0.5)
    parser.add_argument('--is_bidirect', default=True)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--emb_dim', type=int, default=768)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--lstm_dim', type=int, default=384)
    parser.add_argument('--attention_heads', default=1, type=int)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--mem_dim', type=int, default=768)
    parser.add_argument('--gcn_dropout', type=float, default=0.2)
    parser.add_argument('--bert_feature_dim', type=int, default=768)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--sen_filter_threshold', type=float, default=0.6)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--inter_dropout_rate', type=float, default=0.1)
    parser.add_argument('--model_path', default='./saved_model/14res', type=str,)
    parser.add_argument('--save_path', type=str, default="./saved_model/14res",)

    parser.add_argument('--max_span_size', type=int, default=8)
    parser.add_argument('--lowercase', action='store_true', default=True)
    parser.add_argument('--max_pairs', type=int, default=1000)
    parser.add_argument('--sampling_limit', type=int, default=100)
    parser.add_argument('--neg_entity_count', type=int, default=100)
    parser.add_argument('--neg_triple_count', type=int, default=100)

    parser.add_argument('--tokenizer_path', default='./bert/base-uncased', type=str)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--size_embedding', type=int, default=25)
    parser.add_argument('--sampling_processes', type=int, default=4)
    parser.add_argument('--prop_drop', type=float, default=0.1)
    parser.add_argument('--freeze_transformer', action='store_true', default=False)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.1)
    parser.add_argument('--log_path', type=str,default="./log/")
    parser.add_argument('--train_log_iter', type=int, default=1)
    parser.add_argument('--pretrained_bert_name', default='./bert/base-uncased', type=str)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--init_eval', action='store_true', default=False)
    parser.add_argument('--final_eval', action='store_true', default=False)
    parser.add_argument('--store_predictions', action='store_true', default=True)
    parser.add_argument('--store_examples', action='store_true', default=True)
    parser.add_argument('--example_count', type=int, default=None)
    parser.add_argument('--save_optimizer', action='store_true')
    parser.add_argument('--device', default="cuda", type=str)


    opt = parser.parse_args()
    opt.label = opt.dataset

    opt.dataset_file = dataset_files[opt.dataset]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    return opt
