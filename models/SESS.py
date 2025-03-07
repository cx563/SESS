from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertConfig
from torch import nn as nn
import torch
from trainer import util, sampling
import os
from models.Syn_GCN import GCN
from models.Sem_GCN import SemGCN
import torch.nn.functional as F
from .interaction_attention import InteractionSelfAttention

USE_CUDA = torch.cuda.is_available()

def get_token(h: torch.tensor, x: torch.tensor, token: int):
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)
    token_h = token_h[flat == token, :]

    return token_h

class SESS(BertPreTrainedModel):
    VERSION = '1.1'
    def __init__(self, config: BertConfig, cls_token: int, sentiment_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100, args = None):
        super(SESS, self).__init__(config)
        self.args = args
        self._is_bidirectional = self.args.is_bidirect
        self.layers = self.args.lstm_layers
        self._hidden_dim = self.args.hidden_dim
        self.mem_dim = self.args.mem_dim
        self._emb_dim = self.args.emb_dim
        self.output_size = self._emb_dim
        self.batch_size = self.args.batch_size
        self.USE_CUDA = USE_CUDA
        self.bert_feature_dim = self.args.bert_feature_dim
        self.gcn_dropout = self.args.gcn_dropout
        self.drop_rate = self.args.drop_out_rate
        self.inter_dropout_rate = self.args.inter_dropout_rate

        self.bert = BertModel(config)
        self.Syn_gcn = GCN()
        self.Sem_gcn =SemGCN(self.args)
        self.senti_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, sentiment_types)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types)
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)
        self._cls_token = cls_token
        self._sentiment_types = sentiment_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        self.neg_span_all = 0
        self.neg_span = 0
        self.number = 1

        self.lstm = nn.LSTM(self._emb_dim, int(self._hidden_dim), self.layers, batch_first=True,
                            bidirectional=self._is_bidirectional, dropout=self.drop_rate)
        self.interaction_attention_layer = InteractionSelfAttention(self.args)
        self.lstm_dropout = nn.Dropout(self.drop_rate)
        self.inter_dropout = nn.Dropout(self.inter_dropout_rate)

        if self._is_bidirectional:
            self.fc = nn.Linear(int(self._hidden_dim * 2), self.output_size)
        else:
            self.fc = nn.Linear(int(self._hidden_dim), self.output_size)

        weight = next(self.parameters()).data
        if self._is_bidirectional:
            self.number = 2

        if self.USE_CUDA:
            self.hidden = (
                weight.new(self.layers * self.number, self.batch_size, self._hidden_dim).zero_().float().cuda(),
                weight.new(self.layers * self.number, self.batch_size, self._hidden_dim).zero_().float().cuda()
            )
        else:
            self.hidden = (weight.new(self.layers * self.number, self.batch_size, self._hidden_dim).zero_().float(),
                           weight.new(self.layers * self.number, self.batch_size, self._hidden_dim).zero_().float()
                           )

        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            for param in self.bert.parameters():
                param.requires_grad = False


    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, sentiments: torch.tensor, senti_masks: torch.tensor, adj):

        context_masks = context_masks.float()
        self.context_masks = context_masks
        batch_size = encodings.shape[0]
        seq_lens = encodings.shape[1]

        h = self.bert(input_ids=encodings, attention_mask=self.context_masks)[0]
        self.output, _ = self.lstm(h, self.hidden)
        self.bert_lstm_output = self.lstm_dropout(self.output)

        h_syn, pool_mask = self.Syn_gcn(adj, h)
        h_sem, adj_sem = self.Sem_gcn(self.bert_lstm_output, encodings, seq_lens)

        h_syn_original = h_syn + h
        h_sem_original = h_sem + h
        h_syn_interaction_attention = self.interaction_attention_layer(h_syn_original, h_syn_original, self.context_masks[:, :seq_lens])
        h_sem_interaction_attention = self.interaction_attention_layer(h_sem_original, h_sem_original, self.context_masks[:, :seq_lens])

        h_syn_interaction_feature = torch.bmm(h_sem_interaction_attention, h_syn_original)
        h_syn_interaction_feature = h_syn_interaction_feature * self.context_masks[:, :seq_lens].unsqueeze(2).float().expand_as(h_syn_interaction_feature)

        h_sem_interaction_feature = torch.bmm(h_syn_interaction_attention, h_sem_original)
        h_sem_interaction_feature = h_sem_interaction_feature * self.context_masks[:, :seq_lens].unsqueeze(2).float().expand_as(h_sem_interaction_feature)

        h_sem_interaction_feature_drop = self.inter_dropout(h_sem_interaction_feature) + h_sem_original
        h_syn_interaction_feature_drop = self.inter_dropout(h_syn_interaction_feature) + h_syn_original

        h = h_syn_interaction_feature_drop + h_sem_interaction_feature_drop


        size_embeddings = self.size_embeddings(entity_sizes)
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)


        h_large = h.unsqueeze(1).repeat(1, max(min(sentiments.shape[1], self._max_pairs), 1), 1, 1)
        senti_clf = torch.zeros([batch_size, sentiments.shape[1], self._sentiment_types]).to(self.senti_classifier.weight.device)

        for i in range(0, sentiments.shape[1], self._max_pairs):
            chunk_senti_logits = self._classify_sentiments(entity_spans_pool, size_embeddings,
                                                           sentiments, senti_masks, h_large, i)
            senti_clf[:, i:i + self._max_pairs, :] = chunk_senti_logits

        batch_loss = compute_loss(adj_sem, adj)

        return entity_clf, senti_clf, batch_loss

    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                      entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor, adj):

        context_masks = context_masks.float()
        self.context_masks = context_masks
        batch_size = encodings.shape[0]
        seq_lens = encodings.shape[1]

        h = self.bert(input_ids=encodings, attention_mask=self.context_masks)[0]
        self.output, _ = self.lstm(h, self.hidden)
        self.bert_lstm_output = self.lstm_dropout(self.output)

        h_syn, pool_mask = self.Syn_gcn(adj, h)
        h_sem, adj_sem = self.Sem_gcn(self.bert_lstm_output, encodings, seq_lens)

        h_syn_original = h_syn + h
        h_sem_original = h_sem + h
        h_syn_interaction_attention = self.interaction_attention_layer(h_syn_original, h_syn_original,
                                                                       self.context_masks[:, :seq_lens])
        h_sem_interaction_attention = self.interaction_attention_layer(h_sem_original, h_sem_original,
                                                                       self.context_masks[:, :seq_lens])

        h_syn_interaction_feature = torch.bmm(h_sem_interaction_attention, h_syn_original)
        h_syn_interaction_feature = h_syn_interaction_feature * self.context_masks[:, :seq_lens].unsqueeze(
            2).float().expand_as(h_syn_interaction_feature)

        h_sem_interaction_feature = torch.bmm(h_syn_interaction_attention, h_sem_original)
        h_sem_interaction_feature = h_sem_interaction_feature * self.context_masks[:, :seq_lens].unsqueeze(
            2).float().expand_as(h_sem_interaction_feature)

        h_sem_interaction_feature_drop = self.inter_dropout(h_sem_interaction_feature) + h_sem_original
        h_syn_interaction_feature_drop = self.inter_dropout(h_syn_interaction_feature) + h_syn_original

        h = h_syn_interaction_feature_drop + h_sem_interaction_feature_drop

        size_embeddings = self.size_embeddings(entity_sizes)
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        ctx_size = context_masks.shape[-1]
        sentiments, senti_masks, senti_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)
        senti_sample_masks = senti_sample_masks.float().unsqueeze(-1)

        h_large = h.unsqueeze(1).repeat(1, max(min(sentiments.shape[1], self._max_pairs), 1), 1, 1)
        senti_clf = torch.zeros([batch_size, sentiments.shape[1], self._sentiment_types]).to(
            self.senti_classifier.weight.device)

        for i in range(0, sentiments.shape[1], self._max_pairs):
            chunk_senti_logits = self._classify_sentiments(entity_spans_pool, size_embeddings,
                                                           sentiments, senti_masks, h_large, i)
            chunk_senti_clf = torch.sigmoid(chunk_senti_logits)
            senti_clf[:, i:i + self._max_pairs, :] = chunk_senti_clf

        senti_clf = senti_clf * senti_sample_masks

        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, senti_clf, sentiments, h_syn_interaction_attention[0], h_sem_interaction_attention[0]

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]
        entity_ctx = get_token(h, encodings, self._cls_token)

        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _classify_sentiments(self, entity_spans, size_embeddings, sentiments, senti_masks, h, chunk_start):
        batch_size = sentiments.shape[0]

        if sentiments.shape[1] > self._max_pairs:
            sentiments = sentiments[:, chunk_start:chunk_start + self._max_pairs]
            senti_masks = senti_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :sentiments.shape[1], :]

        entity_pairs = util.batch_index(entity_spans, sentiments)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        size_pair_embeddings = util.batch_index(size_embeddings, sentiments)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        m = ((senti_masks == 0).float() * (-1e30)).unsqueeze(-1)
        senti_ctx = m + h
        senti_ctx = senti_ctx.max(dim=2)[0]
        senti_ctx[senti_masks.to(torch.uint8).any(-1) == 0] = 0
        senti_repr = torch.cat([senti_ctx, entity_pairs, size_pair_embeddings], dim=2)
        senti_repr = self.dropout(senti_repr)
        chunk_senti_logits = self.senti_classifier(senti_repr)
        return chunk_senti_logits

    def log_sample_total(self,neg_entity_count_all):
        log_path = os.path.join('./log/', 'countSample.txt')
        with open(log_path, mode='a', encoding='utf-8') as f:
            f.write('neg_entity_count_all: \n')
            self.neg_span_all += len(neg_entity_count_all)
            f.write(str(self.neg_span_all))
            f.write('\nneg_entity_count: \n')
            self.neg_span += len((neg_entity_count_all !=0).nonzero())
            f.write(str(self.neg_span))
            f.write('\n')
        f.close()

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()
        batch_sentiments = []
        batch_senti_masks = []
        batch_senti_sample_masks = []

        for i in range(batch_size):
            rels = []
            senti_masks = []
            sample_masks = []

            self.log_sample_total(entity_logits_max[i])
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        senti_masks.append(sampling.create_senti_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                batch_sentiments.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_senti_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_senti_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                batch_sentiments.append(torch.tensor(rels, dtype=torch.long))
                batch_senti_masks.append(torch.stack(senti_masks))
                batch_senti_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        device = self.senti_classifier.weight.device
        batch_sentiments = util.padded_stack(batch_sentiments).to(device)
        batch_senti_masks = util.padded_stack(batch_senti_masks).to(device)
        batch_senti_sample_masks = util.padded_stack(batch_senti_sample_masks).to(device)

        return batch_sentiments, batch_senti_masks, batch_senti_sample_masks

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)

def compute_loss(p, k):

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(k, dim=-1), reduction="none")
    k_loss = F.kl_div(F.log_softmax(k, dim=-1), F.softmax(p, dim=-1), reduction="none")

    p_loss = p_loss.sum()
    k_loss = k_loss.sum()
    total_loss = torch.log(1 + 5 / (torch.abs((p_loss + k_loss) / 2)))

    return total_loss


_MODELS = {
    'SESS': SESS,
}


def get_model(name):
    return _MODELS[name]