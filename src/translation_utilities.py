import numpy as np
import transformers
import torch


class PartiallyFrozenEmbeddings(torch.nn.Module):
    # Inspired by https://stackoverflow.com/questions/54924582/is-it-possible-to-freeze-only-certain-embedding-weights-in-the-embedding-layer-i/54952825#54952825
    def __init__(self, old_embeddings: torch.nn.Embedding, frozen_part: int):
        super().__init__()
        self.pretrained_embeddings = torch.nn.Embedding.from_pretrained(old_embeddings.weight[:frozen_part],
                                                                       padding_idx=frozen_part-1, freeze=True)
        self.trainable_embeddings = torch.nn.Embedding.from_pretrained(old_embeddings.weight[frozen_part:], freeze=False)

        self.register_parameter('shared', None)
        
    def forward(self, batch):
        # Which tokens in batch do not have representation, should have indices BIGGER
        # than the pretrained ones, adjust your data creating function accordingly
        mask = batch >= self.pretrained_embeddings.num_embeddings
    
        # You may want to optimize it, you could probably get away without copy, though
        # I'm not currently sure how
        pretrained_batch = torch.clone(batch)
        pretrained_batch[mask] = 0
    
        embedded_batch = self.pretrained_embeddings(pretrained_batch)
    
        # Every token without representation has to be brought into appropriate range
        batch -= self.pretrained_embeddings.num_embeddings
        # Zero out the ones which already have pretrained embedding
        batch[~mask] = 0
        non_pretrained_embedded_batch = self.trainable_embeddings(batch)
    
        # And finally change appropriate tokens from placeholder embeddings created by
        # pretrained into trainable embeddings.
        embedded_batch[mask] = non_pretrained_embedded_batch[mask]

        return embedded_batch


class PartiallyFrozenLinear(torch.nn.Module):
    
    def __init__(self, embeddings: PartiallyFrozenEmbeddings):
        super().__init__()
        
        emb_dim = embeddings.pretrained_embeddings.embedding_dim
        pretrained_num = embeddings.pretrained_embeddings.num_embeddings
        trainable_num = embeddings.trainable_embeddings.num_embeddings
        
        self.pretrained_linear = torch.nn.Linear(in_features=emb_dim, out_features=pretrained_num, bias=False)
        self.trainable_linear = torch.nn.Linear(in_features=emb_dim, out_features=trainable_num, bias=False)

        self.pretrained_linear.weight = embeddings.pretrained_embeddings.weight
        self.trainable_linear.weight = embeddings.trainable_embeddings.weight
        
        self.register_parameter('lm_head', None)
        
    def forward(self, input: torch.Tensor):
        
        output_pretrained = self.pretrained_linear.forward(input)
        output_trainable = self.trainable_linear.forward(input)
        
        return torch.cat((output_pretrained, output_trainable), dim=-1)


def partially_freeze_embeddings(model, pad_token_id):
    old_embeddings = model.base_model.shared
    new_embeddings = PartiallyFrozenEmbeddings(old_embeddings, pad_token_id + 1)
    
    model.base_model.shared = new_embeddings
    model.base_model.encoder.set_input_embeddings(new_embeddings)
    model.base_model.decoder.set_input_embeddings(new_embeddings)
    model.lm_head = PartiallyFrozenLinear(new_embeddings)
    
    
def restore_model_structure(model):
    old_embeddings = model.base_model.shared
    new_weight = torch.cat((old_embeddings.pretrained_embeddings.weight,
                            old_embeddings.trainable_embeddings.weight), dim=0)
    new_embeddings=torch.nn.Embedding.from_pretrained(new_weight,
                                                      padding_idx=old_embeddings.pretrained_embeddings.num_embeddings-1)
    new_lm_head = torch.nn.Linear(in_features=new_embeddings.embedding_dim,
                                  out_features=new_embeddings.num_embeddings, bias=False)
    new_lm_head.weight = new_embeddings.weight
    
    model.base_model.shared = new_embeddings
    model.base_model.encoder.set_input_embeddings(new_embeddings)
    model.base_model.decoder.set_input_embeddings(new_embeddings)
    model.lm_head = new_lm_head


def tie_partially_frozen_weights(embeddings: PartiallyFrozenEmbeddings,lm_head: PartiallyFrozenLinear):
    
        lm_head.pretrained_linear.weight = embeddings.pretrained_embeddings.weight
        lm_head.trainable_linear.weight = embeddings.trainable_embeddings.weight