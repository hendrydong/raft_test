



#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""Alignment tuning example, such as RLHF."""

import logging
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import HfArgumentParser, pipeline, AutoTokenizer

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline

import warnings

@dataclass
class RewardArguments:
    reward_type: Optional[str] = field(
        default="hf_pipeline",
        metadata={
            "help": (
                "type of reward model, support huggingface pipeline. Will"
                " support \"customized\" torch.nn.modules in the future."
            ),
        },
    )
    reward_model_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "reward model name (huggingface) or its path"
            ),
        },
    )
    reward_task: Optional[str] = field(
        default="sentiment-analysis",
        metadata={
            "help": "type of reward task, such as sentiment-analysis, detoxic."
        },
    )
    reward_model_args: Optional[str] = field(
        default="return_all_scores=True, function_to_apply=\"none\", batch_size=1",
        metadata={
            "help": (
                "extra arguments required by different type of reward models."
            ),
        },
    )



from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
import torch.nn as nn
import torch
from typing import Optional, List

class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward( # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        transformer_outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,                               
                            )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)
        
        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
        rewards = torch.gather(rewards, 1, ends)
        
        return rewards
    




def get_reward_function(reward_args, pipeline_args):
    if reward_args.reward_model_or_path is None:
        warnings.warn("No reward model is provided.")
        return None
    args = reward_args

    tokenizer = LlamaTokenizer.from_pretrained("openbmb/UltraRM-13b")
    model = LlamaRewardModel.from_pretrained("openbmb/UltraRM-13b").to(torch.bfloat16)
    model = model.to(f"cuda:{pipeline_args.local_rank}")

    rm_tokenizer = AutoTokenizer.from_pretrained("openbmb/UltraRM-13b")
  
    def reward_func(dataset: Dataset):
        if dataset.type != "text_only":
            raise NotImplementedError(
                "reward function only accept \"text_only\" datasets"
            )
        pipe_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": 1
        }   
        def get_reward(texts):
            scores = []
            for txt in texts:
                inputs = tokenizer(txt, return_tensors="pt").to(f"cuda:{pipeline_args.local_rank}")
                chosen_reward = model(**inputs).item()
                scores.append(chosen_reward)
            return scores

        data_dict = dataset.to_dict()
        texts_for_rewards = [
            sample["text"] for sample in data_dict["instances"]
        ]
        rewards = get_reward(texts_for_rewards)
            
            

        reward_dataset = Dataset.create_from_dict({
            "type": "float_only",
            "instances": [
                { "value": reward } for reward in rewards
            ]
        })
        return reward_dataset

    return reward_func



def main():
	# Parses arguments
    pipeline_name = "raft_aligner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        DatasetArguments,
        PipelineArguments,
        RewardArguments,
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, pipeline_args, reward_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, pipeline_args, reward_args = parser.parse_args_into_dataclasses()

    # Initializes pipeline, dataset and model for reward training
    aligner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)
    model = AutoModel.get_model(model_args)

    # Initializes reward function
    reward_function = get_reward_function(reward_args, pipeline_args)

    reward_model_args = ModelArguments(arch_type="text_regression")
    reward_model = AutoModel.get_model(reward_model_args)
    reward_model.register_inference_function(reward_function)

    # Aligns model with rewards
    aligned_model = aligner.align(
        model=model,
        dataset=dataset,
        reward_model=reward_model,
    )


if __name__ == '__main__':
    main()