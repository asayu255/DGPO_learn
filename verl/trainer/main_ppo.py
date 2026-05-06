# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """
    #初期化とプロパティ
    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        # トークナイザー（数値をテキストに戻すための辞書）
        self.tokenizer = tokenizer
        # コンソールにデバッグ表示（print）するデータ件数の上限
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # 形式に関するデフォルトのスコア（フォーマットが正しいだけで与える点数など）
        self.format_score = format_score

        # 以下は、WandBなどのログに記録するための統計データを貯める空のリスト
        self.search_count_lst = []         # 検索を行った回数
        self.search_presence_lst = []      # 検索を1回でも行ったか（0 or 1）
        self.info_contains_answer_lst = [] # 検索結果の中に「正解」が含まれていたか（0 or 1）

    @property
    def extra_metrics(self):
        # 上記で貯めたリストを辞書形式で返すプロパティ。学習フレームワークがログを取るために呼び出します。
        return {
            "search_count": self.search_count_lst,
            "search_present": self.search_presence_lst,
            "info_contains_answer": self.info_contains_answer_lst,
        }
    
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # 報酬を格納するための空のテンソル（行列）を作成。
        # サイズはモデルの出力（responses）と同じで、初期値はすべて 0.0。
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        # どのデータセット（nq, triviaqaなど）を何回printしたかを記録する辞書
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts'] # 問題文のトークンID（数値の配列）を取得
            prompt_length = prompt_ids.shape[-1]    # 問題文の最大長を取得

            # attention_mask（有効な文字が1、パディングが0の配列）を使って、本当の文字数を計算
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # パディングを除外した、純粋な問題文のトークンIDだけを切り出す
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] # 回答のトークンIDを取得
            # 回答部分の本当の文字数を計算
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            # パディングを除外した、純粋な回答のトークンIDだけを切り出す
            valid_response_ids = response_ids[:valid_response_length]

            # 問題文と回答のトークンIDを連結する
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            # トークナイザーを使って、数値の配列を人間が読める文字列（テキスト）に戻す
            sequences_str = self.tokenizer.decode(sequences)

            # メタデータから、この問題の「本当の正解（Ground Truth）」を取得
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # データセット名（nq, triviaqaなど）を取得
            data_source = data_item.non_tensor_batch['data_source']
            # データセットに応じた採点関数（完全一致を見る qa_em など）を選択
            compute_score_fn = _select_rm_score_fn(data_source)

            # 採点関数を実行し、スコア（報酬値）を算出
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            solution_str = sequences_str # 変数名を見やすくするために代入
            targets = ground_truth.get('target', []) # 正解の文字列リストを取得
            
            # 正解が単一の文字列だった場合、リスト形式に変換する（後の処理でエラーにならないように）
            if isinstance(targets, str):
                targets = [targets]

            # テキスト内に "<search>" というタグが何回出現したかカウントする
            # （-1している理由は、プロンプトの初期指示文などにデフォルトで1つ含まれているためだと思われます）
            search_count = solution_str.count('<search>')-1
            self.search_count_lst.append(search_count) # ログ用リストに追加
            self.search_presence_lst.append(1 if search_count > 0 else 0) # 1回以上検索したら 1、そうでなければ 0

            found = 0
            # 正規表現を使って、"<information>内容</information>" に囲まれた検索結果テキストをすべて抽出する
            for match in re.finditer(r'<information[^>]*>(.*?)</information>', solution_str, flags=re.DOTALL | re.IGNORECASE):
                text = match.group(1) # タグの中身（抽出された検索結果）
                # 検索結果テキストの中に、正解（targets）が含まれているかチェック
                if any(t.strip() and t in text for t in targets):
                    found = 1 # 見つかったらフラグを1にする
                    break
            self.info_contains_answer_lst.append(found) # ログ用リストに追加

            
            # 【重要】計算したスコアを、回答の「最後のトークン」の位置にセットする
            # （強化学習では、一連の行動が終わった最後に報酬を与えるのが一般的なため）
            reward_tensor[i, valid_response_length - 1] = score

            # このデータセットのprint回数カウンターを初期化
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # 上限（num_examine）に達していなければ、生成されたテキストをコンソールに出力してデバッグしやすくする
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # 最終的に、バッチ全員分の報酬が詰まったテンソル（行列）を返して終了！
        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
