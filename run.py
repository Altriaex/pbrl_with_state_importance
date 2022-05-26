'''
Descripttion: 
Author: CoeusZhang
Date: 2022-02-22 09:57:22
LastEditTime: 2022-05-26 17:46:20
'''
import os

lasso_bt3_l1 = {"BeamRider": 0.01, "Enduro": 0.01, "Breakout": 0.01,
      "Qbert": 0.01, "Hero": 0.001, "Pong": 0.01,
      "Seaquest": 0.01, "Alien": 0.1, "PrivateEye": 0.001,
      "Asterix": 0.1, "Boxing": 0.01, "Amidar": 0.1, "Assault": 0.01,"RoadRunner":0.1, "BattleZone": 0.001}
lasso_bt3_l2 = {"BeamRider": 0.01, "Enduro": 0.01, "Breakout": 0.001,
      "Qbert": 0.1, "Hero": 0.1, "Pong": 0.001,
      "Seaquest": 0.001, "Alien": 0.001, "PrivateEye": 0.001,
      "Asterix": 0.001, "Boxing": 0.01, "Amidar": 0.1, "Assault": 0.1,"RoadRunner":0.001, "BattleZone": 0.01}

base_str = '''python -um crowd_pbrl.main --exp_path G:\crowd_pbrl\experiments\\trajectory_ranking_expert  --gpu_id 0'''

for game in ["Enduro"]:
      for task in ["sample_trajectories", 'sample_queries', 'generate_bt_answers', 'generate_trex_answers']:
            for split in ["1", "2", "3", "4","5"]:
                  for method in ["bt"]:
                        cmd_str = base_str +  f''' --split {split} --game {game} --task {task} --method {method}'''
                        print("******")
                        print(cmd_str)
                        os.system(cmd_str)   
for game in ["Enduro"]:
      for task in ["train_reward_model", "infer_rewards"]:
            for split in ["1", "2", "3", "4","5"]:
                  for method in ["bt", "lasso_bt3", "trex"]:
                        cmd_str = base_str +  f''' --split {split} --game {game} --task {task} --method {method}'''
                        if method == 'lasso_bt3':
                              cmd_str += f''' --l1_weight {lasso_bt3_l1[game]} --temporal_l2_weight {lasso_bt3_l2[game]}'''
                        print("******")
                        print(cmd_str)
                        os.system(cmd_str)   