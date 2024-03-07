from TensorRTS import Agent, GameResult, Interactive_TensorRTS, GameRunner
from entity_gym.env import Observation
from entity_gym.env import *
from typing import Dict, List, Mapping, Tuple, Set
# from tournament_runner import Bot, Matchup, Bracket, Tournament
from transformers import DecisionTransformerModel
import torch
import numpy as np

import functools

MODEL_PATH = "./DTbot/model-4"


def flatten_dict_of_arrays(data, prefix=""):
  flattened_list = []
  def flatten_inner(value, key):
    if isinstance(value, list):
      for i, v in enumerate(value):
        flatten_inner(v, f"{prefix}{key}[{i}]")
    else:
      flattened_list.append((f"{prefix}{key}", value))

  for key, value in data.items():
    flatten_inner(value, key)

  return [value for _, value in flattened_list]


def value_to_discrete_4(input):
    output = [0,0,0,0]
    output[input] = 1
    return output

def get_move_no_lookback(model,device,game_state:Observation):
    state = np.array(flatten_dict_of_arrays(game_state.features))

    state_dim = 20
    act_dim = 4
    states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
    target_return = torch.tensor(10, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)


    with torch.no_grad():
        state_preds, action_preds, return_preds = model(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=target_return,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
            # use_mps_device=True

        )
    return action_preds

class DTbot(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> None: 
        super().__init__(init_observation, action_space)
        self.model = DecisionTransformerModel.from_pretrained(MODEL_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]:
        possible_moves = {
            0 : "advance",
            1 : "retreat",
            2 : "rush",
            3 : "boom"
        }
        action = {}

        current_game_state.features
        model_out = get_move_no_lookback(self.model,self.device,current_game_state)
        action['Move'] = GlobalCategoricalAction(0, possible_moves[torch.argmax(model_out).item()])
        
        return action
    
    def on_game_start(self) -> None:
        return super().on_game_start()
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)

def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> Agent: 
    """Creates an agent of this type

    Returns:
        Agent: _description_
    """
    return DTbot(init_observation, action_space)

def student_name_hook() -> str: 
    """Provide the name of the student as a string

    Returns:
        str: Name of student
    """
    return 'Denis Govorov'