from TensorRTS import Agent, GameResult, Interactive_TensorRTS, GameRunner
from entity_gym.env import Observation
from entity_gym.env import *
from typing import Dict, List, Mapping, Tuple, Set
# from tournament_runner import Bot, Matchup, Bracket, Tournament


from DTbot import agent as DTagent




class Rush_Agent_Player_2(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> None: 
        super().__init__(init_observation, action_space)

    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]:
        mapping = {}


        if current_game_state.features['Tensor'][1][2] > 0 :
            #rush
            mapping['Move'] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[1])
        else:
            #advance
            mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[2])

        return mapping
    
    def on_game_start(self) -> None:
        return super().on_game_start()
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    


if __name__ == "__main__":
    game_runner = GameRunner(enable_printouts=True)

    observation = game_runner.game.observe()
    action_space = game_runner.game.action_space()

    # dTbot = Bot("./DTbot/agent.py")
    player1 = DTagent.DTbot(observation,action_space)
    # player1 = dTbot.create_instance(observation,action_space)

    # player1 = Rush_Agent_Player_1(observation,action_space)

    player2 = Rush_Agent_Player_2(observation,action_space)

    game_runner.assign_players(player1, player2)


    game_runner.run()


    # env = TensorRTS()
    # # The `CliRunner` can run any environment with a command line interface.
    # CliRunner(env).run()    