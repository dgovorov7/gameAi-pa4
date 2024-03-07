# gameAi-pa4


Implementation of decision transfomers for [VectorRTS](https://github.com/drchangliu/RL4SE/tree/main/enn/TensorRTS)

Currently alot of the code is based off of this [notebook](https://github.com/huggingface/blog/blob/main/notebooks/101_train-decision-transformers.ipynb) and using many of the huggingface transfomers libraries.
Alot of my implemenation feels very hacky, be it because huggingface's libraries arent designed with reinforcment learning in mind, and the unfamilirization I have with these librares. At some point I plan to dig through the code from the original [decison transformers gitub](https://github.com/kzl/decision-transformer) and other papers that build apon this.


### Running DTbot

On top of entity-gym the following two libraries are necessary ``` transformers pytorch ```

The DTbot folder should have everything needed to run the agent for the tournament runner, but the ```MODEL_PATH``` variable inside agent.py may need to be changed

The following code sets up the environment and runs DTbot against a rush opponent:

```
conda create -n huggingface python=3.8  
conda activate
conda install pip
python -m pip install entity_gym   
conda install transformers pytorch
python test_agent.py
```

### Training DTbot

Needs a lot of work, much of is currently in trainingPlayground.ipynb

The training works by first generating a dataset of games, logging every observation, action, and reward for each game state. This dataset is then feed into huggingface's trainer library which trains the decision transfomer. The current model uses a dataset of 1000 games of a random bot vs a rush bot. The current preformance of the bot is very poor, but it hopfully can be improved by using a better dataset (possibly using someones ENN implementation) as by nature transformers are an imitation learning algorithm.

[wandb run](https://wandb.ai/dgovorov7/VectorRTS-transformer/runs/iynzvft8?workspace=user-dgovorov7) as stated before huggingface's transformer trainer library is not made with reinforcement learning in mind so getting an evaluation during training is tricky
