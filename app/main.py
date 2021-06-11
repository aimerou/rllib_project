from fastapi import FastAPI
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

app = FastAPI()

@app.get("/train_agent/")
async def train_agent():
    ray.init()
    trainer = PPOTrainer(config={
        "num_gpus": 0,
        "num_workers": 1
    }, env="CartPole-v0")
    result = trainer.train()
    return str(result)

@app.get("/multi_agent/")
async def multi_agent():
    return {}
