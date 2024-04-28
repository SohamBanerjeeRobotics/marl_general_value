from tasks import make_global_navigation_tasks, make_navigation_goals_and_embeddings
import jax.numpy as jnp
import equinox as eqx
import optax
import tqdm
import pandas as pd
from modules import Block
import jax
from sentence_transformers import SentenceTransformer
#from angle_emb import AnglE

llms = {
    #AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to("cpu"): "WhereIsAI/UAE-Large-V1",
    #SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True): "Alibaba-NLP/gte-large-en-v1.5",
    #SentenceTransformer('paraphrase-MiniLM-L6-v2'): "paraphrase-MiniLM-L6-v2",
    #SentenceTransformer('all-MiniLM-L6-v2'): "all-MiniLM-L6-v2",
    #SentenceTransformer('all-distilroberta-v1'): "all-distilroberta-v1",
    SentenceTransformer('sentence-transformers/all-mpnet-base-v2'): "sentence-transformers/all-mpnet-base-v2",
    #SentenceTransformer('sentence-transformers/LaBSE'): "sentence-transformers/LaBSE",
    #SentenceTransformer('paraphrase-albert-small-v2'): "paraphrase-albert-small-v2",
    #SentenceTransformer('Supabase/gte-small'): "Supabase/gte-small",
    #SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=512): "mixedbread-ai/mxbai-embed-large-v1",
}


class Decoder(eqx.Module):
    l0: eqx.nn.Linear
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    l3: eqx.nn.Linear
    l4: eqx.nn.Linear

    def __init__(self, emb_size):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        hidden_size = 256
        self.l0 = Block(emb_size, hidden_size, 0, key=keys[0])
        self.l1 = Block(hidden_size, hidden_size, 0, key=keys[1])
        self.l2 = Block(hidden_size, hidden_size, 0, key=keys[2])
        self.l3 = Block(hidden_size, hidden_size, 0, key=keys[3])
        self.l4 = eqx.nn.Linear(hidden_size, 2, key=keys[4])

    def __call__(self, embed):
        x = self.l0(embed)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.l4(x)

    def to_embedding(self, embed):
        x = self.l0(embed)
        x = self.l1(x)
        x = self.l2(x)
        return self.l3(x)


def loss_fn(model, emb, goal):
    pred = model(emb) 
    return jnp.mean(jnp.abs(pred - goal))


results = {}
for llm, llm_name in llms.items():
    batch_size = 100
    epochs = 500
    key = jax.random.PRNGKey(0)
    key, model_key, data_key = jax.random.split(key, 3)

    train = make_navigation_goals_and_embeddings(llm, 3000)
    test = make_navigation_goals_and_embeddings(llm, 50)

    #  TODO: Should include velocity
    model = eqx.filter_jit(eqx.filter_vmap(Decoder(train["task_embedding"].shape[1])))

    lr_schedule = optax.constant_schedule(0.00001)
    opt = optax.adamw(lr_schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    pbar = tqdm.tqdm(total=epochs * train["task_embedding"].shape[0] // batch_size)
    best_val_loss = jnp.inf
    for epoch in range(epochs):
        for i in range(train["task_embedding"].shape[0] // batch_size):
            batch = {k: v[i * batch_size : (i + 1) * batch_size] for k, v in train.items()}
            emb = batch["task_embedding"]
            goal = batch["goal"]
            loss, grad = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model, emb, goal)
            updates, opt_state = eqx.filter_jit(opt.update)(
                grad, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
            )
            model = eqx.filter_jit(eqx.apply_updates)(model, updates)

            v_emb = test["task_embedding"]
            v_goal = test["goal"]
            val_loss = eqx.filter_jit(loss_fn)(model, v_emb, v_goal)
            best_val_loss = min(val_loss, best_val_loss)

            pbar.update()
            pbar.set_description(f"LLM: {llm_name}, epoch: {epoch}, loss: {loss:0.4f}, val_loss: {val_loss:0.4f}, best val_loss: {best_val_loss:0.4f}")
    results[llm_name] = best_val_loss

print(results)
