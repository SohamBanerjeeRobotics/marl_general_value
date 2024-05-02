from dataset import ACTION_IDX
from dynamics_model import StateTransitionModel
import equinox as eqx
import jax.numpy as jnp
import pygame
import jax
from constants import ARENA_BOUNDS_E, ARENA_BOUNDS_N


def to_screen(pos, borders):
    x_scale = WIDTH / (borders[1, 0] - borders[0, 0])
    y_scale = HEIGHT / (borders[1, 1] - borders[0, 1])
    print(x_scale, y_scale)
    screen_pos = pos * jnp.array([x_scale, y_scale]) + jnp.array([(WIDTH + W_PADDING) / 2, (HEIGHT + H_PADDING) / 2])
    return screen_pos.tolist()

model = StateTransitionModel(state_size=4, num_actions=9, dropout=0, key=jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves("data/dynamics_model_weights.eqx", model)
WIDTH = 600
HEIGHT = 600
W_PADDING = 0.2 * WIDTH
H_PADDING = 0.2 * HEIGHT
#borders = jnp.array([[-1.5, -1.0], [1.5, 1.0]])
borders = jnp.array([ARENA_BOUNDS_N, ARENA_BOUNDS_E]).T

border_vis = pygame.Rect(*to_screen(borders[0], borders), WIDTH, HEIGHT)
pygame.init()
screen = pygame.display.set_mode((WIDTH + W_PADDING, HEIGHT + H_PADDING))
clock = pygame.time.Clock()
running = True

agent_state = jnp.array([0.0, 0.0, 0, 0])
dt = 5

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        keys = pygame.key.get_pressed()

        if (keys[pygame.K_w] and keys[pygame.K_a]) or keys[pygame.K_z]:
            # TODO: Why is N/S inverted? 
            command = ACTION_IDX["NW"]
        elif (keys[pygame.K_w] and keys[pygame.K_d]) or keys[pygame.K_c]:
            command = ACTION_IDX["NE"]
        elif (keys[pygame.K_s] and keys[pygame.K_a]) or keys[pygame.K_q]:
            command = ACTION_IDX["SW"]
        elif (keys[pygame.K_s] and keys[pygame.K_d]) or keys[pygame.K_e]:
            command = ACTION_IDX["SE"]
        elif keys[pygame.K_s]:
            command = ACTION_IDX["S"]
        elif keys[pygame.K_w]:
            command = ACTION_IDX["N"]
        elif keys[pygame.K_d]:
            command = ACTION_IDX["E"]
        elif keys[pygame.K_a]:
            command = ACTION_IDX["W"]
        else:
            command = ACTION_IDX["0"]

    screen.fill((255, 255, 255))
    pygame.draw.rect(screen, "gray", border_vis)
    pygame.draw.circle(screen, "red", to_screen(agent_state[:2], borders), 20)
    pygame.display.flip()
    agent_state = model(agent_state, command)
    # Clamp to train domain
    agent_state = agent_state.at[0].set(jnp.clip(agent_state[0], borders[0,0], borders[1,0]))
    agent_state = agent_state.at[1].set(jnp.clip(agent_state[1], borders[0,1], borders[1,1]))
    clock.tick(dt)


for action, idx in ACTION_IDX.items():
    print(f"action {action}: {model(jnp.array([1.0, 1.0, 0, 0, 0]), idx)}")
for action, idx in ACTION_IDX.items():
    print(f"action {action} delta: {jnp.array([1.0, 1.0, 0, 0, 0]) - model(jnp.array([1.0, 1.0, 0, 0, 0]), idx)}")
