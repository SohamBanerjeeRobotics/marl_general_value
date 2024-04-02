import jax
import jax.numpy as jnp
import equinox as eqx

def huber(x):
    return jax.lax.select(
        jnp.abs(x) < 1.0,
        0.5 * x ** 2,
        jnp.abs(x) - 0.5 
    )

def soft_update(network, target, tau):
    """Update the parameters of a target net using Polyak averaging."""
    def polyak(param, target_param):
        return target_param * (1 - tau) + param * tau

    params, _ = eqx.partition(network, eqx.is_inexact_array)
    target_params, static = eqx.partition(target, eqx.is_inexact_array)
    updated_params = jax.tree_map(polyak, params, target_params)
    target = eqx.combine(static, updated_params)
    target = eqx.tree_inference(target, True)
    return target

def critic_loss(q_network, q_target, policy, tape, gamma, noise_scale, key):
    """DDPG critic loss"""
    q_value = q_network(
        tape["observation"], tape["action"], key=key
    )
    action = policy(tape['next_observation'], noise_scale, key=key)

    next_q = jax.lax.stop_gradient(q_target(
        tape["next_observation"], action, key=key
    ))

    target = tape["next_reward"] + (1.0 - tape["next_done"]) * gamma * next_q 
    error = q_value - target
    [td_error] = huber(error)
    return td_error, td_error


def general_critic_loss(q_network, q_target, data, gamma, key):
    """critic loss"""
    q_value = q_network(
        data["state"], data["task_embedding"], key=key
    )[data["action"].squeeze(0)]

    next_q = jax.lax.stop_gradient(q_target(
        data["next_state"], data["task_embedding"], key=key
    )).max()

    target = data["next_reward"] + (1.0 - data["next_done"]) * gamma * next_q 
    error = q_value - target
    [td_error] = huber(error)
    return td_error, td_error

def dqn_ensemble_loss(q_network, q_target, tape, gamma, noise_scale, key):
    """Q Loss for a discrete Q function"""
    q_value = q_network(tape["state"], key=key)
    # Argmax over the action dim, but not ensemble dim
    next_q = jax.lax.stop_gradient(q_target(
        tape["next_state"], key=key
    )).max(axis=-1)

    target = tape["next_reward"] + (1.0 - tape["next_done"]) * gamma * next_q 
    error = q_value - target
    [td_error] = huber(error)
    return td_error, td_error

def policy_loss(policy, q_network, tape, key):
    """DDPG actor loss"""
    actions = policy(tape['observation'], noise_scale=jnp.array(0), key=key)
    [q_value] = q_network(tape['observation'], actions)
    return -q_value

def mean_reduce(fn, *args, **kwargs):
    """Given a tree of gradients produced by fn with dims [Batch, Agent, Params],
    reduce the gradient tree via mean to [Params].
    
    fn should be wrapped in eqx.filter_value_and_grad or jax.value_and_grad
    """
    outputs, grad = fn(*args, **kwargs)
    reduced_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=(0,1)), grad)
    return outputs, reduced_grad

def update_qnet(q_network, q_target, tape, opt, opt_state, gamma, tau, key):
    """Updates the discrete Q network. This function will vmap over the agent dimension,
    assuming all agents follow the same q function/policy."""
    loss = eqx.filter_value_and_grad(critic_loss, has_aux=True)
    B = tape['next_reward'].shape[0]
    A  = tape['next_reward'].shape[1]
    keys = jax.random.split(key, B * A).reshape(B, A, -1)
    marl_loss = eqx.filter_vmap(loss, in_axes=(None, None, None, 0, None, None, 0)) 
    batch_loss = eqx.filter_vmap(marl_loss, in_axes=(None, None, None, 0, None, None, 0)) 
    (value, td_error), grad = mean_reduce(batch_loss, q_network, q_target, tape, gamma, keys)
    updates, opt_state = opt.update(
        grad, opt_state, params=eqx.filter(q_network, eqx.is_inexact_array)
    )
    q_network = eqx.apply_updates(q_network, updates)
    q_target = soft_update(q_network, q_target, tau=tau)
    return q_network, td_error, value


def update_general_qnet(q_network, q_target, data, opt, opt_state, gamma, tau, key):
    """Updates the discrete Q network. This function will vmap over the task and batch dims."""
    loss_fn = eqx.filter_value_and_grad(general_critic_loss, has_aux=True)
    B = data['next_reward'].shape[0]
    A  = data['next_reward'].shape[1]
    keys = jax.random.split(key, B * A).reshape(B, A, -1)
    # loss_fn args: q_network, q_target, data, gamma, key
    task_loss_fn = eqx.filter_vmap(loss_fn, in_axes=(None, None, 0, None, 0)) 
    batch_loss_fn = eqx.filter_vmap(task_loss_fn, in_axes=(None, None, 0, None, 0))  
    (value, td_error), grad = mean_reduce(batch_loss_fn, q_network, q_target, data, gamma, keys)
    updates, opt_state = opt.update(
        grad, opt_state, params=eqx.filter(q_network, eqx.is_inexact_array)
    )
    q_network = eqx.apply_updates(q_network, updates)
    q_target = soft_update(q_network, q_target, tau=tau)
    return q_network, q_target, td_error, value
    


def update_critic(q_network, q_target, policy, tape, opt, opt_state, gamma, noise_scale, tau, key):
    """Updates the critic. This function will vmap over the agent dimension,
    assuming all agents follow the same q function/policy."""
    loss = eqx.filter_value_and_grad(critic_loss, has_aux=True)
    B = tape['next_reward'].shape[0]
    A  = tape['next_reward'].shape[1]
    keys = jax.random.split(key, B * A).reshape(B, A, -1)
    marl_loss = eqx.filter_vmap(loss, in_axes=(None, None, None, 0, None, None, 0)) 
    batch_loss = eqx.filter_vmap(marl_loss, in_axes=(None, None, None, 0, None, None, 0)) 
    (value, td_error), grad = mean_reduce(batch_loss, q_network, q_target, policy, tape, gamma, noise_scale, keys)
    updates, opt_state = opt.update(
        grad, opt_state, params=eqx.filter(q_network, eqx.is_inexact_array)
    )
    q_network = eqx.apply_updates(q_network, updates)
    q_target = soft_update(q_network, q_target, tau=tau)
    return q_network, td_error, value

def update_actor(policy, q_network, tape, opt, opt_state, noise_scale, key):
    """Updates the policy. This function will vmap over the agent dimension,
    assuming all agents follow the same q function/policy."""
    loss = eqx.filter_value_and_grad(policy_loss)
    B = tape['next_reward'].shape[0]
    A  = tape['next_reward'].shape[1]
    keys = jax.random.split(key, B * A).reshape(B, A, -1)
    marl_loss = eqx.filter_vmap(loss, in_axes=(None, None, 0, 0)) # Not random!
    batch_loss = eqx.filter_vmap(marl_loss, in_axes=(None, None, 0, 0)) # Not random!
    value, grad = mean_reduce(batch_loss, policy, q_network, tape, keys)
    updates, opt_state = opt.update(
        grad, opt_state, params=eqx.filter(policy, eqx.is_inexact_array)
    )
    policy = eqx.apply_updates(policy, updates)
    return policy, value