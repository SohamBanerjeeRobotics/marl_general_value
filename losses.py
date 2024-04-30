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

def general_critic_loss(q_network, q_target, data, gamma, key):
    """critic loss"""
    q_value = q_network(
        data["state"], data["task_embedding"], key=key
    )
    taken_q_value = q_value[data["action"]].squeeze(0)

    next_q = jax.lax.stop_gradient(q_target(
        data["next_state"], data["task_embedding"], key=key
    )).mean()

    target = data["next_reward"] + (1.0 - data["next_done"]) * gamma * next_q 
    error = taken_q_value - target.squeeze(0)
    td_error = huber(error)
    return td_error, (td_error, taken_q_value, next_q)

def general_cql_loss(q_network, q_target, data, gamma, key):
    """critic loss"""
    q_value = q_network(
        data["state"], data["task_embedding"], key=key
    )
    taken_q_value = q_value[data["action"]].squeeze(0)

    next_action = jax.lax.stop_gradient(q_network(
        data["next_state"], data["task_embedding"], key=key
    ).argmax())

    next_q = jax.lax.stop_gradient(q_target(
        data["next_state"], data["task_embedding"], key=key
    ))[next_action]

#    uniform_weighting = 0.8 * (jnp.ones((num_actions,)) / num_actions)
#    greedy_weighting = jnp.zeros((num_actions,)).at[next_action].set(0.2)
#    weighting = uniform_weighting + greedy_weighting
#
#    next_q = jnp.sum(next_q * weighting)
       
    target = data["next_reward"] + (1.0 - data["next_done"]) * gamma * next_q 
    error = taken_q_value - target.squeeze(0)
    cql = jax.nn.logsumexp(q_value) - taken_q_value
    td_error = huber(error) + 0.1 * cql
    return td_error, (td_error, taken_q_value, next_q)


def general_critic_loss_simple(q_network, q_target, data, gamma, key):
    """critic loss"""
    q_value = q_network(
        data["state"], data["task_embedding"], key=key
    )[data["action"]]

    next_q = jax.lax.stop_gradient(q_target(
        data["next_state"], data["task_embedding"], key=key
    )).max()

    target = data["next_reward"] + (1.0 - data["next_done"]) * gamma * next_q 
    error = q_value - target
    [td_error] = huber(error)
    return td_error, (td_error, q_value, next_q)

def mean_reduce(fn, *args, **kwargs):
    """Given a tree of gradients produced by fn with dims [Batch, Task, Params],
    reduce the gradient tree via mean to [Params].
    
    fn should be wrapped in eqx.filter_value_and_grad or jax.value_and_grad
    """
    outputs, grad = fn(*args, **kwargs)
    reduced_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=(0,1)), grad)
    return outputs, reduced_grad


def vmap_task(loss_fn):
    return eqx.filter_vmap(
        loss_fn, 
        in_axes=(
            # qnet
            None, 
            # qtarget
            None,
            # data, recall the shape is [Batch, Task, ...]
            {
                "action": None,
                "next_reward": 0,
                "next_done": 0,
                "next_state": None,
                "state": None,
                "task_embedding": 0,
            }, 
            # gamma
            None, 
            # key
            0,
        )
    ) 

def vmap_batch(loss_fn):
    return eqx.filter_vmap(
        loss_fn, 
        in_axes=(
            # qnet
            None, 
            # qtarget
            None,
            # data, recall the shape is [Batch, Task, ...]
            {
                "action": 0,
                "next_reward": 0,
                "next_done": 0,
                "next_state": 0,
                "state": 0,
                "task_embedding": None,
            }, 
            # gamma
            None, 
            # key
            0,
        )
    ) 

def update_general_qnet(q_network, q_target, data, opt, opt_state, gamma, tau, key):
    """Updates the discrete Q network. This function will vmap over the task and batch dims."""
    loss_fn = eqx.filter_value_and_grad(general_critic_loss, has_aux=True)
    B = data['next_reward'].shape[0]
    A  = data['next_reward'].shape[1]
    keys = jax.random.split(key, B * A).reshape(B, A, -1)

    # We keep the data with singleton dims to help understand which dims map to which axes
    # but vmap does not handle singleton dims well, so we remove them here
    # Squeeze out Task dims
    data = {
        k: v.squeeze(1) if k in ["state", "next_state", "action"] else v for k, v in data.items() 
    }
    # Squeeze out Batch dims
    data = {
        k: v.squeeze(0) if k in ["task_embedding"] else v for k, v in data.items()
    }

    batch_loss_fn = vmap_batch(vmap_task(loss_fn))
    outputs, grad = mean_reduce(batch_loss_fn, q_network, q_target, data, gamma, keys)
    _, (td_error, q_value, q_target_value) = outputs
    updates, opt_state = opt.update(
        grad, opt_state, params=eqx.filter(q_network, eqx.is_inexact_array)
    )
    q_network = eqx.apply_updates(q_network, updates)
    q_target = soft_update(q_network, q_target, tau=tau)
    return q_network, q_target, td_error, q_value, q_target_value


def update_general_qnet_simple(q_network, q_target, data, opt, opt_state, gamma, tau, key):
    """Updates the discrete Q network. This function will vmap over the task and batch dims."""
    loss_fn = eqx.filter_value_and_grad(general_critic_loss_simple, has_aux=True)
    B = data['next_reward'].shape[0]
    keys = jax.random.split(key, B)

    # We keep the data with singleton dims to help understand which dims map to which axes
    # but vmap does not handle singleton dims well, so we remove them here
    # Squeeze out Task dims

    batch_loss_fn = eqx.filter_vmap(loss_fn, in_axes=(None, None, 0, None, 0))
    outputs, grad = mean_reduce(batch_loss_fn, q_network, q_target, data, gamma, keys)
    _, (td_error, q_value, q_target_value) = outputs
    updates, opt_state = opt.update(
        grad, opt_state, params=eqx.filter(q_network, eqx.is_inexact_array)
    )
    q_network = eqx.apply_updates(q_network, updates)
    q_target = soft_update(q_network, q_target, tau=tau)
    return q_network, q_target, td_error, q_value, q_target_value
