import gymnasium as gym
import numpy as np
import tensorflow as tf
from actor_critic import ActorCritic
from collections import deque
from PIL import Image

# Number of hidden, shared neurons in the neural net.  This can be as low as
# 32, but is more stable with more neurons without much performance difference.
NUM_HIDDEN_UNITS = 512

# Usually denoted γ, the discount factor determines how much to favor future
# rewards over immediate ones.  A value of 0 makes the agent only consider
# immediate rewards (myopic).
GAMMA = 0.99

# Learning rate, used with the optimizer.
ALPHA = 0.01

# For preventing division by zero.
EPSILON = np.finfo(np.float32).eps.item()

# Render episodes during training every N episodes.
RENDER_FREQ = 1000

# When rendering an episode to a GIF, every Nth screen is rendered.
RENDER_SCREEN_FREQ = 4

# Gym environment, and the "success" criteria: average episode reward over
# some number of episodes.
GYM_ENV = "LunarLander-v3"
MEAN_REWARD_EPISODES = 100
MEAN_REWARD_TARGET = 200

"""
Run a single episode, and return action probabilities (actor), values (critic),
and rewards.
"""
def run_episode(env, model, initial_state):
  """
  Wrapper for env.step, wrapped so that it can be compiled into a TensorFlow
  graph.  This can only accept numpy arrays as parameters, hence the closure
  over env.
  """
  @tf.numpy_function(Tout=[tf.float32, tf.int32, tf.int32, tf.int32])
  def step(action):
    state, reward, done, truncated, _ = env.step(action)

    return (
      state.astype(np.float32),
      np.array(reward, np.int32),
      np.array(done, np.int32),
      np.array(truncated, np.int32)
    )

  # Episodes don't have a fixed length, so TensorArrays are used to allow for
  # training on variable-length data.
  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  write_ind = 0

  episode_over = tf.constant(False)
  state = initial_state
  state_shape = state.shape

  while not episode_over:
    # Forward passes are performed in batches, so the state must be expanded to
    # a batch of 1.
    state = tf.expand_dims(state, 0)

    # Actor policy (logits) and critic value.
    action_logits, value = model(state)

    # Select an action from the logits, then convert the logits to a probability
    # distribution.
    action = tf.random.categorical(action_logits, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits)

    state, reward, done, truncated = step(action)
    reward = reward - done

    # This gives TF details about the shape of the state Tensor at runtime,
    # since TF can't infer the shape on its own.
    state = tf.ensure_shape(state, state_shape)

    # Store the probability of the chosen action, the value squeezed down a
    # dimension (batch size of one; inverse of tf.expand_dims), and
    # the reward.
    action_probs = action_probs.write(write_ind, action_probs_t[0, action])
    values = values.write(write_ind, tf.squeeze(value))
    rewards = rewards.write(write_ind, reward)
    write_ind += 1

    episode_over = tf.cast(done, tf.bool) or tf.cast(truncated, tf.bool)
    episode_over = tf.ensure_shape(episode_over, ())

  # All of these become 1D.  They'll be used for computing loss.
  return action_probs.stack(), values.stack(), rewards.stack()

"""
Takes a TensorArray of rewards (typically denoted R) from an episode, one reward
per timestep, and converts them to returns (typically G).  A return for a
timestep is the reward for that timestep plus all future rewards.

Future rewards are discounted by a discount factor (denoted γ) to prioritize
current rewards over future ones, and to ensure convergence.

The returns are standardize using Z-score normalization
(https://en.wikipedia.org/wiki/Standard_score) such that they have a 0 mean and
standard deviation of 1.  This stabilizes training.
"""
def get_returns(rewards):
  num_rewards = tf.size(rewards)
  rewards = tf.cast(rewards, tf.float32)
  returns = tf.TensorArray(dtype=tf.float32, size=num_rewards)
  accumulated_reward = tf.constant(0.0)
  return_shape = accumulated_reward.shape

  for t in tf.range(num_rewards):
    # From the last reward to the first, sum up rewards into an accumulator.
    index = num_rewards - 1 - t
    accumulated_reward = rewards[index] + GAMMA * accumulated_reward
    accumulated_reward = tf.ensure_shape(accumulated_reward, return_shape)

    # Store the accumulated return (from the end to the beginning).
    returns = returns.write(index, accumulated_reward)

  returns = returns.stack()

  # Standardize: z = (x - μ) / σ, where x is the returns, μ is the mean of the
  # returns, and σ is the standard deviation of the returns.
  mean = tf.math.reduce_mean(returns)
  std_deviation = tf.math.reduce_std(returns)
  returns = (returns - mean) / tf.math.maximum(std_deviation, EPSILON)

  return returns

"""
Run one episode and train the model on the results.
"""
@tf.function
def train_step(env, model, initial_state):
  # Handles automatic differentiation.
  with tf.GradientTape() as tape:
    action_probs, values, rewards = run_episode(env, model, initial_state)
    returns = get_returns(rewards)

    # Batch for computing loss.
    action_probs = tf.expand_dims(action_probs, 1)
    values = tf.expand_dims(values, 1)
    returns = tf.expand_dims(returns, 1)

    loss = model.compute_loss(action_probs, values, returns)

  # Compute the gradient from the loss and apply it to the model.
  gradients = tape.gradient(loss, model.trainable_variables)
  model.apply_gradients(gradients)

  # Return the total score for the episode.
  return tf.reduce_sum(rewards)

"""
Run an episode and render it to an animated GIF.
"""
def render_episode(env, model, image_file):
  state, _ = env.reset()
  images = [Image.fromarray(env.render())]
  total_reward = 0
  episode_over = False
  episode_num = 0

  while not episode_over:
    episode_num += 1

    action_logits, _ = model(tf.expand_dims(state, 0))
    action = np.argmax(np.squeeze(action_logits))
    state, reward, done, truncated, _ = env.step(action)

    total_reward += reward
    episode_over = done or truncated

    if episode_num % RENDER_SCREEN_FREQ == 0:
      images.append(Image.fromarray(env.render()))

  # Save to an animated GIF that loops with 1ms between frames.
  images[0].save(
    image_file,
    save_all=True,
    append_images=images[1:],
    loop=0,
    duration=1
  )

  return total_reward

"""
Entrypoint.  Train the model, and render the result.
"""
def main():
  env = gym.make(GYM_ENV)
  render_env = gym.make(GYM_ENV, render_mode="rgb_array")
  model = ActorCritic(env.action_space.n, NUM_HIDDEN_UNITS, ALPHA)
  episode_rewards = deque(maxlen=MEAN_REWARD_EPISODES)
  episode_num = 0
  mean_reward = 0
  done = False

  while not done:
    episode_num += 1
    state, _ = env.reset()

    episode_reward = int(train_step(env, model, state))
    episode_rewards.append(episode_reward)
    mean_reward = np.mean(episode_rewards)

    done = mean_reward >= MEAN_REWARD_TARGET

    if episode_num % MEAN_REWARD_EPISODES == 0 or done:
      print(f"Episode {episode_num} mean reward: {mean_reward}")

    # Every once in a while, render an animated gif of an episode.
    if episode_num % RENDER_FREQ == 0 or done:
      image_file = f"../data/{GYM_ENV}-{episode_num}.gif"
      rendered_episode_reward = render_episode(render_env, model, image_file)
      print(f"Rendered episode {episode_num} to {image_file}. Reward: {rendered_episode_reward}")

if __name__ == "__main__":
  main()
