import tensorflow as tf

"""
One Model for both Actor and Critic to speed up training.
"""
class ActorCritic(tf.keras.Model):
  def __init__(self, num_actions, num_hidden_units, learning_rate):
    super().__init__()

    # Shared layer for both Actor and Critic, with a rectivied linear unit
    # activation function for non-linearity (f(x) = max(0, x)).
    self.common = tf.keras.layers.Dense(num_hidden_units, activation="relu")

    # The actor is the policy function π.  It's the raw prediction values
    # (logits) for each action given a state, and these predictions will be
    # converted to a probability distribution via softmax.
    self.actor = tf.keras.layers.Dense(num_actions)

    # The critic is the value function V.  It estimates the value (total reward)
    # of a state if π is followed.
    self.critic = tf.keras.layers.Dense(1)

    # For the Critic loss (see compute_loss).
    self.critic_loss_func = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    # For applying gradients.
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  """
  Forward pass.
  """
  def call(self, state):
    common_outputs = self.common(state)

    return self.actor(common_outputs), self.critic(common_outputs)

  """
  Only one model is used, so the Actor and Critic loss are combined by adding
  them together.

  For the Actor loss, the Advantage is calculated.  It's the standardized
  returns less the values from the Critic, G(s_t,a_t) - V(s_t), which describes
  how much better it is to take an action from a given state than taking a
  random action.  More detail can be found, here:
  https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#advantage-functions
  The Actor loss is then the log of the action probabilities multiplied by the
  advantange, summed:

  L_actor = -Σ(log[π(a_t|s_t)] * [G(s_t,a_t) - V(s_t)])

  Since the Actor and Critic losses are combined, the Actor loss is made
  negative. This maximizes the probabilities of choosing the actions with the
  highest rewards by minimizing combined loss.

  More detail can be found in this lenghty lecture (the timestamp shows the
  formula): https://www.youtube.com/watch?v=EKqxumCuAAY&t=3743s

  The Critic loss is simpler.  The Critic is trained to be as closs as possible
  to the returns, G.  Huber loss is used, where values is the "expected" and
  returns is the "actual" (or "predicted" and "true").
  """
  def compute_loss(self, action_probs, values, returns):
    actor_loss = -tf.reduce_sum(tf.math.log(action_probs) * (returns - values))
    critic_loss = self.critic_loss_func(returns, values)

    return actor_loss + critic_loss

  """
  Apply gradients to the model's trainable parameters using the optimizer.
  """
  def apply_gradients(self, gradients):
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
