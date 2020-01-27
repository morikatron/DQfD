"""Deep Q model

The functions in this model:

======= step ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: tensor
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps: float
        update epsilon a new value, if negative not update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


(NOT IMPLEMENTED YET)
======= step (in case of parameter noise) ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps: float
        update epsilon to a new value, if negative no update happens
        (default: no update)
    reset: bool
        reset the perturbed policy by sampling a new perturbation
    update_param_noise_threshold: float
        the desired threshold for the difference between non-perturbed and perturbed policy
    update_param_noise_scale: bool
        whether or not to update the scale of the noise for the next time it is re-perturbed

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s',d) and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * (1-d) * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import tensorflow as tf


@tf.function
def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


class DQfD(tf.Module):

    def __init__(self,
                 q_func,
                 observation_shape,
                 num_actions,
                 lr,
                 grad_norm_clipping=None,
                 gamma=0.99,
                 n_step=10,
                 exp_margin=0.8,
                 lambda1=1.0,
                 lambda2=1.0,
                 lambda3=1e-5,
                 double_q=True,
                 param_noise=False,
                 param_noise_filter_func=None):

        self.num_actions = num_actions
        self.gamma = gamma
        self.n_step = n_step
        self.exp_margin = exp_margin
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.double_q = double_q
        self.param_noise = param_noise
        self.param_noise_filter_func = param_noise_filter_func
        self.grad_norm_clipping = grad_norm_clipping

        self.optimizer = tf.keras.optimizers.Adam(lr)

        with tf.name_scope('q_network'):
            self.q_network = q_func(observation_shape, num_actions)
        with tf.name_scope('target_q_network'):
            self.target_q_network = q_func(observation_shape, num_actions)
        self.eps = tf.Variable(0., name="eps")

    @tf.function
    def step(self, obs, stochastic=True, update_eps=-1):
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            q_values = self.q_network(obs)
            deterministic_actions = tf.argmax(q_values, axis=1)
            batch_size = tf.shape(obs)[0]
            random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions, dtype=tf.int64)
            chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < self.eps
            stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        if stochastic:
            output_actions = stochastic_actions
        else:
            output_actions = deterministic_actions

        if update_eps >= 0:
            self.eps.assign(update_eps)

        return output_actions, self.eps, None, None

    @tf.function()
    def train(self, obs0, actions, rewards, obs1, dones, is_demos, importance_weights, obsn=None, rewards_n=None, dones_n=None):
      with tf.GradientTape() as tape:

        # ====================1-step loss===================
        q_t = self.q_network(obs0)
        one_hot_actions = tf.one_hot(actions, self.num_actions, dtype=tf.float32)
        q_t_selected = tf.reduce_sum(q_t * one_hot_actions, 1)

        q_tp1 = self.target_q_network(obs1)

        if self.double_q:
            q_tp1_using_online_net = self.q_network(obs1)
            q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, self.num_actions, dtype=tf.float32), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        dones = tf.cast(dones, q_tp1_best.dtype)
        q_tp1_best_masked = (1.0 - dones) * q_tp1_best

        q_t_selected_target = rewards + self.gamma * q_tp1_best_masked
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        loss_dq = huber_loss(td_error)

        # ====================n-step loss===================
        if obsn is not None:
            q_tpn = self.target_q_network(obsn)
            if self.double_q:
                q_tpn_using_online_net = self.q_network(obsn)
                q_tpn_best_using_online_net = tf.argmax(q_tpn_using_online_net, 1)
                q_tpn_best = tf.reduce_sum(q_tpn * tf.one_hot(q_tpn_best_using_online_net, self.num_actions, dtype=tf.float32), 1)
            else:
                q_tpn_best = tf.reduce_max(q_tpn, 1)
            dones_n = tf.cast(dones_n, q_tpn_best.dtype)
            q_tpn_best_masked = (1.0 - dones_n) * q_tpn_best

            q_tn_selected_target = rewards_n + (self.gamma ** self.n_step) * q_tpn_best_masked
            n_td_error = q_t_selected - tf.stop_gradient(q_tn_selected_target)
            loss_n = self.lambda1 * huber_loss(n_td_error)
        else:
            loss_n = tf.constant(0.)

        # ==========large margin classification loss=========
        is_demo = tf.cast(is_demos, q_tp1_best.dtype)
        margin_l = self.exp_margin * (tf.ones_like(one_hot_actions, dtype=tf.float32) - one_hot_actions)
        margin_masked = tf.reduce_max(q_t + margin_l, 1)
        loss_E = self.lambda2 * is_demo * (margin_masked - q_t_selected)

        # ==========L2 loss=========
        loss_l2 = self.lambda3 * tf.reduce_sum([tf.reduce_sum(tf.square(variables)) for variables in self.q_network.trainable_variables])

        all_loss = loss_n + loss_dq + loss_E  # + loss_l2 ?
        weighted_error = tf.reduce_mean(importance_weights * all_loss) + loss_l2

      grads = tape.gradient(weighted_error, self.q_network.trainable_variables)
      if self.grad_norm_clipping:
        clipped_grads = []
        for grad in grads:
            clipped_grads.append(tf.clip_by_norm(grad, self.grad_norm_clipping))
            grads = clipped_grads
      grads_and_vars = zip(grads, self.q_network.trainable_variables)
      self.optimizer.apply_gradients(grads_and_vars)

      return td_error, n_td_error, loss_dq, loss_n, loss_E, loss_l2, weighted_error

    @tf.function(autograph=False)
    def update_target(self):
      q_vars = self.q_network.trainable_variables
      target_q_vars = self.target_q_network.trainable_variables
      for var, var_target in zip(q_vars, target_q_vars):
        var_target.assign(var)
