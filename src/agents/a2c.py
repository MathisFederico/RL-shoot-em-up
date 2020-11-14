import learnrl as rl
import tensorflow as tf
import tensorflow_probability as tfp

from gym.spaces import MultiBinary

from agents.memory import TFMemory

kl = tf.keras.layers

class VPnetwork(tf.keras.Model):

    def __init__(self, action_shape):
        super().__init__()
        self.conv_1 = kl.Conv2D(16, (5, 5), strides=(1, 1), dilation_rate=(3, 3), activation='relu')
        self.conv_2 = kl.Conv2D(16, (5, 5), strides=(2, 2), activation='relu')
        self.conv_2 = kl.Conv2D(16, (5, 5), strides=(2, 2), activation='relu')

        self.flatten_1 = kl.GlobalMaxPooling2D()

        self.dense_1 = kl.Dense(64, activation='relu')
        self.dense_2 = kl.Dense(64, activation='relu')

        self.v_out = kl.Dense(1, activation='linear')

        self.p_out = kl.Dense(action_shape, activation='sigmoid')
    
    def preprocess(self, inputs):
        return inputs / 255
    
    def __call__(self, inputs):
        x = self.preprocess(inputs)
        
        x = self.conv_1(x)
        x = self.conv_2(x)

        x = self.flatten_1(x)

        x = self.dense_1(x)
        x = self.dense_2(x)

        v = self.v_out(x)
        p = self.p_out(x)

        return v[:, 0], p

class ActorCriticAgent(rl.Agent):

    def __init__(self, action_space, p_threshold=0.5, sample_size=32,
                       entropy_weight=1, learning_rate=1e-4, discount_factor=0.99,
                       learn_skip=100, remember_skip=9,
                       exploration=.05, exploration_decay=1e-4):
        self.action_space = action_space

        if isinstance(action_space, MultiBinary):
            self.action_shape = action_space.n
        else:
            raise NotImplementedError

        self.value_policy = VPnetwork(self.action_shape)
        self.value_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.policy_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.p_threshold = p_threshold

        self.memory = TFMemory(1000)
        self.sample_size = sample_size
        self.entropy_weight = entropy_weight
        self.discount_factor = discount_factor

        self.learn_skip = learn_skip
        self._learn_to_skip = learn_skip

        self.remember_skip= remember_skip
        self._remember_to_skip = remember_skip

        self.exploration = exploration
        self.exploration_decay = exploration_decay
    
    def act(self, observation, greedy=False):
        _, p = self.value_policy(tf.expand_dims(observation, axis=0))
        if not greedy:
            p = self.exploration * tf.random.uniform(p.shape, 0, 1, dtype=tf.float32) + \
                (1 - self.exploration) * p #pylint: disable=all
        action = tf.squeeze(p >= self.p_threshold)
        return tf.cast(action, tf.float32).numpy()

    def learn(self):
        metrics = {
            'learn_to_skip': self._learn_to_skip,
            'remember_to_skip': self._remember_to_skip,
            'exploration': self.exploration
        }

        if self._learn_to_skip > 0:
            self._learn_to_skip -= 1
        else:
            self._learn_to_skip = self.learn_skip

            self.exploration *= 1 - self.exploration_decay

            observation, action, reward, done, next_observation = self.memory.sample(self.sample_size)
            action = tf.cast(action, tf.float32)
            expected_returns = self.evaluate(reward, done, next_observation)

            with tf.GradientTape() as tape_v, tf.GradientTape() as tape_p:
                V, P = self.value_policy(observation)
                button_pressed = action == 1
                P = tf.where(button_pressed, P, 1-P)
                P = tf.reduce_prod(P, axis=-1)
                logP = tf.math.log(P)
                A = expected_returns - V

                value_loss = tf.reduce_mean(tf.square(expected_returns - V))

                entropy_loss = tf.reduce_mean(P * logP)
                advantage_loss = - tf.reduce_mean(A * logP)
                policy_loss = advantage_loss + self.entropy_weight * entropy_loss

            value_grad = tape_v.gradient(value_loss, self.value_policy.trainable_variables)
            policy_grad = tape_p.gradient(policy_loss, self.value_policy.trainable_variables)

            self.value_opt.apply_gradients([
                (grad, var) 
                for (grad, var) in zip(value_grad, self.value_policy.trainable_variables) 
                if grad is not None
            ])
            
            self.policy_opt.apply_gradients([
                (grad, var) 
                for (grad, var) in zip(policy_grad, self.value_policy.trainable_variables) 
                if grad is not None
            ])

            metrics.update({
                'value_loss': value_loss.numpy(),
                'entropy_loss': entropy_loss.numpy(),
                'advantage_loss': advantage_loss.numpy(),
                'loss': policy_loss.numpy()
            })

        return metrics


    def evaluate(self, reward, done, next_observation):
        futur_reward = reward
        not_done = tf.logical_not(done)

        if tf.reduce_any(not_done):
            not_done_indicies = tf.where(not_done)
            next_value, _ = self.value_policy(next_observation)
            try:
                tf.tensor_scatter_nd_add(futur_reward, not_done_indicies, self.discount_factor * next_value)
            except:
                print(futur_reward, not_done_indicies, next_value)

        return futur_reward

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        if self._remember_to_skip > 0:
            self._remember_to_skip -= 1
        else:
            self._remember_to_skip = self.remember_skip
            self.memory.remember(
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                next_observation=next_observation
            )