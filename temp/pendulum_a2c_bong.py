import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from datetime import datetime
import random

EPISODES = 2000


class A2CAgent:
    def __init__(self, state_size, action_size, state_size_2, action_size_2):
        self.render = True
        self.load_model = True
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.state_size_2 = state_size_2
        self.action_size_2 = action_size_2
        self.value_size = 1
        
        self.epsilon = 0.99
        self.epsilon_decator = 0.99
        
        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.actor_lr = 0.00025
        self.critic_lr = 0.00025

        # 정책신경망과 가치신경망 생성
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()
        self.actor_2 = self.build_actor_2()
        self.critic_2 = self.build_critic_2()
        self.actor_updater_2 = self.actor_optimizer_2()
        self.critic_updater_2 = self.critic_optimizer_2()

        if self.load_model:
            self.actor.load_weights("./pendulum_actor.h5")
            self.critic.load_weights("./pendulum_critic.h5")
            
    # actor: 상태를 받아 각 행동의 확률을 계산
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_shape=[3,], activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary()
        return actor

    # critic: 상태를 받아서 상태의 가치를 계산
    def build_critic(self):
        critic = Sequential()
        # input_dim=self.state_size, 
        critic.add(Dense(24, input_shape=[3,], activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(24, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        return critic
    
    # actor: 상태를 받아 각 행동의 확률을 계산
    def build_actor_2(self):
        actor2 = Sequential()
        actor2.add(Dense(24, input_shape=[4,], activation='relu',
                        kernel_initializer='he_uniform'))
        actor2.add(Dense(self.action_size_2, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor2.summary()
        return actor2

    # critic: 상태를 받아서 상태의 가치를 계산
    def build_critic_2(self):
        critic2 = Sequential()
        # input_dim=self.state_size, 
        critic2.add(Dense(24, input_shape=[4,], activation='relu',
                         kernel_initializer='he_uniform'))
        critic2.add(Dense(24, activation='relu',
                         kernel_initializer='he_uniform'))
        critic2.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic2.summary()
        return critic2

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state, epsilon):
        policy = self.actor.predict(state, batch_size=1).flatten()
        if random.random() < epsilon:
            action = random.randrange(0, self.action_size)
        else:
            action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action
    
    def get_action_2(self, state, epsilon):
        policy = self.actor_2.predict(state, batch_size=1).flatten()
        if random.random() > epsilon:
            action = random.randrange(0, self.action_size_2)
        else:
            action = np.random.choice(self.action_size_2, 1, p=policy)[0]
        return action

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [],
                           updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        target = K.placeholder(shape=[None, ])

        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)

        return train
    
    # 정책신경망을 업데이트하는 함수
    def actor_optimizer_2(self):
        action = K.placeholder(shape=[None, self.action_size_2])
        advantage = K.placeholder(shape=[None, ])
        
        action_prob = K.sum(action * self.actor_2.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor_2.trainable_weights, [], loss)
        train = K.function([self.actor_2.input, action, advantage], [],
                           updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer_2(self):
        target = K.placeholder(shape=[None, ])

        loss = K.mean(K.square(target - self.critic_2.output))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic_2.trainable_weights, [], loss)
        train = K.function([self.critic_2.input, target], [], updates=updates)

        return train

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, state_2, action, action_2, reward, next_state, next_state_2, done):
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]
        
        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        # 벨만 기대 방정식를 이용한 어드벤티지와 업데이트 타깃
        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value
        self.actor_updater([state, act, advantage])
        self.critic_updater([state, target])
        
        value = self.critic_2.predict(state_2)[0]
        next_value = self.critic_2.predict(next_state_2)[0]
        
        act = np.zeros([1, self.action_size_2])
        act[0][action_2] = 1

        # 벨만 기대 방정식를 이용한 어드벤티지와 업데이트 타깃
        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value
        self.actor_updater_2([state_2, act, advantage])
        self.critic_updater_2([state_2, target])


if __name__ == "__main__":
    print(str(datetime.now()) + ' started')
    env = gym.make('Pendulum-v0')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = 3
    action_size = 2
    state_size_2 = 4
    action_size_2 = 10
    action_list = [0.0, 0.2, 0.4, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(state_size, action_size, state_size_2, action_size_2)

    turn_off_random_action = False
    
    scores, episodes = [], []
    max_score = 0
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()
            action_left_or_right = agent.get_action(state, agent.epsilon)
            state_left_or_right = np.array([list(np.append(state[0], np.array([action_left_or_right,]), axis=0))])
            
            action_power = agent.get_action_2([state_left_or_right], agent.epsilon)
            if action_left_or_right:
                real_action = np.array([action_list[action_power]])
            else:
                real_action = np.array([-action_list[action_power]])
            
            next_state, reward, done, info = env.step(real_action)
            
            next_state_1 = np.reshape(next_state, [1, state_size])
            
            next_state_2 = np.array(list(np.append(np.array(next_state), np.array([action_left_or_right]), axis=0)))
            reward += 5
            
            
            next_state_2 = np.reshape(next_state_2, [1, state_size_2])

            agent.train_model(state, state_left_or_right,
                              action_left_or_right, action_power,
                              reward, next_state_1, next_state_2, done)

            score += reward
            state = next_state_1

            if done:
                agent.epsilon *= agent.epsilon_decator
                scores.append(score)
                episodes.append(e)
                # 에피소드마다 학습 결과 출력
                print("episode:", e, "  score:", score)
    pylab.plot(episodes, scores, 'b')
    pylab.savefig("./pendulum_a2c.png")
    agent.actor.save_weights("./pendulum_actor.h5")
    agent.critic.save_weights("./pendulum_critic.h5")
