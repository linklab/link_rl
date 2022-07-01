## Plan A
### ModelCreator
- SingleModelCreator
  - QModelCreator
  - ReinforceModelCreator
- DoubleModelCreator
  - A2cModelCreator(ppo)
  - DdpgModelCreator
  - Td3ModelCreator
  - SacModelCreator

- SingleModelCreator
  - self._model = self._create_model
  - self.get_model

- DoubleModelCreator
  - self._actor_model = self._create_actor_model
  - self._critic_model = self._create_critic_model
  - self.get_actor_model
  - self.get_critic_model


### JB
- main
  - runner
    - mac
  - leaner Algorithm
    - mac Unique
      - agent getter setter 들
      - agent(nn.Module)
    - target_mac

### Class
- main
  - runner  # 배제
  - leaner (Algorithm)  # 배제
    - agent (Unique) - call train(), get_action() | new class
      - model_controller DQN, DrQ, Curl | agent of past link_rl
        - model(nn.Module) (Algorithm, layers)

### 순서
0. leaner 완전 배제
1. model
2. agent method 정리
3. renamed agent -> model controller
4. Add agent which contains controllers
5. registry
6. pytest
