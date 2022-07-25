###		body, connections = sample_robot((5, 5))
###		env = gym.make(env_spec.id, body=body, connections=connections)
 1: env_id:              Walker-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (64,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (9,), float64)
 2: env_id:        BridgeWalker-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (69,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (8,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (8,), float64)
 3: env_id:         CaveCrawler-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (96,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (9,), float64)
 4: env_id:              Jumper-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (81,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (11,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (11,), float64)
 5: env_id:             Flipper-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (71,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (9,), float64)
 6: env_id:            Balancer-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (67,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (9,), float64)
 7: env_id:            Balancer-v1 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (71,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (8,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (8,), float64)
 8: env_id:           UpStepper-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (72,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (5,), RANGE: Box([0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6], (5,), float64)
 9: env_id:         DownStepper-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (84,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (10,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (10,), float64)
10: env_id:   ObstacleTraverser-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (74,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (6,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6], (6,), float64)
11: env_id:   ObstacleTraverser-v1 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (74,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (8,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (8,), float64)
12: env_id:             Hurdler-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (84,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (13,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (13,), float64)
13: env_id:           GapJumper-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (84,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (12,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (12,), float64)
14: env_id:      PlatformJumper-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (80,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (9,), float64)
15: env_id:           Traverser-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (86,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (9,), float64)
16: env_id:              Lifter-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (75,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (12,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (12,), float64)
17: env_id:             Carrier-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (76,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (11,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (11,), float64)
18: env_id:             Carrier-v1 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (72,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (9,), float64)
19: env_id:              Pusher-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (66,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (9,), float64)
20: env_id:              Pusher-v1 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (70,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (9,), float64)
21: env_id:         BeamToppler-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (75,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (10,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (10,), float64)
22: env_id:          BeamSlider-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (75,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (8,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (8,), float64)
23: env_id:             Thrower-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (68,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (7,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6], (7,), float64)
24: env_id:             Catcher-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (65,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (5,), RANGE: Box([0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6], (5,), float64)
25: env_id:       AreaMaximizer-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (62,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (9,), float64)
26: env_id:       AreaMinimizer-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (66,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (10,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (10,), float64)
27: env_id:   WingspanMazimizer-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (70,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (11,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (11,), float64)
28: env_id:     HeightMaximizer-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (62,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (9,), float64)
29: env_id:             Climber-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (66,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (12,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (12,), float64)
30: env_id:             Climber-v1 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (70,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (10,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6], (10,), float64)
31: env_id:             Climber-v2 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (66,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (4,), RANGE: Box([0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6], (4,), float64)
32: env_id: BidirectionalWalker-v0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (65,) -      | ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (7,), RANGE: Box([0.6 0.6 0.6 0.6 0.6 0.6 0.6], [1.6 1.6 1.6 1.6 1.6 1.6 1.6], (7,), float64)

register(
    id = 'Walker-v0',
    entry_point = 'evogym.envs.walk:WalkingFlat',
    max_episode_steps=500
)

register(
    id = 'BridgeWalker-v0',
    entry_point = 'evogym.envs.walk:SoftBridge',
    max_episode_steps=500
)

register(
    id = 'CaveCrawler-v0',
    entry_point = 'evogym.envs.walk:Duck',
    max_episode_steps=1000
)

register(
    id = 'Jumper-v0',
    entry_point = 'evogym.envs.jump:StationaryJump',
    max_episode_steps=500
)

register(
    id = 'Flipper-v0',
    entry_point = 'evogym.envs.flip:Flipping',
    max_episode_steps=600
)

register(
    id = 'Balancer-v0',
    entry_point = 'evogym.envs.balance:Balance',
    max_episode_steps=600
)

register(
    id = 'Balancer-v1',
    entry_point = 'evogym.envs.balance:BalanceJump',
    max_episode_steps=600
)

register(
    id = 'UpStepper-v0',
    entry_point = 'evogym.envs.traverse:StepsUp',
    max_episode_steps=600
)

register(
    id = 'DownStepper-v0',
    entry_point = 'evogym.envs.traverse:StepsDown',
    max_episode_steps=500
)

register(
    id = 'ObstacleTraverser-v0',
    entry_point = 'evogym.envs.traverse:WalkingBumpy',
    max_episode_steps=1000
)

register(
    id = 'ObstacleTraverser-v1',
    entry_point = 'evogym.envs.traverse:WalkingBumpy2',
    max_episode_steps=1000
)

register(
    id = 'Hurdler-v0',
    entry_point = 'evogym.envs.traverse:VerticalBarrier',
    max_episode_steps=1000
)

register(
    id = 'GapJumper-v0',
    entry_point = 'evogym.envs.traverse:Gaps',
    max_episode_steps=1000
)

register(
    id = 'PlatformJumper-v0',
    entry_point = 'evogym.envs.traverse:FloatingPlatform',
    max_episode_steps=1000
)

register(
    id = 'Traverser-v0',
    entry_point = 'evogym.envs.traverse:BlockSoup',
    max_episode_steps=600
)

## PACKAGE ##
register(
    id = 'Lifter-v0',
    entry_point = 'evogym.envs.manipulate:LiftSmallRect',
    max_episode_steps=300
)

register(
    id = 'Carrier-v0',
    entry_point = 'evogym.envs.manipulate:CarrySmallRect',
    max_episode_steps=500
)

register(
    id = 'Carrier-v1',
    entry_point = 'evogym.envs.manipulate:CarrySmallRectToTable',
    max_episode_steps=1000
)

register(
    id = 'Pusher-v0',
    entry_point = 'evogym.envs.manipulate:PushSmallRect',
    max_episode_steps=500
)

register(
    id = 'Pusher-v1',
    entry_point = 'evogym.envs.manipulate:PushSmallRectOnOppositeSide',
    max_episode_steps=600
)

register(
    id = 'BeamToppler-v0',
    entry_point = 'evogym.envs.manipulate:ToppleBeam',
    max_episode_steps=1000
)

register(
    id = 'BeamSlider-v0',
    entry_point = 'evogym.envs.manipulate:SlideBeam',
    max_episode_steps=1000
)

register(
    id = 'Thrower-v0',
    entry_point = 'evogym.envs.manipulate:ThrowSmallRect',
    max_episode_steps=300
)

register(
    id = 'Catcher-v0',
    entry_point = 'evogym.envs.manipulate:CatchSmallRect',
    max_episode_steps=400
)

### SHAPE ###
register(
    id = 'AreaMaximizer-v0',
    entry_point = 'evogym.envs.change_shape:MaximizeShape',
    max_episode_steps=600
)

register(
    id = 'AreaMinimizer-v0',
    entry_point = 'evogym.envs.change_shape:MinimizeShape',
    max_episode_steps=600
)

register(
    id = 'WingspanMazimizer-v0',
    entry_point = 'evogym.envs.change_shape:MaximizeXShape',
    max_episode_steps=600
)

register(
    id = 'HeightMaximizer-v0',
    entry_point = 'evogym.envs.change_shape:MaximizeYShape',
    max_episode_steps=500
)

### CLIMB ###
register(
    id = 'Climber-v0',
    entry_point = 'evogym.envs.climb:Climb0',
    max_episode_steps=400
)

register(
    id = 'Climber-v1',
    entry_point = 'evogym.envs.climb:Climb1',
    max_episode_steps=600
)

register(
    id = 'Climber-v2',
    entry_point = 'evogym.envs.climb:Climb2',
    max_episode_steps=1000
)

### MULTI GOAL ###
register(
    id = 'BidirectionalWalker-v0',
    entry_point = 'evogym.envs.multi_goal:BiWalk',
    max_episode_steps=1000
)