Playing table-hockey

# FIELD
1 ('action_dim', 
       [
              [Box([-100.], [200.], (1,), float32), Box([-30.], [30.], (1,), float32)], 
              [Box([-100.], [200.], (1,), float32), Box([-30.], [30.], (1,), float32)]
       ]
)
2 ('agent_nums', [1, 1])
3 ('all_observes', 
       [
              {
                     'obs': {
                            'agent_obs': array(
                                   [
                                          [0., 0., 0., ..., 0., 0., 0.],
                                          [0., 0., 0., ..., 0., 0., 0.],
                                          [0., 0., 0., ..., 0., 0., 0.],
                                          ...,
                                          [0., 0., 0., ..., 0., 0., 0.],
                                          [0., 0., 0., ..., 0., 0., 0.],
                                          [0., 0., 0., ..., 0., 0., 0.]
                                   ]
                            ), 
                            'id': 'team_0', 
                            'game_mode': 'NEW GAME', 
                            'energy': 1000
                     }, 
                     'controlled_player_index': 0
              }, 
              {
                     'obs': {
                            'agent_obs': array(
                                   [
                                          [0., 0., 0., ..., 0., 0., 0.],
                                          [0., 0., 0., ..., 0., 0., 0.],
                                          [0., 0., 0., ..., 0., 0., 0.],
                                          ...,
                                          [0., 0., 0., ..., 0., 0., 0.],
                                          [0., 0., 0., ..., 0., 0., 0.],
                                          [0., 0., 0., ..., 0., 0., 0.]
                                   ]
                            ), 
                            'id': 'team_1', 
                            'game_mode': 'NEW GAME', 
                            'energy': 1000
                     }, 
                     'controlled_player_index': 1
              }
       ]
)
4 ('board_height', 700)
5 ('board_width', 700)
6 ('current_state', 
       [
              {
                     'agent_obs': array(
                            [
                                   [0., 0., 0., ..., 0., 0., 0.],
                                   [0., 0., 0., ..., 0., 0., 0.],
                                   [0., 0., 0., ..., 0., 0., 0.],
                                   ...,
                                   [0., 0., 0., ..., 0., 0., 0.],
                                   [0., 0., 0., ..., 0., 0., 0.],
                                   [0., 0., 0., ..., 0., 0., 0.]
                            ] --> (40, 40)
                     ), 
                     'id': 'team_0', 
                     'game_mode': 'NEW GAME', 
                     'energy': 1000
              }, 
              {
                     'agent_obs': array(
                            [
                                   [0., 0., 0., ..., 0., 0., 0.],
                                   [0., 0., 0., ..., 0., 0., 0.],
                                   [0., 0., 0., ..., 0., 0., 0.],
                                   ...,
                                   [0., 0., 0., ..., 0., 0., 0.],
                                   [0., 0., 0., ..., 0., 0., 0.],
                                   [0., 0., 0., ..., 0., 0., 0.]
                            ]
                     ), 
                     'id': 'team_1', 
                     'game_mode': 'NEW GAME', 
                     'energy': 1000
              }
       ]
)
7 ('done', False)
8 ('env_core', <olympics_engine.AI_olympics.AI_Olympics object at 0x102527d00>)
9 ('game_name', 'integrated')
10 ('init_info', None)
11 ('is_act_continuous', True)
12 ('is_obs_continuous', True)
13 ('joint_action_space', 
       [
              [Box([-100.], [200.], (1,), float32), Box([-30.], [30.], (1,), float32)], 
              [Box([-100.], [200.], (1,), float32), Box([-30.], [30.], (1,), float32)]
       ]
)
14 ('max_step', 2000)
15 ('n_player', 2)
16 ('n_return', [0, 0])
17 ('obs_type', ['vector', 'vector'])
18 ('seed', None)
19 ('step_cnt', 0)
20 ('won', {})


# METHOD
1 ('check_win', <bound method OlympicsIntegrated.check_win of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
2 ('create_seed', <function OlympicsIntegrated.create_seed at 0x160559b80>)
3 ('decode', <bound method OlympicsIntegrated.decode of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
4 ('get_all_observes', <bound method OlympicsIntegrated.get_all_observes of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
5 ('get_config', <bound method Game.get_config of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
6 ('get_next_state', <bound method Game.get_next_state of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
7 ('get_render_data', <bound method Game.get_render_data of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
8 ('get_reward', <bound method OlympicsIntegrated.get_reward of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
9 ('get_single_action_space', <bound method OlympicsIntegrated.get_single_action_space of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
10 ('is_terminal', <bound method OlympicsIntegrated.is_terminal of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
11 ('is_valid_action', <bound method OlympicsIntegrated.is_valid_action of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
12 ('reset', <bound method OlympicsIntegrated.reset of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
13 ('set_action_space', <bound method OlympicsIntegrated.set_action_space of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
14 ('set_current_state', <bound method Game.set_current_state of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
15 ('set_n_return', <bound method OlympicsIntegrated.set_n_return of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
16 ('set_seed', <bound method OlympicsIntegrated.set_seed of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
17 ('step', <bound method OlympicsIntegrated.step of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)
18 ('step_before_info', <bound method OlympicsIntegrated.step_before_info of <olympics_env.olympics_integrated.OlympicsIntegrated object at 0x163235220>>)