import pybullet_envs

envs = pybullet_envs.getList()

for env in envs:
    print(env)
