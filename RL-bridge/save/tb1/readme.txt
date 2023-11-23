AAMAS2023 桥牌赛道    及第平台承办

batch size 4096，sleep 1s per train step，lr=2.5e-4

loss = -(policy_loss - 0.5 * value_loss + 0.1 * entropy_loss)

gamma=0.99, lam=0.95

MLPnet

MemPool(5000)

train on node2，4GPUs，170actor

训练时长：17小时