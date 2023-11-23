AAMAS2023 麻将赛道    及第平台承办

batch size 6000 ，sleep 0.2s per train step，lr=2.5e-4

loss = -(policy_loss - 0.5 * value_loss + 0.05 * entropy_loss)

gamma=0.99, lam=0.95

MemPool(10000)

train on node2，4GPUs，100actor