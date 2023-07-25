import numpy as np

# 定义动作空间
actions = ['前进', '后退', '左转', '右转']

# 初始化Q-table
q_table = np.zeros((num_states, num_actions))

# 定义训练参数
num_episodes = 1000
max_steps_per_episode = 100
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

# 训练智能体
for episode in range(num_episodes):
    state = env.reset()
    done = False
    step = 0

    for step in range(max_steps_per_episode):
        # 选择行动
        exploration_threshold = np.random.uniform(0, 1)

        if exploration_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = np.random.randint(0, num_actions)

        # 执行行动并观察下一个状态、奖励和是否结束
        new_state, reward, done, _ = env.step(action)

        # 更新Q-table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                 learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]))

        state = new_state

        if done:
            break

    # 逐渐减小探索率
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

# 使用训练完成的Q-table测试模型
state = env.reset()
done = False

while not done:
    action = np.argmax(q_table[state, :])
    new_state, reward, done, _ = env.step(action)
    state = new_state

    # 更新环境并可视化
    env.render()

env.close()
