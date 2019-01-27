import gym
from agents import DQNAgent
import matplotlib.pyplot as plt

def learning_curve(data, x_index = 0, y1_index = 1, y2_index = None, title = "", 
                   x_name = "", y_name = "",
                   y1_legend = "", y2_legend=""):
    '''根据统计数据绘制学习曲线，
    Args:
        statistics: 数据元组，每一个元素是一个列表，各列表长度一致 ([], [], [])
        x_index: x轴使用的数据list在元组tuple中索引值
        y_index: y轴使用的数据list在元组tuple中的索引值
        title:图标的标题
        x_name: x轴的名称
        y_name: y轴的名称
        y1_legend: y1图例
        y2_legend: y2图例
    Return:
        None 绘制曲线图
    '''
    fig, ax = plt.subplots()
    x = data[x_index]
    y1 = data[y1_index]
    ax.plot(x, y1, label = y1_legend)
    if y2_index is not None:
        ax.plot(x, data[y2_index], label = y2_legend)
    ax.grid(True, linestyle='-.')
    ax.tick_params(labelcolor='black', labelsize='medium', width=1)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.legend()
    plt.show()  

env = gym.make('CartPole-v0')
agent = DQNAgent(env, hidden_dim = 10) #hidden_layer含有10个神经元

data = agent.learning(gamma = 0.99, # 衰减因子
                        epsilon = 0.1, # 小于epsilon时, 随机选取行为
                        decaying_epsilon = False, 
                        alpha = 0.02, # learning rate
                        max_episode_num = 200, # 学习回合次数
                        display = False) # 显示模型

learning_curve(data, 2, 1, title="DQN performance on CartPole",
                            x_name="episodes", 
                            y_name="rewards of episode")
env.close()