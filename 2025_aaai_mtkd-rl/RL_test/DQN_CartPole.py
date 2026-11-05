import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import collections

#创建环境
env=gym.make('CartPole-v1')
#(小车位置，速度，杆角度，角速度)
state_dim=env.observation_space.shape[0]
#(向左，向右)
action_dim=env.action_space.n
print(f"状态空间维度：{state_dim},动作空间大小：{action_dim}")


#定义Q网络
class DQN(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(DQN,self).__init__()
        self.fc1=nn.Linear(state_dim,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,action_dim)  #每一个动作对应一个Q值


    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        return self.fc3(x)


#定义经验回放池:用于存储和采样过去的经验

class ReplayBuffer:
    def __init__(self,capacity):
        #双端队列，自动淘汰旧经验
        self.buffer=collections.deque(maxlen=capacity)

    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):
        transitions=random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done=zip(*transitions)
        return (np.array(state),action,reward,np.array(next_state),done)

    def size(self):
        return len(self.buffer)



#定义DQN智能体
class DQNAgent:
    def __init__(self,state_dim,action_dim,lr=1e-3,gamma=0.98,epsilon=0.01,target_update_freq=10,buffer_size=10000,batch_size=64):
        self.action_dim=action_dim
        self.q_net=DQN(state_dim,action_dim)    #在线网络
        self.target_q_net=DQN(state_dim,action_dim) #目标网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer=optim.Adam(self.q_net.parameters(),lr=lr)
        self.gamma=gamma #折扣因子
        self.epsilon=epsilon #探索率
        self.target_update_freq=target_update_freq #目标网路更新频率
        self.batch_size=batch_size
        self.buffer=ReplayBuffer(buffer_size)
        self.count=0 #记录更新步数

    def take_action(self,state,epsilon=None):
        #根据epsilon-greedy策略选择动作
        if epsilon is None:
            epsilon=self.epsilon
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            state=torch.tensor(state,dtype=torch.float).unsqueeze(0) #增加批次维度
            with torch.no_grad():
                    q_values=self.q_net(state)
                    return q_values.argmax().item()   #利用：选择Q值最大的动作
            
    def update(self):
        #从经验回放池采样并更新网络
        if self.buffer.size() < self.batch_size:
            return 

        #采样
        states,actions,rewards,next_states,dones=self.buffer.sample(self.batch_size)
        states=torch.tensor(states,dtype=torch.float)
        actions=torch.tensor(actions).unsqueeze(1)
        rewards=torch.tensor(rewards,dtype=torch.float).unsqueeze(1)
        next_states=torch.tensor(next_states,dtype=torch.float)
        dones=torch.tensor(dones,dtype=torch.float).unsqueeze(1)

        #计算Q值
        # Q(s,a)
        current_q_values=self.q_net(states).gather(1,actions)

        #计算目标Q值 计算目标Q值 (r + γ * max_a' Q_target(s', a'))
        with torch.no_grad():
            next_q_values=self.target_q_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values=rewards+self.gamma*next_q_values*(1-dones) #如果回合结束(done=1)，则没有未来奖励

        #计算损失
        loss=nn.MSELoss()(current_q_values,target_q_values)

        #梯度下降更新在线网络
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.count+=1

        #定期更新网络
        if self.count % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())



#训练循环
def train_agent(env,agent,num_episodes=500,max_steps=500,initial_epsilon=0.9,epsilon_decay=0.995):
    return_list=[]
    epsilon=initial_epsilon

    for i_episode in range(num_episodes):
        state,_=env.reset()
        episode_return = 0
        done=False

        for step in range(max_steps):
            action=agent.take_action(state,epsilon)
            next_state,reward,done,truncated,_=env.step(action)

            agent.buffer.add(state,action,reward,next_state,done)
            state=next_state
            episode_return+=reward

            agent.update()

            if done or truncated:
                break


        epsilon=max(agent.epsilon,epsilon*epsilon_decay)

        return_list.append(episode_return)
        if (i_episode+1)%50==0:
            print(f"回合: {i_episode+1},平均奖励（最近50回合）：{np.mean(return_list[-50:]):.1f},探索率：{epsilon:.3f}")
    print(f"训练完成！")
    return return_list


def test_agent(env,agent,num_episodes=5,render=True):
    total_rewards=[]
    for i in range(num_episodes):
        state,_=env.reset()
        episode_return=0
        done=False
        while not done:
            if render:
                env.render()
            action=agent.take_action(state)
            next_state,reward,done,truncated,_=env.step(action)
            state=next_state
            episode_return+=reward
            if done or truncated:
                break
        total_rewards.append(episode_return)
        print(f"测试回合 {i+1}: 总奖励 = {episode_return}")
    env.close()
    print(f"平均测试奖励：{np.mean(total_rewards):.1f}")


        
if '__main__' == __name__:
    #创建智能体并开始训练
    agent=DQNAgent(state_dim,action_dim,lr=2e-3,gamma=0.99,epsilon=0.01)
    returns=train_agent(env,agent,num_episodes=300)

    
    # 测试智能体
    test_agent(env, agent, num_episodes=3, render=True) # 在无GUI环境中设置render=False
