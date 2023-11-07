import habitat_sim
import cv2
import torch
import os
from torch.autograd import Variable
from model import Net
import numpy as np
import math
import torch.nn.functional as F
import pandas as pd
model = Net()

# 初始化position
position = np.array([-2.4452953 ,  0.17515154,  2.349534  ], dtype='float32')

# 初始化rotation
rotation = np.quaternion(-0.150860279798508, 0, 0.988555133342743, 0)

# 初始化sensor_states
sensor_states = {
    'rgb': habitat_sim.agent.SixDOFPose(
        position=np.array([-2.306779, 1.6751516, -0.6472672], dtype='float32'),
        rotation=np.quaternion(-0.150860279798508, 0, 0.988555133342743, 0)
    )
}
zero_state = habitat_sim.AgentState(position=position,rotation=rotation,sensor_states=sensor_states)

def TDistance(a,b):
    return math.sqrt((a[0]-b[0])^2+(a[1]-b[1])^2+(a[2]-b[2])^2)

STEP_SIZE = 0.25
LR = 0.01

# MAX_TIME = 20 # s
# MOVE_REWARD = -1
# SUCCESS_REWARD = 10000
# FAIL_REWARD = -100


def reset_agent():
    agent.set_state(zero_state)

def make_configuration():
    # 模拟器配置
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "data/scene_datasets/gibson/Beach.glb"
    backend_cfg.enable_physics = True
    backend_cfg.allow_sliding = True

    # agent 配置
    CAMsensor_cfg = habitat_sim.CameraSensorSpec()
    CAMsensor_cfg.sensor_type = habitat_sim.sensor.SensorType.COLOR
    CAMsensor_cfg.uuid = "rgb"
    CAMsensor_cfg.resolution = [256,256]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [CAMsensor_cfg]

    return habitat_sim.Configuration(backend_cfg,[agent_cfg])

cfg = make_configuration()
sim = habitat_sim.Simulator(cfg)
agent = sim.initialize_agent(0)
agent_state = agent.get_state()
follower = sim.make_greedy_follower(agent_id=0, goal_radius=STEP_SIZE,stop_key="Stop")
observations = sim.reset()


observations = sim.reset()

#load data
path = '/home/yuanzhao/PycharmProjects/Newnav/train_data/img/Beach/'
picfiles = {}
picfiles['path'] = []
picfiles['epoch'] = []

f = os.walk(path)
for a,b,names in f:
    for i in range(len(names)):
        for name in names:
           if(name == "listpd.csv"):
               continue
           if name.split("_",3)[1] == str(i):
               picfiles['path'].append(name)
               picfiles['epoch'].append(name.split("_",3)[2])
# path中按list.txt中的顺序存储地址，对应位置的epoch存储经过的步骤（理论最短）


MAX_EPISODE = 1800 # 训练前1500张图片
MAX_EPOCH = 100 # 每轮最多100步
LOOPTIME = 2    # 每张图训练两次
save = [200,500,800,1000,1300,1500,1799] #保存模型

data = pd.read_csv("/home/yuanzhao/PycharmProjects/Newnav/train_data/img/Beach/listpd.csv",)



for episode in range(MAX_EPISODE):
    print("EPISODE:",episode,"/1800")
    # 获取目标点
    nav_point = np.array(eval(data.location[episode]))
    target_image = cv2.imread(path + picfiles['path'][episode])

    sim.reset()
    reset_agent()
    for round in save:
        if episode == round:
            save_name = "/home/yuanzhao/PycharmProjects/Newnav/model/"+str(round)+"_"+"model.pt"
            torch.save(model.state_dict(), save_name)

    for i in range(LOOPTIME):
        sim.reset()
        reset_agent()
        for epoch in range(MAX_EPOCH):
            try:
                idel_action = follower.next_action_along(goal_pos=nav_point)
                if idel_action == "move_forward":
                    idel_int = 1
                elif idel_action == "turn_left":
                    idel_int = 2
                elif idel_action == "turn_right":
                    idel_int = 3
                elif idel_action == "Stop":
                    idel_int = 0
            except habitat_sim.errors.GreedyFollowerError:
                break
            target_image_tensor = torch.tensor(target_image, dtype=torch.float32)
            obs_rgb_tensor = torch.tensor(observations['rgb'], dtype=torch.float32)
            target_image_tensor = target_image_tensor.permute(2, 0, 1).unsqueeze(0)
            obs_rgb_tensor = obs_rgb_tensor.permute(2, 0, 1).unsqueeze(0)
            action_torchlist = model.forward(target_image_tensor, obs_rgb_tensor)

            # action_torchlist = model.forward(target_image,observations['rgb'])
            # action_int = torch.argmax(action_torchlist,dim = 0).item()
            target_torchlist = torch.zeros_like(action_torchlist)
            target_torchlist[0][idel_int] = 1

            # print("target:",target_torchlist,"   now:",action_torchlist)
            # cross_entropy_loss = F.cross_entropy(action_torchlist.unsqueeze(0), target_torchlist.unsqueeze(0))
            # cross_entropy_loss.backward()
            mse_loss = LR*F.mse_loss(action_torchlist, target_torchlist)
            mse_loss.backward()
            if idel_action != "Stop":
                observations = sim.step(idel_action)
                # cv2.imshow("rgb", observations['rgb'])
                # cv2.waitKey(1)
            else:
                break


# torch.save(model.state_dict(),'/home/yuanzhao/PycharmProjects/Newnav/model/model.pt')

















