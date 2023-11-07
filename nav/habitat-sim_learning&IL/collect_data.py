import habitat_sim
import cv2
import numpy as np
import pandas as pd

STEP_SIZE = 0.25

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



def make_configuration():
    # 模拟器配置
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "data/scene_datasets/gibson/Beach.glb"
    backend_cfg.enable_physics = True
    # backend_cfg.allow_sliding = True

    # agent 配置
    CAMsensor_cfg = habitat_sim.CameraSensorSpec()
    CAMsensor_cfg.sensor_type = habitat_sim.sensor.SensorType.COLOR
    CAMsensor_cfg.uuid = "rgb"
    CAMsensor_cfg.resolution = [256,256]
    # DepSensor_cfg = habitat_sim.CameraSensorSpec()
    # DepSensor_cfg.uuid = "depth"
    # DepSensor_cfg.resolution = [256,256]
    # DepSensor_cfg.sensor_type = habitat_sim.sensor.SensorType.DEPTH # 深度
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [CAMsensor_cfg]

    return habitat_sim.Configuration(backend_cfg,[agent_cfg])

cfg = make_configuration()
sim = habitat_sim.Simulator(cfg)
agent = sim.initialize_agent(0)
agent_state = agent.get_state()
follower = sim.make_greedy_follower(agent_id=0, goal_radius=STEP_SIZE,stop_key="Stop")
observations = sim.reset()

MAX_EPISODE = 5000
MAX_EPOCH = 1000

listpd = pd.DataFrame(columns=['location'])

for episode in range(MAX_EPISODE):
    print("EPISODE:",episode,"/5000")
    nav_point = sim.pathfinder.get_random_navigable_point()
    observations = sim.reset()
    agent.set_state(zero_state)
    cv2.imshow("rgb",observations['rgb'])
    for epoch in range(MAX_EPOCH):
        try:
            action = follower.next_action_along(goal_pos=nav_point)
        except habitat_sim.errors.GreedyFollowerError:
            break
        # print(action)
        if action !=  "Stop":
           observations = sim.step(action)
           # cv2.imshow("rgb",observations['rgb'])
           # cv2.waitKey(1)
        else:
            imname ="/home/yuanzhao/PycharmProjects/Newnav/train_data/img/Beach/" +"_" + str(episode) + "_" +str(epoch) + "_" +".png"
            cv2.imwrite(imname,observations['rgb'])
            listpd.loc[len(listpd)] = str(nav_point.ravel().tolist())
            break

listpd.to_csv("/home/yuanzhao/PycharmProjects/Newnav/train_data/img/Beach/listpd.csv")