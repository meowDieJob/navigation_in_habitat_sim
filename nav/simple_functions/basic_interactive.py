import habitat_sim
import cv2
import _magnum
import numpy as np



def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "data/scene_datasets/gibson/Shelbiana.glb"
    backend_cfg.enable_physics = True
    backend_cfg.allow_sliding = True


    # agent configuration
    CAMsensor_cfg = habitat_sim.CameraSensorSpec()
    CAMsensor_cfg.sensor_type = habitat_sim.sensor.SensorType.COLOR
    CAMsensor_cfg.uuid = "rgb1"
    CAMsensor_cfg.resolution = [480,640]
    CAMsensor_cfg.hfov = 50
    CAMsensor_cfg.position = [0,0,0]


    CAM1sensor_cfg = habitat_sim.CameraSensorSpec()
    CAM1sensor_cfg.sensor_type = habitat_sim.sensor.SensorType.COLOR
    CAM1sensor_cfg.uuid = "rgb2"
    CAM1sensor_cfg.resolution = [480,640]
    CAM1sensor_cfg.hfov = 90
    CAM1sensor_cfg.position = [0,0,0]

    depthSensor_cfg = habitat_sim.CameraSensorSpec()
    depthSensor_cfg.sensor_type = habitat_sim.sensor.SensorType.DEPTH
    depthSensor_cfg.uuid = "depth"
    depthSensor_cfg.resolution = [960,1080]
    depthSensor_cfg.position=[0,0,0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [CAMsensor_cfg,CAM1sensor_cfg,depthSensor_cfg]
    print(agent_cfg.action_space)

    return habitat_sim.Configuration(backend_cfg,[agent_cfg])



cfg = make_configuration()
env = habitat_sim.Simulator(cfg)
agent = env.initialize_agent(0) # "0" is the name of your agent
agent.agent_config.action_space['move_forward'].actuation.amount = 0.0378
agent.agent_config.action_space['turn_left'].actuation.amount = 5.216
agent.agent_config.action_space['turn_right'].actuation.amount = 5.216
agent_state = agent.get_state() # The way to get your agent's state




observations = env.reset()
print(agent.agent_config.action_space)


is_end = False
while not is_end:
    keystroke = cv2.waitKey(0)
    if keystroke == ord('w'):
        observations = env.step("move_forward")
        # print('forward')
    elif keystroke == ord('a'):
        observations = env.step("turn_left")
        # print('left')
    elif keystroke == ord('d'):
        observations = env.step("turn_right")
        # print('right')
    elif keystroke == ord('f'):
        is_end = True
        # print('stop')
    else:
        print('invalid key')

    obs_depth = observations['depth']
    cv2.imshow("DEPTH", obs_depth)

    cv2.imshow('RGB1', observations['rgb1'])
    agent_state = agent.get_state()
    print(agent_state.rotation)
    # print(obj.rotation)
    # print(obj.translation)

    cv2.imshow('RGB2', observations['rgb2'])
    # cv2.waitKey(0)
