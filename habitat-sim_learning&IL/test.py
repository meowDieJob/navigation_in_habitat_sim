import habitat_sim
import cv2


def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "data/scene_datasets/gibson/Beach.glb"
    backend_cfg.enable_physics = True
    backend_cfg.allow_sliding = True

    # agent configuration
    CAMsensor_cfg = habitat_sim.CameraSensorSpec()
    CAMsensor_cfg.sensor_type = habitat_sim.sensor.SensorType.COLOR
    CAMsensor_cfg.uuid = "rgb"
    CAMsensor_cfg.resolution = [256,256]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [CAMsensor_cfg]

    return habitat_sim.Configuration(backend_cfg,[agent_cfg])

cfg = make_configuration()
env = habitat_sim.Simulator(cfg)
agent = env.initialize_agent(0) # "0" is the name of your agent
agent_state = agent.get_state() # The way to get your agent's state
# print(agent_state)
# env.navmesh_visualization = True # This will let you see the mesh of the map

observations = env.reset()
cv2.imshow('RGB',observations['rgb'])
is_end = False
while not is_end:
    keystroke = cv2.waitKey(0)
    if keystroke == ord('w'):
        observations = env.step("move_forward")
        print('forward')
    elif keystroke == ord('a'):
        observations = env.step("turn_left")
        print('left')
    elif keystroke == ord('d'):
        observations = env.step("turn_right")
        print('right')
    elif keystroke == ord('f'):
        is_end = True
        print('stop')
    else:
        print('invalid key')
    cv2.imshow('RGB', observations['rgb'])
    agent_state = agent.get_state()
    #print(agent_state)
    cv2.imshow('RGB', observations['rgb'])
    cv2.waitKey(0)