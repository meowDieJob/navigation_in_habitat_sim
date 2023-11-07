import cv2
import habitat_sim
import numpy as np
import magnum as mn
import attr

@attr.s(auto_attribs=True, slots=True)
class MoveAndSpinSpec:
    forward_amount: float
    spin_amount: float


@habitat_sim.registry.register_move_fn(body_action=True)
class MoveForwardAndSpin(habitat_sim.SceneNodeControl):
    def __call__(
            self, scene_node: habitat_sim.SceneNode, actuation_spec: MoveAndSpinSpec
    ):
        forward_ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.FRONT
        )
        scene_node.translate_local(forward_ax * actuation_spec.forward_amount)

        # Rotate about the +y (up) axis
        rotation_ax = habitat_sim.geo.LEFT
        scene_node.rotate_local(mn.Deg(actuation_spec.spin_amount), rotation_ax)
        # Calling normalize is needed after rotating to deal with machine precision errors
        scene_node.rotation = scene_node.rotation.normalized()


def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "/home/yuanzhao/PycharmProjects/navigation/data/scene_datasets/gibson/Spencerville_new_plant.glb"



    # agent configuration
    CAMsensor_cfg = habitat_sim.CameraSensorSpec()
    CAMsensor_cfg.sensor_type = habitat_sim.sensor.SensorType.COLOR
    CAMsensor_cfg.uuid = "rgb"
    CAMsensor_cfg.resolution = [480,640]
    CAMsensor_cfg.hfov = 50
    CAMsensor_cfg.position = [0,0,0]


    depthSensor_cfg = habitat_sim.CameraSensorSpec()
    depthSensor_cfg.sensor_type = habitat_sim.sensor.SensorType.DEPTH
    depthSensor_cfg.uuid = "depth"
    depthSensor_cfg.resolution = [480,640]
    depthSensor_cfg.position=[0,0,0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [CAMsensor_cfg,depthSensor_cfg]
    print(agent_cfg.action_space)

    return habitat_sim.Configuration(backend_cfg,[agent_cfg])

cfg = make_configuration()
env = habitat_sim.Simulator(cfg)
habitat_sim.registry.register_move_fn(
        MoveForwardAndSpin, name="lookup", body_action=True
    )
agent = env.initialize_agent(0) # "0" is the name of your agent
agent.agent_config.action_space["lookup"] = habitat_sim.ActionSpec(
        "lookup", MoveAndSpinSpec(0.0, 45.0)
    )
agent.agent_config.action_space["lookdown"] = habitat_sim.ActionSpec(
        "lookup", MoveAndSpinSpec(0.0, -45.0)
    )
agent.agent_config.action_space['move_forward'].actuation.amount = 0.0378
agent.agent_config.action_space['turn_left'].actuation.amount = 5.216
agent.agent_config.action_space['turn_right'].actuation.amount = 5.216
agent_state = agent.get_state() # The way to get your agent's state

print(agent_state)
# env.navmesh_visualization = True # This will let you see the mesh of the map

observations = env.reset()
# agent.set_state(zero_state)
print(observations)
obs_depth = observations['depth']
cv2.imshow("DEPTH",obs_depth)
cv2.imshow('RGB',observations['rgb'])




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
    elif keystroke == ord('t'):
        observations = env.step("lookup")
    elif keystroke == ord('y'):
        observations = env.step("lookdown")
        # print('right')
    elif keystroke == ord('f'):
        is_end = True
        # print('stop')
    else:
        print('invalid key')

    obs_depth = observations['depth']
    cv2.imshow("DEPTH", obs_depth)
    cv2.imshow('RGB', observations['rgb'])
    agent_state = agent.get_state()

