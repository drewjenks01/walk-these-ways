from .sensor import Sensor
from .attached_camera_sensor import AttachedCameraSensor
from .floating_camera_sensor import FloatingCameraSensor
from .joint_position_sensor import JointPositionSensor
from .joint_velocity_sensor import JointVelocitySensor
from .orientation_sensor import OrientationSensor
from .heightmap_sensor import HeightmapSensor
from .rc_sensor import RCSensor
from .action_sensor import ActionSensor
from .last_action_sensor import LastActionSensor
from .clock_sensor import ClockSensor
from .yaw_sensor import YawSensor
from .object_sensor import ObjectSensor
from .timing_sensor import TimingSensor
from .body_velocity_sensor import BodyVelocitySensor
from .object_velocity_sensor import ObjectVelocitySensor
from .restitution_sensor import RestitutionSensor
from .friction_sensor import FrictionSensor
from .ground_friction_sensor import GroundFrictionSensor
from .ground_roughness_sensor import GroundRoughnessSensor
from .egomotion_sensor import EgomotionSensor
from .foot_force_sensor import FootForceSensor

ALL_SENSORS = {
                "ActionSensor": ActionSensor,
                "AttachedCameraSensor": AttachedCameraSensor,
                "BodyVelocitySensor": BodyVelocitySensor,
                "ClockSensor": ClockSensor,
                "EgomotionSensor": EgomotionSensor,
                "FloatingCameraSensor": FloatingCameraSensor,
                "FrictionSensor": FrictionSensor,
                "GroundFrictionSensor": GroundFrictionSensor,
                "GroundRoughnessSensor": GroundRoughnessSensor,
                "HeightmapSensor": HeightmapSensor,
                "JointPositionSensor": JointPositionSensor,
                "JointVelocitySensor": JointVelocitySensor,
                "LastActionSensor": LastActionSensor,
                "ObjectSensor": ObjectSensor,
                "ObjectVelocitySensor": ObjectVelocitySensor,
                "OrientationSensor": OrientationSensor,
                "RCSensor": RCSensor,
                "RestitutionSensor": RestitutionSensor,
                "TimingSensor": TimingSensor,
                "YawSensor": YawSensor,
                "FootForceSensor": FootForceSensor,
                }
