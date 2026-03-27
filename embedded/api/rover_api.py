from drivers.gps.provider import GPSProvider
from drivers.motor.hbridge import DualHBridgeMotorDriver
from drivers.sensors.ultrasonic_array import DualUltrasonicArray
from drivers.camera.picam2 import PiCam2FrameDriver


class RoverAPI:
    def __init__(
        self,
        gps_port='/dev/ttyAMA0',
        gps_baud=9600,
        gps_fallback_file='/home/yasen/gps_fallback.env',
        left_motor_pins=(20, 21),
        right_motor_pins=(16, 12),
        ultrasonic1_pins=(23, 24),
        ultrasonic2_pins=(27, 17),
        ultrasonic3_pins=(5, 6),
    ):
        self.gps = GPSProvider(port=gps_port, baud=gps_baud, fallback_file=gps_fallback_file)
        self.ultrasonic = DualUltrasonicArray(
            sensor1_trig=ultrasonic1_pins[0],
            sensor1_echo=ultrasonic1_pins[1],
            sensor2_trig=ultrasonic2_pins[0],
            sensor2_echo=ultrasonic2_pins[1],
            sensor3_trig=ultrasonic3_pins[0],
            sensor3_echo=ultrasonic3_pins[1],
        )
        self.motor = DualHBridgeMotorDriver(
            left_in1=left_motor_pins[0],
            left_in2=left_motor_pins[1],
            right_in1=right_motor_pins[0],
            right_in2=right_motor_pins[1],
        )
        self.camera = PiCam2FrameDriver()

    def get_gps_values(self, timeout_seconds=2.0, allow_fallback=True):
        return self.gps.get_position(timeout_seconds=timeout_seconds, allow_fallback=allow_fallback)

    def get_gsm_values(self, timeout_seconds=2.0, allow_fallback=True):
        return self.get_gps_values(timeout_seconds=timeout_seconds, allow_fallback=allow_fallback)

    def get_ultrasonic(self, sensor_id=None, timeout_seconds=0.015):
        if sensor_id is None:
            return self.ultrasonic.read_all(timeout_seconds=timeout_seconds)
        return self.ultrasonic.read_sensor(sensor_id=sensor_id, timeout_seconds=timeout_seconds)

    def set_motor(self, side, direction):
        return self.motor.set_motor(side=side, direction=direction)

    def drive(self, left_direction, right_direction):
        return self.motor.drive(left_direction=left_direction, right_direction=right_direction)

    def stop_motors(self):
        self.motor.stop()

    def getframe(self):
        return self.camera.take_picture()

    def take_picture(self):
        return self.getframe()

    def get_camera_frame(self):
        return self.getframe()

    def close(self):
        self.gps.close()
        self.motor.cleanup()
        self.ultrasonic.cleanup()
        self.camera.close()
