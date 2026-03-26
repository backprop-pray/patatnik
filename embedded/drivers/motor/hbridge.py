import RPi.GPIO as GPIO


class DualHBridgeMotorDriver:
    _FORWARD = (1, 0)
    _REVERSE = (0, 1)
    _STOP = (0, 0)

    def __init__(self, left_in1=20, left_in2=21, right_in1=16, right_in2=12, **legacy_kwargs):
        if 'm1_in1' in legacy_kwargs:
            left_in1 = legacy_kwargs['m1_in1']
        if 'm1_in2' in legacy_kwargs:
            left_in2 = legacy_kwargs['m1_in2']
        if 'm2_in1' in legacy_kwargs:
            right_in1 = legacy_kwargs['m2_in1']
        if 'm2_in2' in legacy_kwargs:
            right_in2 = legacy_kwargs['m2_in2']

        self.left_in1 = left_in1
        self.left_in2 = left_in2
        self.right_in1 = right_in1
        self.right_in2 = right_in2
        self.pins = (self.left_in1, self.left_in2, self.right_in1, self.right_in2)

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    def _normalize_direction(self, direction):
        value = str(direction).strip().lower()
        if value in ('forward', 'f', '1', '+1'):
            return self._FORWARD, 'forward'
        if value in ('backward', 'reverse', 'back', 'b', 'r', '-1'):
            return self._REVERSE, 'backward'
        if value in ('stop', 's', '0', 'brake'):
            return self._STOP, 'stop'
        raise ValueError(f'Invalid direction: {direction}')

    def _normalize_side(self, side):
        value = str(side).strip().lower()
        if value in ('left', 'l', 'm1', 'left_motor'):
            return 'left'
        if value in ('right', 'r', 'm2', 'right_motor'):
            return 'right'
        raise ValueError(f'Invalid side: {side}')

    def _set_left_tuple(self, state):
        in1, in2 = state
        GPIO.output(self.left_in1, GPIO.HIGH if in1 else GPIO.LOW)
        GPIO.output(self.left_in2, GPIO.HIGH if in2 else GPIO.LOW)

    def _set_right_tuple(self, state):
        in1, in2 = state
        GPIO.output(self.right_in1, GPIO.HIGH if in1 else GPIO.LOW)
        GPIO.output(self.right_in2, GPIO.HIGH if in2 else GPIO.LOW)

    def set_motor(self, side, direction):
        side_name = self._normalize_side(side)
        state, normalized_direction = self._normalize_direction(direction)

        if side_name == 'left':
            self._set_left_tuple(state)
        else:
            self._set_right_tuple(state)

        return {'side': side_name, 'direction': normalized_direction}

    def drive(self, left_direction, right_direction):
        left_state, left_norm = self._normalize_direction(left_direction)
        right_state, right_norm = self._normalize_direction(right_direction)

        self._set_left_tuple(left_state)
        self._set_right_tuple(right_state)

        return {'left': left_norm, 'right': right_norm}

    def set_states(self, m1_in1, m1_in2, m2_in1, m2_in2):
        self._set_left_tuple((1 if m1_in1 else 0, 1 if m1_in2 else 0))
        self._set_right_tuple((1 if m2_in1 else 0, 1 if m2_in2 else 0))

    def set_both_forward(self):
        self.drive('forward', 'forward')

    def set_both_reverse(self):
        self.drive('backward', 'backward')

    def stop(self):
        self.drive('stop', 'stop')

    def cleanup(self):
        self.stop()
        GPIO.cleanup(self.pins)
