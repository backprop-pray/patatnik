#!/usr/bin/env python3
import time

from drivers.motor.hbridge import DualHBridgeMotorDriver

M1_IN1 = 20
M1_IN2 = 21
M2_IN1 = 16
M2_IN2 = 12
STEP_SECONDS = 2.0


def main():
    driver = DualHBridgeMotorDriver(
        m1_in1=M1_IN1,
        m1_in2=M1_IN2,
        m2_in1=M2_IN1,
        m2_in2=M2_IN2,
    )

    try:
        driver.set_both_forward()
        print(f'Both forward: M1({M1_IN1}=H, {M1_IN2}=L) M2({M2_IN1}=H, {M2_IN2}=L)')
        time.sleep(STEP_SECONDS)

        driver.set_both_reverse()
        print(f'Both reverse: M1({M1_IN1}=L, {M1_IN2}=H) M2({M2_IN1}=L, {M2_IN2}=H)')
        time.sleep(STEP_SECONDS)
    finally:
        driver.cleanup()
        print('Done. All pins set LOW and cleaned up.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
