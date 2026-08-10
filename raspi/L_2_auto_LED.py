import afb1
import time

afb1.gpio.init()  # GPIO 초기화

try:
    while True:
        print("왼쪽 전조등 ON")
        afb1.gpio.led(True, False)
        time.sleep(1)

        print("오른쪽 전조등 ON")
        afb1.gpio.led(False, True)
        time.sleep(1)

        print("전조등 OFF")
        afb1.gpio.led(False, False)
        time.sleep(1)

except KeyboardInterrupt:
    print("LED 제어 종료")

finally:
    afb1.gpio.stop_all()  # 전체 해제