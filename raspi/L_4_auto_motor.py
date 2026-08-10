import afb1
import time

afb1.gpio.init()

try:
    while True:
        print("앞으로 전진")
        afb1.gpio.motor(70)   # 왼쪽 모터
        time.sleep(2)

        print("정지")
        afb1.gpio.motor(0)
        time.sleep(1)

        print("뒤로 후진")
        afb1.gpio.motor(-70)
        time.sleep(2)

        print("정지")
        afb1.gpio.motor(0)
        time.sleep(1)

except KeyboardInterrupt:
    print("사용자 종료")

finally:
    afb1.gpio.stop_all()
