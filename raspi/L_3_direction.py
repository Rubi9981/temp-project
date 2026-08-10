import afb1
import time

afb1.gpio.init()  # GPIO 초기화

try:
    while True:
        angle = int(input("조향 각도 입력 (30~150): "))
        afb1.gpio.servo(angle)
        time.sleep(0.5)

except KeyboardInterrupt:
    print("사용자 종료")

finally:
    afb1.gpio.stop_all()  # 전체 모터/LED/서보 정지

