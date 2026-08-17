"""주행 하드웨어 추상화.

둘 다 같은 인터페이스를 따른다: read() / servo(angle) / motor(speed) / shutdown().

    Afb1Hardware   — 라즈베리파이 실차 (afb1 모듈)
    ReplayHardware — 저장 이미지를 먹여 루프 로직만 검증 (개발 PC)

**read() 는 매번 새 배열을 돌려줘야 한다.** 주행 루프가 프레임을 백그라운드
워커(화면/YOLO/녹화)에 복사 없이 넘기기 때문이다 (loop.Worker 참조).
버퍼를 재사용하도록 바꾸면 워커가 처리하는 동안 내용이 덮어써진다.

afb1 API 는 raspi/ 예제에서 확인된 것만 쓴다:
    gpio.init() / gpio.stby(1) / gpio.servo(30~150) / gpio.motor(speed) / gpio.stop_all()
    camera.init(w, h, fps) / camera.get_image() / camera.release_camera()
"""
import os

import cv2

import config as cfg


class Afb1Hardware:
    """실제 afb1 하드웨어."""

    def __init__(self):
        try:
            import afb1
        except ImportError as exc:
            raise SystemExit(
                'afb1 모듈을 찾을 수 없습니다. 이 스크립트는 라즈베리파이에서 실행해야 합니다.\n'
                '개발 PC에서 루프 로직만 확인하려면: python3 drive.py --replay <이미지폴더>'
            ) from exc
        self.afb1 = afb1

        afb1.gpio.init()
        # raspi/L_5_Capture.py 가 주행 전에 호출한다. 모터 드라이버 standby 해제로
        # 보이며, 빠뜨리면 모터가 돌지 않을 수 있다.
        afb1.gpio.stby(1)
        afb1.camera.init(cfg.W, cfg.H, cfg.CAMERA_FPS)

    def read(self):
        frame = self.afb1.camera.get_image()
        if frame is None:
            return None
        if frame.ndim == 3 and frame.shape[2] == 4:      # RGBA 로 오는 경우 대비
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # raspi/L_5_Capture.py 가 저장 전에 하던 채널 스왑을 그대로 재현한다.
        # captures/ 의 이미지는 이 스왑을 거친 뒤 imwrite 된 것이라, 여기서
        # 똑같이 하지 않으면 오프라인에서 튜닝한 이진화가 다른 색을 보게 된다.
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def servo(self, angle):
        self.afb1.gpio.servo(int(angle))

    def motor(self, speed):
        self.afb1.gpio.motor(int(speed))

    def shutdown(self):
        try:
            self.afb1.gpio.stop_all()
        finally:
            try:
                self.afb1.camera.release_camera()
            except Exception:
                pass


class ReplayHardware:
    """하드웨어 없이 저장 이미지를 먹여 루프 로직만 검증한다.

    실차에서 확인할 수 없는 것(카메라, 서보, 모터)을 뺀 나머지 — 상태 기계,
    실패 카운팅, 평활, 종료 처리 — 를 개발 PC에서 돌려볼 수 있게 한다.
    """

    def __init__(self, directory, loop=False):
        import glob
        self.paths = sorted(
            p for p in glob.glob(os.path.join(directory, '*'))
            if p.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        if not self.paths:
            raise SystemExit(f'이미지가 없습니다: {directory}')
        self.i = 0
        self.loop = loop
        self.commands = []          # (kind, value) 기록 — 사후 점검용

    def read(self):
        if self.i >= len(self.paths):
            if not self.loop:
                return None
            self.i = 0
        img = cv2.imread(self.paths[self.i])
        self.i += 1
        return img

    def servo(self, angle):
        self.commands.append(('servo', int(angle)))

    def motor(self, speed):
        self.commands.append(('motor', int(speed)))

    def shutdown(self):
        pass
