"""
[PC용 실시간 비디오 뷰어 클라이언트]
라즈베리파이에서 실행 중인 자율주행 스트리밍 서버(http://<라즈베리파이_IP>:5000/video_feed)의
화면을 PC의 OpenCV 창에서 초저지연으로 수신하고 확인하는 스크립트입니다.

사용법:
  python pc_viewer.py [라즈베리파이_IP]
  예: python pc_viewer.py 192.168.0.50
"""

import sys
import os
import time
import cv2

def main():
    # 1. 라즈베리파이 IP 주소 설정
    if len(sys.argv) > 1:
        raspi_ip = sys.argv[1].strip()
    else:
        raspi_ip = input("라즈베리파이 IP 주소를 입력하세요 (기본값: localhost): ").strip()
        if not raspi_ip:
            raspi_ip = "localhost"

    stream_url = f"http://{raspi_ip}:5000/video_feed"
    print(f"\n[연결 시도] {stream_url} 스트림에 연결 중...")

    # 2. 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"\n[오류] 스트림에 연결할 수 없습니다: {stream_url}")
        print("1. 라즈베리파이에서 'python auto_drive_stream.py'가 실행 중인지 확인하세요.")
        print("2. PC와 라즈베리파이가 동일한 Wi-Fi 네트워크에 연결되어 있는지 확인하세요.")
        print("3. 라즈베리파이 IP 주소가 올바른지 확인하세요.")
        return

    print("[연결 성공] 실시간 영상 수신 시작!")
    print("-----------------------------------------")
    print(" [단축키]")
    print("   's' : 현재 화면 PC에 스크린샷 저장")
    print("   'q' 또는 ESC : 뷰어 종료")
    print("-----------------------------------------\n")

    capture_dir = os.path.join(os.path.dirname(__file__), 'pc_captures')
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)

    window_name = f"AFB-1 Realtime Viewer [{raspi_ip}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[경고] 프레임을 수신할 수 없습니다. 재연결 대기 중...")
            time.sleep(0.5)
            continue

        # 클라이언트 단 수신 FPS 계산
        now = time.time()
        dt = now - prev_time
        if dt > 0:
            fps_display = 1.0 / dt
        prev_time = now

        # 화면 좌하단에 PC 수신 레이턴시/FPS 표시
        cv2.putText(
            frame,
            f"Client Recv FPS: {fps_display:.1f}",
            (15, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 100),
            2
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' 또는 ESC
            print("[종료] 뷰어를 종료합니다.")
            break
        elif key == ord('s'):  # 스크린샷 저장
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_file = os.path.join(capture_dir, f"pc_snap_{timestamp}.jpg")
            cv2.imwrite(save_file, frame)
            print(f"[저장] PC에 스크린샷 저장 완료: {save_file}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
