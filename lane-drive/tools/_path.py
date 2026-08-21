"""상위 폴더(project_v2)를 import 경로에 넣는다.

이 폴더의 도구들은 한 단계 위의 config / bev / binarize / detect / control 을
쓰는데, `python3 tools/evaluate.py` 로 실행하면 sys.path[0] 가 tools/ 라서
그대로는 찾지 못한다. 각 도구가 형제 모듈보다 **먼저** 이 모듈을 import 한다.

    import _path  # noqa: F401
    import config as cfg

부수효과가 목적인 import 라 이름이 쓰이지 않는다 — 지우면 ImportError 가 난다.
"""
import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
