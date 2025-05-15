"""
data_collect 패키지 초기화 모듈
 - collect_data() 함수를 외부에서 바로 import 할 수 있도록 노출
"""
from .run_data_collection import run as collect_data  # noqa: F401
