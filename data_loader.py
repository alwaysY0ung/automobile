import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Union, Optional # Union과 Optional 임포트 추가

class DataLoader:
    """데이터셋을 로드하고 캐싱하는 클래스"""
    
    def __init__(self, cache_dir='./cache/'):
        """
        데이터로더 초기화
        
        cache_dir (str): 캐시 디렉토리 경로
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True) # 기본 캐시 디렉토리가 존재하는지 확인하고 없으면 생성
    
    def _get_cache_path(self, filename: str) -> Path:
        """
        주어진 파일명(확장자 포함 또는 제외)에 대한 캐시 경로를 Path 객체로 반환.
        이 메서드는 filename이 단순 문자열일 때만 사용되어야 함.
        """
        # 파일명에서 확장자를 제거하고, .parquet 확장자를 다시 붙입니다.
        base_name = Path(filename).stem 
        return self.cache_dir / (base_name + '.parquet')
    
    # mkdir 메서드는 Path.mkdir(parents=True, exist_ok=True)로 대체 가능하여 
    # save_dataframe에서 직접 처리하도록 수정했으므로, 이 메서드는 더 이상 호출되지 않아도 됩니다.
    def mkdir(self, folder_path):
        Path(folder_path).mkdir(parents=True, exist_ok=True)
    
    def check_exist(self, file_path='./cache/df1.parquet') -> Optional[pd.DataFrame]:
        cache_path = Path(file_path)
        # 파일 존재 여부 확인
        if cache_path.exists():
            # 파일이 있다면 직접 pandas로 읽기
            df = pd.read_parquet(cache_path)
            return df
        else:
            return None
    
    def load_dataset(self, filename='can_dump1.txt') -> Optional[pd.DataFrame]:
        """
        데이터셋 로드 (캐시 사용)
        filename (str): 로드할 파일 이름 (예: 'df1.txt' 또는 'my_data')
        """
        # _get_cache_path는 filename이 단순한 파일 이름 문자열일 때만 사용됨
        cache_path = self._get_cache_path(filename) 
        
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        else: return None

    def save_dataframe(self, df: pd.DataFrame, name_or_path: Union[str, Path], write=False) -> bool:
        """
        데이터프레임을 Parquet 형식으로 캐시에 저장함.
        
        Args:
            df (DataFrame): 저장할 데이터프레임
            name_or_path (str or Path): 저장할 파일 이름 (확장자 제외) 또는 전체 Path 객체
            write (Boolean): 동일한 파일명의 파일이 이미 존재할 때, 덮어씌움 여부 (True: 덮어씌움)
            
        Returns:
            bool: 저장 성공 여부
        """
        final_save_path: Path

        if isinstance(name_or_path, Path):
            # name_or_path가 이미 Path 객체(즉, TestPreprocessor에서 넘어온 완전한 경로)이면, 
            # 이를 최종 저장 경로로 직접 사용.
            final_save_path = name_or_path
        else:
            # name_or_path가 문자열(즉, TrainPreprocessor에서 넘어온 파일 이름)이면, 
            # 이를 파일 이름으로 간주하고 DataLoader의 cache_dir과 결합.
            final_save_path = self.cache_dir / (name_or_path + '.parquet')
        
        # 파일이 저장될 부모 디렉토리가 존재하는지 확인하고, 없으면 생성.
        # 이 부분이 'cache/cache/fuzz' 오류를 해결하는 핵심.
        output_dir = final_save_path.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"디렉토리 '{output_dir}'를 생성했습니다.")
        
        # 파일이 이미 존재하는 경우 덮어쓰기 옵션에 따라 처리.
        if final_save_path.exists():
            if not write:
                print(f"'{final_save_path.name}' 파일이 이미 존재합니다. (덮어쓰지 않음)")
                return False
            else:
                print(f"'{final_save_path.name}' 파일이 이미 존재하지만 덮어씌웁니다.")
        
        print(f"DataFrame을 '{final_save_path}'에 저장 중...")
        try:
            # to_parquet는 기본적으로 TimedeltaIndex와 같은 인덱스를 저장.
            df.to_parquet(final_save_path) 
            print(f"'{final_save_path.name}' 파일 생성 완료.")
            return True
        except Exception as e:
            print(f"'{final_save_path.name}' 파일 생성 실패: {e}")
            return False

    def read_parquet(self, filename: str) -> Optional[pd.DataFrame]:
        """
        지정된 Parquet 파일 읽기
        
        Args:
            filename (str): 읽을 파일 이름 (확장자 제외)
            
        Returns:
            DataFrame 또는 None: 로드된 데이터프레임 또는 파일이 없을 경우 None
        """
        cache_path = self._get_cache_path(filename)
        
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            print(f"'{cache_path.name}' 파일 로드 완료. 크기: {df.shape}")
            return df
        else:
            print(f"'{cache_path.name}' 파일이 존재하지 않습니다.")
            return None