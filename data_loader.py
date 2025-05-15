import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from tqdm import trange
import os

class DataLoader:
    """데이터셋을 로드하고 캐싱하는 클래스"""
    
    def __init__(self, cache_dir='./cache/'):
        """
                            데이터로더 초기화
                            
                            cache_dir (str): 캐시 디렉토리 경로
                            """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, filename='df1.txt'):
        """
                            주어진 파일명에 대한 캐시 경로 반환
                            
                            filename (str): 원본 파일명
                                
                            Return: Path 캐시 파일 경로 (.parquet 형식)
                            """
        cache_path = self.cache_dir.joinpath(filename)
        return cache_path.with_suffix('.parquet')
    
    def mkdir(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    def check_exist(self, file_path='./cache/df1.parquet'):
        cache_path = Path(file_path)
        # 파일 존재 여부 확인
        if cache_path.exists():
            # 파일이 있다면 직접 pandas로 읽기
            df = pd.read_parquet(cache_path)
            return df
        else:
            return None
    
    def load_dataset(self, filename='can_dump1.txt'):
        """
                            데이터셋 로드 (캐시 사용)
                            
                            filename (str): 로드할 파일 이름
                                
                            Return: DataFrame, 로드된 데이터프레임
                            """
        cache_path = self._get_cache_path(filename)
        
        # 캐시가 있으면 캐시에서 로드
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        else: return None

    def save_dataframe(self, df, filename, write=False):
        """
                            데이터프레임을 Parquet 형식으로 캐시에 저장
                            
                            df (DataFrame): 저장할 데이터프레임
                            filename (str): 저장할 파일 이름 (확장자 제외)
                            write (Boolean): 동일한 파일명의 파일이 이미 존재할 때, 덮어씌움 여부 (True: 덮어씌움)
                                
                            Return: bool 저장 성공 여부
                            """
        # 캐시 경로 가져오기
        cache_path = self._get_cache_path(filename)
        
        # 파일이 이미 존재하는지 확인
        if cache_path.exists() and not write:
            print(f"{cache_path.name} 파일이 이미 존재합니다.")
            return False
        
        # 파일이 이미 존재하는지 확인
        if cache_path.exists() and write:
            print(f"{cache_path.name} 파일이 이미 존재하지만 덮어씌웁니다.")
            df.to_parquet(cache_path)
            return True
        
        # 데이터프레임을 Parquet 형식으로 저장
        df.to_parquet(cache_path)
        
        # 저장 확인
        if cache_path.exists():
            print(f"{cache_path.name} 파일 생성 완료.")
            return True
        else:
            print(f"{cache_path.name} 파일 생성 실패.")
            return False
            
    def read_parquet(self, filename):
        """
        지정된 Parquet 파일 읽기
        
        매개변수:
            filename (str): 읽을 파일 이름 (확장자 제외)
            
        반환:
            DataFrame 또는 None: 로드된 데이터프레임 또는 파일이 없을 경우 None
        """
        cache_path = self._get_cache_path(filename)
        
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            print(f"{cache_path.name} 파일 로드 완료. 크기: {df.shape}")
            return df
        else:
            print(f"{cache_path.name} 파일이 존재하지 않습니다.")
            return None

# from hashlib import md5
# enc = hashlib.md5()

# # 문자열 -> 바이너리 ->