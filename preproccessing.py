import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import DataLoader # 캐시 관리
from raw_data_processor import RawDataParser # step1 관련
from data_decoder import DataDecoder # step2 관련
import cantools

class Preprocessor():
    def __init__(self,
                 raw_path='./datasets/logfile2022-11-01_11-56-50.txt',
                 cache_path='./cache/df1.parquet'):
        self.raw_path = raw_path
        self.cache_path = Path(cache_path)
        self.df1 = None
        self.Identifiers = None
        self.df_merged_path = None
        self.df_merged = None
        self.loader = DataLoader()

    def step1(self):
        df1 = self.df1
        try:
            # deserialize 전 cache 저장한 dataframe 불러오기
            loader = self.loader
            
            if self.cache_path.exists(): # ex) './cache/df1.parquet'
                df1 = pd.read_parquet(self.cache_path)

            else: # cache에 파싱한 dataframe이 저장되어있지 않은 경우
                loader.mkdir('./cache') # cache 폴더가 없을 경우 생성
                
                print("파싱된 데이터가 캐시에 존재하지 않아, raw data 파싱을 시작.")
                # load raw dataset
            
                # 데이터 읽기 + DataFrame 변환
                rdp = RawDataParser(self.raw_path)
                df1 = rdp.parse_txt()
                
                # 파싱 dataframe을 cache에 저장
                name = self.cache_path.stem # str(self.cache_path).split('/')[2].split('.')[0]
                loader.save_dataframe(df1, name)
                print("\n파싱 dataframe 파일 Parquet 형식으로 저장 완료.")
        
        except Exception as e:
            print(f"오류 발생: {e}") # self.df1 = None

        self.df1 = df1
        return self.df1

    def step2(self, df_merged_path='./cache/df_merged_0025.parquet',
              time=False,
              dbc_path='datasets/hyundai_2015_ccan.dbc'):
        """
                            desirialize하는 메서드

                            df_merged_path
                            time
                            dbc_path

                            Returns: 모든 aid에 대해 merge된 데이터프레임
                            """
        self.df_merged_path = Path(df_merged_path) # './cache/df_merged_0.0025s.parquet'
        df_merged = None  # <-- 기본값 초기화
        
        try:
            loader = self.loader
            
            if self.df_merged_path.exists():
                df_merged = pd.read_parquet(self.df_merged_path)

            else: # cache에 파싱한 dataframe이 저장되어있지 않은 경우
                dd = DataDecoder(self.df1, dbc_path)
                dd.initialize()
                df_merged = dd.merge(time)
                
                # 파싱 dataframe을 cache에 저장
                name = self.df_merged_path.stem # str(self.df_merged_path).split('/')[2].split('.')[0]
                loader.save_dataframe(df_merged, name)
                print("\n파싱 dataframe 파일 Parquet 형식으로 저장 완료.")
        
        except Exception as e:
            print(f"오류 발생: {e}") # self.df1 = None

        self.df_merged = df_merged

        return self.df_merged