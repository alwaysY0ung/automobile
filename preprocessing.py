import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import DataLoader # 캐시 관리
from raw_data_processor import RawDataParser # step1 관련
from data_decoder import DataDecoder # step2 관련
import cantools
import os
import re

class TrainPreprocessor():
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

    def step2(self, df_merged_path='./cache/df_merged_0005.parquet',
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
                dd.initialize() # DataFrame의 고유 Identifier 목록을 가져오고 DBC 파일에서 메시지 및 시그널 정보를 파싱
                df_merged = dd.merge(time)
                
                # 파싱 dataframe을 cache에 저장
                name = self.df_merged_path.stem # str(self.df_merged_path).split('/')[2].split('.')[0]
                loader.save_dataframe(df_merged, name)
                print("\n파싱 dataframe 파일 Parquet 형식으로 저장 완료.")
        
        except Exception as e:
            print(f"오류 발생: {e}") # self.df1 = None

        self.df_merged = df_merged

        return self.df_merged
    
class TestPreprocessor():
    """
    사용 예시: TestPreprocessor 인스턴스를 유지하면서, 필요에 따라 step1()을 호출할 때마다 다른 공격 유형 목록을 지정하여 전처리를 수행할 수 있음
    test_pp = TestPreprocessor(...)
    results_fab = test_pp.step1(attack_types_to_process=['fabrication'])
    results_fuzz = test_pp.step1(attack_types_to_process=['fuzz'])
    results_all = test_pp.step1() # 모든 공격 유형
    """
    def __init__(self,
                 raw_data_dir='datasets/intrusion_datasets', # 디렉토리 경로
                 dbc_path='datasets/hyundai_2015_ccan.dbc',
                 cache_dir='./cache/'):
        self.raw_data_dir = Path(raw_data_dir) # Path 객체로
        self.dbc_path = dbc_path
        self.cache_dir = Path(cache_dir)
        self.loader = DataLoader(cache_dir=cache_dir) # 기존 DataLoader 재사용


    def step1(self, time='0.005s', attack_types_to_process=None):
        """
                Args:
                    time_resample_freq (str): 데이터를 리샘플링할 시간 간격.
                    attack_types_to_process (list, optional): 전처리할 공격 유형 목록 (예: ['fabrication', 'fuzz']).
                                                            None 또는 빈 리스트인 경우 입력한 경로에 있는 모든 파일, 즉 모든 공격 유형을 처리.
                Returns:
                    dict: 전처리 결과를 요약한 딕셔너리.
                        {'processed_attack_types': list, 'cache_paths': dict, 'message': str, 'status': str}
                        (어차피 캐싱되므로 데이터 자체를 return하는 것은 바람직하지 않을 것임.)
                """
        # 캐시 경로들을 저장할 딕셔너리 초기화
        processed_attack_types = []
        cache_paths_info = {}

        # 공격 유형별로 Raw 데이터 준비
        dfs = self._prepare_raw_data_by_attack_file(attack_types_to_process)
        if not dfs: return {} # 처리할 test datasets가 없는 경우

        # 개별 df에 대해 처리
        for attack_type, path_list in dfs.items():
            processed_attack_types.append(attack_type) # 처리할 공격 유형 리스트에 추가
            cache_paths_info[attack_type] = [] # 해당 공격 유형의 캐시 경로들을 담을 리스트

            # 공격 유형별 캐시 폴더 생성
            self.loader.mkdir(str(self.cache_dir / attack_type)) # DataLoader의 mkdir 사용

            for i, path in enumerate(path_list):
                processed_path = self.cache_dir / attack_type / f"test_{(path.stem).replace('dataset_','')}_{str(time).replace('.', '').replace('s', '')[1:]}.parquet"
                # 해당 개별 파일의 최종 결과가 이미 캐시되어 있는지 확인
                df = self.loader.check_exist(str(processed_path))
                print("1: ",df)
                if df is not None:
                    # print(f"    파일 '{path.name}' 최종 전처리된 데이터가 캐시({processed_path})에 존재하여 로드.")
                    continue # 다음 개별 파일로 넘어감
                
                # 캐시되지 않은 경우, 컬럼명이 변경된 원본 파일에서 DataFrame을 다시 로드하여 전처리 수행
                # print(f"    파일 '{path.name}' 데이터 전처리를 시작합니다. (캐시 없음)")
                try:
                    raw_df = pd.read_parquet(path) # 파일 경로를 사용하여 DataFrame 로드
                except Exception as e:
                    # print(f"경고: 전처리 중 '{path.name}' 파일 로드 실패: {e}. 이 파일은 건너뛰게 됨.")
                    continue # 파일 로드 실패 시 해당 파일 건너뛰기



                # (기존 train과 구조가 동일한 부분) <-> (시간 + label) 이렇게 두 부분으로 나누기
                # (기존 train과 구조가 동일한 부분): 기존 data_decoder.py의 deserialize 메서드 적용
                X_df = raw_df.drop(columns=['label'])
                print("2: ", raw_df)
                print("2: ", X_df)
                X_df = self._process_features_part(
                    df_raw_for_features=X_df,
                    time=time
                )
                print("전")
                # (시간 + label 처리 프로세스) (new): 현재 파일의 label_to_gt 메서드 이용
                y_df = raw_df[['Time', 'label']].copy()
                y_df = self.label_to_gt(y_df, time)
                print("3: ", raw_df)
                print("3: ", X_df)
                print("3: ", y_df)
                print("후")
                # 나눠서 처리한 두 부분 합치기 (new)
                y_df = y_df.loc[X_df.index]
                result = pd.merge(X_df, y_df, left_index=True, right_index=True, how='left')
                index_new = pd.timedelta_range(start=result.index.min(), end=result.index.max(), freq=time)
                result = result.reindex(index=index_new).fillna(0)
                result['label'] = result['label'].astype(int)
                print("4: ", result)
                # 캐시 경로에 저장: 기존 data_loader.py 이용
                self.loader.save_dataframe(result, processed_path, write=True)
                cache_paths_info[attack_type].append(str(processed_path)) # 최종 캐시 경로 정보 추가
        return {
            'processed_attack_types': processed_attack_types,
            'cache_paths': cache_paths_info
        }
    
    def get_attack_type_from_filename(self, filename): # models_main에서도 호출됨
        """
        개별 파일 이름에서 공격 유형과 세부 정보를 추출하는 메서드
        ex. 'dataset_fabrication_aid=044.parquet' -> ('fabrication', 'aid=044')"""
        attack_type = "unknown" # 알 수 없음, 기본 공격 유형
        sub_identifier = "unknown"

        # 공격유형
        match = re.match(r'dataset_([a-zA-Z]+)(_.*)?\.parquet', filename) # '+' : 앞의 형식이 한 글자 이상 반복되는 조건을 의미,  '?': (_.*)가 있을 수도 있고 없을 수도 있음을 의미
        if match == None:
            match = re.match(r'test([a-zA-Z]+)(_.*)?\.parquet', filename)
        if match:
            attack_type = match.group(1).lower() # 정규표현식 중 두 번째 그룹, 즉 ([a-zA-Z]+) 부분을 의미

        # 세부정보 - 정규식을 사용하여 `dataset_공격유형` 부분을 제외한 나머지 추출
        exp = Path(filename).stem # 확장자 제거
        if 'aid' in exp:
            sub_identifier = exp.split('_')[2]
        elif attack_type == 'fuzz': # fuzz
            sub_identifier = exp.split('=')[1]
        elif attack_type == 'replay': # replay
            parts = exp.split('=')
            sub_identifier = parts[1].replace('_to', '-') + parts[2] # 120_to=240   ->   120-240
        else: # 모르는 파일 형식일 경우, 전체 파일명을 식별자로 사용
            sub_identifier = exp
        
        return attack_type, sub_identifier
    
    def _prepare_raw_data_by_attack_file(self, attack_types_to_process=None): # TestProcessor 클래스 내부에서만 호출될 것이라는 표시를 위해 메서드 앞에 '_'를 붙였다
        """
        이때 요청된 공격 유형이 실제 파일로 존재하지 않으면 처리되지 않음. 별도 경고 없음.

        지정된 디렉토리에서 Parquet 테스트 파일을 로드,
        컬럼명을 DataDecoder가 기대하는 형태로 변경한 후, 원본 파일을 덮어씌워 저장.
        요청된 공격 유형별로 개별 파일의 경로 리스트를 딕셔너리로 반환.
        
        Args:
            attack_types_to_process (list, optional): 전처리할 공격 유형 목록. 
                                                     None 또는 빈 리스트인 경우 모든 공격 유형을 처리.
        Returns:
            dict: {attack_type: [Path, ...]} 형태의 딕셔너리.
                  각 공격 유형 내의 개별 파일 경로 리스트.
                                                                                        """
        
        all_files = list(self.raw_data_dir.glob('*.parquet'))
        all_files = [f for f in all_files if f.name.startswith('dataset_')] # 'dataset_'으로 시작하는 파일만 필터링
        
        df_lists_by_attack_type = {} # 공격 유형별로 DataFrame 리스트를 저장할 딕셔너리

        attack_types_in_files = set()
        for file_path in all_files:
            attack_type, _ = self.get_attack_type_from_filename(file_path.name) # 튜플 언팩킹
            # 전처리할 공격 유형 목록이 지정되지 않았거나(None/빈 리스트), 현재 파일의 공격 유형이 지정된 목록에 포함되어 있을 때만 처리
            if attack_types_to_process is None or len(attack_types_to_process) == 0:
                # attack_types_to_process가 None 또는 비어 있으면 모든 공격 유형을 처리
                pass # 이 경우 아무 것도 건너뛰지 않음
            elif attack_type not in attack_types_to_process:
                # attack_types_to_process가 지정되었지만 현재 attack_type이 목록에 없으면 건너뛰기
                continue

            attack_types_in_files.add(attack_type) # 처리되든 처리 안 되든 입력한 폴더 경로에 있는 모든 공격 유형을 담을 집합임

            if attack_type not in df_lists_by_attack_type:
                df_lists_by_attack_type[attack_type] = []

            df_lists_by_attack_type[attack_type].append(file_path)


        rename_map = { # 컬럼명 적절하게 바꾸기 (기존 data_decoder 메서드 재사용이 잘 동작하기 위해서임)
            'arbitration_id': 'Identifier',
            'dlc': 'DLC',
            'data': 'Data',
            'timestamp' : 'Time'
        }
        dfs = {} # 처리 예정인 파케이들
        for attack_type, file_paths in df_lists_by_attack_type.items(): # "dos": ["file1.parquet", "file2.parquet"]
            dfs[attack_type] = [] # 공격 유형별로 df가 모이게 됨
            for i, file_path in enumerate(file_paths): # ["file1.parquet", "file2.parquet"]의 요소를 인덱스와 함께 순회
                try:
                    df = pd.read_parquet(file_path)
                    actual_rename_map = {old_name: new_name for old_name, new_name in rename_map.items() if old_name in df.columns} # "있으면" 변경

                    if actual_rename_map: # rename 정의한 것에 해당하는 컬럼이 있으면
                        df.rename(columns=actual_rename_map, inplace=True)

                    # if df.index.name != 'Time': # 이미 전처리된 parquet파일일 수 있으므로
                    #     df = df.set_index('Time')
                    #     print("now: ", df)



                    df.to_parquet(file_path, index=False) 

                    dfs[attack_type].append(file_path)
                except Exception as e:
                    print(f"경고: 파일 '{file_path.name}' 로드/처리/저장 중 오류 발생: {e}. 건너뜀.")
        return dfs

    def _process_features_part(self, df_raw_for_features, time):
        """
        Test datasets의 'label외 컬럼들'을 처리.
        기존 DataDecoder 메서드를 재사용하여 deserialize 및 병합을 수행.
        """
        df = df_raw_for_features

        dd = DataDecoder(df, self.dbc_path)
        dd.initialize()
        result = dd.merge(time = time, flag = False) # time resample freq
        return result

    def label_to_gt(self, df_raw_for_labels, time): # gt = ground truth
        """
        Raw test 데이터에서 'label' 컬럼을 추출하고, 사용자 정의 처리를 거쳐
        Ground Truth 레이블 데이터프레임을 생성.
        """
        df = df_raw_for_labels.copy()

        if 'Time' in df.columns:
            print("label: ", df)
            df['Time'] = pd.to_timedelta(df['Time'], unit='s')
            df = df.set_index('Time')
            df.index = df.index.ceil(time)
            df = df.groupby(df.index)[['label']].max()
            # df = df.groupby(df.index)['label'].max() # keeplast연산 아님여 중복 Time에 대해 label은 최댓값 가져갈거니까요
            # index_new = pd.timedelta_range(start=df.index.min(), end=df.index.max(), freq=time)
            # df = df.reindex(index=index_new).fillna(0)
            # df['label'] = df['label'].astype(int)
        else: print("경고: 'Time' 컬럼이 Label DataFrame에 없음")
        return df