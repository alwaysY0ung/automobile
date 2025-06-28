import pandas as pd
import cantools
from pathlib import Path
import re # 정규 표현식 모듈 임포트
from typing import Union, Optional

class MinMaxScalerFromDBC:
    def __init__(self, dbc_file_path: Union[str, Path] = 'datasets/hyundai_2015_ccan.dbc'):
        self.dbc_file_path = Path(dbc_file_path)
        self.db = None
        self.signal_ranges = {}
        self._load_dbc_and_parse_ranges()

    def _load_dbc_and_parse_ranges(self):
        """
        DBC 파일을 로드하고 각 신호의 Min/Max 값을 파싱.
        각 신호의 이름과 해당 스케일링 정보를 딕셔너리에 저장.
        예: {'ID_SIGNALNAME': {'min': min_val, 'max': max_val}}
        """
        try:
            self.db = cantools.database.load_file(str(self.dbc_file_path))
            print(f"DBC 파일 '{self.dbc_file_path}' 로드 완료.")
        except Exception as e:
            print(f"DBC 파일 로드 중 오류 발생: {e}")
            self.db = None
            return

        for message in self.db.messages:
            message_id = message.frame_id
            for signal in message.signals:
                # DBC 파일의 신호 이름과 데이터프레임 컬럼 이름이 'ID_SIGNALNAME' 형식에 맞게 조합
                # ID를 16진수 3자리로 포맷팅 (예: 044, 5A0)
                column_name = f"{message_id:03X}_{signal.name}"
                
                if signal.is_float or (signal.minimum is not None and signal.maximum is not None):
                    self.signal_ranges[column_name] = {
                        'min': signal.minimum,
                        'max': signal.maximum
                    }
                else:

                    self.signal_ranges[column_name] = {
                        'min': signal.minimum if signal.minimum is not None else 0.0,
                        'max': signal.maximum if signal.maximum is not None else (2**signal.length - 1) * signal.scale + signal.offset
                    }
                # 예외처리
                if self.signal_ranges[column_name]['min'] == self.signal_ranges[column_name]['max']:
                    self.signal_ranges[column_name]['min'] = 0.0 # 스케일링 불가하므로 0~1 범위의 중간값으로 스케일링되도록 기본값 설정
                    self.signal_ranges[column_name]['max'] = 1.0


    def scale(self, df: pd.DataFrame) -> pd.DataFrame:

        df_scaled = df.copy()
        
        # 'label' 컬럼이 있으면 따로 보관
        label_column = None
        if 'label' in df_scaled.columns:
            label_column = df_scaled['label']
            df_scaled = df_scaled.drop(columns=['label'])
            print("'label' 컬럼은 스케일링에서 제외됩니다.")

        columns_to_scale = []
        missing_columns = []

        # 스케일링할 컬럼 필터링
        for col in df_scaled.columns:
            if col in self.signal_ranges:
                columns_to_scale.append(col)
            else:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"경고: 다음 컬럼들은 DBC 파일에서 Min/Max 정보를 찾을 수 없어 스케일링에서 제외됩니다: {missing_columns}")

        # 선택된 컬럼들에 대해 스케일링 적용
        for col in columns_to_scale:
            min_val = self.signal_ranges[col]['min']
            max_val = self.signal_ranges[col]['max']

            if max_val == min_val:
                # min과 max가 같으면 스케일링 무의미 (0으로 채우거나 0.5로 채우는 등 정책 필요)
                # 여기서는 0으로 채우도록 합니다.
                df_scaled[col] = 0.0
                print(f"경고: 컬럼 '{col}'의 Min/Max 값이 동일하여 0으로 스케일링되었습니다. (Min: {min_val}, Max: {max_val})")
            else:
                df_scaled[col] = (df_scaled[col] - min_val) / (max_val - min_val)
                # 스케일링 후 값이 [0, 1] 범위를 벗어날 수 있으므로 클리핑 (이상치 처리)
                df_scaled[col] = df_scaled[col].clip(0.0, 1.0)
                # print(f"컬럼 '{col}' 스케일링 완료. (Min: {min_val}, Max: {max_val})")
        
        # 'label' 컬럼이 있었다면 다시 추가
        if label_column is not None:
            df_scaled['label'] = label_column
            if pd.api.types.is_float_dtype(df_scaled['label']) and (df_scaled['label'] == df_scaled['label'].astype(int)).all():
                 df_scaled['label'] = df_scaled['label'].astype(int)
            elif pd.api.types.is_integer_dtype(df['label']): # 원본이 정수였다면
                 df_scaled['label'] = df_scaled['label'].astype(int) # 다시 정수형으로 변환 시도
            print("'label' 컬럼이 다시 데이터프레임에 추가되었습니다.")

        return df_scaled
