# deserialize.py
# 데이터 역직렬화 및 Pandas DataFrame 변환 관련 함수

import pandas as pd
import numpy as np
import cantools
from data_scaler import MinMaxScalerFromDBC

class DataDecoder:
    def __init__(self, df, dbc_file_path='datasets/hyundai_2015_ccan.dbc'):
        self.df = df
        self.dbc_file_path = dbc_file_path
        self.Identifiers = None
        self.parsed_ID = None # parsing_identifiers
        self.db_2015_ccan = cantools.database.load_file(dbc_file_path)

    def initialize(self):
        self.Identifiers = self.df["Identifier"].unique().tolist() # ['902', '356', '1427', ..., ]
        self.parsed_ID = self.parsing_identifiers(self.dbc_file_path) # parsing_identifiers

    def parsing_identifiers(self, dbc_file_path):
        """
        cantools를 사용하여 dbc 파일에서 메시지 ID, 메시지 이름, 시그널 목록을 파싱합니다.

        Returns:
            dict: {
                message_id: {
                    "name": 메시지 이름 (str),
                    "signals": [시그널1, 시그널2, ...] (list of str)
                },
                ...
            }
        """
        db = cantools.database.load_file(self.dbc_file_path)
        result = {}

        for message in db.messages:
            message_id = message.frame_id
            message_name = message.name
            signal_names = [signal.name for signal in message.signals]
            
            result[message_id] = {
                "name": message_name,
                "signals": signal_names
            }

        return result


    def parsing_identifiers_by_hand(self, dbc_file_path):
        """
                            dbc_file_path: can bus 정보가 있는 파일의 경로 ('datasets/hyundai_2015_ccan.dbc')
                            
                            Return

                            {
                                'ID': {
                                    'name': '메시지 이름',
                                    'signals': ['시그널1', '시그널2', '시그널3', ...]
                                },
                                ...
                            }

                            # 사용 예
                            parsed_ID = (parsing_identifiers('datasets/hyundai_2015_ccan.dbc'))
                            print(parsed_ID)
                            """
        result = {}
        
        self.dbc_file_path = dbc_file_path
        with open(self.dbc_file_path, 'r') as file:
            lines = file.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('BO_'):
                # BO_ 라인 파싱
                parts = line.split()
                if len(parts) >= 3:  # BO_ 1412 AAF11: 8 AAF 형식 확인
                    identifier = int(parts[1])  # CAN ID (예: 1412)
                    message_name = parts[2].rstrip(':')  # 메시지 이름 (예: AAF11)
                    
                    # 결과 딕셔너리에 메시지 ID를 키로 하고 시그널 목록을 저장할 리스트 초기화
                    result[identifier] = {"name": message_name, "signals": []}
                    
                    # 다음 라인부터 순회하면서 SG_ 시그널 정보 수집
                    j = i + 1
                    while j < len(lines) and lines[j].strip().startswith('SG_'):
                        signal_line = lines[j].strip()
                        signal_parts = signal_line.split()
                        if len(signal_parts) >= 2:
                            signal_name = signal_parts[1]  # 시그널 이름
                            result[identifier]["signals"].append(signal_name)
                        j += 1
                    
                    # 다음 BO_ 찾기 위해 인덱스 업데이트
                    i = j - 1  # 다음 반복에서 i += 1이 실행되므로 j-1로 설정
            i += 1
        return result
    
    def merge(self, time='0.005s', flag=True):
        parsed_ID = self.parsed_ID
        store = []
        t = time
        ignore_aid = []

        # 각 그룹(unique한 Identifier)에 대해 for문 실행
        for identifier in self.Identifiers:
            print(f"\n\n{(self.Identifiers.index(identifier)+1)}번 째 / 전체 {len(self.Identifiers)} 번")
            identifier = int(identifier)
            print(f"현재 처리 중인 Identifier: {identifier}")
            if identifier in parsed_ID:

                print(f"{identifier}가 CAN.dbc 파일에 존재합니다. 인덱스는 {t}이며, Deserialize 작업을 시작합니다.")
                
                # deserialize 함수 호출하여 데이터 변환
                new_df = self.deserialize(identifier, t, flag)
                store.append(new_df)
                
            else:
                print(f"{identifier}(이)가 list에 없거나 무언가 잘못되었습니다.")
                ignore_aid.append(identifier)

            print(f"처리되지 않은 aid: {ignore_aid}")

        # concat은 루프 밖에서하고, concat한 데이터프레임 출력
        df_merged = pd.concat(store, axis=1, verify_integrity=False)
        print("1\n",df_merged)
        df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
        print("2\n",df_merged)
        # df_merged = df_merged.ffill().dropna() # ffill -> dropna -> 107개 추출
        print("3\n",df_merged)
        the_107_signals = [
            '044_CR_Datc_OutTempC', '044_CF_Datc_IncarTemp', '080_PV_AV_CAN', '080_N', '080_TQI_ACOR',
            '080_TQFR', '080_TQI', '081_BRAKE_ACT', '081_CF_Ems_EngOperStat', '081_CR_Ems_IndAirTemp',
            '111_TQI_TCU_INC', '111_SWI_GS', '111_TQI_TCU', '111_SWI_CC', '112_VS_TCU', '112_N_INC_TCU',
            '112_VS_TCU_DECIMAL', '113_SLOPE_TCU', '113_CF_Tcu_TarGr', '113_CF_Tcu_ShfPatt', '113_CF_Tcu_ActEcoRdy',
            '162_Clutch_Driving_Tq', '162_Cluster_Engine_RPM', '162_Cluster_Engine_RPM_Flag', '18F_R_PAcnC', '18F_TQI_B',
            '18F_R_NEngIdlTgC', '200_FCO', '220_LAT_ACCEL', '220_LONG_ACCEL', '220_CYL_PRES', '220_YAW_RATE',
            '251_CR_Mdps_StrTq', '251_CR_Mdps_OutTq', '260_TQI_MIN', '260_TQI', '260_TQI_TARGET', '260_TQI_MAX',
            '260_CF_Ems_AclAct', '2B0_SAS_Angle', '2B0_SAS_Speed', '316_PUC_STAT', '316_TQI_ACOR', '316_N', '316_TQI',
            '316_TQFR', '316_VS', '329_TEMP_ENG', '329_MAF_FAC_ALTI_MMV', '329_CLU_ACK', '329_BRAKE_ACT', '329_TPS',
            '329_PV_AV_CAN', '381_CR_Mdps_StrAng', '383_CR_Fatc_OutTemp', '383_CR_Fatc_OutTempSns', '386_WHL_SPD_FL',
            '386_WHL_SPD_FR', '386_WHL_SPD_RL', '386_WHL_SPD_RR', '387_WHL_PUL_FL', '387_WHL_PUL_FR', '387_WHL_PUL_RL',
            '387_WHL_PUL_RR', '47F_ROL_CNT_ESP', '4F1_CF_Clu_VanzDecimal', '4F1_CF_Clu_Vanz', '4F1_CF_Clu_DetentOut',
            '50C_CF_Clu_AvgFCI', '50C_CF_Clu_DTE', '52A_CF_Clu_VehicleSpeed', '541_CF_Gway_TurnSigLh',
            '541_CF_Gway_HeadLampLow', '541_CF_Gway_TurnSigRh', '545_AMP_CAN', '545_BAT_Alt_FR_Duty',
            '545_VB', '545_TEMP_FUEL', '547_ECGPOvrd', '547_IntAirTemp', '549_BAT_SNSR_I', '549_BAT_SOC',
            '549_BAT_SNSR_V', '549_BAT_SNSR_Temp', '553_CF_Gway_AutoLightValue', '553_CF_Gway_AvTail',
            '553_CF_Gway_ExtTailAct', '553_CF_Gway_IntTailAct', '555_CR_Fpcm_LPActPre', '556_PID_04h',
            '556_PID_05h', '556_PID_0Ch', '556_PID_0Dh', '556_PID_11h', '557_PID_0Bh', '557_PID_23h',
            '58B_CF_Lca_Stat', '58B_CF_Lca_IndLeft', '58B_CF_Lca_IndRight', '58B_CF_Lca_IndBriLeft',
            '58B_CF_Lca_IndBriRight', '593_PRESSURE_FL', '593_PRESSURE_FR', '593_PRESSURE_RL',
            '593_PRESSURE_RR', '5A0_CF_Acu_Dtc', '5B0_CF_Clu_Odometer'
        ]
        df_merged = df_merged[[col for col in the_107_signals if col in df_merged.columns]] # df_merged에 107컬럼에 존재하지 않는 컬럼명이 있으면 에러 발생하므로 이를 방지
                                                                                            # df_merged > 107컬럼 포함관계라면 df_merged = df_merged[the_107_signals] 로 빠르게 연산 가능
        print("4\n",df_merged)
        df_merged = df_merged.ffill().dropna() # ffill -> dropna -> 107개 추출
        return df_merged

    def deserialize(self, identifier, time=False, flag=True):
        df = self.df
        db_2015_ccan = self.db_2015_ccan
        """
                            identifier => 정수 입력
                            df => Time	CAN	Identifier	Flags	DLC	Data	Counter 컬럼들이 있는 데이터프레임

                            # 실행 예
                            세번째 파라미터는 시간 형식으로 입력하거나, 아예 입력하지 않아야 한다.
                            df_deserialized_902 = deserialize(902, df1,'0.25s')
                            df_deserialized_902 = deserialize(902, df1)
                            """
        # filtered_df = df[df["Identifier"]==str(identifier)]
        filtered_df = df[df["Identifier"].astype(str) == str(identifier)] # 6월 말, test 데이터 전처리 중 문제가 생겨서 df 컬럼에도 str 변환 추가함


        # Data컬럼 solution1: apply 메서드
        filtered_df = filtered_df.copy()
        filtered_df['Decoded'] = filtered_df['Data'].apply(lambda x: db_2015_ccan.decode_message(identifier, x))
        # 결과를 데이터프레임으로 변환
        # decoded_series에서 딕셔너리 목록 추출
        decoded_dicts = filtered_df['Decoded'].tolist()
        
        # 딕셔너리 목록을 데이터프레임으로 변환
        result = pd.DataFrame(decoded_dicts, index=filtered_df.index)

        # 원본 데이터프레임의 Time 컬럼 추가
        result['Time'] = filtered_df['Time']

        
        # Time 컬럼을 datetime 형식으로 변환
        result['Time'] = pd.to_timedelta(result['Time'], unit='s')  # 초 단위로 해석
        
        # Time을 인덱스로 설정
        result = result.set_index('Time')

        print(f"time duplicated 처리 전 데이터프레임의 길이: {len(result)}")

        # 파라미터로 입력한 시간을 기준으로 반올림 후 중복된 시간 인덱스 중 가장 마지막 인덱스만 살리는 작업
        if not time:
            # time값이 입력되지 않았으므로 모든 행을 그대로 둠
            pass
        else:
            # time값이 입력되었으므로 작업을 수행함
            result.index = result.index.ceil(time)
            # print(result.to_string())
            result = result[~result.index.duplicated(keep='last')]
            # print(f"time duplicated 처리 후 데이터프레임의 길이: {len(result)}")
        
        if flag:
            # TODO min ~ Max 사이 t 간격으로 모든 index
            if time:
                index_new = pd.timedelta_range(start=result.index.min(),
                                            end=result.index.max(), freq=time)
                result = result.reindex(index=index_new) # 누락된 구간은 NaN으로 채워지므로 후처리가 필요할 수 있음

        # TODO 바꿀 때 arbitrationID_컬럼명 이렇게 바꿔준다 (컬럼명 중복이 가끔 있음)
        result.columns = [f"{int(identifier):03X}_{col}" for col in result.columns]

        scaler = MinMaxScalerFromDBC()
        result = scaler.scale(result)


        return result

