# decoding.py
# 데이터셋 로드 및 디코딩 관련 함수

import pandas as pd
import numpy as np

class RawDataParser:
    """
                            텍스트 데이터를 파싱하는 클래스

                            parse_txt 메서드 실행 시
                                convert_to_dict 메서드 호출
                                convert_to_dataframe 메서드 호출
                                drop_error 메서드 호출
                            """
    def __init__(self, path):
        """
                            텍스트데이터파서 초기화
                            
                            path (str): raw 텍스트 파일 경로
                            """
        self.path = path

    def parse_txt(self):
        path = self.path
        print('=' * 20, "raw data를 dataframe으로 변환 중", '=' * 20)

        dict = self.convert_to_dict()
        df = self.convert_to_dataframe(dict)
        df = self.drop_error(df)

        return df

    def convert_to_dict(self):
        """
                            데이터셋을 로드하는 함수
                            
                            Returns: 데이터셋 객체 (딕셔너리)
                            """
        path = self.path
        columns = { # return
            'Time': [],
            'CAN': [],
            'Identifier': [],
            'Flags': [],
            'DLC': [],
            'Data': [],  # 전체 데이터 문자열
            'Counter': [],
            'Error': []  # 오류 여부를 저장하는 새 컬럼 추가
        }
        
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            # 플래그 변수들
            found_separator = False
            skip_next_line = False
            
            for i, line in enumerate(lines):
                # '=' 구분자를 찾았는지 확인 (30개 이상)
                if '=' * 60 in line:
                    found_separator = True
                    skip_next_line = True
                    continue
                
                # 구분자 다음 줄은 건너뛰기 (Trigger 정보 등)
                if skip_next_line:
                    skip_next_line = False
                    continue
                    
                # 구분자를 찾았고, 그 다음다음 줄부터 데이터 처리
                if found_separator and not skip_next_line:
                    # 공백으로 분리하여 데이터 추출
                    parts = line.strip().split()
                    
                    if len(parts) < 4:  # 최소한 Time, CAN, Identifier, Flags는 있어야 함
                        print(f"Line {i+1}: 데이터 부족 - {line.strip()}")
                        continue
                    
                    try:
                        # 기본 컬럼 추출
                        time_val = parts[0]
                        can_channel = parts[1]
                        identifier = parts[2]
                        flags = parts[3]
                        
                        # ErrorFrame 특별 처리
                        if "ErrorFrame" in identifier:
                            # ErrorFrame이면 항상 "ERROR FRAME"이 있고, 마지막 필드가 Counter
                            # DLC 필드는 사용하지 않음
                            dlc = -1  # ErrorFrame의 경우 특수 값으로 표시
                            data_str = "ERROR FRAME"
                            counter = parts[-1]
                            is_error = True
                        else:
                            # 일반 프레임 처리
                            dlc = int(parts[4])  # DLC를 정수로 변환
                            # DLC 값에 따라 데이터 바이트 추출
                            data_bytes = parts[5:5+dlc]
                            data_str = " ".join(data_bytes)
                            # 남은 부분은 Counter
                            counter = parts[5+dlc] if 5+dlc < len(parts) else ""
                            is_error = False
                        
                        # 딕셔너리에 저장
                        columns['Time'].append(time_val)
                        columns['CAN'].append(can_channel)
                        columns['Identifier'].append(identifier)
                        columns['Flags'].append(flags)
                        columns['DLC'].append(dlc)
                        columns['Data'].append(data_str)
                        columns['Counter'].append(counter)
                        columns['Error'].append(is_error)
                        
                    except (ValueError, IndexError) as e:
                        # 파싱 오류 발생 시 처리
                        print(f"Line {i+1} 데이터 파싱 오류: {e}")
                        print(f"오류 발생 line: {line.strip()}")
                        
                        # ErrorFrame 특별 처리를 다시 시도
                        if "ErrorFrame" in line and "ERROR" in line and "FRAME" in line:
                            # ErrorFrame 형식에 맞게 처리
                            error_parts = line.strip().split()
                            columns['Time'].append(error_parts[0] if len(error_parts) > 0 else "")
                            columns['CAN'].append(error_parts[1] if len(error_parts) > 1 else "")
                            columns['Identifier'].append("ErrorFrame")
                            columns['Flags'].append(error_parts[3] if len(error_parts) > 3 else "")
                            columns['DLC'].append(-1)
                            columns['Data'].append("ERROR FRAME")
                            columns['Counter'].append(error_parts[-1] if len(error_parts) > 5 else "")
                            columns['Error'].append(True)
                        else:
                            # 일반 오류 처리
                            columns['Time'].append(parts[0] if len(parts) > 0 else "")
                            columns['CAN'].append(parts[1] if len(parts) > 1 else "")
                            columns['Identifier'].append(parts[2] if len(parts) > 2 else "")
                            columns['Flags'].append(parts[3] if len(parts) > 3 else "")
                            columns['DLC'].append(-2)  # 기타 오류를 위한 또 다른 특수값
                            columns['Data'].append("")
                            columns['Counter'].append(parts[-1] if len(parts) > 4 else "")
                            columns['Error'].append(True)
        
        # 데이터 확인
        print(f"총 {len(columns['Time'])}개의 로그 항목을 읽었습니다.")
        print(f"정상 항목: {len(columns['Time']) - sum(columns['Error'])}개")
        print(f"오류 항목: {sum(columns['Error'])}개")
        
        # ErrorFrame 예시 출력 (data1 객체)
        error_indices = [i for i, err in enumerate(columns['Error']) if err]
        if error_indices:
            print("\nErrorFrame 예시:")
            idx = error_indices[0]
            print(f"Time: {columns['Time'][idx]}")
            print(f"CAN: {columns['CAN'][idx]}")
            print(f"Identifier: {columns['Identifier'][idx]}")
            print(f"Flags: {columns['Flags'][idx]}")
            print(f"DLC: {columns['DLC'][idx]}")
            print(f"Data: {columns['Data'][idx]}")
            print(f"Counter: {columns['Counter'][idx]}")
        
        return columns

    def convert_to_dataframe(self,data_dict):
        """
                            딕셔너리를 판다스 DataFrame으로 변환하는 함수
                        
                            data_dict (dict): read_txt 함수로 얻은 딕셔너리
                        
                            Returns: pandas.DataFrame 변환된 데이터프레임
                            """
        
        # 입력 데이터 유효성 검사
        if not isinstance(data_dict, dict):
            raise TypeError(f"입력 데이터가 딕셔너리가 아닙니다. 현재 타입: {type(data_dict)}")
    
        # 빈 딕셔너리 처리
        if not data_dict:
            print("경고: 빈 딕셔너리가 입력되었습니다.")
            return pd.DataFrame()
        
        # Data 컬럼을 16진수로 변환 (존재하는 경우)
        if 'Data' in data_dict:
            # 'Data' 컬럼의 각 항목을 16진수로 변환
            try:
                new_data = []
                for data_item in data_dict['Data']:
                    if pd.notna(data_item) and data_item != '':
                        # 'ERROR' 문자열 처리
                        if data_item == 'ERROR FRAME':
                            new_data.append('ERROR FRAME')
                            continue
                            
                        try:
                            # 공백으로 구분된 문자열을 숫자 리스트로 변환
                            nums = str(data_item).split()
                            
                            # 숫자들을 바이트 배열로 변환
                            byte_array = bytes([int(num) for num in nums])
                            
                            # 바이트 배열을 바이트 리터럴 형태로 저장
                            new_data.append(byte_array)
                        except ValueError:
                            print(f"경고: 숫자로 변환할 수 없는 Data 값 발견: {data_item}")
                            new_data.append(str(data_item))
                    else:
                        new_data.append(b'')
                data_dict['Data'] = new_data
            except Exception as e:
                print(f"Data 컬럼 바이트 변환 중 오류 발생: {e}")
    
        # 1. 각 키의 값 길이가 동일한지 확인
        try:
            lengths = {}
            for key, value in data_dict.items():
                try:
                    lengths[key] = len(value)
                except TypeError:
                    print(f"경고: '{key}' 키의 값이 길이를 측정할 수 없는 타입입니다: {type(value)}")
                    lengths[key] = 1  # 기본값 설정
        
            length_values = list(lengths.values())
        except Exception as e:
            print(f"길이 검사 중 오류 발생: {e}")
            return pd.DataFrame()  # 오류 발생 시 빈 DataFrame 반환
    
        if len(set(length_values)) != 1:
            # 길이가 일치하지 않는 경우
            print(f"경고: 딕셔너리의 값 길이가 일치하지 않습니다.")
            print("각 키별 길이:")
            for key, length in lengths.items():
                print(f"  - {key}: {length}")
        
            # 가장 짧은 길이로 자르기
            min_length = min(length_values)
            print(f"모든 값을 가장 짧은 길이({min_length})로 자릅니다.")
        
            # 각 값을 자르기
            for key in data_dict.keys():
                try:
                    data_dict[key] = data_dict[key][:min_length]
                except TypeError:
                    print(f"경고: '{key}' 키의 값은 슬라이스할 수 없습니다.")
    
        # 2. 데이터프레임으로 변환
        try:
            df = pd.DataFrame(data_dict)
        except Exception as e:
            print(f"DataFrame 생성 중 오류 발생: {e}")
            return pd.DataFrame()
    
        # 3. 데이터 타입 변환
        try:
            # Time을 float로 변환
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        
            # DLC를 정수로 변환 (ErrorFrame의 경우 -1로 이미 설정됨)
            df['DLC'] = pd.to_numeric(df['DLC'], errors='coerce').astype('Int64')  # nullable integer
        
            # Counter를 정수로 변환 (빈 문자열은 NaN으로)
            if 'Counter' in df.columns:
                df['Counter'] = pd.to_numeric(df['Counter'], errors='coerce').astype('Int64')
        
            # Error를 bool로 변환
            if 'Error' in df.columns:
                df['Error'] = df['Error'].astype(bool)
        except Exception as e:
            print(f"데이터 타입 변환 중 오류 발생: {e}")
    
        # 5. 결과 요약 정보 출력
        print(f"변환 완료: {len(df)}행, {len(df.columns)}열")
    
        try:
            if 'Error' in df.columns:
                error_count = df['Error'].sum()
                print(f"ErrorFrame 수: {error_count}개")
        except Exception as e:
            print(f"ErrorFrame 개수 계산 중 오류 발생: {e}")
    
        return df

    def drop_error(self, df):
        """
                            rat txt data를 파싱한 df의 Error 컬럼의 값이 True이면 drop함
                            """
        df = df[df["Error"] != True]
        df = df.drop("Error", axis=1)

        return df
