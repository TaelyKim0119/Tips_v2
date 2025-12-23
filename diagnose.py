#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import json

# Windows 인코딩 문제 해결
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("Cursor Python 인터프리터 문제 진단")
print("=" * 70)

# 1. Python 실행 파일 경로
print("\n[1] Python 실행 파일:")
print(f"    {sys.executable}")

# 2. Python 버전
print(f"\n[2] Python 버전:")
print(f"    {sys.version}")

# 3. Jupyter 커널 확인
print("\n[3] Jupyter 커널 목록:")
try:
    result = subprocess.run(
        ['jupyter', 'kernelspec', 'list', '--json'],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    if result.returncode == 0:
        kernels = json.loads(result.stdout)
        for name, info in kernels['kernelspecs'].items():
            print(f"    - {name}: {info['spec']['display_name']}")
            print(f"      경로: {info['resource_dir']}")
    else:
        print(f"    [오류] {result.stderr}")
except Exception as e:
    print(f"    [오류] {e}")

# 4. IPython 커널 설치 확인
print("\n[4] IPython 커널 패키지:")
try:
    import ipykernel
    print(f"    [OK] ipykernel 설치됨 - 버전: {ipykernel.__version__}")
except ImportError:
    print(f"    [X] ipykernel이 설치되지 않음!")
    print(f"    해결: pip install ipykernel")

# 5. Jupyter Lab/Notebook 설치 확인
print("\n[5] Jupyter 설치 상태:")
packages = ['jupyter', 'jupyterlab', 'notebook']
for pkg in packages:
    try:
        module = __import__(pkg.replace('-', '_'))
        version = getattr(module, '__version__', 'N/A')
        print(f"    [OK] {pkg}: {version}")
    except ImportError:
        print(f"    [X] {pkg}: 미설치")

# 6. VS Code Python 확장용 설정 파일 생성
print("\n[6] VS Code/Cursor 설정 파일 생성:")
vscode_dir = os.path.join(os.getcwd(), '.vscode')
settings_file = os.path.join(vscode_dir, 'settings.json')

if not os.path.exists(vscode_dir):
    os.makedirs(vscode_dir)
    print(f"    [생성] .vscode 디렉토리")

settings = {
    "python.defaultInterpreterPath": sys.executable,
    "jupyter.jupyterServerType": "local",
    "python.terminal.activateEnvironment": True,
    "jupyter.kernels.filter": [],
    "notebook.output.textLineLimit": 5000
}

with open(settings_file, 'w', encoding='utf-8') as f:
    json.dump(settings, f, indent=4, ensure_ascii=False)
    
print(f"    [완료] {settings_file}")
print(f"    인터프리터 경로: {sys.executable}")

# 7. Python 커널을 Jupyter에 등록
print("\n[7] Python 커널 등록:")
try:
    cmd = [sys.executable, '-m', 'ipykernel', 'install', '--user', '--name', 'python3', '--display-name', 'Python 3']
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode == 0:
        print(f"    [OK] Python 커널 등록 완료")
        print(f"    {result.stdout.strip()}")
    else:
        print(f"    [주의] {result.stderr}")
except Exception as e:
    print(f"    [오류] {e}")

# 8. 환경 변수
print("\n[8] Conda 환경:")
conda_env = os.environ.get('CONDA_DEFAULT_ENV', '없음')
print(f"    CONDA_DEFAULT_ENV: {conda_env}")

# 9. 간단한 실행 테스트
print("\n[9] 코드 실행 테스트:")
try:
    result = 2 + 2
    print(f"    [OK] 2 + 2 = {result}")
    
    import numpy as np
    arr = np.array([1, 2, 3])
    print(f"    [OK] numpy 배열: {arr}")
    
    import pandas as pd
    df = pd.DataFrame({'A': [1, 2, 3]})
    print(f"    [OK] pandas DataFrame 생성 성공")
    
except Exception as e:
    print(f"    [X] 오류: {e}")

print("\n" + "=" * 70)
print("진단 완료!")
print("=" * 70)
print("\n다음 단계:")
print("1. Cursor를 재시작하세요")
print("2. 노트북 파일을 열고 우측 상단의 'Select Kernel' 클릭")
print("3. 'Python 3' 또는 'Python 3.11.5' 선택")
print("4. 여전히 안 되면 Ctrl+Shift+P -> 'Python: Select Interpreter' 실행")
print("=" * 70)



