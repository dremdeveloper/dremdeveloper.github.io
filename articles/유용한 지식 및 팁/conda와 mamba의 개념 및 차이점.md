# conda와 mamba의 개념 및 차이점

## 1. 개요

conda는 패키지(package), 의존성(dependency), 가상환경(environment)을 함께 관리하는 도구이다.[1] 사용자는 conda를 이용해 서로 다른 Python 버전과 라이브러리 조합을 프로젝트별로 분리하여 관리할 수 있다.[2] mamba는 conda 생태계와 호환되는 고속 패키지 관리자이며, 대부분의 conda 명령을 같은 방식으로 사용할 수 있다.[3]

정리하면, **conda는 기준이 되는 환경·패키지 관리 도구**이고, **mamba는 conda와 호환되면서 더 빠른 설치와 의존성 해결을 제공하는 구현체**라고 볼 수 있다.[1][3]

> 아래 명령어 결과의 경로, 버전, 채널명은 운영체제와 설치 환경에 따라 달라질 수 있다. 본 문서의 실행 결과는 공식 문서의 형식과 일반적인 CLI 출력 예시를 기준으로 정리하였다.[2][3]

---

## 2. 핵심 개념

### 2.1 패키지(package)
프로그램이나 라이브러리를 설치 가능한 단위로 묶은 것이다. 예를 들어 `python`, `numpy`, `pandas`, `scipy` 등이 패키지에 해당한다.

### 2.2 의존성(dependency)
어떤 패키지가 동작하기 위해 추가로 필요한 다른 패키지나 라이브러리를 의미한다. 예를 들어 특정 패키지는 특정 Python 버전이나 다른 수치 계산 라이브러리를 함께 요구할 수 있다.

### 2.3 가상환경(environment)
프로젝트별로 독립된 실행 공간을 만드는 기능이다. 서로 다른 프로젝트가 서로 다른 Python 버전이나 패키지 버전을 사용해도 충돌 없이 공존할 수 있다.[2]

예를 들면 다음과 같다.

- 프로젝트 A: Python 3.10, NumPy 1.x
- 프로젝트 B: Python 3.12, NumPy 2.x

### 2.4 채널(channel)
패키지를 내려받는 저장소를 의미한다. 대표적으로 `defaults`와 `conda-forge`가 있으며, 같은 패키지라도 채널에 따라 빌드 방식이나 제공 버전이 달라질 수 있다.

---

## 3. conda의 개념

공식 문서는 conda를 **"package, dependency, and environment management for any language"**로 설명한다.[1] 즉, conda는 단순한 패키지 설치기가 아니라 다음 기능을 함께 수행한다.

1. 패키지 검색
2. 의존성 계산
3. 충돌이 없는 버전 조합 선택
4. 특정 환경에 설치
5. 환경 생성, 활성화, 삭제, 내보내기

### 3.1 conda의 특징

- Python뿐 아니라 다양한 언어와 시스템 라이브러리를 함께 다룰 수 있다.[1]
- 환경 생성, 활성화, 목록 조회, 삭제, 내보내기를 지원한다.[2]
- 팀 단위 개발에서 동일한 환경을 재현하기에 유리하다.[2]
- 패키지 간 충돌을 자동으로 계산해 설치한다.[2]

### 3.2 conda를 사용하는 이유

- 데이터 분석, 과학 계산, 머신러닝처럼 의존성이 복잡한 환경을 안정적으로 구성할 수 있다.
- 프로젝트별 환경 분리를 통해 패키지 충돌을 줄일 수 있다.
- 환경 파일을 공유해 같은 실행 환경을 재현할 수 있다.[2]

---

## 4. mamba의 개념

공식 문서는 mamba를 conda 환경을 관리하는 CLI 도구로 소개하며, conda와 같은 명령과 설정 방식을 사용하는 **drop-in replacement**라고 설명한다.[3] 또한 mamba는 `libmamba`를 기반으로 하는 빠르고 견고한 패키지 관리자이다.[6]

### 4.1 mamba의 특징

- conda 패키지와 호환된다.[6]
- 대부분의 conda 명령을 거의 동일한 형태로 사용할 수 있다.[3]
- 환경 생성과 패키지 설치 시 의존성 해결 속도가 빠른 편이다.[6]
- `repoquery` 기능을 통해 패키지 의존성 구조를 분석할 수 있다.[3]

### 4.2 mamba를 사용하는 이유

- 대형 환경 생성이나 다수 패키지 설치 시 시간이 단축되는 경우가 많다.
- 반복 설치가 많은 CI/CD, Docker, 서버 구축 환경에 적합하다.
- conda와 명령 체계가 유사해 학습 비용이 낮다.[3]

---

## 5. conda와 mamba의 차이점

| 항목 | conda | mamba |
|---|---|---|
| 기본 성격 | 표준 패키지·의존성·환경 관리자 | conda 호환 고속 구현 |
| 명령 체계 | 공식 표준 | 거의 동일한 명령 체계 제공[3] |
| 의존성 해결 | 기본 기능 제공, 최신 버전은 `libmamba` solver 사용 가능[4] | `libmamba` 기반 고속 처리[6] |
| 체감 속도 | 일반적 | 대체로 더 빠름 |
| 학습 자료 | 공식 문서와 예제가 많음 | conda를 알면 바로 사용 가능 |
| 부가 기능 | 표준 환경 관리 중심 | `repoquery` 등 분석 기능 제공[3] |

핵심 차이는 다음과 같이 정리할 수 있다.

- **개념의 기준은 conda**이다.
- **실행 속도와 의존성 처리 성능은 mamba가 유리한 경우가 많다.**
- 다만 최신 conda는 `libmamba` solver를 사용할 수 있으므로, 과거보다 속도 차이가 줄어들었다.[4]

---

## 6. 최신 사용 흐름에서 알아둘 점

### 6.1 conda도 예전보다 빨라졌다

conda 공식 릴리스 노트에 따르면 `conda-libmamba-solver`는 주요 conda 설치본에 포함되어 왔으며, conda 23.10.0 이후에는 `libmamba`를 기본 solver로 사용하는 방향으로 전환되었다.[4] 따라서 현재는 단순히 "conda는 느리고 mamba는 빠르다"라고만 설명하면 불완전하다.

### 6.2 Miniforge는 conda와 mamba를 함께 제공한다

conda-forge 다운로드 페이지는 **Miniforge를 preferred installer**로 안내하며, 여기에 `conda`, `mamba`, 관련 의존성이 포함된다고 명시한다.[5] 따라서 현재는 Miniforge를 설치한 뒤, 상황에 따라 conda와 mamba를 함께 사용하는 방식이 일반적이다.[5]

---

## 7. 주요 명령어와 실제 사용 예시

### 7.1 새 환경 생성

```bash
conda create -n demo python=3.11
```

예시 출력:

```text
Collecting package metadata (...)
Solving environment: done

## Package Plan ##

  environment location: /home/username/miniforge3/envs/demo

  added / updated specs:
    - python=3.11

Proceed ([y]/n)? y
```

설명:

- `demo`라는 이름의 새 환경을 생성한다.
- 해당 환경에 Python 3.11을 설치한다.
- 설치 전에 conda가 의존성 계산 결과를 보여 주고 사용자 확인을 받는다.[2]

---

### 7.2 환경 활성화

```bash
conda activate demo
```

예시 출력 또는 프롬프트 변화:

```text
(demo) username@machine:~$
```

설명:

- 현재 셸에서 `demo` 환경이 활성화된다.
- 이후 실행하는 `python`, `pip`, `conda list` 등은 `demo` 환경 기준으로 동작한다.[2]

---

### 7.3 패키지 설치: conda

```bash
conda install numpy pandas
```

예시 출력:

```text
Collecting package metadata (...)
Solving environment: done

## Package Plan ##

  environment location: /home/username/miniforge3/envs/demo

  added / updated specs:
    - numpy
    - pandas

Proceed ([y]/n)? y
```

설명:

- 현재 활성화된 환경에 `numpy`, `pandas`를 설치한다.
- conda는 설치 전에 호환 가능한 버전 조합을 먼저 계산한다.[7]

---

### 7.4 패키지 설치: mamba

```bash
mamba install numpy pandas
```

예시 출력:

```text
Looking for: [numpy, pandas]

Transaction

  Prefix: /home/username/miniforge3/envs/demo

  Install:
    - numpy
    - pandas

Confirm changes: [Y/n]
```

설명:

- 목적은 `conda install numpy pandas`와 동일하다.
- 일반적으로 의존성 계산과 transaction 처리 속도가 더 빠르게 체감된다.[3][6]

---

### 7.5 환경 목록 확인

```bash
conda info --envs
```

예시 출력:

```text
# conda environments:
#
base               /home/username/miniforge3
demo            *  /home/username/miniforge3/envs/demo
```

설명:

- `base`와 `demo` 환경이 존재함을 보여 준다.
- `*` 표시는 현재 활성화된 환경을 의미한다.[8]

---

### 7.6 현재 환경의 패키지 목록 확인

```bash
conda list
```

예시 출력:

```text
# packages in environment at /home/username/miniforge3/envs/demo:
#
# Name                    Version
numpy                     2.x
pandas                    2.x
python                    3.11.x
```

설명:

- 현재 환경에 설치된 패키지와 버전을 확인하는 명령이다.

---

### 7.7 환경 파일로 내보내기

```bash
conda env export --from-history > environment.yml
```

예시 출력 파일(`environment.yml`):

```yaml
name: demo
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pandas
```

설명:

- 사용자가 직접 지정해 설치한 패키지 중심으로 환경을 기록한다.
- 팀원에게 같은 환경을 전달하거나 재현할 때 유용하다.[9]

참고로 최신 conda에서는 `conda export`도 지원한다.[9]

---

### 7.8 환경 파일로 다시 생성하기

```bash
conda env create -f environment.yml
```

또는

```bash
mamba env create -f environment.yml
```

설명:

- 저장된 `environment.yml` 파일을 바탕으로 같은 환경을 다시 만든다.[2][3]
- 팀 단위 협업에서 가장 많이 쓰이는 방식 중 하나이다.

---

### 7.9 환경 삭제

```bash
conda remove --name demo --all
```

예시 출력:

```text
Remove all packages in environment /home/username/miniforge3/envs/demo:

Proceed ([y]/n)? y
```

설명:

- `demo` 환경 전체를 삭제한다.[10]

---

### 7.10 mamba의 의존성 분석 기능 예시

```bash
mamba repoquery depends -t xtensor
```

공식 문서 예시 출력 일부:

```text
xtensor == 0.21.5
├─ libgcc-ng [>=7.3.0]
│ ├─ _libgcc_mutex [0.1 conda_forge]
│ └─ _openmp_mutex [>=4.5]
├─ libstdcxx-ng [>=7.3.0]
└─ xtl [>=0.6.9,<0.7]
```

설명:

- 특정 패키지가 어떤 의존성을 가지는지 트리 형태로 보여 준다.[3]
- conda 기본 명령보다 의존성 구조를 확인하기 편리하다.

---

## 8. 언제 conda를 쓰고, 언제 mamba를 쓰는가

### 8.1 conda가 적합한 경우

- 환경 관리 개념을 처음 학습하는 경우
- 공식 문서 예제를 그대로 따라가려는 경우
- 환경 생성, 활성화, 삭제, 내보내기 중심의 표준 흐름이 필요한 경우

### 8.2 mamba가 적합한 경우

- 패키지 수가 많은 대형 환경을 자주 생성하는 경우
- 서버, Docker, CI/CD처럼 반복 설치가 많은 경우
- 의존성 계산 속도가 중요한 경우

### 8.3 가장 현실적인 사용 방식

현재는 다음과 같은 방식이 많이 사용된다.

- 개념과 표준 명령 흐름은 **conda 기준으로 이해**한다.
- 실제 대형 설치와 업데이트는 **mamba로 실행**한다.
- 최신 conda는 `libmamba` solver를 활용할 수 있으므로, 상황에 따라 둘 중 하나를 선택한다.[4]

---

## 9. 결론

conda는 패키지, 의존성, 가상환경을 통합 관리하는 표준 도구이며, 프로젝트별로 독립된 실행 환경을 구성하는 데 적합하다.[1][2] mamba는 conda와 높은 호환성을 유지하면서 더 빠른 의존성 해결과 설치 성능을 제공하는 도구이다.[3][6] 따라서 두 도구는 경쟁 관계라기보다, **같은 생태계 안에서 서로 보완적으로 사용되는 도구**로 이해하는 것이 가장 적절하다.

요약하면 다음과 같다.

- **환경 관리의 기준 도구는 conda이다.**
- **고속 설치와 의존성 처리에는 mamba가 강점이 있다.**
- **최신 conda는 libmamba solver를 활용할 수 있어, 예전보다 conda와 mamba의 성능 차이가 줄어들었다.**[4]

---

## 10. 참고 자료

[1] Conda Documentation: https://docs.conda.io/  
[2] Managing environments — conda documentation: https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html  
[3] Mamba User Guide: https://mamba.readthedocs.io/en/stable/user_guide/mamba.html  
[4] Conda Release Notes (`conda-libmamba-solver` 관련): https://docs.conda.io/projects/conda/en/stable/release-notes.html  
[5] conda-forge Download / Miniforge: https://conda-forge.org/download/  
[6] Mamba Documentation: https://mamba.readthedocs.io/  
[7] `conda install` command reference: https://docs.conda.io/projects/conda/en/stable/commands/install.html  
[8] `conda env list` command reference: https://docs.conda.io/projects/conda/en/stable/commands/env/list.html  
[9] `conda env export` and `conda export` command reference: https://docs.conda.io/projects/conda/en/stable/commands/env/export.html , https://docs.conda.io/projects/conda/en/latest/commands/export.html  
[10] `conda remove` command reference: https://docs.conda.io/projects/conda/en/stable/commands/remove.html
