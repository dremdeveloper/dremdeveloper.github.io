# tmux 초보자 가이드

## 문서 정보

- 문서명: tmux Beginner Guide
- 문서형식: 운영/개발자용 실무 문서
- 대상: tmux를 처음 사용하는 개발자, 서버 작업 입문자
- 범위: 설치, 기본 조작, 세션/윈도우/패인 관리, 설정, 실전 예시
- 기준 Prefix: `Ctrl+b`

---

## 1. 목적

- 터미널 작업을 **세션 단위로 유지**한다.
- SSH 연결이 끊겨도 작업 상태를 **복구**한다.
- 하나의 터미널에서 여러 작업을 **윈도우/패인으로 분리**한다.
- 최소한의 설정으로 **개발 실무에 바로 적용**한다.

---

## 2. 사전 지식

- 터미널 기본 사용 가능
- 리눅스/맥 기본 명령어 이해
- SSH 접속 경험이 있으면 활용도가 높음

---

## 3. 핵심 개념

| 개념 | 설명 | 예시 |
|---|---|---|
| Session | 작업 묶음의 최상위 단위 | 프로젝트별 세션: `api`, `infra` |
| Window | 세션 안의 탭 개념 | 로그용 창, 서버 실행용 창 |
| Pane | 창을 분할한 영역 | 좌측 에디터, 우측 로그 |
| Prefix | tmux 명령 시작 키 | 기본값 `Ctrl+b` |
| Detach | tmux를 종료하지 않고 화면만 빠져나옴 | `Ctrl+b`, `d` |
| Attach | 기존 세션에 다시 붙음 | `tmux attach -t api` |

---

## 4. 설치

### 4.1 macOS

```bash
brew install tmux
```

### 4.2 Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y tmux
```

### 4.3 Fedora

```bash
sudo dnf install -y tmux
```

### 4.4 Arch Linux

```bash
sudo pacman -S tmux
```

### 4.5 설치 확인

```bash
tmux -V
```

예상 출력 예시:

```bash
tmux 3.x
```

---

## 5. 가장 먼저 익힐 5개 명령

```bash
tmux new -s dev          # dev 세션 생성 및 진입
tmux ls                  # 세션 목록 조회
tmux attach -t dev       # dev 세션 재진입
tmux kill-session -t dev # dev 세션 종료
tmux                     # 이름 없는 새 세션 생성
```

---

## 6. 빠른 시작

### 6.1 새 세션 생성

```bash
tmux new -s myapp
```

### 6.2 세션에서 빠져나오기

키 입력:

```text
Ctrl+b, d
```

### 6.3 세션 목록 확인

```bash
tmux ls
```

### 6.4 기존 세션 다시 접속

```bash
tmux attach -t myapp
```

### 6.5 세션 종료

```bash
tmux kill-session -t myapp
```

---

## 7. Prefix 키 사용 규칙

tmux 내부 단축키는 대부분 다음 순서로 사용한다.

```text
1) Ctrl+b 입력
2) 손을 뗀다
3) 다음 키 입력
```

예시:

- `Ctrl+b`, `c` → 새 창 생성
- `Ctrl+b`, `%` → 좌우 분할
- `Ctrl+b`, `"` → 상하 분할
- `Ctrl+b`, `d` → 분리(detach)

---

## 8. 세션 관리

### 8.1 세션 생성

```bash
tmux new -s backend
tmux new -s frontend
tmux new -s ops
```

### 8.2 세션 목록 보기

```bash
tmux ls
```

예상 출력 예시:

```text
backend: 1 windows (created Mon Mar 24 09:00:00 2026)
frontend: 2 windows (created Mon Mar 24 09:10:00 2026)
ops: 1 windows (created Mon Mar 24 09:20:00 2026)
```

### 8.3 세션 재접속

```bash
tmux attach -t backend
```

### 8.4 세션 이름 변경

```bash
tmux rename-session -t backend api
```

### 8.5 세션 종료

```bash
tmux kill-session -t api
```

### 8.6 현재 실행 중인 tmux 전체 종료

```bash
tmux kill-server
```

주의:

- 모든 세션이 종료된다.
- 작업 중인 프로세스도 함께 영향받을 수 있다.

---

## 9. 윈도우(Window) 관리

### 9.1 새 윈도우 생성

키 입력:

```text
Ctrl+b, c
```

### 9.2 윈도우 이동

| 동작 | 키 |
|---|---|
| 다음 윈도우 | `Ctrl+b`, `n` |
| 이전 윈도우 | `Ctrl+b`, `p` |
| 번호로 이동 | `Ctrl+b`, `0~9` |
| 목록 보기 | `Ctrl+b`, `w` |

### 9.3 윈도우 이름 변경

키 입력:

```text
Ctrl+b, ,
```

또는 명령:

```bash
tmux rename-window -t 1 logs
```

### 9.4 윈도우 종료

현재 윈도우에서 셸 종료:

```bash
exit
```

또는 키 입력:

```text
Ctrl+b, &
```

---

## 10. 패인(Pane) 관리

### 10.1 좌우 분할

키 입력:

```text
Ctrl+b, %
```

### 10.2 상하 분할

키 입력:

```text
Ctrl+b, "
```

### 10.3 패인 간 이동

| 방향 | 키 |
|---|---|
| 왼쪽 | `Ctrl+b`, `←` |
| 오른쪽 | `Ctrl+b`, `→` |
| 위 | `Ctrl+b`, `↑` |
| 아래 | `Ctrl+b`, `↓` |

대안 명령:

```bash
tmux select-pane -L
tmux select-pane -R
tmux select-pane -U
tmux select-pane -D
```

### 10.4 패인 크기 조절

```text
Ctrl+b, Ctrl+←
Ctrl+b, Ctrl+→
Ctrl+b, Ctrl+↑
Ctrl+b, Ctrl+↓
```

환경에 따라 키 동작이 다를 수 있다. 동작하지 않으면 명령 사용:

```bash
tmux resize-pane -L 5
tmux resize-pane -R 5
tmux resize-pane -U 3
tmux resize-pane -D 3
```

### 10.5 패인 닫기

현재 패인에서 셸 종료:

```bash
exit
```

또는 키 입력:

```text
Ctrl+b, x
```

### 10.6 레이아웃 정리

키 입력:

```text
Ctrl+b, Space
```

자주 쓰는 레이아웃:

- even-horizontal
- even-vertical
- tiled
- main-horizontal
- main-vertical

명령 예시:

```bash
tmux select-layout tiled
```

---

## 11. 복사/스크롤

### 11.1 기본 스크롤

키 입력:

```text
Ctrl+b, [
```

동작:

- 복사 모드 진입
- 방향키 / PageUp / PageDown 사용
- `q` 또는 `Enter`로 종료

### 11.2 마우스 스크롤 사용

`.tmux.conf`에 아래 설정 추가:

```tmux
set -g mouse on
```

적용:

```bash
tmux source-file ~/.tmux.conf
```

### 11.3 vi 스타일 복사 모드 사용

`.tmux.conf`:

```tmux
setw -g mode-keys vi
```

자주 쓰는 키:

| 동작 | 키 |
|---|---|
| 위로 이동 | `k` |
| 아래로 이동 | `j` |
| 페이지 업 | `Ctrl+b` |
| 검색 | `/` |
| 복사 시작 | `Space` |
| 복사 완료 | `Enter` |

---

## 12. 자주 쓰는 tmux 명령어

### 12.1 세션 관련

```bash
tmux new -s dev
tmux ls
tmux attach -t dev
tmux switch-client -t dev
tmux rename-session -t dev api
tmux kill-session -t api
```

### 12.2 윈도우 관련

```bash
tmux new-window -n logs
tmux list-windows
tmux select-window -t 1
tmux rename-window -t 1 server
```

### 12.3 패인 관련

```bash
tmux split-window -h
tmux split-window -v
tmux list-panes
tmux select-pane -L
tmux select-pane -R
tmux resize-pane -Z
```

`resize-pane -Z`:

- 현재 패인 최대화/복원 토글

---

## 13. 초보자 필수 단축키 치트시트

| 기능 | 키 |
|---|---|
| Prefix | `Ctrl+b` |
| detach | `Ctrl+b`, `d` |
| 새 윈도우 | `Ctrl+b`, `c` |
| 다음 윈도우 | `Ctrl+b`, `n` |
| 이전 윈도우 | `Ctrl+b`, `p` |
| 윈도우 목록 | `Ctrl+b`, `w` |
| 윈도우 이름 변경 | `Ctrl+b`, `,` |
| 패인 좌우 분할 | `Ctrl+b`, `%` |
| 패인 상하 분할 | `Ctrl+b`, `"` |
| 패인 닫기 | `Ctrl+b`, `x` |
| 패인 이동 | `Ctrl+b`, 방향키 |
| 패인 최대화 | `Ctrl+b`, `z` |
| 레이아웃 순환 | `Ctrl+b`, `Space` |
| 복사 모드 | `Ctrl+b`, `[` |
| 명령 프롬프트 | `Ctrl+b`, `:` |
| 세션 목록/이동 | `Ctrl+b`, `s` |

---

## 14. 실전 사용 패턴

### 14.1 백엔드 개발

권장 구조:

| Window | 용도 |
|---|---|
| `editor` | 코드 편집/탐색 |
| `server` | 애플리케이션 실행 |
| `logs` | 로그 모니터링 |
| `db` | DB 접속/쿼리 |

실행 예시:

```bash
tmux new -s backend
```

tmux 내부 작업 예시:

```text
1) Ctrl+b, c          # editor
2) Ctrl+b, ,          # 이름 editor
3) Ctrl+b, c          # server
4) Ctrl+b, ,          # 이름 server
5) Ctrl+b, c          # logs
6) Ctrl+b, ,          # 이름 logs
```

### 14.2 SSH 원격 서버 작업

권장 순서:

```bash
ssh user@server
tmux new -s deploy
```

배포 중 연결이 끊겨도 다음으로 복구:

```bash
ssh user@server
tmux attach -t deploy
```

### 14.3 로그 확인 + 명령 실행 동시 작업

구성 예시:

- 좌측 패인: 서버 실행
- 우측 상단 패인: `tail -f` 로그
- 우측 하단 패인: 테스트 명령

생성 예시:

```bash
tmux new -s monitor
tmux split-window -h
tmux split-window -v
```

---

## 15. 동기 입력 (여러 패인에 같은 명령 전송)

다중 서버 또는 다중 셸에 같은 명령을 넣을 때 사용.

tmux 명령 프롬프트 실행:

```text
Ctrl+b, :
```

입력:

```tmux
setw synchronize-panes on
```

해제:

```tmux
setw synchronize-panes off
```

주의:

- 모든 패인에 동일한 명령이 입력된다.
- 운영 서버에서 실수 가능성이 높으므로 상태 확인 후 사용한다.

---

## 16. 최소 권장 설정 파일

파일 경로:

```bash
~/.tmux.conf
```

예시:

```tmux
# 마우스 사용
set -g mouse on

# 복사 모드에서 vi 키 사용
setw -g mode-keys vi

# 인덱스를 1부터 시작
set -g base-index 1
setw -g pane-base-index 1

# 설정 파일 다시 불러오기 단축키: Prefix + r
bind r source-file ~/.tmux.conf \; display-message "tmux config reloaded"

# 패인 이동을 더 쉽게
bind -r h select-pane -L
bind -r j select-pane -D
bind -r k select-pane -U
bind -r l select-pane -R
```

적용:

```bash
tmux source-file ~/.tmux.conf
```

---

## 17. 추천 추가 설정

### 17.1 Prefix를 `Ctrl+a`로 변경

GNU Screen 사용자에게 익숙한 방식.

```tmux
set -g prefix C-a
unbind C-b
bind C-a send-prefix
```

### 17.2 패인 분할 시 현재 경로 유지

```tmux
bind '"' split-window -v -c '#{pane_current_path}'
bind % split-window -h -c '#{pane_current_path}'
bind c new-window -c '#{pane_current_path}'
```

효과:

- 새 창/패인이 현재 작업 디렉터리 기준으로 열린다.

---

## 18. 자주 발생하는 문제

### 18.1 `sessions should be nested with care` 메시지

원인:

- tmux 안에서 다시 tmux 실행

대응:

- 기존 세션 내부인지 확인
- 필요 시 현재 세션에서 새 윈도우/패인 사용
- 중첩이 꼭 필요하면 환경을 명확히 구분

### 18.2 방향키/단축키가 기대대로 동작하지 않음

점검 항목:

- 터미널 앱 키 매핑 충돌 여부
- tmux 버전 확인
- SSH/로컬 환경 차이 확인
- `.tmux.conf` 사용자 정의 바인딩 충돌 여부 확인

### 18.3 마우스 스크롤이 안 됨

점검 순서:

```tmux
set -g mouse on
```

적용 후:

```bash
tmux source-file ~/.tmux.conf
```

### 18.4 세션에 다시 붙지 못함

확인:

```bash
tmux ls
```

세션이 없으면:

- 이미 종료됨
- 서버 재시작으로 tmux 서버 종료됨
- 직접 `kill-session` 또는 `kill-server` 실행됨

---

## 19. 운영 기준 권장 사용 규칙

- 프로젝트별로 **세션 이름 고정**
  - 예: `api`, `web`, `infra`, `deploy`
- 창 이름은 **역할 기준**으로 지정
  - 예: `server`, `logs`, `shell`, `db`
- 장기 실행 작업은 가능하면 **tmux 내부에서 실행**
- 원격 작업 시작 직후 **tmux 먼저 실행**
- `kill-server` 사용 전 **세션 목록 확인**
- 공용 서버에서는 동기 입력 사용 시 **환경 확인 필수**

---

## 20. 최소 학습 순서

### Day 1

- `tmux new -s <name>`
- `Ctrl+b`, `d`
- `tmux ls`
- `tmux attach -t <name>`

### Day 2

- `Ctrl+b`, `c`
- `Ctrl+b`, `n`
- `Ctrl+b`, `p`
- `Ctrl+b`, `%`
- `Ctrl+b`, `"`

### Day 3

- `Ctrl+b`, `[`
- `Ctrl+b`, `:`
- `.tmux.conf` 작성
- `tmux source-file ~/.tmux.conf`

---

## 21. 즉시 사용 가능한 예제

### 21.1 개발 세션 생성

```bash
tmux new -s dev
```

tmux 내부:

```text
Ctrl+b, c   -> server
Ctrl+b, c   -> logs
Ctrl+b, c   -> git
```

### 21.2 로그 전용 창 생성

```bash
tmux new-window -n logs
tail -f /var/log/app.log
```

### 21.3 현재 패인 최대화

```text
Ctrl+b, z
```

### 21.4 설정 변경 즉시 반영

```bash
tmux source-file ~/.tmux.conf
```

---

## 22. 한 페이지 요약

### 꼭 기억할 명령

```bash
tmux new -s work
tmux ls
tmux attach -t work
tmux kill-session -t work
```

### 꼭 기억할 단축키

```text
Ctrl+b, d   # 분리
Ctrl+b, c   # 새 창
Ctrl+b, %   # 좌우 분할
Ctrl+b, "   # 상하 분할
Ctrl+b, [   # 스크롤/복사 모드
Ctrl+b, z   # 현재 패인 최대화
```

### 초보자 권장 습관

- 로컬/원격 모두 작업 시작 시 tmux 먼저 실행
- 세션 이름을 프로젝트 기준으로 고정
- 창 이름을 역할 기준으로 지정
- `.tmux.conf`는 최소 설정부터 시작

---

## 23. 부록: 명령어 요약표

| 작업 | 명령 |
|---|---|
| 새 세션 생성 | `tmux new -s <name>` |
| 세션 목록 | `tmux ls` |
| 세션 접속 | `tmux attach -t <name>` |
| 세션 종료 | `tmux kill-session -t <name>` |
| 새 윈도우 생성 | `tmux new-window -n <name>` |
| 새 패인 좌우 분할 | `tmux split-window -h` |
| 새 패인 상하 분할 | `tmux split-window -v` |
| 패인 목록 | `tmux list-panes` |
| 윈도우 목록 | `tmux list-windows` |
| 설정 다시 로드 | `tmux source-file ~/.tmux.conf` |

---

## 24. 완료 기준

다음 작업을 혼자 수행할 수 있으면 입문 완료:

- 새 세션 생성
- detach / attach
- 창 3개 생성 및 이름 지정
- 패인 2개 이상 분할
- 로그 확인용 창 운영
- 최소 `.tmux.conf` 작성 및 적용

