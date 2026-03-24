# bash 명령어

이 문서는 개발자가 로컬 환경, 원격 서버, CI/CD 러너, 컨테이너 호스트에서 바로 써먹을 수 있는 Bash 명령을 **실전 기준**으로 정리한 가이드입니다.  
문법만 나열하지 않고, **언제 쓰는지**, **어떤 결과가 나오는지**, **그 결과를 어떻게 해석해야 하는지**까지 같이 담았습니다.

기준 환경은 다음과 같습니다.

- Linux + Bash + GNU 계열 도구
- macOS/BSD에서는 일부 옵션(`-maxdepth`, `-printf`, `du -d`, `sed -r` 등)이 다를 수 있음
- 예시 출력은 이해를 돕기 위한 샘플이며 실제 값은 환경마다 다름
- 삭제·정리 작업은 항상 **미리보기 → 검증 → 실행** 순서로 진행

---

## 먼저 기억할 운영 습관

### 1) 삭제 전에 반드시 출력으로 검증

바로 지우지 말고 먼저 대상을 확인합니다.

```bash
find ./tmp -type f -name "*.tmp" -mtime +7 -print
```

예시 결과:

```text
./tmp/cache-20260310.tmp
./tmp/render-1821.tmp
```

해석:

- `7일보다 오래된 .tmp 파일` 두 개가 정리 대상이라는 뜻
- 이 단계에서 경로가 예상과 다르면 삭제 명령을 실행하면 안 됨

실제 삭제는 그 다음입니다.

```bash
find ./tmp -type f -name "*.tmp" -mtime +7 -delete
```

예시 결과:

```text
# 출력 없음
```

해석:

- `find -delete`는 보통 성공해도 별도 출력이 없음
- 그래서 더더욱 `-print`로 먼저 검증하는 습관이 중요함

---

### 2) 공백이 있는 파일명은 `-print0` + `xargs -0`

```bash
find ./trash -type f -name "*.bak" -print0 | xargs -0 rm -v
```

예시 결과:

```text
removed './trash/old report (final).bak'
removed './trash/user backup 01.bak'
```

해석:

- 공백, 괄호, 특수문자가 있어도 파일명이 안전하게 전달됨
- 대량 삭제, 대량 압축, 대량 이동 작업에서 매우 중요함

---

### 3) 와일드카드는 따옴표로 감싸기

```bash
find . -name "*.log"
grep -R "ERROR" .
```

예시 결과:

```text
# 첫 번째 명령은 .log 파일을 찾고
# 두 번째 명령은 ERROR가 포함된 줄을 출력
```

해석:

- `*.log`를 따옴표 없이 쓰면 셸이 먼저 확장해 버릴 수 있음
- 특히 `find . -name *.log` 같은 형태는 현재 디렉터리 상황에 따라 오동작 가능

---

## 1. 파일 검색과 일괄 처리: `find`

`find`는 디렉터리 트리를 순회하면서 이름, 타입, 수정 시간, 크기, 소유자, 권한 등을 기준으로 파일을 찾습니다.  
실무에서는 단순 검색뿐 아니라 **정리, 압축, 삭제, 권한 점검, 대량 처리**까지 이어지는 경우가 많습니다.

---

### 1-1. 확장자로 파일 찾기

```bash
find . -type f -name "*.js"
```

예시 결과:

```text
./src/index.js
./src/utils/date.js
./tests/index.test.js
```

해석:

- `.` 아래 전체를 탐색
- `-type f`라서 디렉터리는 제외
- `.js` 확장자 파일만 출력
- 코드베이스에서 특정 언어 파일 목록을 빠르게 볼 때 좋음

---

### 1-2. 대소문자 무시하고 파일 찾기

```bash
find . -type f -iname "*.jpg"
```

예시 결과:

```text
./images/banner.jpg
./images/BANNER.JPG
./archive/Banner.Jpg
```

해석:

- `-iname`은 대소문자 무시
- 사용자 업로드 파일처럼 확장자 표기가 제각각일 때 유용

---

### 1-3. 최근 1시간 안에 수정된 파일 찾기

```bash
find . -type f -mmin -60
```

예시 결과:

```text
./logs/app.log
./tmp/render.cache
./dist/manifest.json
```

해석:

- `-mmin -60`은 최근 60분 이내 수정된 파일
- 배포 직후 어떤 파일이 갱신됐는지 점검할 때 좋음

---

### 1-4. 오래된 로그 파일 찾기

```bash
find /var/log/myapp -type f -name "*.log.*" -mtime +14
```

예시 결과:

```text
/var/log/myapp/app.log.1
/var/log/myapp/app.log.2.gz
/var/log/myapp/error.log.2026-03-01.gz
```

해석:

- `14일보다 오래된 회전 로그`를 찾음
- 바로 삭제하지 말고 먼저 목록을 보고 보존 정책과 맞는지 검증해야 함

---

### 1-5. 큰 파일 찾기

```bash
find . -type f -size +500M -exec ls -lh {} \;
```

예시 결과:

```text
-rw-r--r-- 1 dev dev 1.2G Mar 24 09:10 ./artifacts/model.bin
-rw-r--r-- 1 dev dev 768M Mar 24 09:14 ./backups/db.dump
```

해석:

- `500MB 초과 파일`을 찾고 사람이 읽기 좋은 크기로 표시
- `find`만 쓰는 것보다 `ls -lh`를 붙이는 편이 용량 판단이 쉬움
- 디스크 부족 원인을 추적할 때 매우 자주 쓰는 패턴

---

### 1-6. 특정 디렉터리를 제외하고 검색하기

```bash
find . \
  -path "./node_modules" -prune -o \
  -path "./.git" -prune -o \
  -type f -name "*.ts" -print
```

예시 결과:

```text
./src/main.ts
./src/jobs/sync.ts
./tests/main.spec.ts
```

해석:

- `node_modules`, `.git`은 아예 내려가지 않고 건너뜀
- 대규모 저장소에서는 검색 성능 차이가 큼
- 결과에 제3자 코드나 Git 내부 파일이 섞이지 않아 잡음도 줄어듦

---

### 1-7. 빈 디렉터리 찾기

```bash
find . -type d -empty
```

예시 결과:

```text
./tmp/cache
./build/old
./uploads/staging
```

해석:

- 안에 아무 것도 없는 디렉터리만 찾음
- 정리 스크립트 작성 전에 “비어 있는 폴더가 얼마나 있는지” 확인하기 좋음

---

### 1-8. 조건에 맞는 파일에 명령 실행하기

```bash
find ./logs -type f -name "*.log" -mtime +3 -exec gzip {} +
```

예시 결과:

```text
# 성공 시 대부분 별도 출력 없음
```

압축 후 확인:

```bash
find ./logs -type f -name "*.gz"
```

예시 결과:

```text
./logs/app.log.gz
./logs/error.log.gz
```

해석:

- `3일 지난 .log` 파일을 `gzip`으로 압축
- `\;`는 파일마다 한 번씩 실행, `+`는 여러 파일을 묶어서 실행하므로 더 효율적
- 보존은 하되 용량은 줄이고 싶을 때 적합

---

### 1-9. 안전하게 삭제하는 3단계

1단계: 대상 확인

```bash
find ./tmp -type f -name "*.tmp" -mtime +7 -print
```

예시 결과:

```text
./tmp/a.tmp
./tmp/b.tmp
```

해석:

- 삭제 후보 두 개를 먼저 눈으로 확인하는 단계

2단계: 자세히 보기

```bash
find ./tmp -type f -name "*.tmp" -mtime +7 -exec ls -lh {} \;
```

예시 결과:

```text
-rw-r--r-- 1 dev dev 1.4M Mar 10 01:12 ./tmp/a.tmp
-rw-r--r-- 1 dev dev  84K Mar 11 03:44 ./tmp/b.tmp
```

해석:

- 파일 크기와 수정 시각까지 함께 검증 가능
- “지워도 되는지” 판단에 도움이 됨

3단계: 실제 삭제

```bash
find ./tmp -type f -name "*.tmp" -mtime +7 -delete
```

예시 결과:

```text
# 출력 없음
```

삭제 후 확인:

```bash
find ./tmp -type f -name "*.tmp" -mtime +7 -print
```

예시 결과:

```text
# 출력 없음
```

해석:

- 같은 조건으로 재검색했을 때 출력이 없으면 삭제 완료
- `find -delete`는 강력하므로 시작 경로를 항상 좁게 잡아야 함

---

### `find` 실수 방지 포인트

- `find . -name *.log`처럼 따옴표 없이 쓰지 않기
- 루트 전체(`/`)를 무작정 검색하지 않기
- `-delete` 전에 `-print` 또는 `-exec ls -lh {} \;`로 확인
- 공백이 있는 파일명을 다룰 때는 `-print0` + `xargs -0`

---

## 2. 문자열·패턴 검색: `grep`

`grep`은 파일이나 표준 입력에서 **특정 문자열이 들어 있는 줄**을 찾습니다.  
로그 분석, 설정 탐색, 소스 코드 검색, 실시간 필터링에 매우 자주 쓰입니다.

---

### 2-1. 로그에서 에러 찾기

```bash
grep -n "ERROR" app.log
```

예시 결과:

```text
128:2026-03-24 10:14:03 ERROR database connection refused
271:2026-03-24 10:18:51 ERROR timeout while calling payment API
```

해석:

- `128`, `271`은 줄 번호
- 문제 시점과 메시지를 빠르게 찾을 수 있음
- 긴 로그에서 “어디부터 봐야 하는지” 감을 잡기 좋음

---

### 2-2. 대소문자 무시하고 찾기

```bash
grep -ni "warning" app.log
```

예시 결과:

```text
83:WARNING cache miss for key user:1902
84:warning retrying request
```

해석:

- `WARNING`, `warning`, `Warning`을 모두 찾음
- 로그 포맷이 팀마다 다를 때 유용

---

### 2-3. 여러 패턴 중 하나라도 찾기

```bash
grep -E "ERROR|WARN|FATAL" app.log
```

예시 결과:

```text
2026-03-24 10:10:20 WARN retry scheduled
2026-03-24 10:14:03 ERROR database connection refused
2026-03-24 10:21:42 FATAL worker crashed
```

해석:

- `-E`를 쓰면 `|` 같은 확장 정규식을 편하게 사용 가능
- 심각도별 로그를 한 번에 훑어볼 때 좋음

---

### 2-4. 노이즈가 많은 줄은 제외하고 보기

```bash
grep -vE "healthcheck|/metrics" access.log
```

예시 결과:

```text
10.0.0.4 - - [24/Mar/2026:10:11:08 +0900] "POST /login HTTP/1.1" 500 612
10.0.0.6 - - [24/Mar/2026:10:11:12 +0900] "GET /api/orders HTTP/1.1" 200 983
```

해석:

- 지나치게 자주 찍히는 헬스체크·메트릭 요청을 제외
- 실제 사용자 요청만 보고 싶을 때 많이 씀

---

### 2-5. 코드베이스 전체에서 재귀 검색

```bash
grep -Rni --exclude-dir=node_modules --exclude-dir=.git "TODO" src/
```

예시 결과:

```text
src/api/user.js:18:// TODO add validation
src/jobs/sync.js:102:// TODO remove retry hack
```

해석:

- `-R` 재귀 검색
- `-n` 줄 번호 출력
- `-i` 대소문자 무시
- `node_modules`, `.git` 제외로 성능과 정확성을 둘 다 챙길 수 있음

---

### 2-6. 앞뒤 문맥까지 함께 보기

```bash
grep -nC 2 "panic" server.log
```

예시 결과:

```text
310-2026-03-24 10:43:18 INFO starting background cleanup
311-2026-03-24 10:43:19 INFO loading cache
312:2026-03-24 10:43:20 panic: nil pointer dereference
313-2026-03-24 10:43:20 goroutine 82 [running]:
314-2026-03-24 10:43:20 service.(*Worker).Run(...)
```

해석:

- `-C 2`는 앞뒤 2줄까지 같이 보여줌
- 에러 직전과 직후 상황을 함께 봐야 원인 추정이 쉬움

---

### 2-7. 매칭된 파일 이름만 보기

```bash
grep -Rl "DATABASE_URL" .
```

예시 결과:

```text
./.env
./docker-compose.yml
./config/prod.env
```

해석:

- 어느 파일에 문자열이 들어 있는지만 빠르게 확인
- 설정 키가 어디에 선언·복제됐는지 볼 때 유용

---

### 2-8. 특수문자가 들어 있는 문자열은 고정 문자열로 찾기

```bash
grep -F "user[id]=42" app.log
```

예시 결과:

```text
2026-03-24 10:16:08 DEBUG query=user[id]=42 retry=0
```

해석:

- `[]`, `.`, `*`를 정규식이 아니라 문자 그대로 처리
- 사용자가 입력한 쿼리 스트링, JSON 일부, SQL 조각 찾을 때 안전함

---

### 2-9. 실시간 로그에서 에러만 추리기

```bash
tail -F app.log | grep --line-buffered -E "ERROR|WARN"
```

예시 결과:

```text
2026-03-24 10:44:10 WARN redis timeout, retrying
2026-03-24 10:44:18 ERROR payment API returned 502
```

해석:

- `tail -F`는 로그 회전에도 따라감
- `--line-buffered`를 주면 파이프 지연을 줄여 거의 실시간으로 보임
- 장애 재현 중 핵심 메시지만 보고 싶을 때 좋은 패턴

---

### 2-10. 개수만 세기

```bash
grep -c " 500 " access.log
```

예시 결과:

```text
37
```

해석:

- `500` 상태 코드가 찍힌 줄이 37개라는 뜻
- 비율 계산까지는 아니지만 에러 증가 여부를 빠르게 판단 가능

---

### `grep` 실수 방지 포인트

- 재귀 검색 시 `node_modules`, `.git`, 바이너리 디렉터리를 제외하기
- 특수문자가 포함된 문자열은 `-F`를 우선 고려하기
- `ps aux | grep nginx`처럼 자기 자신이 잡히는 패턴은 `grep "[n]ginx"` 또는 `pgrep -af nginx`로 대체하기

---

## 3. 프로세스 확인: `ps`

`ps`는 현재 실행 중인 프로세스 상태를 보여줍니다.  
프로세스가 떠 있는지, CPU나 메모리를 많이 쓰는지, 부모-자식 관계가 어떤지 확인할 때 기본적으로 사용합니다.

---

### 3-1. 전체 프로세스 보기

```bash
ps aux | head
```

예시 결과:

```text
USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
root         1  0.0  0.1 169480 11328 ?        Ss   Mar23   0:03 /sbin/init
deploy    18342 72.4  8.3 2416120 684320 ?     Sl   10:15   4:12 python app.py
deploy    18490  2.1  1.2  826520  99840 ?     S    10:16   0:07 nginx: worker process
```

해석:

- `PID`: 프로세스 ID
- `%CPU`: CPU 사용률
- `%MEM`: 메모리 사용률
- `RSS`: 실제 물리 메모리 사용량
- `STAT`: 상태 코드
- `COMMAND`: 실행 명령
- 순간적으로 어떤 프로세스가 눈에 띄는지 1차 탐색할 때 좋음

---

### 3-2. CPU 많이 쓰는 프로세스 상위 보기

```bash
ps aux --sort=-%cpu | head
```

예시 결과:

```text
USER       PID %CPU %MEM COMMAND
deploy    18342 72.4  8.3 python app.py
postgres   9241 18.8  6.1 postgres: writer process
root       2211  7.4  0.4 /usr/sbin/rsyslogd
```

해석:

- `--sort=-%cpu`는 CPU 사용률 내림차순
- “지금 CPU를 누가 먹고 있는지” 가장 빨리 파악하는 방법 중 하나

---

### 3-3. 메모리 많이 쓰는 프로세스 상위 보기

```bash
ps aux --sort=-%mem | head
```

예시 결과:

```text
USER       PID %CPU %MEM   RSS COMMAND
deploy    18342 51.9  8.3 684320 python app.py
java      14300 12.1  7.8 642180 java -jar api.jar
redis      2011  1.2  3.4 281440 /usr/bin/redis-server
```

해석:

- 메모리 누수 의심 시 가장 먼저 보는 명령 중 하나
- `%MEM`만 보지 말고 `RSS`도 같이 보는 습관이 좋음

---

### 3-4. 특정 PID만 자세히 보기

```bash
ps -p 18342 -o pid,ppid,user,%cpu,%mem,rss,vsz,lstart,cmd
```

예시 결과:

```text
  PID  PPID USER     %CPU %MEM   RSS    VSZ                  STARTED CMD
18342 18290 deploy   72.4  8.3 684320 2416120 Tue Mar 24 10:15:02 2026 python app.py
```

해석:

- 한 개 프로세스에 대해 필요한 컬럼만 뽑아봄
- 스크립트 자동화, 모니터링 점검, 특정 PID 조사에 적합

---

### 3-5. 부모-자식 관계 트리 보기

```bash
ps -ef --forest
```

예시 결과:

```text
root      1001     1  0 Mar23 ?        00:00:00 sshd: /usr/sbin/sshd -D
deploy   18290  1001  0 10:14 ?        00:00:00  \_ bash
deploy   18342 18290 72 10:15 ?        00:04:12      \_ python app.py
deploy   18410 18342  1 10:15 ?        00:00:02          \_ worker --queue email
```

해석:

- 어떤 프로세스가 누구에 의해 생성됐는지 한눈에 보임
- `supervisor`, `systemd`, 쉘 스크립트, 워커 프로세스 관계를 추적할 때 좋음

---

### 3-6. 좀비 프로세스 찾기

```bash
ps -el | grep ' Z '
```

예시 결과:

```text
0 Z  1000 22510 22490  0  80   0 -     0 -      ?        00:00:00 defunct
```

해석:

- `Z`는 zombie 상태
- 자식 프로세스가 종료됐지만 부모가 아직 수거하지 못한 상태
- 하나 정도는 즉시 치명적이지 않을 수 있지만 계속 쌓이면 부모 프로세스를 확인해야 함

---

### 3-7. 이름으로 프로세스 찾기: `pgrep`

```bash
pgrep -af "python|gunicorn|uvicorn"
```

예시 결과:

```text
18342 python app.py
18521 gunicorn app.wsgi:application
```

해석:

- `ps aux | grep`보다 깔끔하게 프로세스를 찾을 수 있음
- `-a`는 전체 명령줄, `-f`는 전체 커맨드라인 매칭

---

### 3-8. 종료 신호 보내기: `kill`

```bash
kill 18342
```

예시 결과:

```text
# 출력 없음
```

종료 확인:

```bash
ps -p 18342
```

예시 결과:

```text
# 출력 없음
```

해석:

- 기본 `kill`은 `TERM` 신호
- 대부분의 애플리케이션은 이 신호를 받고 정상 종료를 시도함
- 바로 `kill -9`를 쓰기보다 `TERM` → 확인 → 필요할 때만 `-9` 순으로 가는 것이 좋음

---

### `ps` 출력에서 자주 보는 상태 코드

| 코드 | 의미 |
|---|---|
| `R` | Running |
| `S` | Sleep |
| `D` | I/O 대기 등 인터럽트 불가 sleep |
| `T` | 중지 |
| `Z` | Zombie |
| `l` | 멀티스레드 |
| `s` | 세션 리더 |

예를 들어 `Sl`은 “대기 상태이지만 멀티스레드 프로세스”로 읽으면 됩니다.

---

## 4. 리소스 확인: CPU / Memory / Disk / GPU / Network

장애 대응에서 자주 묻는 질문은 거의 비슷합니다.

- CPU가 바쁜가?
- 메모리가 부족한가?
- 디스크가 찼는가?
- GPU가 꽉 찼는가?
- 어떤 포트를 누가 잡고 있는가?

이 섹션은 그 질문에 바로 답하기 위한 명령 묶음입니다.

---

### 4-1. CPU 부하 확인: `uptime`

```bash
uptime
```

예시 결과:

```text
10:42:31 up 5 days, 3:44, 2 users, load average: 0.62, 0.74, 0.81
```

해석:

- 마지막 세 숫자는 1분, 5분, 15분 평균 부하(load average)
- 해석 기준은 **CPU 코어 수**
- 8코어 머신에서 `0.81`은 여유가 있는 편
- 2코어 머신에서 `8.0`이면 큐가 많이 밀린 상태일 가능성이 큼

---

### 4-2. CPU 구조 확인: `lscpu`

```bash
lscpu
```

예시 결과:

```text
Architecture:        x86_64
CPU(s):              8
Thread(s) per core:  2
Core(s) per socket:  4
Model name:          Intel(R) Xeon(R) CPU
```

해석:

- 총 CPU 개수, 코어 수, 스레드 수 확인
- `uptime`의 load average를 해석할 때 기준점이 됨

---

### 4-3. 실시간 전체 상태: `top`

```bash
top
```

예시 결과:

```text
top - 10:44:15 up 5 days, 3:46, 2 users, load average: 2.44, 1.98, 1.30
Tasks: 284 total,   2 running, 282 sleeping,   0 stopped,   0 zombie
%Cpu(s): 71.2 us,  9.8 sy,  0.0 ni, 17.1 id,  1.1 wa,  0.0 hi,  0.8 si,  0.0 st
MiB Mem : 15925.4 total, 1021.2 free, 9210.7 used, 5693.5 buff/cache
MiB Swap:  2048.0 total, 2048.0 free,    0.0 used. 6123.1 avail Mem
```

해석:

- `us`: 사용자 영역 CPU 사용
- `sy`: 커널 영역 CPU 사용
- `id`: idle
- `wa`: I/O wait
- `avail Mem`: 실제로 쓸 수 있는 여유 메모리
- `wa`가 높으면 CPU보다 디스크나 네트워크 I/O 병목을 의심해야 함

---

### 4-4. 특정 프로세스의 스레드 보기

```bash
top -H -p 18342
```

예시 결과:

```text
PID   USER   PR NI  VIRT   RES  SHR S %CPU %MEM     TIME+ COMMAND
18342 deploy 20  0 2359m 668m  12m S 12.0  4.2   0:32.11 python
18357 deploy 20  0 2359m 668m  12m R 88.4  4.2   3:21.44 python
18360 deploy 20  0 2359m 668m  12m S  1.2  4.2   0:04.28 python
```

해석:

- 같은 PID의 개별 스레드 사용량을 보여줌
- 멀티스레드 애플리케이션에서 “어느 스레드가 CPU를 먹는지” 확인할 때 좋음

---

### 4-5. 메모리 요약: `free -h`

```bash
free -h
```

예시 결과:

```text
               total        used        free      shared  buff/cache   available
Mem:            15Gi       9.0Gi       1.0Gi       512Mi       5.0Gi       6.0Gi
Swap:          2.0Gi         0Bi       2.0Gi
```

해석:

- `free`만 보고 메모리가 없다고 판단하면 안 됨
- Linux는 남는 메모리를 캐시로 적극 활용함
- 실제로 중요하게 볼 값은 `available`
- `available`이 낮고 `swap`이 증가 중이면 메모리 압박 가능성이 큼

---

### 4-6. 짧은 간격으로 변화를 보기: `vmstat`

```bash
vmstat 1 5
```

예시 결과:

```text
procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
 r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
 1  0      0 1034528 256320 5120016   0    0     2     5  320  811 12  4 83  1  0
 3  1      0  834120 256320 5121100   0    0    20   930 1220 3110 41 12 39  8  0
```

해석:

- `r`: 실행 대기 중인 프로세스 수
- `b`: I/O 대기 등으로 블록된 프로세스 수
- `si`, `so`: swap in / swap out
- `wa`: I/O wait
- `so`가 계속 크면 메모리 부족 가능성, `wa`가 높으면 I/O 병목 가능성

---

### 4-7. 파일시스템 용량 확인: `df -h`

```bash
df -h
```

예시 결과:

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        80G   71G  5.2G  94% /
/dev/sdb1       200G  112G   79G  59% /data
tmpfs           7.8G     0  7.8G   0% /run/user/1000
```

해석:

- `/`가 94%면 시스템 전체가 불안정해질 수 있음
- 로그, 패키지 캐시, 빌드 산출물, Docker 레이어 같은 후보를 바로 의심해야 함

---

### 4-8. 큰 디렉터리 찾기: `du`

```bash
du -xhd1 /var | sort -h
```

예시 결과:

```text
12M   /var/tmp
180M  /var/cache
1.4G  /var/log
8.2G  /var/lib
9.8G  /var
```

해석:

- `/var` 아래 어떤 디렉터리가 용량을 차지하는지 한 단계 깊이로 요약
- `-x`는 다른 파일시스템으로 넘어가지 않음
- `df`에서 문제 파일시스템을 찾고, `du`로 범위를 좁히는 식으로 자주 사용

---

### 4-9. 어떤 포트를 누가 잡고 있는지: `ss`

```bash
ss -tulpn
```

예시 결과:

```text
Netid State  Recv-Q Send-Q Local Address:Port  Peer Address:Port Process
tcp   LISTEN 0      128    0.0.0.0:22          0.0.0.0:*         users:(("sshd",pid=1010,fd=3))
tcp   LISTEN 0      511    0.0.0.0:80          0.0.0.0:*         users:(("nginx",pid=18490,fd=6))
tcp   LISTEN 0      128    127.0.0.1:5432      0.0.0.0:*         users:(("postgres",pid=9241,fd=7))
```

해석:

- `LISTEN` 중인 포트와 해당 프로세스를 보여줌
- “8080 포트 점유 프로세스가 누구인지” 같은 질문에 바로 답 가능

특정 포트만 보려면:

```bash
ss -tulpn | grep ":8080"
```

예시 결과:

```text
tcp   LISTEN 0      128    0.0.0.0:8080   0.0.0.0:*   users:(("java",pid=14300,fd=118))
```

해석:

- 8080 포트를 `java` 프로세스가 점유 중이라는 뜻
- 포트 충돌 원인 분석에 자주 씀

---

### 4-10. GPU 상태 확인: `nvidia-smi`

```bash
nvidia-smi
```

예시 결과:

```text
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|  0   RTX 4090               On | 00000000:65:00.0 Off |                  Off |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
| 35%   52C    P2   180W / 450W |  12288MiB / 24564MiB |     91%      Default |
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   PID   Type   Process name                               GPU Memory    |
|    0  18342    C    python train.py                              12280MiB    |
+-----------------------------------------------------------------------------+
```

해석:

- `Memory-Usage`: GPU 메모리 점유량
- `GPU-Util`: 실제 연산 사용률
- 프로세스 목록으로 어떤 PID가 GPU를 쓰는지 확인 가능
- 메모리는 찼는데 `GPU-Util`이 낮으면 데이터 로딩 병목이나 대기 시간을 의심할 수 있음

---

### 4-11. GPU 프로세스만 간단히 보기: `nvidia-smi pmon`

```bash
nvidia-smi pmon -c 1
```

예시 결과:

```text
# gpu        pid  type    sm   mem   enc   dec   command
    0      18342     C    89    48     0     0   python
```

해석:

- 어떤 프로세스가 GPU 연산과 메모리를 얼마나 쓰는지 요약해서 보여줌
- `nvidia-smi` 전체 표보다 빠르게 프로세스 중심으로 볼 수 있음

---

## 5. 로그 확인과 출력 저장

로그 섹션에서는 다음 작업이 핵심입니다.

- 최근 로그 보기
- 실시간 추적
- 서비스 단위 로그 보기
- 커널 레벨 문제 보기
- 커맨드 출력을 파일로 저장
- stdout과 stderr를 구분해서 남기기

---

### 5-1. stdout을 파일에 저장하기

```bash
./app.sh > app.log
```

실행 후 파일 확인:

```bash
tail -n 3 app.log
```

예시 결과:

```text
[INFO] server starting
[INFO] config loaded
[INFO] listening on :8080
```

해석:

- 표준 출력(stdout)만 `app.log`에 저장
- 기존 파일 내용은 덮어씀
- 터미널에는 기본적으로 아무 것도 안 보일 수 있음

---

### 5-2. stdout을 이어쓰기

```bash
./batch.sh >> batch.log
```

실행 후 파일 확인:

```bash
tail -n 3 batch.log
```

예시 결과:

```text
[INFO] job started at 10:10:00
[INFO] processed 120 rows
[INFO] job completed
```

해석:

- 기존 내용 뒤에 이어서 기록
- 크론 작업, 배치 히스토리 누적에 자주 씀

---

### 5-3. stderr만 따로 저장하기

```bash
./app.sh 2> error.log
```

실행 후 파일 확인:

```bash
cat error.log
```

예시 결과:

```text
connect: connection refused
failed to open config file: ./config/prod.yml
```

해석:

- 표준 에러(stderr)만 `error.log`로 보냄
- 실패 원인만 따로 모으고 싶을 때 유용
- stdout은 여전히 터미널에 출력될 수 있음

---

### 5-4. stdout + stderr를 한 파일에 저장하기

```bash
./deploy.sh > deploy.log 2>&1
```

실행 후 확인:

```bash
tail -n 5 deploy.log
```

예시 결과:

```text
[INFO] pulling image myapp:2026.03.24
[INFO] stopping old container
[WARN] healthcheck delayed
[INFO] starting new container
[INFO] deploy completed
```

해석:

- `2>&1`은 stderr를 stdout이 향하는 곳으로 보냄
- 배포나 마이그레이션처럼 “모든 출력”을 한 파일에 남길 때 좋음
- 순서가 중요하므로 `> file 2>&1` 형태를 습관적으로 쓰는 편이 안전함

---

### 5-5. 최근 로그 보기: `tail -n`

```bash
tail -n 100 app.log
```

예시 결과:

```text
2026-03-24 10:12:01 INFO server started
2026-03-24 10:12:19 INFO connected to redis
2026-03-24 10:14:03 ERROR database connection refused
```

해석:

- 큰 로그 파일에서도 최근 N줄만 빠르게 확인 가능
- 장애 직전 상황을 볼 때 가장 먼저 쓰는 명령 중 하나

---

### 5-6. 실시간 로그 추적: `tail -f`

```bash
tail -f app.log
```

예시 결과:

```text
2026-03-24 10:44:05 INFO request_id=ab12 started
2026-03-24 10:44:06 WARN retrying external API
2026-03-24 10:44:07 INFO request_id=ab12 completed
```

해석:

- 파일 끝에 새 줄이 추가될 때마다 계속 출력
- 같은 파일에 계속 로그를 쓰는 프로세스를 추적할 때 적합

---

### 5-7. 로그 회전까지 따라가기: `tail -F`

```bash
tail -F app.log
```

예시 결과:

```text
==> app.log <==
2026-03-24 10:58:59 INFO request finished

==> app.log <==
2026-03-24 11:00:00 INFO log reopened after rotation
2026-03-24 11:00:01 ERROR reconnecting to database
```

해석:

- 로그 파일이 교체되거나 회전돼도 다시 따라감
- 운영 서버에서는 `-f`보다 `-F`가 더 안전한 경우가 많음

---

### 5-8. 큰 로그를 탐색하기: `less`

```bash
less app.log
```

예시 결과:

```text
2026-03-24 10:12:01 INFO server started
2026-03-24 10:12:19 INFO connected to redis
2026-03-24 10:14:03 ERROR database connection refused
2026-03-24 10:14:04 INFO retry scheduled
```

해석:

- 출력 자체보다 **탐색 기능**이 핵심
- `/ERROR`로 검색, `n`으로 다음 결과, `G`로 파일 끝 이동이 가능
- `cat`보다 훨씬 실용적임

실시간 모드로 열고 싶다면:

```bash
less +F app.log
```

예시 결과:

```text
2026-03-24 10:44:05 INFO request started
2026-03-24 10:44:06 WARN retrying external API
```

해석:

- `tail -f`처럼 따라가다가 `Ctrl+C`로 실시간 모드를 빠져나와 과거 로그를 탐색할 수 있음

---

### 5-9. systemd 서비스 로그 보기: `journalctl`

```bash
journalctl -u myapp.service -n 100 --no-pager
```

예시 결과:

```text
Mar 24 10:12:01 api-01 systemd[1]: Started myapp.service.
Mar 24 10:14:03 api-01 myapp[18342]: ERROR database connection refused
Mar 24 10:14:04 api-01 myapp[18342]: retrying in 5 seconds
```

해석:

- 특정 서비스 단위로 최근 로그 확인
- 애플리케이션 로그와 systemd 상태 변화가 함께 보이므로 서비스 재시작 원인 파악에 좋음

---

### 5-10. 특정 시점 이후 로그만 보기

```bash
journalctl -u myapp.service --since "2026-03-24 10:00" --no-pager
```

예시 결과:

```text
Mar 24 10:03:10 api-01 myapp[18342]: INFO deploy version=2026.03.24
Mar 24 10:14:03 api-01 myapp[18342]: ERROR database connection refused
Mar 24 10:14:04 api-01 myapp[18342]: retrying in 5 seconds
```

해석:

- 배포 이후, 특정 장애 시각 이후 로그만 추릴 수 있음
- 로그량이 많은 서비스일수록 시간 필터가 중요함

---

### 5-11. 서비스 로그 실시간 추적

```bash
journalctl -u myapp.service -f
```

예시 결과:

```text
Mar 24 10:44:12 api-01 myapp[18342]: INFO received job_id=9281
Mar 24 10:44:13 api-01 myapp[18342]: WARN job_id=9281 retry=1
Mar 24 10:44:15 api-01 myapp[18342]: INFO job_id=9281 completed
```

해석:

- `tail -F`와 비슷하지만 파일이 아니라 서비스 기준
- 로그 파일 경로를 몰라도 된다는 장점이 있음

---

### 5-12. 커널 메시지 확인: `dmesg`

```bash
dmesg -T | tail -n 20
```

예시 결과:

```text
[Tue Mar 24 10:21:42 2026] Out of memory: Killed process 18342 (python) total-vm:2416120kB
[Tue Mar 24 10:21:42 2026] oom_reaper: reaped process 18342 (python)
```

해석:

- OOM, 디스크 I/O 오류, 드라이버 문제 같은 커널 레벨 이슈를 확인
- 애플리케이션 로그에는 이유가 없는데 프로세스가 죽었을 때 가장 먼저 확인할 가치가 큼

---

### 5-13. 화면에도 보이고 파일에도 남기기: `tee`

```bash
./deploy.sh 2>&1 | tee deploy-2026-03-24-1015.log
```

예시 결과(터미널에 보이는 내용):

```text
[INFO] pulling image myapp:2026.03.24
[INFO] stopping old container
[WARN] healthcheck delayed
[INFO] deploy completed
```

생성된 로그 파일 확인:

```bash
tail -n 4 deploy-2026-03-24-1015.log
```

예시 결과:

```text
[INFO] pulling image myapp:2026.03.24
[INFO] stopping old container
[WARN] healthcheck delayed
[INFO] deploy completed
```

해석:

- 터미널에서 상황을 보면서 동시에 파일에도 저장
- 수동 배포, 장애 대응 기록, 운영 작업 로그 보존에 매우 유용

---

### 5-14. access log 상태 코드 집계

```bash
awk '{print $9}' access.log | sort | uniq -c | sort -nr
```

예시 결과:

```text
1452 200
  81 404
  37 500
  12 302
```

해석:

- 보통 Nginx/Apache access log에서 9번째 필드가 상태 코드
- 500이 얼마나 나왔는지 빠르게 감을 잡을 수 있음
- 에러 비율 계산 전 1차 점검용으로 좋음

---

### 5-15. 가장 많이 요청된 경로 상위 보기

```bash
awk '{print $7}' access.log | sort | uniq -c | sort -nr | head
```

예시 결과:

```text
900 /health
420 /api/orders
215 /api/login
```

해석:

- 어떤 경로가 가장 많이 호출됐는지 파악 가능
- `/health`가 압도적이면 실제 사용자 트래픽과 구분해서 봐야 함

---

## 6. 파일 정리와 디스크 회수

이 섹션은 “디스크가 부족할 때 무엇부터 확인하고 어떻게 정리할지”에 초점을 맞춥니다.  
핵심은 **크기 파악 → 후보 확인 → 삭제 또는 압축 → 재검증** 흐름입니다.

---

### 6-1. 현재 디렉터리에서 큰 항목 찾기

```bash
du -sh ./* | sort -h
```

예시 결과:

```text
84K   ./scripts
320M  ./logs
1.2G  ./build
2.8G  ./tmp
```

해석:

- 어떤 디렉터리가 큰지 한눈에 볼 수 있음
- 정리 우선순위를 잡는 첫 단계로 좋음

---

### 6-2. 오래된 임시 파일 찾기

```bash
find ./tmp -type f -name "*.tmp" -mtime +7 -print
```

예시 결과:

```text
./tmp/cache-20260310.tmp
./tmp/render-1821.tmp
```

해석:

- 7일 이상 지난 임시 파일만 후보로 찾음
- 바로 삭제 전에 반드시 거치는 단계

---

### 6-3. 오래된 임시 파일 삭제

```bash
find ./tmp -type f -name "*.tmp" -mtime +7 -delete
```

예시 결과:

```text
# 출력 없음
```

삭제 확인:

```bash
find ./tmp -type f -name "*.tmp" -mtime +7 -print
```

예시 결과:

```text
# 출력 없음
```

해석:

- 삭제 자체는 출력이 없더라도, 재검색 결과가 없으면 완료된 것
- 운영 디렉터리에서는 항상 범위를 좁게 주는 것이 중요

---

### 6-4. 빈 디렉터리 찾기

```bash
find . -type d -empty -print
```

예시 결과:

```text
./tmp/cache
./uploads/staging
```

해석:

- 비어 있는 디렉터리를 먼저 확인
- 삭제보다 먼저 “어떤 폴더가 비어 있는지” 파악하는 단계

---

### 6-5. 빈 디렉터리 삭제

```bash
find . -type d -empty -delete
```

예시 결과:

```text
# 출력 없음
```

삭제 후 확인:

```bash
find . -type d -empty -print
```

예시 결과:

```text
# 출력 없음
```

해석:

- 빈 디렉터리 정리가 완료되면 재검색 결과가 없어야 함
- 상위 디렉터리가 연쇄적으로 비게 될 수 있으므로 반복 실행이 필요할 수도 있음

---

### 6-6. 오래된 로그를 삭제 대신 압축

```bash
find ./logs -type f -name "*.log" -mtime +3 -exec gzip {} +
```

예시 결과:

```text
# 성공 시 대부분 출력 없음
```

압축 결과 확인:

```bash
find ./logs -type f -name "*.gz" -print
```

예시 결과:

```text
./logs/app.log.gz
./logs/error.log.gz
```

해석:

- 로그를 바로 지우기 아까울 때 가장 무난한 정리 방법
- 보존 요구사항이 있는 운영 환경에서 특히 유용

---

### 6-7. 큰 로그·덤프 파일만 따로 찾기

```bash
find . -type f \( -name "*.log" -o -name "*.dump" -o -name "*.tar" \) -size +100M -exec ls -lh {} \;
```

예시 결과:

```text
-rw-r--r-- 1 dev dev 1.1G Mar 24 09:00 ./backups/app.dump
-rw-r--r-- 1 dev dev 420M Mar 24 09:10 ./logs/server.log
```

해석:

- 무작정 전체를 지우는 대신 “공간을 많이 쓰는 파일 유형”만 우선 추릴 수 있음
- 운영 로그, 백업 덤프, 아카이브 파일 점검에 좋음

---

### 6-8. 지우기 전에 아카이브하기

```bash
tar -czf archive-2026-03-24.tar.gz build/ logs/
```

예시 결과:

```text
# 성공 시 일반적으로 출력 없음
```

생성 파일 확인:

```bash
ls -lh archive-2026-03-24.tar.gz
```

예시 결과:

```text
-rw-r--r-- 1 dev dev 428M Mar 24 10:55 archive-2026-03-24.tar.gz
```

해석:

- 삭제 전에 백업본을 남기고 싶을 때 적합
- 빌드 산출물, 로그, 배포 결과물 보관에 자주 사용

---

### 6-9. Python 캐시 정리

```bash
find . -type d -name "__pycache__" -print
```

예시 결과:

```text
./src/__pycache__
./tests/__pycache__
```

해석:

- 캐시 디렉터리가 어디 있는지 먼저 확인
- Python 프로젝트에서 흔한 정리 대상

실제 삭제:

```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
```

예시 결과:

```text
# 출력 없음
```

삭제 확인:

```bash
find . -type d -name "__pycache__" -print
```

예시 결과:

```text
# 출력 없음
```

해석:

- 캐시가 사라졌는지 재검색으로 검증
- `rm -rf`가 들어가므로 반드시 검색 범위를 좁게 잡아야 함

---

### 6-10. `.pyc` 파일 삭제

```bash
find . -type f -name "*.pyc" -delete
```

예시 결과:

```text
# 출력 없음
```

삭제 전후 확인:

```bash
find . -type f -name "*.pyc" -print
```

예시 결과:

```text
# 출력 없음
```

해석:

- Python 바이트코드 파일 정리
- 빌드 산출물/캐시 정리 스크립트에 자주 포함됨

---

### 6-11. Git 무시 파일만 안전하게 정리

미리보기:

```bash
git clean -ndX
```

예시 결과:

```text
Would remove dist/
Would remove coverage/
Would remove .pytest_cache/
```

해석:

- `.gitignore`에 의해 무시되는 파일만 후보로 보여줌
- 가장 안전한 1차 검토 단계

실제 삭제:

```bash
git clean -fdX
```

예시 결과:

```text
Removing dist/
Removing coverage/
Removing .pytest_cache/
```

해석:

- 무시 파일과 디렉터리를 실제로 삭제
- 빌드 찌꺼기, 캐시, 테스트 산출물 정리에 매우 유용
- `git clean -fd`는 미추적 파일까지 지울 수 있어 더 위험하므로 차이를 알고 써야 함

---

### 6-12. 공백 있는 파일명을 안전하게 지우기

```bash
find ./trash -type f -name "*.bak" -print0 | xargs -0 rm -v
```

예시 결과:

```text
removed './trash/old report (final).bak'
removed './trash/user backup 01.bak'
```

해석:

- 공백과 괄호가 있는 파일명도 깨지지 않음
- 오래된 수동 백업 파일 정리에 자주 쓰는 패턴

---

## 7. 실무에서 자주 쓰는 조합

---

### 케이스 1. 디스크가 찼는데 어디가 큰지 모르겠다

```bash
df -h
du -xhd1 /var | sort -h
find /var -xdev -type f -size +200M -exec ls -lh {} \;
```

예시 결과:

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        80G   76G  2.1G  98% /

12M   /var/tmp
1.4G  /var/log
8.2G  /var/lib

-rw-r--r-- 1 root root 1.8G Mar 24 08:12 /var/log/myapp/server.log
-rw-r--r-- 1 root root 920M Mar 24 08:20 /var/lib/docker/overlay2/.../diff/app/cache.bin
```

해석:

- `df`로 문제 파일시스템을 찾고
- `du`로 큰 디렉터리를 좁히고
- `find`로 실제 큰 파일까지 특정하는 흐름
- 장애 대응 시 가장 기본적인 디스크 추적 패턴

---

### 케이스 2. 특정 서비스가 자꾸 죽는다

```bash
ps aux | grep "[m]yapp"
journalctl -u myapp.service -n 50 --no-pager
dmesg -T | tail -n 20
```

예시 결과:

```text
deploy   18342  0.0  0.0  0  0 ?  Z 10:21 0:00 [myapp] <defunct>

Mar 24 10:21:41 api-01 myapp[18342]: INFO starting request batch
Mar 24 10:21:42 api-01 systemd[1]: myapp.service: Main process exited, code=killed, status=9/KILL

[Tue Mar 24 10:21:42 2026] Out of memory: Killed process 18342 (myapp)
```

해석:

- 프로세스 상태 확인
- 서비스 로그 확인
- 커널 OOM 여부 확인
- 이렇게 3단계로 보면 “앱 문제인지, systemd 문제인지, 시스템 리소스 문제인지” 빠르게 좁힐 수 있음

---

### 케이스 3. 배포 직후 에러만 실시간으로 보고 싶다

```bash
journalctl -u myapp.service -f | grep --line-buffered -iE "error|timeout|refused"
```

예시 결과:

```text
Mar 24 10:44:18 api-01 myapp[18342]: ERROR payment API timeout
Mar 24 10:44:25 api-01 myapp[18342]: ERROR database connection refused
```

해석:

- 서비스 로그 전체가 아니라 장애 관련 줄만 추려서 볼 수 있음
- 배포 직후 smoke check에 특히 유용

---

### 케이스 4. 코드 전체에서 설정 키가 어디 쓰이는지 찾고 싶다

```bash
grep -Rni --exclude-dir=node_modules --exclude-dir=.git "DATABASE_URL" .
```

예시 결과:

```text
./.env:4:DATABASE_URL=postgres://...
./docker-compose.yml:18:      DATABASE_URL: ${DATABASE_URL}
./src/config/db.ts:7:const url = process.env.DATABASE_URL
```

해석:

- 환경 변수 선언 위치
- 배포 설정 위치
- 실제 코드 참조 위치를 한 번에 확인
- 환경 변수 충돌, 중복 선언, 배포 누락 점검에 좋음

---

### 케이스 5. Git 프로젝트의 빌드 찌꺼기만 정리하고 싶다

```bash
git clean -ndX
git clean -fdX
```

예시 결과:

```text
Would remove dist/
Would remove coverage/
Would remove .pytest_cache/

Removing dist/
Removing coverage/
Removing .pytest_cache/
```

해석:

- 첫 줄은 미리보기, 둘째 줄은 실제 실행
- 로컬 실험 파일까지 날리지 않으면서 `.gitignore` 대상만 정리할 때 매우 안전함

---

## 8. 빠른 치트시트

### 파일 찾기

```bash
find . -type f -name "*.log"
find . -type f -mmin -30
find . -type f -size +100M -exec ls -lh {} \;
find . -type d -empty
```

예시 결과:

```text
./logs/app.log
./tmp/cache.db
-rw-r--r-- 1 dev dev 420M Mar 24 09:10 ./logs/server.log
./tmp/old-cache
```

해석:

- 파일 검색, 최근 수정 파일 확인, 큰 파일 탐지, 빈 디렉터리 점검에 바로 사용 가능

---

### 텍스트 검색

```bash
grep -Rni "TODO" src/
grep -E "ERROR|WARN" app.log
grep -nC 2 "panic" server.log
grep -Rl "DATABASE_URL" .
```

예시 결과:

```text
src/jobs/sync.js:102:// TODO remove retry hack
2026-03-24 10:14:03 ERROR database connection refused
312:2026-03-24 10:43:20 panic: nil pointer dereference
./.env
```

해석:

- 코드 탐색, 로그 필터링, 문맥 확인, 파일 위치 파악까지 한 세트로 자주 쓰임

---

### 프로세스

```bash
ps aux --sort=-%cpu | head
ps aux --sort=-%mem | head
ps -ef --forest
pgrep -af nginx
```

예시 결과:

```text
deploy    18342 72.4  8.3 python app.py
java      14300 12.1  7.8 java -jar api.jar
deploy   18342 18290 72 10:15 ? 00:04:12 \_ python app.py
18490 nginx: worker process
```

해석:

- CPU/메모리 상위 프로세스, 부모-자식 관계, 특정 프로세스 존재 여부를 빠르게 확인 가능

---

### 리소스

```bash
uptime
free -h
df -h
ss -tulpn
nvidia-smi
```

예시 결과:

```text
load average: 0.62, 0.74, 0.81
Mem: 15Gi total, 6.0Gi available
/dev/sda1 80G 71G 5.2G 94% /
tcp LISTEN 0 511 0.0.0.0:80 users:(("nginx",pid=18490,fd=6))
GPU-Util 91%  Memory-Usage 12288MiB / 24564MiB
```

해석:

- CPU 부하, 메모리 여유, 디스크 부족, 포트 점유, GPU 상태를 한 번에 훑는 기본 묶음

---

### 로그

```bash
tail -n 100 app.log
tail -F app.log
journalctl -u myapp.service -n 100 --no-pager
dmesg -T | tail -n 20
./deploy.sh 2>&1 | tee deploy.log
```

예시 결과:

```text
2026-03-24 10:14:03 ERROR database connection refused
2026-03-24 11:00:01 ERROR reconnecting to database
Mar 24 10:14:03 api-01 myapp[18342]: ERROR database connection refused
[Tue Mar 24 10:21:42 2026] Out of memory: Killed process 18342 (python)
[INFO] deploy completed
```

해석:

- 최근 로그, 회전 로그 추적, 서비스 로그, 커널 로그, 배포 로그 저장을 바로 수행 가능

---

### 파일 정리

```bash
du -sh ./* | sort -h
find ./tmp -type f -name "*.tmp" -mtime +7 -print
find . -type d -empty -delete
git clean -ndX
git clean -fdX
```

예시 결과:

```text
84K ./scripts
./tmp/cache-20260310.tmp
# 출력 없음
Would remove dist/
Removing dist/
```

해석:

- 용량 파악, 삭제 후보 확인, 빈 디렉터리 정리, Git 무시 파일 미리보기·실행까지 실무적인 정리 흐름을 구성할 수 있음

---

## 9. 마지막 체크리스트

삭제나 운영 명령을 실행하기 전에는 아래 순서를 습관처럼 확인하면 사고를 크게 줄일 수 있습니다.

1. **시작 경로가 맞는가**
2. **와일드카드를 따옴표로 감쌌는가**
3. **먼저 `-print`, `ls -lh`, `git clean -n`으로 검증했는가**
4. **`sudo`가 정말 필요한가**
5. **삭제 대신 압축이나 백업이 더 적절하지 않은가**
6. **실행 후 재확인 명령까지 준비했는가**

---

필요하다면 이 문서를 다음 형태로도 재가공할 수 있습니다.

- 팀 위키용 축약본
- 온콜/장애 대응 플레이북 스타일
- Docker / systemd / Nginx / PostgreSQL 확장판
- 신입 개발자 온보딩용 “터미널 필수 명령” 버전
