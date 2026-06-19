# LoRA Maker

이미지를 넣으면 자동으로 최적화된 LoRA를 만들어주는 AI 이미지 학습 도구입니다.  
FastAPI 백엔드 + 웹 UI, 선택적으로 Electron 데스크톱 앱으로 실행할 수 있습니다.

---

## 주요 기능

- **4단계 마법사** — 목적 선택 → 기본 설정 → 이미지 업로드 → GPU 선택으로 프로젝트 생성
- **자동 전처리** — 얼굴/오브젝트 감지, 크롭, 리사이즈 자동화 (OpenCV + LBP 애니메이션 얼굴 감지 포함)
- **자동 캡셔닝** — BLIP / WD14 Tagger 기반 태그 자동 생성
- **kohya-sd-scripts 연동** — SDXL LoRA 학습 파이프라인 직접 실행
- **실시간 학습 모니터링** — WebSocket으로 로그·에폭·손실값 실시간 표시
- **체크포인트 관리** — 에폭별 `.safetensors` 저장 및 목록 조회
- **LoRA 검증기** — 가중치 분석 + 이미지 생성 before/after 비교 테스트
- **Vast.ai 연동** — 클라우드 GPU 인스턴스 생성·관리 (API 키 설정)
- **Electron 래퍼** — Windows 데스크톱 앱으로 패키징 가능
- **프리셋** — 캐릭터 / 얼굴 / 오브젝트 / 스타일 LoRA 최적 설정 내장

---

## 요구 사항

- Python 3.10+
- CUDA 지원 GPU (VRAM 8GB+ 권장, 학습 시)
- Node.js 18+ (Electron 앱 빌드 시에만)
- kohya-sd-scripts (별도 설치, 아래 참고)

---

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/Raaatel/lora_maker.git
cd lora_maker
```

### 2. 자동 설치 (Windows)

```bat
install.bat
```

또는 Python 스크립트로:

```bash
python install.py
```

가상환경 생성 → 패키지 설치 → kohya-sd-scripts 클론까지 자동으로 진행됩니다.

### 3. 수동 설치

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

# kohya-sd-scripts 설치
git clone https://github.com/kohya-ss/sd-scripts.git kohya-sd-scripts
cd kohya-sd-scripts
pip install -r requirements.txt
cd ..
```

---

## 실행

### 웹 앱 (권장)

```bat
start.bat
```

또는:

```bash
python start.py
```

브라우저에서 `http://localhost:8000` 으로 접속하세요.

### Electron 데스크톱 앱

```bat
start_electron.bat
```

또는:

```bash
cd electron
npm install
npm start
```

---

## 프로젝트 구조

```
lora-maker/
├── app.py                    # FastAPI 앱 진입점
├── start.py / start.bat      # 실행 스크립트
├── install.py / install.bat  # 설치 스크립트
├── requirements.txt
│
├── server/
│   ├── app_factory.py        # FastAPI 앱 팩토리
│   ├── database.py           # SQLite (aiosqlite) DB 레이어
│   └── routers/
│       ├── api_projects.py   # 프로젝트 CRUD
│       ├── api_upload.py     # 이미지 업로드
│       ├── api_checkpoints.py# 체크포인트 관리
│       ├── api_validation.py # LoRA 검증 (가중치 분석 + 추론 테스트)
│       ├── api_vastai.py     # Vast.ai 연동
│       ├── ws.py             # WebSocket 학습 스트림
│       └── pages.py          # HTML 라우트
│   └── services/
│       ├── training_runner.py    # kohya 학습 프로세스 실행
│       ├── training_manager.py   # 학습 상태 관리
│       ├── preprocess_runner.py  # 이미지 전처리
│       ├── caption_runner.py     # 자동 캡셔닝
│       ├── config_builder.py     # kohya TOML 설정 생성
│       ├── checkpoint_manager.py # 체크포인트 감시
│       ├── validator.py          # 가중치 분석 + 추론
│       ├── vastai_service.py     # Vast.ai API 클라이언트
│       └── websocket_manager.py  # WebSocket 브로드캐스트
│   └── scripts/
│       ├── preprocess.py         # 전처리 스크립트 (얼굴감지·크롭)
│       └── auto_caption.py       # 캡셔닝 스크립트
│
├── config/
│   ├── defaults.yaml             # 기본 학습 설정
│   └── presets/
│       ├── character_lora.yaml
│       ├── face_lora.yaml
│       ├── object_lora.yaml
│       └── style_lora.yaml
│
├── templates/
│   └── index.html                # 메인 UI
├── static/
│   ├── css/app.css
│   └── js/app.js
│
├── electron/
│   ├── main.js                   # Electron 메인 프로세스
│   ├── preload.js
│   └── package.json
│
└── data/                         # 런타임 생성 (git 제외)
    ├── db/app.db                 # SQLite DB
    └── jobs/<project_id>/        # 프로젝트별 이미지·체크포인트
```

---

## 사용법

1. **새 프로젝트** — 사이드바의 `+ 새 프로젝트` 클릭
2. **목적 선택** — 캐릭터 / 얼굴 / 오브젝트 / 스타일 중 선택 (프리셋 자동 적용)
3. **이미지 업로드** — 10~30장 권장, 자동 전처리 및 캡셔닝 실행
4. **GPU 선택** — 로컬 GPU 또는 Vast.ai 클라우드 GPU 선택
5. **학습 시작** — 실시간 로그·손실 그래프 확인
6. **검증** — 에폭별 체크포인트를 클릭해 가중치 분석 또는 이미지 생성 테스트

### Vast.ai 설정

사이드바 하단 ⚙️ 설정에서 Vast.ai API 키와 SSH 개인 키 경로를 입력하세요.  
API 키는 [Vast.ai → Account → API Keys](https://vast.ai/console/account) 에서 발급받을 수 있습니다.

---

## API 엔드포인트 (주요)

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/projects` | 프로젝트 목록 |
| POST | `/api/projects` | 프로젝트 생성 |
| POST | `/api/projects/{id}/start` | 학습 시작 |
| POST | `/api/projects/{id}/stop` | 학습 중단 |
| GET | `/api/projects/{id}/checkpoints` | 체크포인트 목록 |
| GET | `/api/projects/{id}/validate/weight/{epoch}` | 가중치 분석 |
| POST | `/api/projects/{id}/validate/inference` | 추론 테스트 |
| POST | `/api/validate/file` | 파일 직접 검증 |
| WS | `/ws/{project_id}` | 학습 로그 스트림 |

---

## 라이선스

MIT License

---

## 관련 프로젝트

- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) — LoRA 학습 백엔드
- [Vast.ai](https://vast.ai) — 클라우드 GPU 마켓플레이스
