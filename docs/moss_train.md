# moss_train.py - MOSS-Image full fine-tune: user guide

Status: implemented 2026-11-17 in the worktree `backend/sd_scripts_moss_full` (branch `feat/moss-full-ft`) and tested on
the CPU only. The GPU smoke run (design section 7) has not been run yet, so every speed, VRAM and RAM figure below is an
estimate from `docs/moss_full_finetune_design.md`. Implementation details and deviations: `docs/moss_ft_impl_notes.md`.
The live text encoder (`moss_te_live`, plan `docs/moss_ft_te_live_fp16_plan.md` part A) was added the same day, also
tested on the CPU only. fp16 compute with dynamic loss scaling (part B) followed the same day, also CPU-tested only
(the CPU cannot run fp16 autocast through accelerate; smoke item G3 checks it on the GPU). The Carousel mode (rotating
blocks, section 9, plan `docs/moss_ft_carousel_plan.md`) was added the same evening, CPU-tested only (smoke C1-C4).
The deterministic validation loss (`moss_ft_val_dataset_config`, section 10) was added 2026-11-18, CPU-tested only.

---

## 한국어 요약

- **무엇:** MOSS-Image Base DiT(5.08B)의 가중치 전체를 학습합니다(LoRA 가 아님).
  - 결과는 Base 와 같은 키 463개의 bf16 safetensors 입니다. ComfyUI 에서 그대로 읽힙니다.
- **실행:** `venv_t214\Scripts\python.exe moss_train.py --config_file <설정.toml>`. 프로세스 1개, GPU 1개만 됩니다.
- **기본 경로 `adamw_host`:**
  - fp32 master 와 AdamW m·v 는 호스트 RAM 에 둡니다. GPU 에는 bf16 계산 사본만 둡니다.
  - 마지막 마이크로스텝의 backward 안에서 CPU 가 AdamW 를 돌립니다.
- **시작 전에 닫을 것:**
  - ComfyUI(8188), 브라우저 등 RAM 을 많이 쓰는 프로그램을 닫습니다. ComfyUI 는 VRAM 도 차지합니다.
  - GA>1 이면 RAM 이 약 85 GiB(GA=1 이면 약 66 GiB) 필요하고, 여유 8 GiB 를 더 봅니다.
  - 모자라면 시작할 때 이유와 함께 거부합니다.
- **디스크:**
  - 학습 state 1개가 61 GB, DiT 저장 1개가 10 GB 입니다.
  - 시작할 때 이 런이 가장 많이 남기는 양을 계산해서, 빈 공간이 모자라면 거부합니다.
    - state: 남길 에폭·스텝 state + LR 단계 경계 state + 쓰는 중 1개.
    - DiT: 남길 에폭·스텝 저장 + 최종 1개.
  - 예시 설정은 state 4개(244 GB)입니다. DiT 저장은 `save_last_n_steps` 가 없으면 24개(244 GB)입니다.
    그래서 예시에 `save_last_n_steps = 2000` 을 넣었습니다(11개, 112 GB). 합계 약 356 GB 입니다.
  - 데이터셋(캐시가 이미지 옆에 생김), output_dir, state 폴더는 모두 D: 에 둡니다.
  - TE 캐시는 캡션 변형 수에 비례합니다. 변형은 1~3개만 쓰세요.
- **이어하기:** `--resume <state 폴더>` (output_dir 이 아니라 그 안의 state 폴더).
  - 에폭 끝 state 는 옵티마이저 상태·LR(어떤 스케줄러든)이 정확히 이어집니다.
  - 배치 순서까지 같으려면 `max_data_loader_n_workers = 0` 이어야 합니다(CPU 시험에서 비트 단위로 같음). 워커가 있으면 데이터 순서·증강이 달라집니다.
  - 에폭 중간 state 는 그 에폭을 새로 섞어서 남은 배치 수만큼 돕니다(kohya 와 같음).
  - 덜 쓰인 state 는 이유와 함께 거부합니다.
  - 같은 폴더에서 이전 state 로 되돌아가 다시 돌려도, 새로 쓴 state 는 지우지 않습니다. 버린 런의 더 뒤 state 는 새 런이 그 지점을 지날 때까지 남겨 둡니다.
- **DataLoader 워커:** 4개 이하로 둡니다. 워커 1개가 호스트 RAM 을 약 1 GiB 씁니다. kohya 기본값은 8개입니다.
- **라이브 TE (`moss_te_live = true`, 11-17 구현, CPU 시험만):** TE 캐시 없이 학습 중에 캡션을 인코딩합니다.
  - Qwen3.5-9B 디코더 32층(12.9 GiB)은 호스트 RAM 에 두고, 한 번에 1층씩 GPU 링(기본 2칸, 0.8 GiB)으로 보냅니다.
    임베딩 표(1.9 GiB)는 파일에서 CPU 로 바로 읽고 GPU 로 보내지 않습니다.
  - 다음 `moss_te_live_ahead` 마이크로배치(GA 배수로 올림)의 캡션을 한 패스에 모아 인코딩합니다. 가중치 전송 1번을
    그 캡션들이 나눠 씁니다. ahead 는 **마이크로배치 수**라서 한 패스의 캡션 수 = ahead x 배치입니다. 배치 4 면
    ahead 8 로 32 캡션(배치 1 의 ahead 32 와 같은 효율)이고, GPU 에 들고 있는 출력·latent(`te_held`)도
    ahead x 배치 x 토큰 수에 비례하므로 예시는 8 입니다.
  - DataLoader 워커가 있으면 `prefetch_factor` 를 ahead / 워커 수(올림)로 올려, 한 패스 분량을 워커가 미리 준비합니다
    (CPU 배치라 VRAM 은 그대로).
  - 캡션마다 캐시 경로와 똑같이 혼자, 패딩 없이 돌립니다. CPU 시험에서 캐시 런과 손실·저장 DiT 가 비트 단위로 같습니다.
  - **묶어 돌리기 `moss_te_live_batch = N` (11-18 구현, CPU 시험만, 선택):** 기본 1 은 위 그대로(캡션마다 혼자, 캐시와
    비트 단위로 같음)입니다. N > 1 이면 한 패스의 캡션을 길이순으로 정렬해 N 개 이하씩 고르게 묶고(32 캡션·N 8 이면
    8 x 4, 33·32 면 17 + 16), 묶음의 가장 긴 캡션 길이까지 오른쪽을 패딩해 층마다 묶음 하나를 한 번에 돌립니다.
    11-17 측정에서 패스 시간의 대부분이 캡션 1개씩 32층을 도는 작은 커널 호출이었기 때문입니다(Carousel 1024 b1 스텝의
    약 16 %).
    - 패딩은 각 캡션의 마지막 토큰을 반복합니다(실제 토큰이라 패딩 위치 값도 유한). 인과(causal) 어텐션·짧은 conv·
      gated delta rule 이 모두 뒤만 보므로 패딩은 진짜 토큰에 닿지 않습니다(같은 모양에 패딩 내용만 바꾸면 비트 단위로
      같음을 시험으로 확인). 다만 행렬 모양이 바뀌어 커널이 달라지므로 **비트 단위로 같지는 않고 가깝습니다**:
      CPU fp32 작은 모델에서 최대 차이 약 1e-5, CPU e2e 에서 손실 차이 약 5e-5 (상대). GPU 실측은
      `_chains\20261117_moss_ft_smoke\run_te.ps1`(te_batch_bench.py)로 잽니다.
    - VRAM: 한 번의 층 호출 중간값은 행 수 x 그 묶음의 가장 긴 길이에 비례합니다. 그래서 호출 하나는 행
      min(N, 캡션 수)개·패딩 포함 N x 300 토큰까지만 담습니다. **300 토큰을 넘는 캡션은 자기들끼리** 이 한도 안에서
      더 작게 묶이고(한도보다 긴 캡션 하나는 혼자), 나머지는 전과 똑같이 묶입니다. 시작 시 예산 줄은 이 호출을 셉니다
      (300 토큰 기준 행당 약 65 MiB: N 8 이면 패스 최대 약 1.5 GiB, 32 면 약 3.0 GiB). 캡션이 모두 300 토큰 이하이면
      어떤 패스든(로더 마지막의 짧은 패스 포함) 이 값 안입니다. 첫 학습 패스는 묶음 크기·패딩 길이·패딩 비율과
      그 모양의 패스 최대 추정(`ft_budget.te_pass_gpu_bytes`)을 한 번 로그로 남깁니다.
    - 11-18 리뷰 반영: 처음 구현은 행 수로만 묶어 가장 긴 캡션들이 첫(가장 큰) 묶음에 모였습니다. 스모크 서브셋 캡션
      32개(10~770 토큰)에서 N 32 는 32 x 770 한 번(패딩 포함 토큰 3.4배, 패스 약 6.1 GiB로 추정의 두 배)이었습니다.
      지금은 [8 x 770, 24 x 284] 두 번(토큰 1.8배, 약 2.4 GiB)이고 N 8 은 1.2배·약 1.4 GiB, N 16 은 1.4배·약 1.8 GiB 입니다.
      적당한 N 은 GPU 벤치 결과로 고릅니다.
    - `moss_te_live` 없이 쓰거나(캐시 경로는 캡션을 하나씩 한 번 인코딩하므로 묶을 것이 없음) 1 보다 작은 정수가 아니면
      거부합니다.
  - 매 스텝 새로 뽑히는 것: `shuffle_caption` + `keep_tokens_separator = "|||"`(HEAD 고정·TAGS 섞기·문장은 끝),
    `keep_tokens`, 여러 줄 캡션 + `enable_wildcard`(줄 하나를 무작위로 = 긴·중간·짧은 변형), `caption_dropout_rate`
    (빈 캡션 = 공식 무조건 프롬프트), `caption_tag_dropout_rate`(보호 태그는 유지), token warmup.
  - RAM: copy 모드에서 +13.9 GiB. adamw_host GA>1 이면 합계 약 99 GiB + 여유 8 GiB 라서 128 GB 에서 빠듯합니다.
    모자라면 `moss_te_live_host = "mmap"` 또는 GA 1 로 합니다. mmap 은 RAM 을 디스크 읽기와 바꾸는 것입니다:
    사적 RAM 은 거의 0 이지만 OS 파일 캐시가 층 12.9 GiB 를 계속 들고 있어야 하고, 못 들면 패스마다 약 13.8 GB 를
    디스크에서 읽습니다(패스당 ~7 s, 추정). 시작 시 남는 RAM 이 (필요량 + 12.9 GiB) 보다 적으면 그 시간을 붙여 경고합니다.
  - 고정(page-locked) 메모리 합계(링 + unsloth 체크포인트 입력 + TE 스테이징 + pinned 모드의 TE 층)를 시작 시 한 줄로
    보여 주고, Windows 에서 실제 RAM 의 50 % 를 넘으면 경고합니다(나중의 cudaHostAlloc 실패 예방).
  - 디스크: TE 캐시가 없습니다(12.5만 장 x 변형 3 이면 약 0.9 TB 를 아낍니다).
  - 속도: 이미지당 약 0.06 s(추정, 1024² b1 의 약 5 %). GPU 스모크 런에서 잽니다.
  - TE 캐시 플래그나 `cache_caption_variants` 와 같이 쓰면 거부합니다. 캐시 경로에서 셔플·드롭아웃·token warmup 을
    쓰면 `--moss_te_live` 를 안내하며 거부합니다. `|||` 가 캡션에 있는데 `keep_tokens_separator` 가 없으면 경고합니다.
  - 여러 줄 캡션 파일이 있는데 그 서브셋에 `enable_wildcard` 가 꺼져 있으면 경고합니다(모든 줄을 이어 붙여 한 프롬프트로
    인코딩됨 = 긴·중간·짧은 변형이 한꺼번에 들어가고 토큰도 VRAM 추정치 300 을 크게 넘음). `enable_wildcard` 를 켠
    서브셋의 캡션에 `{a|b}` 가 있으면 무작위로 하나만 남는다고 경고합니다.
  - 로그 `te/s_per_image` 는 런 전체의 패스 시간 / 인코딩한 캡션 수입니다(CUDA 에선 이벤트로 잰 GPU 시간). 패스 하나가
    여러 스텝을 먹이므로 스텝마다의 `te/pass_s` 는 패스가 든 스텝에만 값이 있습니다.
- **fp16 계산 (`mixed_precision = "fp16"`, 11-17 구현, CPU 시험만):** 조인트 Linear 는 fp16, 잔차 스트림과 나머지는 fp32 입니다.
  - `--moss_stream_fp32` 가 꼭 있어야 합니다(없으면 거부). 사용자 측정: fp16 + fp32 스트림 오차 18.6 %(cos 0.989),
    bf16 + bf16 스트림 82 %(cos 0.58). VRAM·RAM 은 bf16 과 같습니다.
  - 동적 loss scale: 2^15(32768)에서 시작, 그래디언트 오버플로가 나면 그 스텝에서 절반, 2,000 스텝 연속 정상이면 2배.
    항상 2의 거듭제곱이라, 호스트 AdamW 가 클립 계수에 1/S 를 곱해 한 번에 언스케일해도 값이 정확합니다
    (CPU 시험: 스케일한 런과 안 한 런이 비트 단위로 같음).
  - loss 자체가 inf/nan 인 스텝(forward 의 fp16 오버플로)은 스케일을 내리지 않습니다. loss 는 S 를 곱하기 전에
    계산되므로 S 와 무관하고, 내려 봐야 그래디언트 언더플로만 늘기 때문입니다. 그 스텝은 건너뛰고
    `--moss_ft_max_skipped_steps` 연속 횟수에만 셉니다. 진행 막대의 평균 loss 에는 넣지 않습니다.
  - `moss_ft_loss_scale_growth_interval = 0` 은 "키우지 않음"입니다(오버플로 때 절반으로 내리는 건 그대로).
  - 오버플로 스텝은 기본(`partial`)으로 그 파라미터만 건너뜁니다(나머지는 정상 갱신). 단 `exact` 클립 스텝은
    정책과 상관없이 스텝 전체를 건너뜁니다(모든 갱신이 backward 뒤라 가능하고, 남은 파라미터 norm 만으로 클립하면
    덜 클립되기 때문). `moss_ft_overflow_policy = "window"` 는 `exact` 클립에서 같은 동작을 명시합니다.
  - 학습 중 fp32 master 가 fp16 범위(65504)를 넘으면 GPU 가중치가 조용히 inf 가 되지 않도록 파라미터 이름과 함께
    멈춥니다(이전 state 에서 LR 을 낮춰 이어 가거나 bf16 으로).
  - 스케일과 카운터는 state 에 저장되어 이어하기가 정확합니다. bf16 state 로 fp16 런을 이어도 됩니다(초기 스케일부터).
    이어할 때 `moss_ft_loss_scale` 을 저장된 런과 다른 값으로 주면 스케일을 그 값으로 다시 시작합니다(무너진 스케일
    되돌리기). 같은 값이면 저장된 스케일 그대로입니다.
  - `adafactor_fused`·`full_fp16` 과는 같이 쓸 수 없습니다(거부). bf16 도 그대로 쓸 수 있습니다.
- **Carousel 모드 (`moss_carousel = true`, 11-17 구현, CPU 시험만, 9절):** LISA / BAdam 처럼 블록을 돌려 가며 학습합니다.
  - 이름은 임시입니다. `library/moss/ft_budget.py` 의 `CAROUSEL = "carousel"` 한 줄을 바꾸면 플래그(`--moss_<이름>_*`)·
    로그 키·메타데이터가 모두 따라 바뀝니다. state 파일 이름(`rotation.json` 등)은 고정이라 이름을 바꿔도 이어하기가 됩니다.
  - 학습 단위(unit) 44개: 텍스트 refiner 블록 2개 + joint 블록 42개(`rotate_stem` 이면 stem 을 더해 45개). 한 번에 K개만
    학습하고, `moss_carousel_period`(기본 50) 스텝마다 새 K개로 바꿉니다. 고르는 방법: `random`(기본) / `round_robin` /
    `window`. `random` 은 한 바퀴(cycle)마다 모든 unit 이 정확히 한 번씩 뽑힙니다. 바퀴 경계에서도 같은 unit 이 두 번
    연속 활성이 되지 않습니다(K ≤ unit 수의 절반일 때).
  - 활성 unit 의 fp32 master·fp32 기울기 누적·옵티마이저 상태는 **GPU** 에 둡니다. 스텝도 GPU 에서 돕니다.
    호스트에는 모든 블록의 fp32 master(18.9 GiB)와, `keep`(기본)이면 비활성 unit 의 옵티마이저 상태를 둡니다.
  - 옵티마이저: `optimizer_type = "AdamW"`(torch fused) / `"AdamW8bit"`(bitsandbytes, CUDA 전용, paged 는 거부) /
    `"CAME"`(ft_optim 의 CAME, bias_correction 기본 켬; 끄고 beta3 > 0.99 면 거부, `reset` 과 함께 끄면 거부).
  - 상태 정책: `keep`(기본, 비트 그대로 왕복) 또는 `reset`(LISA 식으로 들어올 때마다 새로 시작).
    CAME 만 `moss_carousel_state_dtype = "bf16"` 으로 호스트의 exp_avg 를 bf16 으로 줄일 수 있습니다(손실 있음).
    `reset` 은 들어올 때마다 t=1 부터라, CAME 의 bias_correction 을 끄면 초기 증폭이 교체마다 되풀이됩니다. 그래서 거부합니다.
  - 항상 학습하는 작은 그룹 `moss_carousel_always`:
    - `small`(기본, 임베더·time MLP·공용 modulation·head, 99.1M): 역전파가 **매 스텝 42블록 전체**를 지나갑니다(가중치
      기울기는 활성 unit 만). 즉 "가장 낮은 활성 블록 아래로는 역전파 안 함"은 이 기본값에선 일어나지 않습니다.
    - `head`(19.3M): 가장 낮은 활성 블록 아래는 재계산도 역전파도 하지 않습니다. 대신 stem 은 끝까지 학습되지 않습니다.
    - `rotate_stem`(11-17 추가): head 는 항상, stem(79.8M)은 45번째 회전 unit 이 됩니다. stem 도 K/45 만큼 학습되고,
      stem 이 비활성인 동안에는 `head` 처럼 역전파가 잘립니다. random K=8 기준 추정 비용 127.5/168(small 134, head 126).
    - 기본값을 `small` 로 둔 것은 11-17 요구("임베더 등은 항상 학습, 기본 켬")를 따른 것입니다. 바꿀지는 사용자 결정입니다.
  - K: `moss_carousel_active` 로 직접 주거나, 안 주면 VRAM 예산(`moss_carousel_vram_gib`, 기본 30.5 GiB)으로 계산합니다.
    K 는 런 전체에 고정이라 **가장 큰 버킷·배치**로 정합니다. 한 unit 도 안 들어가면 시작할 때 거부하고, 메시지에
    `moss_carousel_active N`·unsloth·b1+GA 를 안내합니다(`active` 를 직접 주면 30.5 GiB 경고만 냅니다).
    - 활성화 추정: 11-17 스모크 실측(adamw_host, fp16 + fp32 스트림, 일반 체크포인트: b1 4.03, b4 11.63 GiB) 직선
      맞춤에 여유 0.5 GiB 를 더한 값을 씁니다. 같은 조건이고 설계 추정보다 작을 때만 씁니다(b4 에선 설계가 4.2 GiB 과대).
    - K 표(30.5 GiB, fp16 + fp32 스트림, small, 순서 adamw / adamw8bit / came):
      1024 b1 6 / 10 / 8 · 1024 b2 4 / 7 / 5 · 1024 b4 1 / 2 / 1 · 2048 b1 1 / 3 / 2 · 1536 b2 거부 / 2 / 1 ·
      unsloth 1024 b4 4 / 7 / 6 · unsloth 2048 b1 4 / 8 / 6. `rotate_stem` 은 0~1개 더 들어갑니다.
    - **권장: Carousel 은 b1 + GA.** 이미지/초가 b2·b4 와 거의 같고 K 가 가장 큽니다. 2048 이 섞인 데이터셋은
      `unsloth_offload_checkpointing` 을 쓰세요.
  - unit 하나당 GPU 추가: adamw 1.69 GiB, adamw8bit 1.06 GiB, came 1.27 GiB. 호스트 RAM(keep, 44개 모두 방문):
    master 18.9 + 상태 adamw 37.1 / adamw8bit 9.4 / came 18.6(bf16 이면 9.3) GiB + 프로세스 8 GiB (+ 라이브 TE copy 12.9 GiB).
    adamw keep 은 합계 약 77 GiB 로 adamw_host GA2 실측(RSS 94.5 GiB)보다 약 18 GiB 적습니다.
  - 속도: 시작 로그의 추정은 설계값(1.1 s/이미지 x 134/168)이라 1024 b1 GA4 에서 이미지당 0.95 s(TE ahead 32) /
    1.06 s(ahead 8) 로 나옵니다. 실측 기반 어림(g3 의 GPU 0.82 s/이미지 x 134/168 + TE 0.157 s + 교체 분할)은
    **이미지당 약 0.85 s** 로, adamw_host b4xGA2 실측 1.21 s/이미지보다 약 1.4배 빠릅니다. 대신 각 블록은 전체
    스텝의 K/44 만큼만 학습됩니다. 교체 1회는 adamw keep K=6 에서 약 1.9 s(P=50 이면 스텝당 약 0.04 s).
  - **라이브 TE 비용이 그대로 스텝 시간에 더해집니다.** adamw_host 에서는 호스트 대기 뒤에 숨었지만, Carousel 은 GPU 가
    병목입니다(실측 0.13~0.16 s/이미지 = 1024 b1 스텝의 15~20 %). b1 에서는 `moss_te_live_ahead = 32` 로 두세요(예시도 32).
    디스크 TE 캐시를 쓰면 이 시간과 호스트 RAM 12.9 GiB 가 없어지지만, 캡션 셔플·드롭아웃·와일드카드를 못 씁니다.
  - fp16 + 동적 loss scale 그대로 쓸 수 있습니다. 오버플로 스텝은 통째로 건너뜁니다. 클립은 항상 정확(exact)합니다.
  - **weight decay·클립의 실제 세기:** decay 는 활성일 때만 걸리므로 블록별 실제 decay 는 대략 K/44 배입니다(전체 FT 와
    같은 세기를 원하면 `weight_decay` 를 44/K 배로). 클립 norm 도 활성 집합만으로 재므로 전체 FT 보다 느슨합니다.
  - `moss_carousel_host_pinned = "masters"` 는 교체 바이트 중 master 부분(adamw keep 이면 약 1/3)만 빨라집니다(추정
    1.9 → 약 1.5 s). 19 GiB page-lock 은 Windows 에서 검증된 적이 없으니 기본값 `none` 을 권합니다.
  - 저장: DiT 는 모든 master 를 모아 기존 형식(bf16/fp16/float)으로. state 는 기존 폴더 + `rotation.json` +
    `rotation_optimizer.safetensors`. 에폭 끝 state 에서 이어하면 비트 단위로 같습니다(CPU 시험, 교체 경계 포함).
    마지막 스텝 뒤에는 교체하지 않습니다(최종 저장 직전의 쓸모없는 D2H/H2D 없음).
  - 이어하기: `moss_carousel_seed` 를 안 줬으면 저장된 seed 로 계속 고릅니다(`seed` 가 없으면 런마다 새로 뽑혀서,
    예전엔 이어하면 선택 순서가 바뀌었음). 직접 준 seed 가 저장값과 다르면 경고하고 새 값을 씁니다.
  - 로그: `carousel/lowest_active_block`(활성 joint 블록 중 가장 낮은 번호, 없으면 42), `carousel/grad_floor`(역전파가
    내려가는 가장 낮은 블록; `small` 이면 늘 0), `carousel/grad_depth`, `carousel/active_units`,
    `carousel/trainable_params`, `time/optimizer_step`, 교체 스텝에 `carousel/switch_s`·`switch_d2h_gib`·`switch_h2d_gib`.
    교체마다 나간/들어온 블록 한 줄.
  - OCGOptV2 는 상태가 AdamW 와 같은 8 B/param 이라 메모리 이득이 없어 넣지 않았습니다.
  - GPU 스모크 C1~C4(7절)는 GPU 사용 허락을 받은 뒤 합니다.
- **검증 손실 (`moss_ft_val_dataset_config`, 11-18 구현, CPU 시험만, 10절):** 학습에 안 쓰는 그림으로 매번 같은 조건의
  손실을 잽니다. 런끼리(옵티마이저·Carousel 설정 등) 비교하는 용도입니다.
  - 플래그: `moss_ft_val_dataset_config`(검증 그림의 데이터셋 TOML, 없으면 검증 안 함), `moss_ft_val_every_n_steps`
    (N 스텝마다, 기본 0 = 첫 스텝 전과 마지막 스텝 뒤만), `moss_ft_val_sigmas`(고정 노이즈 세기, 기본 8개
    0.0625, 0.1875, ..., 0.9375: [0, 1] 을 8등분한 칸의 가운데라 평균이 "sigma 를 고르게 뽑았을 때의 평균 손실"의
    근사), `moss_ft_val_seed`(노이즈 시드, 기본 1234).
  - 고정되는 것: 그림과 배치(TOML 의 batch_size 로 버킷마다, 이미지 키 순서; num_repeats·flip_aug·color_aug·
    random_crop·해상도 jitter 는 끔), 캡션(여러 줄이면 **첫 줄**, 와일드카드는 첫 선택지, `|||` 는 학습 캡션처럼 ", " 로
    이음, 셔플·태그 드롭아웃·캡션 드롭아웃 없음), TE 출력(시작할 때 한 번, 캡션마다 혼자 인코딩 = 캐시 경로와 같은 값,
    `moss_te_live_batch` 를 써도 혼자; TOML 옆 `<TOML 이름>_moss_val_te/` 에 저장해 다음 런은 읽기만), 노이즈(시드·
    파일 이름·sigma 로만 정해짐, 학습 RNG 안 씀), 손실 식(학습 손실과 같은 target·weighting·loss_type·CFM·마스크·
    min-SNR·loss_multiplier; sigma 뽑기·OT·antithetic·ip_noise_gamma 만 뺌).
  - 같은 모델을 두 번 검증하면 비트 단위로 같은 값입니다. 캐시 경로 런과 라이브 TE 런도 같은 값입니다(CPU 시험).
  - 학습에 영향 없음: `torch.no_grad()` 로 돌아서 adamw_host·Carousel 의 기울기 훅이 불리지 않고, 체크포인트·Carousel
    no_grad 래퍼는 그냥 지나갑니다. 옵티마이저 상태·loss scale·RNG(전부 저장 후 복원)·DataLoader 순서·라이브 TE 피드·
    샘플/저장 일정은 그대로입니다. CPU 시험에서 검증을 켠 런과 끈 런의 스텝별 손실·최종 DiT 가 비트 단위로 같습니다
    (캐시 경로·라이브 TE·Carousel).
  - 언제: 스텝 0(업데이트 전), N 스텝마다(옵티마이저 스텝과 그 로그 뒤, GA 창 밖), 마지막 스텝 뒤. 이어하기 런은 시작할
    때 검증하지 않고, 같은 폴더면 val_log.jsonl 에 이어 씁니다.
  - 출력: `<output_dir>/val_log.jsonl` 에 검증마다 한 줄 `{"step", "images_seen", "loss", "loss_by_sigma", "n",
    "seconds"}`(images_seen = 지금까지 학습에 실제로 쓴 이미지 수, state 에 저장돼 이어하기 때 이어짐), TensorBoard
    `val/loss`·`val/loss_s<sigma>`, 콘솔 한 줄 `VAL step N: loss X (n images, s s)`. 새 런(이어하기 아님)이 기존
    val_log.jsonl 을 만나면 `val_log.<시각>.jsonl` 로 옮겨 보관합니다(지우지 않음). 손실이 유한하지 않으면(fp16
    forward 오버플로) 그 값은 `null` 로 씁니다(NaN/Infinity 는 JSON 이 아니라 JS·jq 가 못 읽음). 읽는 쪽은 `null` 을
    빠진 점으로 다뤄야 합니다. TensorBoard 와 콘솔은 nan/inf 그대로입니다.
  - 비용: 검증 1번 = 배치 수 x sigma 수 번의 no-grad forward. 1024² forward 1번은 학습 마이크로스텝의 약 1/3(추정)이라
    32장 x 8 sigma 면 1분 남짓(추정, GPU 미측정). 16~64장 정도의 작은 세트를 권합니다.
  - VRAM: no-grad forward 는 블록 입력을 저장하지 않아 조인트 토큰당 약 0.29 MB 입니다(학습 마이크로스텝은 약 0.91 MB +
    0.5 GiB, fp32 스트림). 그래서 검증 배치의 토큰이 학습 배치의 약 3배를 넘을 때(더 큰 버킷·배치)만 예산 합계에 더하고,
    시작 줄에 더했는지 적습니다. Carousel 의 K 계산에도 같은 값이 들어갑니다. 호스트 RAM: 검증 캡션의 TE 출력(캡션당 약
    2.4 MB)을 CPU 에 둡니다. latent 는 학습과 같은 캐시(`cache_latents_to_disk` 면 그림 옆 npz)에서 매번 읽습니다.
  - 거부: 학습 TOML 안의 `validation_split`·`is_val`(별도 TOML 로), 검증 TOML 안의 `validation_split`, TOML 없이 준 검증
    플래그, 없는 파일, (0, 1] 밖이거나 중복된 sigma·64개 초과, 음수 N·seed, 캐시된 latent 가 없는 검증 그림. 검증 그림이
    학습 그림과 겹치면 경고만 합니다.
  - 주의: 손실 설정(loss_type, weighting_scheme, cfm_lambda, min_snr_gamma, loss_multiplier)이 다른 런끼리는 검증 손실
    식도 달라서 숫자를 비교하면 안 됩니다. 검증 세트·sigma·seed 도 같아야 비교됩니다.
- **먼저 할 것:** 설계 7절의 GPU 스모크 런. 사용자 허락을 받고, GPU 가 비어 있을 때 합니다.

---

## 1. Running it

```
cd <LoRA_Easy_Training_Scripts>\backend\sd_scripts_moss_full   (or the repository after the merge)
venv_t214\Scripts\python.exe moss_train.py --config_file <output_root>\<run>\moss_fft.toml
```

- **Processes:** one process, one GPU. `accelerate launch --num_processes 1` should work too (not tried); more processes
  are refused.
- **Interpreter:** use `venv_t214`. Run it from the repository folder so that `library` is imported from there.
- **Launch:** start it the way the user starts long trainings (a new PowerShell window via WMI) and read the log from files.
- **`--help` when redirected:** set `PYTHONIOENCODING=utf-8`. kohya's Japanese help strings do not fit the cp949
  console encoding; `anima_train.py` has the same problem.
- **`--show_timesteps console` (or `image`):** prints the timestep distribution on the CPU and exits, as with
  `moss_train_network.py`. Nothing is cached or loaded.

**Files it needs:**
- `pretrained_model_name_or_path`: `moss_image_base_bf16.safetensors`. A heal-layout file with fused q/k/v also loads;
  the saves use the split Base layout.
- `vae`: the MOSS VAE (fp32).
- `moss_text_encoder`: the Qwen3.5-9B text encoder. On the cached path it is loaded only while the text-encoder caches
  are written, and never when they are complete. With `moss_te_live` its decoder layers stay in host RAM for the whole
  run (see "Live text encoder" below).
- `moss_tokenizer_dir`: the folder with the tokenizer.

**Dataset TOML:**
- Bucket sizes must be multiples of 16 (`bucket_reso_steps` 64 is fine). `batch_size` is set per dataset.
- The caches (`*_moss.npz`, and `*_moss_te.npz` on the cached path) are written next to the images, so keep the images
  on D:.
- Above 2,000 images both `*_to_disk` cache flags are required (only `cache_latents_to_disk` with `moss_te_live`).

**Start-up order:**
1. The arguments are checked.
2. The dataset is read. The LR scheduler is built once on a stand-in optimizer, so a setting kohya's
   `get_scheduler_fix` refuses fails here. The caption settings are checked against the text path. The GPU / host RAM /
   disk budget is logged and checked.
3. The latents are cached. Cached path: the text-encoder outputs are cached and the text encoder is released. Live
   path: the encoder's layers are read into host RAM (`moss_te_live_host`) and the sample prompts are encoded.
4. Only then are the host pools allocated and the DiT loaded.

**Live text encoder (`moss_te_live = true`, plan `docs/moss_ft_te_live_fp16_plan.md` part A):**
- **What it does:** the DataLoader runs kohya's caption processing and the tokenizer at every fetch. The main process
  encodes the captions of the next `moss_te_live_ahead` (32) micro-batches in one pass, rounded up to a multiple of
  GA, so a pass always starts a window. A pass streams the 32 decoder layers to the GPU one at a time through
  `moss_te_live_ring_slots` (2) slots: layer l + 1 uploads while layer l runs every caption of the pass. Each caption
  runs alone and unpadded, as the cached path encodes it, so the outputs are the cached outputs (bit for bit on the
  CPU; the same check on the GPU is smoke item G1).
- **Host forms (`moss_te_live_host`):** `copy` (default: 12.9 GiB private + 1 GiB pinned staging), `pinned` (12.9 GiB
  page-locked in place, no staging, CUDA only), `mmap` (no private copy; it trades RAM for disk reads: the layers come
  from the OS file cache, which then wants ~13.8 GB, else each pass reads them from the disk, ~7 s per pass at 2 GB/s,
  an estimate; the start-up check warns when the available RAM cannot hold them next to the run).
- **`moss_te_live_ahead` counts micro-batches:** a pass encodes ahead x batch captions and keeps their outputs and
  latents on the GPU until their micro-steps (`te_held` scales with ahead x batch x text tokens). Batch 4 with ahead 8
  is 32 captions per pass, the same amortization as batch 1 with ahead 32.
- **Batched layer calls (`moss_te_live_batch`, default 1; added 11-18, CPU-tested only):** 1 is the path above (every
  caption alone, bit for bit the cached outputs). N > 1 sorts the pass's captions by length, cuts them into
  ceil(captions / N) groups of as even a size as possible (32 at N = 8: 4 x 8; 33 at 32: 17 + 16), right-pads each
  group to its longest caption with that caption's own last token, and runs every layer once per group instead of once
  per caption (the 11-17 GPU measurement: the per-caption pass was ~0.14 s per caption, ~16 % of a Carousel 1024 b1
  step, mostly small kernels of the pure-torch gated delta rule). The padding cannot reach a real token (causal
  attention, a causal short conv and a causal chunked recurrence; nothing normalizes across the sequence; the pad is a
  real token, so its activations stay finite; the proof is in `library/moss/te_live.py`'s header and
  `tests/test_moss_ft_te_live_batch.py`), but other matrix shapes pick other kernels: the rows are close to, not
  bit-identical with, the cached rows (fp32 CPU tiny model: max abs ~1e-5; CPU e2e: losses within ~5e-5 relative).
  VRAM: a layer call's temporaries scale with its rows x its longest caption, so a call holds at most min(N, captions)
  rows and N x 300 padded tokens: captions past 300 tokens are grouped among themselves within that budget (one past
  it alone), the rest exactly as before. The start-up budget line charges that call (~65 MiB per 300-token row: a pass
  peaks at ~1.5 GiB at N = 8, ~3.0 GiB at N = 32 instead of ~1.0 GiB), and with every caption at most 300 tokens any
  pass stays within it (shorter passes at the loader's end included); the first training pass logs its group sizes,
  padded lengths, padding share and the pass peak of those shapes (`ft_budget.te_pass_gpu_bytes`) once. Reviewer
  11-18: the first version grouped by row count alone, which put the longest captions into the first and largest
  group; the smoke subset's first 32 captions (10-770 tokens) became one call of 32 x 770 at N = 32 (3.4x the tokens,
  a ~6.1 GiB pass, twice the estimate). Now [8 x 770, 24 x 284] (1.8x, ~2.4 GiB); N = 8: 1.2x, ~1.4 GiB; N = 16:
  1.4x, ~1.8 GiB. Refused without `moss_te_live` (the TE
  cache encodes each caption once, alone) and below 1. The GPU speed / VRAM / closeness check is
  `_chains\20261117_moss_ft_smoke\run_te.ps1` (`te_batch_bench.py`: batch 1 vs 8, 16, 32 on the real encoder).
- **Captions per fetch:** `shuffle_caption` with `keep_tokens_separator = "|||"` for `HEAD ||| TAGS ||| SENTENCES`
  (HEAD stays first, TAGS are shuffled and tag-dropped, SENTENCES stay last; kohya splits at the separator only with
  shuffle, tag dropout or token warmup on), `keep_tokens` for plain tag lists, multi-line caption files with
  `enable_wildcard = true` (one line per fetch: long / medium / short variants), `caption_dropout_rate` (the caption
  becomes `""`, the official unconditional prompt), `caption_dropout_every_n_epochs`, `caption_tag_dropout_rate` with
  protected tags, `token_warmup_step`.
- **Refused:** `cache_text_encoder_outputs(_to_disk)` or `cache_caption_variants` with `moss_te_live`; the
  `moss_te_live_*` settings (`moss_te_live_batch` included) without `moss_te_live`; `pinned` without CUDA. On the cached path, `shuffle_caption`,
  `caption_tag_dropout_rate` and `caption_dropout_rate` without `cache_caption_variants` > 1, `token_warmup_step` and
  `caption_dropout_every_n_epochs` are refused with a pointer to `moss_te_live`.
- **Warned:** captions with `|||` while `keep_tokens_separator` is unset, or with the separator set but no shuffle / tag
  dropout / warmup (the encoder would see a literal `|||`); multi-line caption files in a subset with
  `enable_wildcard` off (every line joined into one prompt, far past the ~300 text tokens the VRAM estimate assumes);
  `{...}` in captions of a subset with `enable_wildcard` on (each `{a|b}` becomes one random option).
- **Workers:** with `max_data_loader_n_workers = 0` the dataset's random draws happen in the main process; pulling
  `ahead` batches early interleaves them differently with the training RNG (valid, not bit-identical to a cached run
  with random captions). The DataLoader workers never get the encoder. With workers, the DataLoader's
  `prefetch_factor` is ceil(ahead / workers) (at least 2), so the workers have a whole pass ready when it starts.

## 2. Example config (design 8.1 with the live text encoder; the values are to be confirmed by the smoke run)

```toml
pretrained_model_name_or_path = "<forge>/models/Stable-diffusion/moss_image_base_bf16.safetensors"
vae = "<forge>/models/VAE/moss_image_vae_fp32.safetensors"
moss_text_encoder = "<forge>/models/text_encoder/qwen3.5_9b_moss_image_sft_bf16.safetensors"
moss_tokenizer_dir = "<forge>/extensions/forge_sam3_extension/assets/qwen35_tokenizer"
dataset_config = "D:/<dataset>/moss_fft_1024.toml"   # resolution 1024, batch_size 4, bucket_reso_steps 64, latents next to images on D:
mixed_precision = "fp16"                               # fp16 joint Linears + the fp32 stream (bf16 also works)
moss_stream_fp32 = true                                # required with fp16 (and with bf16)
moss_ft_loss_scale = 32768                             # dynamic loss scale: start (x0.5 on a gradient overflow, x2 per 2,000 clean steps)
attn_mode = "flash"
gradient_checkpointing = true
gradient_accumulation_steps = 2
optimizer_type = "AdamW"                               # -> adamw_host
optimizer_args = ["betas=(0.9, 0.999)", "eps=1e-8", "weight_decay=0.0"]
moss_ft_lr_phases = "2:5e-6,6:1e-6"
lr_warmup_steps = 100
max_grad_norm = 0.0                                    # first run: read grad_norm, then set ~2-3x its median
cache_latents = true
cache_latents_to_disk = true
moss_te_live = true                                    # no TE cache: captions encoded at train time
moss_te_live_ahead = 8                                 # micro-batches per encoder pass: 8 x batch 4 = 32 captions
moss_te_live_host = "copy"                             # "mmap" when host RAM is short (GA > 1 needs ~99 GiB + 8)
# in the dataset TOML ([[datasets.subsets]]): keep_tokens_separator = "|||", shuffle_caption = true,
# enable_wildcard = true (multi-line captions = variants), caption_dropout_rate = 0.05, caption_tag_dropout_rate = 0.1
save_precision = "bf16"
save_every_n_epochs = 1                                # 7 epoch DiT saves are kept (save_last_n_epochs would cap them)
save_state = true
save_last_n_epochs_state = 1
save_every_n_steps = 1000
save_last_n_steps = 2000                               # step DiT saves of the last 2,000 steps only (3 files, not 16)
save_last_n_steps_state = 1
output_dir = "<output_root>/<run>"
sample_prompts = "path/to/sample.txt"
sample_at_first = true
sample_every_n_epochs = 1
max_data_loader_n_workers = 3
persistent_data_loader_workers = true
seed = 1234
```

**What the key lines do:**
- `mixed_precision = "fp16"`: fp16 compute with a dynamic loss scale (section 6 logs `train/loss_scale`). bf16 is the
  fallback if the smoke run G3 shows trouble; switching between them on a resume is allowed (the scale restarts at
  `moss_ft_loss_scale` from a bf16 state and is ignored by a bf16 run).
- `moss_te_live`: the live text encoder (section 1). The cached alternative of design 8.1 replaces the three
  `moss_te_live*` lines with `cache_text_encoder_outputs = true`, `cache_text_encoder_outputs_to_disk = true` and
  `cache_caption_variants = 3` (then shuffle / dropout come only from the cached variants).
- `moss_ft_lr_phases`: 5e-6 for 2 epochs, then 1e-6 for 6 epochs.
  - It sets `learning_rate` and the run length. Leave `max_train_steps` out.
  - `lr_scheduler` must stay `constant` (or `constant_with_warmup`).
  - `lr_warmup_steps` must end inside phase 1.
  - `scale_resume` is refused with phases: its extra epochs would have no LR. Add a last phase instead.
- **Without phases** any kohya scheduler works (`lr_scheduler`, or `lr_scheduler_type` + `lr_scheduler_args`, the
  LoraEasyCustomOptimizer restart schedulers included). It is built at start-up step 2, and a resume continues it
  exactly. `zero_lr_warmup` is refused: kohya's wrapper cannot be resumed from a state. Use `lr_warmup_steps`.
- `save_last_n_steps` counts steps (kohya): with saves every 1,000 steps, 2000 keeps the last 3 step files.
- **Timestep distribution:** `timestep_sampling = "moss"` (the Base's pretraining) or the LoRA choice `"sigmoid"`. This
  is the user's recipe decision.
- **Optimizer modes:**

| `optimizer_type` | Mode | Use |
|---|---|---|
| `AdamW` (or empty) | `adamw_host` | the default: fp32 master, bias-corrected, keeps 1e-6 updates exactly |
| `CAME` | `came_host` | about 2.5x slower per step on this CPU. β3 > 0.99 needs `optimizer_args = ["bias_correction=True", ...]` (design 3.3) |
| `Adafactor` + `fused_backward_pass = true` | `adafactor_fused` | memory-light fallback: GA 1, stochastic rounding noise 2-9x the intended move at 1e-6 |
| `moss_carousel = true` + `AdamW` / `AdamW8bit` / `CAME` | `carousel` | rotating blocks with the optimizer on the GPU (section 9) |

## 3. Before a run: memory

**Host RAM (estimates, design 2.2):**

| Mode | Host RAM |
|---|---|
| `adamw_host`, GA > 1 | ~85 GiB |
| `adamw_host`, GA = 1 | ~66 GiB |
| `came_host` | ~66 GiB |
| `adafactor_fused` | ~8 GiB |
| + `moss_te_live`, host `copy` | +13.9 GiB (12.9 layers + 1 pinned staging): `adamw_host` GA > 1 ~99 GiB, GA 1 ~80 GiB |
| + `moss_te_live`, host `pinned` | +12.9 GiB page-locked |
| + `moss_te_live`, host `mmap` | +1 GiB staging (the layers come from the OS file cache, ~13.8 GB of it, or from the disk every pass) |

- The start-up check refuses when the RAM available is less than the estimate + 8 GiB. With the live encoder in `copy`
  mode at GA > 1 that is ~107 GiB on a 128 GB machine: tight with Windows and a desktop. The refusal then suggests
  `moss_te_live_host = "mmap"` (RAM traded for disk reads) or GA 1. With `mmap`, a warning gives the per-pass disk
  time when the available RAM is below the need + 12.9 GiB.
- **Page-locked memory:** a start-up line sums what is pinned (rings, unsloth checkpoint inputs, TE staging, the TE
  layers in `pinned` mode); on Windows a warning comes past 50 % of physical RAM, where a later `cudaHostAlloc` can
  fail mid-run.
- **Close RAM-heavy programs first:** ComfyUI with models loaded, browsers.
- `--moss_ft_skip_resource_check` turns the refusal into a warning. The estimates are still logged.
- **DataLoader workers:** the ~8 GiB "process" share covers up to 4 workers. Each worker is a spawned process that
  imports torch and this script again (about 1 GiB each), so each one above 4 adds 1 GiB and logs a warning. kohya's
  default is 8; 2-4 are enough with cached latents and text-encoder outputs.
- **`--unsloth_offload_checkpointing`** keeps every joint block's input in pinned host RAM. torch's pinned allocator
  rounds each block up to a power of two and keeps 42 blocks of every size class the run reaches, including a bucket's
  short last batch. At 1024² batch 4 that is 18.4 GiB of page-locked RAM, not the 8.45 GiB of design 2.2. The budget
  counts it that way.
- `--moss_ft_ring_mib` / `--moss_ft_upload_ring_mib` must be powers of two: the allocator would round them up anyway.
- On CUDA, `host/pinned_cache_gib` logs everything the pinned allocator holds (smoke run: compare with the estimate).

**GPU (estimates, design 2.1):**
- About 14.6 / 18.3 / 25.8 GiB allocated at 1024² batch 1 / 2 / 4 with `moss_stream_fp32`.
- **`moss_te_live`:** a pass needs ~1.0 GiB (the 2-slot ring 0.81 GiB + the rows' activations) but runs between
  micro-steps, when the DiT's activations are freed, so it does not raise the peak. What stays is the encoded text and
  latents of the batches waiting for their micro-step: ~0.14 GiB at 1024² batch 1 with `ahead` 32 (x batch size).
  With `moss_te_live_batch` N the pass adds the temporaries of its largest layer call instead of one caption's:
  ~1.5 GiB at N = 8, ~3.0 GiB at N = 32 (min(N, captions) rows of 300 tokens; a call never holds more than N x 300
  padded tokens - longer captions run in smaller groups of their own).
- About 2.3 GiB more is used outside the trainer (desktop, CUDA context).
- Close ComfyUI (it held up to 27 GB of VRAM before), and make sure no other training runs.
- There is no VRAM cap and no `empty_cache()` per step: an OOM surfaces. `--cuda_memory_fraction` and
  `--cuda_empty_cache_*` are ignored with a warning.

## 4. Disk

| Item | Size |
|---|---|
| Trained DiT (`<output_name>.safetensors`, `-NNNNNN`, `-stepNNNNNNNN`) | 10.16 GB each (bf16); 20.3 GB with `save_precision = "float"` |
| Training state, `adamw_host` | 61.0 GB |
| Training state, `came_host` | 40.7 GB |
| TE cache | 2.46 MB per image per caption variant (300 tokens): 16k images x 3 variants = ~118 GB, 125k x 3 = ~0.92 TB; none with `moss_te_live` |
| Latents at 1024² | 2 MiB per image (1 MiB with `cache_latents_dtype = "bf16"`); double with `flip_aug` |

**Planning the space:**
- **States at the peak:** the kept epoch states + the kept step states + the pinned LR-phase states + 1 being written.
  The example config keeps 1 epoch + 1 step + the pinned end of phase 1 + 1 being written = 4 x 61 = 244 GB.
- **DiT files at the peak:** the kept epoch saves + the kept step saves + the final file. kohya's `save_last_n_epochs`
  counts saves, `save_last_n_steps` counts steps. Without them the example keeps 7 epoch + 16 step + 1 final = 24 x
  10.16 = 244 GB; with `save_last_n_steps = 2000` it keeps 7 + 3 + 1 = 11 files (112 GB).
- **The start-up check** computes both from the run's length and save settings and refuses when the free space at
  `output_dir` / the state folder is less (summed when they are on one drive). The start-up log shows the counts.
  The example needs about 356 GB free on D:.
- Keep the dataset, `output_dir` and `--moss_ft_state_dir` on D:.

## 5. Outputs, states and resume

**In `output_dir`:**
- The final DiT `<output_name>.safetensors`.
- Epoch saves `<output_name>-NNNNNN.safetensors` (never the last epoch: the final file is that).
- Step saves `<output_name>-stepNNNNNNNN.safetensors`.
- Samples in `sample/`.
- `val_log.jsonl` with `moss_ft_val_dataset_config` (section 10).
- Old DiT saves are removed by kohya's `save_last_n_epochs` / `save_last_n_steps`.

**Training states** (`--moss_ft_state_dir`, default `output_dir`):
- Names: `<output_name>-NNNNNN-state` (epoch), `<output_name>-stepNNNNNNNN-state` (step), `<output_name>-state` (train end).
- Each state is written atomically: the folder only counts once its `COMPLETE.json` exists.
- **Keep policy:** the newest `save_last_n_epochs_state` epoch states and the newest `save_last_n_steps_state` step states
  are kept (both default to 1 with `save_state`). These count states, not steps.
  - The state just written is never deleted.
  - States numbered past the run's position belong to another run in the same folder, for example the run abandoned
    when phase 2 is re-run from its pinned state. They are left alone (logged once) until this run passes them.
- **Pinned LR-phase states:** with `save_state`, the epoch state at the end of every LR phase but the last is written
  even when `save_every_n_epochs` misses that epoch, and it is never deleted. Phase 2 can then be re-run from it.
  - A phase that ends inside an epoch (e.g. `"1.5:5e-6,..."`) has no epoch state to pin.
  - Without `save_state` (only `save_state_on_train_end`) nothing is pinned.
  - `--moss_ft_no_keep_phase_states` turns this off.
- **Each state's `master.safetensors`** is itself a loadable fp32 DiT. `param_groups.pt` holds the LR and what an LR
  scheduler keeps in the param groups.

**Resume:** `resume = "<output_root>/<run>/<name>-state"` (a state folder, not `output_dir` itself: that is refused with the
name of the newest state inside it).
- By default, the saved LR and scheduler state are restored. A changed CLI LR or schedule produces a warning with
  the changed values and the override flag. The existing `moss_ft_lr_phases` exception remains: it applies the current
  phase configuration at the saved scheduler step and warns when it differs.
- Opt in with `--moss_ft_resume_new_lr_schedule` (TOML: `moss_ft_resume_new_lr_schedule = true`) to start the current
  CLI schedule at its beginning, including warmup and restart cycles. This retains model weights, optimizer moments
  and parameter step counts, `global_step`, data position and loss scaler. The new scheduler's step zero is relative
  to the resume, while logging/save step numbers remain cumulative. For example, `--learning_rate 5e-6` with this
  flag starts a new schedule based on 5e-6; without the flag the saved LR is restored. This option also restarts an
  explicitly configured phase schedule from its beginning.
- **An epoch-end state** continues the optimizer state, the LR (any scheduler: the param groups are part of the state),
  the clipping counters and the noise exactly. A step state written at an epoch's last window counts as an epoch-end
  state.
  - The batches are the same only with `max_data_loader_n_workers = 0`; that case is tested bit for bit on the CPU.
  - With workers (the example config uses 3) each worker seeds its RNG when it starts, so the data order and the
    augmentation draws after a resume differ.
- **A mid-epoch state** skips the batches already trained in that epoch, drawn from a new shuffle of the epoch (kohya
  behavior).
- **Refusals:** a state without `COMPLETE.json` or with a truncated file is refused, and the message names the most
  recently written complete state next to it. So is an accelerate state, another optimizer mode, or another
  `--moss_ft_freeze`.
- **Warnings:** changed hyperparameters, GA or dataset size are logged as warnings.
- A resumed run does not sample at its start.

## 6. Logs

`global_step` counts accumulation windows that update at least one parameter. An all-overflow skip or a stale-clip
measurement window does not advance LR, Carousel periods, the progress bar, step logs, validation, step sampling or
step saves. Partial overflow with some updated parameters counts as one step. Loss-scale recovery and consumed
images/batches advance on every attempted window, so resume retains the correct data position. Epoch-end sampling
and saving still follow consumed epochs; the configured epoch budget is unchanged, so skips can leave fewer
successful updates than `max_train_steps` at train end.

DiT metadata `ss_moss_ft_epoch` is the one-based epoch containing the save, consistently for step, epoch-end and
final saves (including a final save inside an epoch). It is zero only before any batch has been consumed. Training
state `completed_epochs` remains a count of fully consumed epochs, with `epoch_step` locating an unfinished epoch.

TensorBoard via `log_with` / `logging_dir`.

**Every step:**
- `loss` (the window mean) and `lr/decay`, `lr/no_decay`: the LR this step used.
- `grad_norm`, `clip_coef`, `skipped_params`. A step that skipped parameters for non-finite gradients logs no
  `grad_norm`: its norm is not finite. The warning line gives the norm of the stepped part.
- `time/step`, `time/tail` (the CPU wait for the host step), `time/hook_wait`, `time/worker_busy`, `time/data`.
- `throughput/tok_per_s`, `throughput/img_per_s`.
- With `moss_te_live`: `te/pass_s` (host seconds of the encoder passes in this step's window, 0 in the steps between
  passes; not in `time/data`), `te/pass_images` (the captions those passes encoded) and `te/s_per_image` (the run's
  pass time per encoded caption; on CUDA the GPU time from events around each pass, read once they completed, since
  the host time there is mostly kernel launches), plus `te/gpu_s_total` on CUDA. The console line every
  `--moss_ft_log_every` steps adds "text encoder N s".
- With fp16: `train/loss_scale` (the scale this step's loss was multiplied by), `train/overflow` (1 when a gradient
  overflowed in fp16) and `train/overflow_windows` (the run's count). A step that overflowed skips the overflowed
  parameters (the whole step with exact clipping; an INFO line, not a warning) and the scale halves - unless the loss
  itself was not finite (a forward overflow the scale cannot cause): then the scale stays. A scale change is logged.
  `grad_norm` is always the unscaled norm. `--moss_ft_max_skipped_steps` then counts only overflows the scale cannot
  explain (a non-finite loss or an overflow at `--moss_ft_loss_scale_min`). A non-finite loss is left out of the
  progress bar's `avr_loss` and of `loss/epoch`.

**Every `--moss_ft_log_every` (10) steps:**
- `vram/peak_allocated`, `vram/peak_reserved`, `vram/device_peak`.
- `host/rss_gib`, `host/available_gib`, `host/pinned_gib` (the rings and the norm table, as the pinned allocator
  rounds them), and on CUDA `host/pinned_cache_gib` (everything the pinned allocator holds).
- The per-tier timing (`tier_time/*`), plus a console line with the same.

**At each validation** (`moss_ft_val_dataset_config`, section 10): `val/loss` and `val/loss_s<sigma>` at the step it
ran after (step 0 = before any update), one line in `val_log.jsonl` and the console line
`VAL step N: loss X (n images, s s)`.

**At start:** one line per budget item.

**Gradient clipping** (`--moss_ft_clip_mode stale`, the default):
- It uses the norm of the previous step that skipped nothing. After a step that skipped parameters, the next step
  keeps the last clean norm.
- **A step without a known previous norm** (the first one, or a resume from a state without one):
  - with GA > 1 it runs exact;
  - with GA 1 there is no accumulator, so it only measures its norm and changes no weights (a log line says so). An
    unclipped first step would put the unclipped gradient into AdamW's / CAME's moments and shrink the clipped steps
    after it.
- First run: `max_grad_norm = 0`, read `grad_norm`, then set it to 2-3x the median.

## 7. Smoke plan

Design section 7: about 60-70 minutes on the GPU. It needs the user's go-ahead, the GPU free, ComfyUI closed and no
other training.

| # | Run | Decides |
|---|---|---|
| 7.1 | PCIe probe | whether the final-step overlap is copy-bound |
| 7.2 | bf16 vs fp16 precision probe with the fp32 stream | whether fp16 compute (phase 1b) is needed |
| 7.3 | speed / memory at batch 1 x GA 8, 2 x 4, 4 x 2 | the run shape (largest batch with the device peak <= 30.5 GiB) |
| 7.4 | state save / resume on the GPU | state size, write and load times; GPU copies == master |
| 7.5 | DiT save + one sample | the file loads (later also in ComfyUI) |
| G1 | live vs cached text-encoder rows on the real weights (64 captions) | expected `torch.equal`; else the error next to bf16 rounding |
| G2 | live pass time (stage / H2D / compute; `te/s_per_image` from CUDA events) and peak VRAM, `reserved` before / after each pass, `ahead` 8 / 32, host copy / pinned / mmap | the TE share of a step (estimate ~0.06 s/image); whether a pass grows the allocator |
| G3 | 200 fp16 steps at 1024 b1 against a bf16 run | the scale trajectory, overflow steps, loss vs bf16, VRAM equal to bf16 |
| C1 | carousel adamw8bit: a real round trip (bitsandbytes on CUDA) and its first steps against bnb without switching (K = 44) | the 8-bit state codec on the real kernels |
| C2 | carousel VRAM peak at b1 / b4 for each kind (K from the budget), fp16 overflows, the I1 guard silent | the K formula of plan 3.8 |
| C3 | carousel switch time (pageable / `host_pinned masters`), step time, `head` + `window` vs `small` + `random` | the switch cost and the truncation speed-up |
| C4 | carousel 200 fp16 steps (fp32 stream, loss scale) for adamw and came | stability |

Until 7.3 is done, the batch / GA in the example config and every time estimate are unconfirmed.

## 8. Limits (phase 1)

- **Hardware:** one GPU, one process.
- **Precision:** `mixed_precision` is fp16 or bf16, both with `--moss_stream_fp32` (or `no` = fp32, a reference).
  fp16 needs a host optimizer mode (`adamw_host` / `came_host`) or `carousel`. Loss-scale flags without fp16 are refused.
  `--moss_ft_overflow_policy window` needs `--moss_ft_clip_mode exact` with `max_grad_norm > 0`.
- **Live text encoder:** one caption at a time per layer (the cached path's math); a padded batched layer forward
  would be faster but not bit-identical to the cache (later, after the smoke run).
- **Refused:** block compile (`--moss_block_compile`), block swap, RamTorch, DeepSpeed, `full_bf16` / `full_fp16`,
  kohya's validation sets inside the training TOML (`validation_split`, `is_val` subsets: the held-out set is its own
  TOML, `moss_ft_val_dataset_config`, section 10), `--cache_aug_refresh_epochs`, `--zero_lr_warmup`, and `--scale_resume` with
  `--moss_ft_lr_phases`. Each is refused with the reason. An LR scheduler setting that kohya refuses (e.g.
  `lr_warmup_steps` with `lr_scheduler = "constant"`) is refused before the caching starts.
- **`adafactor_fused`** has no per-parameter skip of non-finite gradients: a non-finite loss stops the run, because the
  weights already took that step.
- **Per-tier timing** runs always and reports every `--moss_ft_log_every` steps. The LoRA trainer's `--moss_tier_timing`
  flags are not used.

## 9. Carousel: rotating blocks (`moss_carousel`, plan `docs/moss_ft_carousel_plan.md`)

Working name: every flag, log key and metadata key is built from `ft_budget.CAROUSEL`; renaming the mode is that one
line plus the TOML keys. The state files keep stable names (`rotation.json`, `rotation_optimizer.safetensors`, format
`moss-ft-rotation/1`), so a state written under the old name still resumes.

**What it does:**
- 44 units in forward order: `text_embedder.blocks.0/1` (113.28M each) and `layers.0 .. 41` (113.25M each); a unit
  holds every trainable tensor of its block, its four norms included. `--moss_ft_freeze` still applies (a block with
  nothing trainable left is not a unit). With `moss_carousel_always = "rotate_stem"` the stem (video / text
  embedders, time MLP, shared modulation, 79.8M) is a 45th unit, first in forward order.
- K units are trainable at a time; every `moss_carousel_period` optimizer steps (default 50) a new set is chosen:
  `random` (default; every unit exactly once per cycle, also across a cycle boundary - the boundary set's new units
  avoid the carried and the previous set's units, so with K <= U/2 no unit is active in two consecutive sets; the set
  is a pure function of `moss_carousel_seed` (default `seed`) and the switch index - no RNG state, a resume needs only
  the step), `round_robin` (fixed sets spread over the depth) or `window` (K contiguous units). No switch follows the
  run's last step.
- Active units: fp32 master, fp32 gradient accumulator and optimizer state on the GPU; the step runs on the GPU after
  the window's last backward (unscale, non-finite check, exact global clip over the trainable tensors, step, the new
  master rounded into the compute copy). Inactive units: only the compute copy (as in the other modes), no weight
  gradients. Host: the fp32 master of every trained tensor, and (`keep`) the inactive units' optimizer states.
- Switch (between steps): outgoing units first (master and state to the host, memory freed), then incoming (master and
  state to the GPU; each compute copy checked against its master: invariant I1). fp32 round trips are exact.
- `moss_carousel_always`: `small` (default) also trains the embedders, time MLP, shared modulation and the head (99.1M)
  at every step; their gradients need the activation gradient at every block, so every step backpropagates through all
  42 blocks - the default does NOT truncate the backward (weight gradients are still only computed for the active
  units). `head` trains only the head (19.3M): nothing below the lowest active block is recomputed or backpropagated
  (the blocks there run under `torch.no_grad()`), but the stem is never trained. `rotate_stem` trains the head always
  and rotates the stem as a unit: it trains in K/45 of the steps, and while it is inactive the backward is truncated as
  with `head` (estimated cost at random K=8: 127.5/168 of full FT; `small` 134, `head` 126). `none`: only the units.
  The default stays `small` (the 11-17 requirement: the small groups trainable by default); switching it is the user's
  call.
- `moss_carousel_state`: `keep` (default; exact; CAME may store its exp_avg in bf16 with `moss_carousel_state_dtype =
  "bf16"`) or `reset` (LISA: a unit starts with a fresh state each time it becomes active).
- Optimizers (`optimizer_type`): `AdamW` (torch, fused), `AdamW8bit` (bitsandbytes, CUDA only, not paged), `CAME`
  (ft_optim's cores - came.py's math - on the GPU, `bias_correction` on by default; off with beta3 > 0.99 is refused,
  and off with `reset` is refused: every incoming unit restarts at t = 1, so the early-update amplification would repeat
  at every switch).
- Weight decay is decoupled and applied only on the steps a tensor is active, so a block's effective decay over the run
  is about K/44 of the same `weight_decay` in a full fine-tune (use 44/K times the value for the same strength). The
  clip norm is measured over the active set only, so a given `max_grad_norm` clips less than in a full fine-tune.
- K: `moss_carousel_active`, else from `moss_carousel_vram_gib` (default 30.5 GiB) and the start-up estimate:
  K = floor((G_alloc - weights - always group - optimizer temporaries - activations) / per-unit bytes). Per unit:
  adamw 1.69, adamw8bit 1.06, came 1.27 GiB. K is fixed for the run, so the largest bucket and batch decide it. K < 1 is
  refused at start-up with the ways out (batch 1 + GA, unsloth offload, adamw8bit / came, `head`, or
  `moss_carousel_active N`, which only warns past 30.5 GiB).
- Activations: the 11-17 smoke measured 4.03 GiB (1024 b1) and 11.63 GiB (b4) over the compute copies (adamw_host, fp16
  + fp32 stream, plain checkpointing); the design estimate was 4.2 GiB high at b4. Under those conditions the budget
  uses the linear fit + 0.5 GiB (`ft_budget.measured_activations`) when it is below the design estimate; the start-up
  line says which model it used. K at 30.5 GiB (fp16 + fp32 stream, `small`, adamw / adamw8bit / came): 1024 b1 6 / 10 /
  8, b2 4 / 7 / 5, b4 1 / 2 / 1; 2048 b1 1 / 3 / 2; 1536 b2 refused / 2 / 1; with unsloth 1024 b4 4 / 7 / 6, 2048 b1
  4 / 8 / 6; `rotate_stem` fits 0-1 more. **Use batch 1 + GA** (about the same images/s, the most active units); for a
  dataset that mixes in 2048, use `unsloth_offload_checkpointing`.
- `moss_carousel_host_pinned = "masters"` page-locks the master pool (faster switch copies; refused past 40 % of
  physical RAM in pinned total; CUDA only). It speeds up only the master part of a switch (adamw keep: about a third of
  the bytes; ~1.9 -> ~1.5 s estimated) and page-locking 19 GiB is untested on Windows: keep the default `none`.
- Live text encoder: the GPU is the bottleneck here, so each encoder pass adds to the step time (measured 0.13-0.16 s
  per image, 15-20 % of a 1024 b1 step; adamw_host hid it behind its host wait). At batch 1 use
  `moss_te_live_ahead = 32` (the example). The disk TE cache removes that time and 12.9 GiB of host RAM, but refuses
  caption shuffling, dropout and wildcards.
- Speed: the start-up estimate uses the design 1.1 s per 1024² image x 134/168: 0.95 s per image at b1 GA 4 with ahead
  32 (1.06 s with ahead 8). From the smoke measurements (g3: 0.82 s GPU per image) it is about 0.85 s per image,
  ~1.4x the measured adamw_host b4 x GA2 (1.21 s per image). A switch of K = 6 adamw keep units takes ~1.9 s (pageable),
  ~0.04 s per step at P = 50.
- fp16 with the dynamic loss scale works as in the host modes; an overflow skips the whole step. `--moss_ft_clip_mode`,
  `--moss_ft_overflow_policy` and the ring / thread flags are not used (an info line).
- Saves: the DiT from every master (the active ones flushed from the GPU first), in `save_precision`. Training states:
  the usual directory plus `rotation.json` (config, units, active set) and `rotation_optimizer.safetensors`
  (`<tensor>::<state key>` of every tensor with a state). A resume continues in the saved set and period position;
  without `moss_carousel_seed` it takes the saved seed (an unset `seed` is drawn anew per run, which used to change the
  selection on resume); changed K / period / selection / an explicitly different seed are warned about and the set
  follows the new settings. Another optimizer kind, another trainable set (`moss_ft_freeze`, or `rotate_stem` on / off)
  or a host-mode state is refused.
- Logs: `carousel/lowest_active_block` (the lowest active joint block, 42 when only refiners / the stem are active),
  `carousel/grad_floor` (the lowest block the backward reaches; always 0 with `small`), `carousel/grad_depth`,
  `carousel/active_units`, `carousel/trainable_params`, `time/optimizer_step`; at a switch `carousel/switch_s`,
  `switch_d2h_gib`, `switch_h2d_gib` and one line with the outgoing / incoming units.
- Not done (review 11-17): using the fp32 compute copy itself as the master of the tensors that are already fp32 under
  `moss_stream_fp32` (the always group, 0.37 GiB, and an active refiner, 0.42 GiB) - below one unit's bytes, so K
  rarely changes, and it would bypass the I1 check of the switch; skipping the fp32 accumulator at GA 1 without clipping
  - the fp16 gradients of the joint Linears still need an fp32 buffer for the exact unscale and the fp32 optimizer.

**Example config (to be confirmed by smoke C2 / C3):**

```toml
# carousel example
pretrained_model_name_or_path = "<forge>/models/Stable-diffusion/moss_image_base_bf16.safetensors"
vae = "<forge>/models/VAE/moss_image_vae_fp32.safetensors"
moss_text_encoder = "<forge>/models/text_encoder/qwen3.5_9b_moss_image_sft_bf16.safetensors"
moss_tokenizer_dir = "<forge>/extensions/forge_sam3_extension/assets/qwen35_tokenizer"
dataset_config = "D:/<dataset>/moss_fft_1024_b1.toml"  # batch_size 1: the largest K at about the same images/s
mixed_precision = "fp16"
moss_stream_fp32 = true
attn_mode = "flash"
gradient_checkpointing = true
gradient_accumulation_steps = 4
optimizer_type = "CAME"                                # or "AdamW" (fastest step), "AdamW8bit" (most units)
optimizer_args = ["betas=(0.9, 0.999, 0.9999)", "weight_decay=0.0"]   # bias_correction is on by default here
moss_carousel = true
moss_carousel_period = 50                              # optimizer steps between switches
moss_carousel_select = "random"
moss_carousel_always = "small"                         # "rotate_stem": stem as a unit + truncation; "head": stem frozen
moss_carousel_state = "keep"
moss_ft_lr_phases = "2:5e-6,6:1e-6"
lr_warmup_steps = 100
max_grad_norm = 0.0
cache_latents = true
cache_latents_to_disk = true
moss_te_live = true
moss_te_live_ahead = 32                                # batch 1: 32 captions per pass (the pass time adds to the step)
save_precision = "bf16"
save_state = true
save_every_n_steps = 1000
save_last_n_steps_state = 1
output_dir = "<output_root>/<run>"
max_data_loader_n_workers = 3
persistent_data_loader_workers = true
seed = 1234
```

**Trade-off:** each unit trains in K/44 of the steps. Per hour that is about the block-updates of adamw_host at b1,
with fresher gradients and a much faster always group; whether the quality per hour is better is the open empirical
question (LISA / BAdam report parity on LLMs; not measured for DiTs). Recommended: `small` with `random` (default);
`rotate_stem` when the backward truncation should apply while the stem still trains; `head` with `window` when speed
matters most and the stem may stay frozen.

## 10. Validation loss (`moss_ft_val_dataset_config`, `library/moss/ft_val.py`)

A loss on held-out images that only the weights change: the same images, captions, noise levels and noise at every
validation, so runs (optimizer, Carousel settings, LR) can be compared step by step. Added 11-18, CPU-tested only.

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `moss_ft_val_dataset_config` | none (no validation) | a kohya dataset TOML of the held-out images |
| `moss_ft_val_every_n_steps` | 0 | validate every N optimizer steps; 0 = only before the first step and after the last |
| `moss_ft_val_sigmas` | `0.0625,0.1875,...,0.9375` | the fixed flow-matching noise levels, in (0, 1] (a string, or a TOML array) |
| `moss_ft_val_seed` | 1234 | the noise seed |

The default sigmas are the midpoints of [0, 1] cut into 8 equal bins (k/16 for odd k, exact in fp32 and x1000): their
mean is the midpoint-rule estimate of the loss averaged over a uniform sigma, and the per-sigma values show where a run
gains or loses (low sigma = detail, high sigma = composition).

**What is fixed:**
- **Images and batches:** every image of the TOML once (`num_repeats` ignored), in the bucket `make_buckets` gives it,
  in batches of the TOML's `batch_size` within a bucket, in sorted image-key order (no shuffle, no seed dependence).
  `flip_aug`, `color_aug`, `gamma_aug`, `random_crop` and resolution jitter of that TOML are switched off (a log line
  lists them).
- **Latents:** the training set's cache (`cache_latents`, `cache_latents_to_disk`: `*_moss.npz` next to the images,
  written at start-up in the same VAE pass as the training latents), variant 0, never flipped. An image without cached
  latents is refused at start-up. With `cache_latents_to_disk` they are read from the npz at every validation (no host
  RAM held); without it they stay in memory like the training latents.
- **Captions:** kohya's `process_caption` without its draws: retain the entire multi-line caption when
  `enable_wildcard = false`; with `enable_wildcard`, select the first line and the first `{a|b}` option. Prefix / suffix, the `keep_tokens` /
  `keep_tokens_separator` split joined with ", " like the shuffled training captions (a `HEAD ||| TAGS ||| SENTENCES`
  caption reaches the encoder as `HEAD, TAGS, SENTENCES` in the file's order), no shuffle, no tag dropout, no caption
  dropout, no token warmup (`captions.canonical_caption` on the original caption). A log line lists the random caption
  settings of the TOML that are not applied.
- **Text-encoder outputs:** encoded once at start-up, each caption alone and unpadded (the cached path's encoding; the
  live encoder's pass runs with `batch_rows` 1 even under `moss_te_live_batch` N), kept on the CPU (~2.4 MB per
  300-token caption) and stored in `<val TOML stem>_moss_val_te/` next to the TOML, keyed by the text-encoder signature
  and the text (as the sample prompts). A cached-path run and a live run therefore read the same rows (the CPU test
  checks equal validation numbers), and a later run with the same TOML encodes nothing. On the cached path the 9B
  encoder loads only if a caption is missing there, before the training TE cache step (one load for both).
- **Noise:** image i at sigma s gets `torch.randn` on the CPU from a generator seeded with sha256(seed, the image's file
  name, s): independent of the batch, the order, the device and the training RNG. File names that repeat across
  folders get `#1`, `#2` appended (by image key).
- **Loss:** `ft_loss.moss_ft_val_loss_rows`, the training loss chain at timestep 1000 x s for every sample: target
  `noise - latents` in float64, `weighting_scheme`, `loss_type` / Huber schedule / `loss_scale`, CFM (within the fixed
  batch), masked loss, token mining, the loss weights, min-SNR and `loss_multiplier` - without the training draws (no
  sigma sampling, no cosine OT, no antithetic pairing, no `ip_noise_gamma`). With the same noise and sigma its batch
  mean equals the training micro-step's loss bit for bit (CPU test). Per sigma: the mean over the images; `loss` = the
  mean over the sigmas. Runs with different loss settings (`loss_type`, `weighting_scheme`, `cfm_lambda`,
  `min_snr_gamma`, `loss_multiplier`) have different validation formulas: compare runs that share them, the
  validation TOML, the sigmas and the seed.

**When:** before the first step (step 0, after the first sample round), after every N-th optimizer step - after the
step, its logs and the fp16 scale update, outside any accumulation window, before that step's samples and saves - and
after the last step. A resumed run does not validate at its start (as it does not sample) and appends to the
`val_log.jsonl` of its `output_dir`; a fresh run moves an existing non-empty `val_log.jsonl` aside to
`val_log.<time>.jsonl` (kept). `images_seen` counts the images of every micro-batch trained (the actual count: short
last batches and partial windows included); it is stored in the training state (`train_state.json`), so a resumed run
continues it (a state written before 11-18 gives an estimate from its epochs, logged).

**What it does not touch:** the validation runs under `torch.no_grad()` between optimizer steps, with the training
forward's autocast and stream dtype (`moss_stream_fp32`), and the model in train mode (the DiT has no dropout; under
no_grad the blocks skip checkpointing and the Carousel's no-grad / grad-input wrappers run the plain forward). No
autograd graph: the `adamw_host` / `came_host` / `adafactor_fused` / Carousel gradient hooks never fire, the optimizer
state, the loss scaler and the LR scheduler are not used. Python's, numpy's and torch's (CPU, and CUDA when
initialised) RNG states are saved and restored around every validation and around the start-up steps of the
validation set (kohya's dataset set-up draws from and reseeds `random`). The DataLoader, the live TE feed and its
queued batches, the Carousel's active set and switches, and the sample / save schedule are not used. The CPU tests run
each mode with and without validation: every step's loss, grad norm and LR, the TE feed's counters and the final DiT are
bit-identical. The time of a validation is not counted in the next step's `time/step` / `time/data`.

**Outputs:**
- `<output_dir>/val_log.jsonl`, one line per validation:
  `{"step": 2, "images_seen": 4, "loss": 1.5708, "loss_by_sigma": {"0.0625": ..., "0.9375": ...}, "n": 3, "seconds": 0.21}`
  (`n` = held-out images; `ft_val.read_val_log` keeps the last line per step). The file is strict JSON: a non-finite
  value (an fp16 forward overflow at a sigma, or diverged weights) is written as `null` - in that sigma's
  `loss_by_sigma` entry and in `loss` (the mean over the sigmas is then non-finite too) - never as `NaN` / `Infinity`,
  which JS `JSON.parse` and jq reject. A reader skips a `null` point.
- TensorBoard: `val/loss` and `val/loss_s<sigma>` (e.g. `val/loss_s0.0625`) at that step.
- Console: `VAL step N: loss X (n images, s s)`; a warning when a sigma's loss is not finite (an fp16 forward overflow).

**Cost (estimate, not measured on the GPU):** a validation is (batches x sigmas) no-grad forwards; one 1024^2 forward is
about a third of a training micro-step, so 32 images x 8 sigmas at batch 1 is roughly a minute. Use a small held-out
set (16-64 images) and N of a few hundred steps; fewer sigmas (`moss_ft_val_sigmas`) cut the time proportionally.

**Memory (budget):**
- GPU: a no-grad forward saves nothing for a backward and checkpoints nothing, so per joint token it holds a block's
  input and output stream (2 x 3072 values, fp32 with `moss_stream_fp32`), one block's forward temporaries (at most the
  recompute's 264.6 KB/token) and the misc per token: ~0.29 MB, against ~0.91 MB per token + 0.5 GiB of gradients in
  flight for the training micro-step. It runs between steps, when the training activations are freed (the live TE's
  held batches stay and are counted). So the start-up budget adds it only when it is larger than the training peak -
  in practice when a validation batch has ~3x the tokens of the largest training batch (a larger bucket or batch size
  in the validation TOML; with `unsloth_offload_checkpointing` less, since the block inputs are then on the host) - and
  logs one line saying whether it was added. The Carousel's K from the budget uses the same maximum.
- Host: the held-out captions' text outputs (~2.4 MB each) and, without `cache_latents_to_disk`, their latents are
  added to the in-memory caches line; their latent npz files to write are added to the disk estimate.

**Refused (start-up):** the `moss_ft_val_*` flags without `moss_ft_val_dataset_config`; a path that is not a file;
N < 0; sigmas outside (0, 1], repeated, more than 64 or printing alike; a seed outside [0, 2^63); `validation_split` in
the validation TOML; a validation TOML without images, with bucket sizes not in steps of 16 or with uncacheable latents;
held-out images without cached latents. kohya's `validation_split` / `is_val` subsets in the training TOML stay
refused with a pointer to `moss_ft_val_dataset_config`. Warned: the validation TOML is the training TOML, or held-out
images are also training images.

**Example:**

```toml
moss_ft_val_dataset_config = "D:/<dataset>/moss_val_1024.toml"   # 32 held-out images, batch_size 1, latents on D:
moss_ft_val_every_n_steps = 250
# moss_ft_val_sigmas = "0.0625,0.1875,0.3125,0.4375,0.5625,0.6875,0.8125,0.9375"   (the default)
# moss_ft_val_seed = 1234                                                          (the default)
```

Tests: `tests/test_moss_ft_val.py` (flags, the fixed plan / captions / noise, the loss against the training formula,
determinism, no side effects with the host pipeline and the Carousel wrappers, the jsonl format, the schedule, the
budget) and `tests/test_moss_ft_val_e2e.py` (cached path, live TE, Carousel, resume; with and without validation).
